"""Causal structural effect model for exp6.6.

Architecture C: F(S,a) → structural effects, O(effects) → future value

The effect model predicts objective-independent structural consequences:
  Δn_components, Δredundancy, Δhub_load, Δspectral_gap

The objective evaluator maps effects to future value using the
ObjectiveSpec. This separation should enable cross-mechanism
generalization: the effect model learns physics, the evaluator
applies the objective.

Also includes:
  Architecture A: ScalarResidualModel (F(S,a) → R)
  Architecture B: ObjectiveConditionedModel (F(S,a,O) → R)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_5.observable_features import (
    extract_observable_features, OBSERVABLE_FEATURE_DIM,
    _compute_degree_stats, _compute_spectral_gap,
)
from ..exp6_4.structural_features import compute_component_info
from ..exp6_3.exact_mpc import apply_action
from .objective_spec import (
    ObjectiveSpec, encode_objective, OBJECTIVE_ENCODING_DIM,
    OBSERVABLE_NAMES,
)


def _get_current_observable_exp66(graph: GraphBuffers, spec: ObjectiveSpec) -> float:
    """Get the current value of the observable for the objective spec.

    Used for correct O(S+ΔS) - O(S) evaluation.
    """
    n = int(graph.num_nodes)

    if spec.observable == "n_components":
        return float(compute_component_info(graph, n).n_components)
    elif spec.observable == "redundancy":
        _, _, _, _, min_deg = _compute_degree_stats(graph, n)
        return float(min_deg)
    elif spec.observable == "hub_load":
        _, _, max_deg, _, _ = _compute_degree_stats(graph, n)
        return float(max_deg)
    elif spec.observable == "spectral_gap":
        return float(_compute_spectral_gap(graph, n))
    else:
        return 0.0


@dataclass
class StructuralEffect:
    """Predicted structural consequences of an action."""
    delta_n_components: float = 0.0
    delta_redundancy: float = 0.0  # change in min degree
    delta_hub_load: float = 0.0   # change in max degree (negative = less hub)
    delta_spectral_gap: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.delta_n_components,
            self.delta_redundancy,
            self.delta_hub_load,
            self.delta_spectral_gap,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "StructuralEffect":
        return cls(
            delta_n_components=float(arr[0]),
            delta_redundancy=float(arr[1]),
            delta_hub_load=float(arr[2]),
            delta_spectral_gap=float(arr[3]),
        )


def compute_effect_labels(
    graph: GraphBuffers, z: torch.Tensor,
    action: tuple[str, int, int, dict],
) -> StructuralEffect:
    """Compute exact structural effect labels for training.

    These are objective-independent: they measure how the action
    changes structural properties regardless of what objective
    is being optimized.
    """
    n = int(graph.num_nodes)
    comp_before = compute_component_info(graph, n)
    _, _, _, max_deg_before, min_deg_before = _compute_degree_stats(graph, n)
    spec_before = _compute_spectral_gap(graph, n)

    next_graph = apply_action(graph, action)
    comp_after = compute_component_info(next_graph, n)
    _, _, _, max_deg_after, min_deg_after = _compute_degree_stats(next_graph, n)
    spec_after = _compute_spectral_gap(next_graph, n)

    return StructuralEffect(
        delta_n_components=float(comp_after.n_components - comp_before.n_components),
        delta_redundancy=float(min_deg_after - min_deg_before),
        delta_hub_load=float(max_deg_after - max_deg_before),
        delta_spectral_gap=float(spec_after - spec_before),
    )


# ---------------------------------------------------------------------------
# Architecture A: Scalar residual model (baseline, no objective info)
# ---------------------------------------------------------------------------

class ScalarResidualModel:
    """Architecture A: F(S,a) → R.

    Same as exp6.4/6.5 scalar MLP. No objective information.
    """

    def __init__(self, hidden_dim: int = 64, n_epochs: int = 300, lr: float = 0.01) -> None:
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self._W1 = None
        self._b1 = None
        self._W2 = None
        self._b2 = 0.0
        self._fitted = False

    def fit(self, X: np.ndarray, y_residual: np.ndarray, **kwargs) -> None:
        rng = np.random.RandomState(42)
        n_feat = X.shape[1]
        self._W1 = rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(1)

        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            pred = (h @ self._W2 + self._b2).flatten()
            err = pred - y_residual
            grad_out = err.reshape(-1, 1) / len(y_residual)
            grad_W2 = h.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ self._W2.T
            grad_h[h <= 0] = 0
            grad_W1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)
            self._W1 -= self.lr * grad_W1
            self._b1 -= self.lr * grad_b1
            self._W2 -= self.lr * grad_W2
            self._b2 -= self.lr * grad_b2
        self._fitted = True

    def predict_residual(self, graph: GraphBuffers, z: torch.Tensor,
                         action: tuple[str, int, int, dict],
                         *, threshold: int = 1, horizon: int = 2,
                         objective: ObjectiveSpec | None = None) -> float:
        if not self._fitted:
            return 0.0
        x = extract_observable_features(graph, z, action, threshold=threshold, horizon=horizon)
        h = np.maximum(0, x @ self._W1 + self._b1)
        return float((h @ self._W2 + self._b2).item())

    @property
    def name(self) -> str:
        return "A_scalar"


# ---------------------------------------------------------------------------
# Architecture B: Objective-conditioned scalar model
# ---------------------------------------------------------------------------

class ObjectiveConditionedModel:
    """Architecture B: F(S,a,O) → R.

    Concatenates the objective encoding with state/action features.
    The model sees what objective is being optimized.
    """

    def __init__(self, hidden_dim: int = 80, n_epochs: int = 300, lr: float = 0.01) -> None:
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self._W1 = None
        self._b1 = None
        self._W2 = None
        self._b2 = 0.0
        self._fitted = False

    def _make_features(self, graph, z, action, objective, threshold, horizon) -> np.ndarray:
        x_state = extract_observable_features(graph, z, action, threshold=threshold, horizon=horizon)
        x_obj = encode_objective(objective) if objective else np.zeros(OBJECTIVE_ENCODING_DIM)
        return np.concatenate([x_state, x_obj])

    def fit(self, X: np.ndarray, y_residual: np.ndarray, **kwargs) -> None:
        """X should already include objective encoding concatenated."""
        rng = np.random.RandomState(42)
        n_feat = X.shape[1]
        self._W1 = rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(1)

        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            pred = (h @ self._W2 + self._b2).flatten()
            err = pred - y_residual
            grad_out = err.reshape(-1, 1) / len(y_residual)
            grad_W2 = h.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ self._W2.T
            grad_h[h <= 0] = 0
            grad_W1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)
            self._W1 -= self.lr * grad_W1
            self._b1 -= self.lr * grad_b1
            self._W2 -= self.lr * grad_W2
            self._b2 -= self.lr * grad_b2
        self._fitted = True

    def predict_residual(self, graph: GraphBuffers, z: torch.Tensor,
                         action: tuple[str, int, int, dict],
                         *, threshold: int = 1, horizon: int = 2,
                         objective: ObjectiveSpec | None = None) -> float:
        if not self._fitted:
            return 0.0
        x = self._make_features(graph, z, action, objective, threshold, horizon)
        h = np.maximum(0, x @ self._W1 + self._b1)
        return float((h @ self._W2 + self._b2).item())

    @property
    def name(self) -> str:
        return "B_objective_conditioned"


# ---------------------------------------------------------------------------
# Architecture C: Causal effect model + objective evaluator
# ---------------------------------------------------------------------------

class CausalEffectModel:
    """Architecture C: F(S,a) → structural effects, O(effects) → R.

    The effect model has REAL supervised heads, each trained on
    actual structural effect labels (not the future residual).

    The objective evaluator maps predicted effects to future value
    using the ObjectiveSpec.
    """

    def __init__(self, hidden_dim: int = 64, n_epochs: int = 300, lr: float = 0.01) -> None:
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        # Shared trunk.
        self._W1 = None
        self._b1 = None
        # 4 supervised heads: components, redundancy, hub_load, spectral_gap.
        self._W_heads = None  # (hidden, 4)
        self._b_heads = None  # (4,)
        self._fitted = False

    def fit(self, X: np.ndarray, y_residual: np.ndarray | None = None,
            y_effects: np.ndarray | None = None, **kwargs) -> None:
        """Fit the effect model on structural effect labels.

        y_effects: (n_samples, 4) array of [delta_comp, delta_red, delta_hub, delta_spec]
        """
        if y_effects is None:
            raise ValueError("CausalEffectModel requires y_effects (structural effect labels)")

        rng = np.random.RandomState(42)
        n_feat = X.shape[1]
        self._W1 = rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
        self._b1 = np.zeros(self.hidden_dim)
        self._W_heads = rng.randn(self.hidden_dim, 4) * np.sqrt(2.0 / self.hidden_dim)
        self._b_heads = np.zeros(4)

        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            pred_effects = h @ self._W_heads + self._b_heads  # (n, 4)

            # Multi-head regression loss on effect labels.
            err = pred_effects - y_effects  # (n, 4)
            grad_out = err / len(y_effects)  # (n, 4)

            grad_W_heads = h.T @ grad_out  # (hidden, 4)
            grad_b_heads = grad_out.sum(axis=0)  # (4,)
            grad_h = grad_out @ self._W_heads.T  # (n, hidden)
            grad_h[h <= 0] = 0
            grad_W1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)

            self._W1 -= self.lr * grad_W1
            self._b1 -= self.lr * grad_b1
            self._W_heads -= self.lr * grad_W_heads
            self._b_heads -= self.lr * grad_b_heads

        self._fitted = True

    def predict_effects(self, graph: GraphBuffers, z: torch.Tensor,
                        action: tuple[str, int, int, dict],
                        *, threshold: int = 1, horizon: int = 2) -> StructuralEffect:
        """Predict structural effects (objective-independent)."""
        if not self._fitted:
            return StructuralEffect()
        x = extract_observable_features(graph, z, action, threshold=threshold, horizon=horizon)
        h = np.maximum(0, x @ self._W1 + self._b1)
        effects = h @ self._W_heads + self._b_heads
        return StructuralEffect.from_array(effects)

    def predict_residual(self, graph: GraphBuffers, z: torch.Tensor,
                         action: tuple[str, int, int, dict],
                         *, threshold: int = 1, horizon: int = 2,
                         objective: ObjectiveSpec | None = None) -> float:
        """Predict future residual via effects → objective evaluation."""
        if not self._fitted:
            return 0.0
        effects = self.predict_effects(graph, z, action, threshold=threshold, horizon=horizon)
        if objective is None:
            return 0.0
        # Compute current observable value for absolute-state evaluation.
        current_value = _get_current_observable_exp66(graph, objective)
        return ObjectiveEvaluator.evaluate(effects, objective, current_value=current_value)

    @property
    def name(self) -> str:
        return "C_causal_effect"


class ObjectiveEvaluator:
    """Maps predicted structural effects to future value under an objective.

    This is the O(effects) → R step in architecture C.
    It uses the ObjectiveSpec to determine which effect matters
    and how to convert it to value.
    """

    @staticmethod
    def evaluate(effects: StructuralEffect, spec: ObjectiveSpec,
                 current_value: float = 0.0) -> float:
        """Evaluate structural effects under the objective specification.

        Computes O(S + ΔS) - O(S), not O(ΔS).

        For threshold objectives:
          predicted_after = current_value + effect_value
          bonus = magnitude if predicted_after reaches threshold, else 0
          minus bonus already collected at current_value.

        This is the mathematically correct evaluation: an effect
        that moves toward but doesn't reach the threshold gets 0
        bonus, not partial credit.
        """
        # Map observable name to effect value.
        effect_map = {
            "n_components": effects.delta_n_components,
            "redundancy": effects.delta_redundancy,
            "hub_load": effects.delta_hub_load,
            "spectral_gap": effects.delta_spectral_gap,
        }

        effect_value = effect_map.get(spec.observable, 0.0)
        predicted_after = current_value + effect_value

        if spec.reward_shape == "threshold":
            # Compute bonus at predicted_after and at current_value.
            if spec.direction == "minimize":
                bonus_after = spec.magnitude if predicted_after <= spec.threshold else 0.0
                bonus_current = spec.magnitude if current_value <= spec.threshold else 0.0
            else:  # maximize
                bonus_after = spec.magnitude if predicted_after >= spec.threshold else 0.0
                bonus_current = spec.magnitude if current_value >= spec.threshold else 0.0
            return bonus_after - bonus_current
        else:
            # Linear reward.
            if spec.direction == "minimize":
                return -effect_value * spec.magnitude
            else:
                return effect_value * spec.magnitude


def get_architecture_ladder() -> list:
    """Get the three architectures for comparison."""
    return [
        ScalarResidualModel(hidden_dim=64, n_epochs=300),
        ObjectiveConditionedModel(hidden_dim=80, n_epochs=300),
        CausalEffectModel(hidden_dim=64, n_epochs=300),
    ]
