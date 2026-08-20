"""Causal effect model v2 for exp6.7.

7 supervised heads + deterministic objective evaluator.
Also includes the scalar baseline for comparison.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_5.observable_features import extract_observable_features, OBSERVABLE_FEATURE_DIM
from ..exp6_6.objective_spec import ObjectiveSpec, encode_objective, OBJECTIVE_ENCODING_DIM
from .extended_effects import ExtendedEffect, EXTENDED_EFFECT_DIM


def _get_current_observable(graph: GraphBuffers, spec: ObjectiveSpec) -> float:
    """Get the current value of the observable for the objective spec."""
    n = int(graph.num_nodes)
    import numpy as np

    if spec.observable == "n_components":
        from ..exp6_4.structural_features import compute_component_info
        return float(compute_component_info(graph, n).n_components)
    elif spec.observable == "redundancy":
        from ..exp6_5.observable_features import _compute_degree_stats
        _, _, _, _, min_deg = _compute_degree_stats(graph, n)
        return float(min_deg)
    elif spec.observable == "hub_load":
        from ..exp6_5.observable_features import _compute_degree_stats
        _, _, max_deg, _, _ = _compute_degree_stats(graph, n)
        return float(max_deg)
    elif spec.observable == "spectral_gap":
        from ..exp6_5.observable_features import _compute_spectral_gap
        return float(_compute_spectral_gap(graph, n))
    elif spec.observable == "path_length":
        from .extended_effects import _compute_avg_path_length
        return float(_compute_avg_path_length(graph, n))
    elif spec.observable == "efficiency":
        from .extended_effects import _compute_global_efficiency
        return float(_compute_global_efficiency(graph, n))
    elif spec.observable == "curvature":
        from .extended_effects import _compute_curvature_proxy
        return float(_compute_curvature_proxy(graph, n))
    else:
        return 0.0


class ScalarResidualModelV2:
    """Architecture A: F(S,a) -> R (scalar baseline)."""

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

    def predict_residual(self, graph, z, action, *, threshold=1, horizon=2, objective=None) -> float:
        if not self._fitted:
            return 0.0
        x = extract_observable_features(graph, z, action, threshold=threshold, horizon=horizon)
        h = np.maximum(0, x @ self._W1 + self._b1)
        return float((h @ self._W2 + self._b2).item())

    @property
    def name(self) -> str:
        return "A_scalar"


class CausalEffectModelV2:
    """Architecture C: F(S,a) -> 7 effects, O(effects) -> R.

    7 supervised heads on real structural effect labels:
      delta_n_components, delta_redundancy, delta_hub_load,
      delta_spectral_gap, delta_path_length, delta_efficiency,
      delta_curvature
    """

    def __init__(self, hidden_dim: int = 64, n_epochs: int = 300, lr: float = 0.01) -> None:
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self._W1 = None
        self._b1 = None
        self._W_heads = None  # (hidden, 7)
        self._b_heads = None  # (7,)
        self._fitted = False

    def fit(self, X: np.ndarray, y_residual=None, y_effects=None, **kwargs) -> None:
        if y_effects is None:
            raise ValueError("CausalEffectModelV2 requires y_effects")
        rng = np.random.RandomState(42)
        n_feat = X.shape[1]
        n_heads = y_effects.shape[1]
        self._W1 = rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
        self._b1 = np.zeros(self.hidden_dim)
        self._W_heads = rng.randn(self.hidden_dim, n_heads) * np.sqrt(2.0 / self.hidden_dim)
        self._b_heads = np.zeros(n_heads)
        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            pred = h @ self._W_heads + self._b_heads
            err = pred - y_effects
            grad_out = err / len(y_effects)
            grad_W_heads = h.T @ grad_out
            grad_b_heads = grad_out.sum(axis=0)
            grad_h = grad_out @ self._W_heads.T
            grad_h[h <= 0] = 0
            grad_W1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)
            self._W1 -= self.lr * grad_W1
            self._b1 -= self.lr * grad_b1
            self._W_heads -= self.lr * grad_W_heads
            self._b_heads -= self.lr * grad_b_heads
        self._fitted = True

    def predict_effects(self, graph, z, action, *, threshold=1, horizon=2) -> ExtendedEffect:
        if not self._fitted:
            return ExtendedEffect()
        x = extract_observable_features(graph, z, action, threshold=threshold, horizon=horizon)
        h = np.maximum(0, x @ self._W1 + self._b1)
        effects = h @ self._W_heads + self._b_heads
        return ExtendedEffect.from_array(effects)

    def predict_residual(self, graph, z, action, *, threshold=1, horizon=2, objective=None) -> float:
        if not self._fitted or objective is None:
            return 0.0
        effects = self.predict_effects(graph, z, action, threshold=threshold, horizon=horizon)
        # Compute current observable value for absolute-state evaluation.
        current_value = _get_current_observable(graph, objective)
        return ObjectiveEvaluatorV2.evaluate(effects, objective, current_value=current_value)

    @property
    def name(self) -> str:
        return "C_causal_effect_v2"


class ObjectiveEvaluatorV2:
    """Maps 7 predicted effects to future value under an objective.

    Supports both threshold and linear reward shapes, and composite
    objectives (multiple effects combined).
    """

    @staticmethod
    def evaluate(
        effects: ExtendedEffect, spec: ObjectiveSpec,
        current_value: float = 0.0,
    ) -> float:
        """Evaluate effects under the objective specification.

        Computes O(S + ΔS) - O(S), not O(ΔS).

        For threshold objectives:
          predicted_after = current_value + effect_value
          bonus = magnitude if predicted_after reaches threshold, else 0
          minus bonus already collected at current_value.

        This is the mathematically correct evaluation: an effect
        that moves toward but doesn't reach the threshold gets 0
        bonus, not partial credit.
        """
        effect_map = {
            "n_components": effects.delta_n_components,
            "redundancy": effects.delta_redundancy,
            "hub_load": effects.delta_hub_load,
            "spectral_gap": effects.delta_spectral_gap,
            "path_length": effects.delta_path_length,
            "efficiency": effects.delta_efficiency,
            "curvature": effects.delta_curvature,
        }

        effect_value = effect_map.get(spec.observable, 0.0)
        predicted_after = current_value + effect_value

        if spec.reward_shape == "threshold":
            # Compute bonus at predicted_after and at current_value.
            if spec.direction == "minimize":
                # Threshold reached when value <= spec.threshold.
                bonus_after = spec.magnitude if predicted_after <= spec.threshold else 0.0
                bonus_current = spec.magnitude if current_value <= spec.threshold else 0.0
            else:  # maximize
                # Threshold reached when value >= spec.threshold.
                bonus_after = spec.magnitude if predicted_after >= spec.threshold else 0.0
                bonus_current = spec.magnitude if current_value >= spec.threshold else 0.0
            return bonus_after - bonus_current
        else:  # linear
            if spec.direction == "minimize":
                return -effect_value * spec.magnitude
            else:
                return effect_value * spec.magnitude

    @staticmethod
    def evaluate_composite(effects: ExtendedEffect, weights: dict[str, float]) -> float:
        """Evaluate effects under a composite objective.

        weights maps observable names to coefficients.
        Example: {"spectral_gap": 2.0, "redundancy": 0.5, "hub_load": -3.0}
        """
        effect_map = {
            "n_components": effects.delta_n_components,
            "redundancy": effects.delta_redundancy,
            "hub_load": effects.delta_hub_load,
            "spectral_gap": effects.delta_spectral_gap,
            "path_length": effects.delta_path_length,
            "efficiency": effects.delta_efficiency,
            "curvature": effects.delta_curvature,
        }
        value = 0.0
        for obs_name, weight in weights.items():
            value += weight * effect_map.get(obs_name, 0.0)
        return value


def get_architecture_ladder_v2() -> list:
    """Get the architecture ladder for exp6.7."""
    return [
        ScalarResidualModelV2(hidden_dim=64, n_epochs=300),
        CausalEffectModelV2(hidden_dim=64, n_epochs=300),
    ]
