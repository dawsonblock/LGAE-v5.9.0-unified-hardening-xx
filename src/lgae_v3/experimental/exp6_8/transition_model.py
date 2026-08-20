"""Exact transition + learned consequential state model for exp6.8.

Architecture:
  G_{t+1} = T_exact(G_t, a_t)    [exact graph transition]
  z_{t+1} = F(G_t, z_t, a_t)     [learned consequential state]

The exact transition uses apply_action_with_status from exp6.3.
The learned model predicts the structural observable vector z
at the next state, given the current graph, observables, and action.

Training labels are exact: z_{t+1} = compute_structural_observables(G_{t+1}).
"""
from __future__ import annotations

import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_3.exact_mpc import apply_action_with_status, apply_action
from ..exp6_7.multi_operator_features import extract_multi_operator_features
from .structural_state import (
    StructuralState, compute_structural_observables,
    STRUCTURAL_OBSERVABLE_DIM, get_observable_value,
)


class ConsequentialStateModel:
    """Learned model: F(G_t, z_t, a_t) -> z_{t+1}.

    Predicts the structural observable vector at the next state.
    Uses a simple MLP with:
      - Multi-operator action features (from exp6.7.1)
      - Current structural observables z_t
      - Concatenated to predict z_{t+1}

    Training uses exact labels: z_{t+1} = compute_structural_observables(G_{t+1}).
    """

    def __init__(self, hidden_dim: int = 128, n_epochs: int = 500, lr: float = 0.01) -> None:
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self._W1 = None
        self._b1 = None
        self._W2 = None
        self._b2 = None
        self._fitted = False
        self._feature_dim = None

    def _make_features(self, graph: GraphBuffers, z: torch.Tensor,
                       z_state: np.ndarray, action: tuple,
                       threshold: int = 1) -> np.ndarray:
        """Build feature vector: [action_features, current_z]."""
        x_action = extract_multi_operator_features(
            graph, z, action, threshold=threshold, horizon=2,
        )
        return np.concatenate([x_action, z_state])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model.

        X: (n_samples, feature_dim) — concatenated [action_features, z_t]
        y: (n_samples, STRUCTURAL_OBSERVABLE_DIM) — z_{t+1} (exact)
        """
        self._feature_dim = X.shape[1]
        rng = np.random.RandomState(42)
        self._W1 = rng.randn(self._feature_dim, self.hidden_dim) * np.sqrt(2.0 / self._feature_dim)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, STRUCTURAL_OBSERVABLE_DIM) * np.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(STRUCTURAL_OBSERVABLE_DIM)

        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            pred = h @ self._W2 + self._b2  # (n, obs_dim)
            err = pred - y  # (n, obs_dim)
            grad_out = err / len(y)
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

    def predict_z(self, graph: GraphBuffers, z: torch.Tensor,
                  z_state: np.ndarray, action: tuple,
                  threshold: int = 1) -> np.ndarray:
        """Predict z_{t+1} = F(G_t, z_t, a_t)."""
        if not self._fitted:
            return z_state.copy()  # identity if not fitted
        x = self._make_features(graph, z, z_state, action, threshold=threshold)
        h = np.maximum(0, x @ self._W1 + self._b1)
        return h @ self._W2 + self._b2

    def predict_z_std(self, graph: GraphBuffers, z: torch.Tensor,
                      z_state: np.ndarray, action: tuple,
                      threshold: int = 1) -> float:
        """Estimate prediction uncertainty (simple proxy: feature norm)."""
        if not self._fitted:
            return 0.0
        x = self._make_features(graph, z, z_state, action, threshold=threshold)
        return float(np.linalg.norm(x))

    @property
    def name(self) -> str:
        return "F_consequential_state"


def exact_transition(graph: GraphBuffers, action: tuple) -> tuple[GraphBuffers, str]:
    """Exact graph transition: G_{t+1} = T_exact(G_t, a_t).

    Returns (new_graph, status). No silent failures.
    """
    result = apply_action_with_status(graph, action)
    return result.graph, result.status


def roll_forward_exact(
    state: StructuralState,
    z: torch.Tensor,
    action: tuple,
    threshold: int = 1,
) -> StructuralState:
    """Roll forward one step using EXACT transition + EXACT observables.

    G_{t+1} = T_exact(G_t, a_t)
    z_{t+1} = compute_structural_observables(G_{t+1})
    """
    new_graph, status = exact_transition(state.graph, action)
    if status != "VALID":
        # No-op: return state unchanged.
        return StructuralState(
            graph=state.graph,
            z=state.z.copy(),
            is_predicted=state.is_predicted,
        )
    return StructuralState.from_graph(new_graph)


def roll_forward_predicted(
    state: StructuralState,
    z: torch.Tensor,
    action: tuple,
    model: ConsequentialStateModel,
    threshold: int = 1,
) -> StructuralState:
    """Roll forward one step using EXACT transition + LEARNED observables.

    G_{t+1} = T_exact(G_t, a_t)           [exact]
    z_{t+1} = F(G_t, z_t, a_t)            [learned]
    """
    new_graph, status = exact_transition(state.graph, action)
    if status != "VALID":
        return StructuralState(
            graph=state.graph,
            z=state.z.copy(),
            is_predicted=True,
        )
    # Predict z from the PRE-transition state (as training does).
    z_pred = model.predict_z(state.graph, z, state.z, action, threshold=threshold)
    return StructuralState.from_predicted(new_graph, z_pred)
