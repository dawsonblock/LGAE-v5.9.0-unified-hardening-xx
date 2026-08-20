"""Learned-state model for exp6.8.1.

Predicts only the LEARNED tier of the split structural state.
Exact and certified tiers are computed deterministically.

  y = learned_state_{t+1}  (3 dims: path_length, efficiency, future_opportunity)
"""
from __future__ import annotations

import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_7.multi_operator_features import extract_multi_operator_features
from .split_state import LEARNED_STATE_DIM, SplitStructuralState, LearnedState


class LearnedStateModel:
    """Predicts only the learned tier: z_{t+1} = F(G_t, z_t, a_t).

    Input features: [action_features, exact_state, certified_state, learned_state]
    Output: learned_state_{t+1} (LEARNED_STATE_DIM = 3)
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model.

        X: (n_samples, feature_dim)
        y: (n_samples, LEARNED_STATE_DIM) — learned state at t+1
        """
        self._feature_dim = X.shape[1]
        rng = np.random.RandomState(42)
        self._W1 = rng.randn(self._feature_dim, self.hidden_dim) * np.sqrt(2.0 / self._feature_dim)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, LEARNED_STATE_DIM) * np.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(LEARNED_STATE_DIM)

        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            pred = h @ self._W2 + self._b2
            err = pred - y
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

    def _make_features(self, graph: GraphBuffers, z: torch.Tensor,
                       state: SplitStructuralState, action: tuple,
                       threshold: int = 1) -> np.ndarray:
        """Build feature vector from split state."""
        x_action = extract_multi_operator_features(
            graph, z, action, threshold=threshold, horizon=2,
        )
        return np.concatenate([
            x_action,
            state.exact.to_array(),
            state.certified.to_array(),
            state.learned.to_array(),
        ])

    def predict_learned(self, graph: GraphBuffers, z: torch.Tensor,
                        state: SplitStructuralState, action: tuple,
                        threshold: int = 1) -> np.ndarray:
        """Predict learned state at t+1."""
        if not self._fitted:
            return state.learned.to_array().copy()
        x = self._make_features(graph, z, state, action, threshold=threshold)
        h = np.maximum(0, x @ self._W1 + self._b1)
        return h @ self._W2 + self._b2

    def predict_uncertainty(self, graph: GraphBuffers, z: torch.Tensor,
                            state: SplitStructuralState, action: tuple,
                            threshold: int = 1) -> float:
        """Estimate prediction uncertainty (feature norm proxy)."""
        if not self._fitted:
            return 0.0
        x = self._make_features(graph, z, state, action, threshold=threshold)
        return float(np.linalg.norm(x))

    @property
    def name(self) -> str:
        return "F_learned_state"

    # Compatibility methods for exp6_8's recursive_causal_mpc interface.
    # These allow LearnedStateModel to be used as a drop-in replacement
    # for ConsequentialStateModel in the recursive planner.

    def predict_z(self, graph: GraphBuffers, z: torch.Tensor,
                  z_state: np.ndarray, action: tuple,
                  threshold: int = 1) -> np.ndarray:
        """Compatible with ConsequentialStateModel.predict_z.

        Takes a flat z_state array and returns a predicted z array.
        Internally builds a SplitStructuralState from the graph.
        """
        if not self._fitted:
            return z_state.copy()
        state = SplitStructuralState.from_graph(graph)
        # Override learned tier with provided z_state (last LEARNED_STATE_DIM elements).
        if len(z_state) >= LEARNED_STATE_DIM:
            state.learned = LearnedState(
                path_length=float(z_state[-LEARNED_STATE_DIM]),
                efficiency=float(z_state[-LEARNED_STATE_DIM + 1]),
                future_opportunity=float(z_state[-LEARNED_STATE_DIM + 2]),
            )
        learned_pred = self.predict_learned(graph, z, state, action, threshold=threshold)
        # Return full z_state with learned portion replaced.
        result = z_state.copy()
        if len(result) >= LEARNED_STATE_DIM:
            result[-LEARNED_STATE_DIM:] = learned_pred
        return result

    def predict_z_std(self, graph: GraphBuffers, z: torch.Tensor,
                      z_state: np.ndarray, action: tuple,
                      threshold: int = 1) -> float:
        """Compatible with ConsequentialStateModel.predict_z_std."""
        if not self._fitted:
            return 0.0
        state = SplitStructuralState.from_graph(graph)
        if len(z_state) >= LEARNED_STATE_DIM:
            state.learned = LearnedState(
                path_length=float(z_state[-LEARNED_STATE_DIM]),
                efficiency=float(z_state[-LEARNED_STATE_DIM + 1]),
                future_opportunity=float(z_state[-LEARNED_STATE_DIM + 2]),
            )
        return self.predict_uncertainty(graph, z, state, action, threshold=threshold)
