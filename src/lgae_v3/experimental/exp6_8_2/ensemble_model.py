"""Ensemble learned state model for exp6.8.2.

Trains M independent models with different random seeds.
Predicts the learned tier (3 dims) with ensemble mean and std.

  mu = (1/M) * sum_m z_hat^(m)
  sigma^2 = (1/M) * sum_m (z_hat^(m) - mu)^2

The ensemble std is used for decision uncertainty, not feature norm.
"""
from __future__ import annotations

import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_7.multi_operator_features import extract_multi_operator_features
from ..exp6_8_1.split_state import (
    LEARNED_STATE_DIM, SplitStructuralState, LearnedState,
)
from ..exp6_8_1.learned_state_model import LearnedStateModel


class EnsembleLearnedModel:
    """Ensemble of M LearnedStateModel instances.

    Each member is initialized with a different random seed.
    Ensemble mean and std provide calibrated uncertainty.
    """

    def __init__(
        self, n_members: int = 5, hidden_dim: int = 128,
        n_epochs: int = 500, lr: float = 0.01,
    ) -> None:
        self.n_members = n_members
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self._members: list[LearnedStateModel] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit all M ensemble members on the same data.

        Each member gets a different random initialization via
        different base seeds, producing diverse predictions.
        """
        self._members = []
        for i in range(self.n_members):
            member = LearnedStateModel(
                hidden_dim=self.hidden_dim,
                n_epochs=self.n_epochs,
                lr=self.lr,
            )
            # Patch the RNG seed in fit by temporarily modifying the method.
            # Each member gets a different initialization.
            original_fit = member.fit
            rng = np.random.RandomState(42 + i * 100)

            # Reimplement fit with member-specific seed.
            n_feat = X.shape[1]
            member._feature_dim = n_feat
            member._W1 = rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
            member._b1 = np.zeros(self.hidden_dim)
            member._W2 = rng.randn(self.hidden_dim, LEARNED_STATE_DIM) * np.sqrt(2.0 / self.hidden_dim)
            member._b2 = np.zeros(LEARNED_STATE_DIM)

            for _ in range(self.n_epochs):
                h = np.maximum(0, X @ member._W1 + member._b1)
                pred = h @ member._W2 + member._b2
                err = pred - y
                grad_out = err / len(y)
                grad_W2 = h.T @ grad_out
                grad_b2 = grad_out.sum(axis=0)
                grad_h = grad_out @ member._W2.T
                grad_h[h <= 0] = 0
                grad_W1 = X.T @ grad_h
                grad_b1 = grad_h.sum(axis=0)

                member._W1 -= self.lr * grad_W1
                member._b1 -= self.lr * grad_b1
                member._W2 -= self.lr * grad_W2
                member._b2 -= self.lr * grad_b2

            member._fitted = True
            self._members.append(member)

        self._fitted = True

    def predict_learned_ensemble(
        self, graph: GraphBuffers, z: torch.Tensor,
        state: SplitStructuralState, action: tuple,
        threshold: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict learned state with ensemble.

        Returns (mean, std) each of shape (LEARNED_STATE_DIM,).
        """
        if not self._fitted:
            return state.learned.to_array().copy(), np.zeros(LEARNED_STATE_DIM)

        preds = []
        for member in self._members:
            p = member.predict_learned(graph, z, state, action, threshold=threshold)
            preds.append(p)

        preds_arr = np.array(preds)  # (M, LEARNED_STATE_DIM)
        mean = preds_arr.mean(axis=0)
        std = preds_arr.std(axis=0)
        return mean, std

    def predict_learned(
        self, graph: GraphBuffers, z: torch.Tensor,
        state: SplitStructuralState, action: tuple,
        threshold: int = 1,
    ) -> np.ndarray:
        """Return ensemble mean prediction."""
        mean, _ = self.predict_learned_ensemble(
            graph, z, state, action, threshold=threshold,
        )
        return mean

    def predict_uncertainty(
        self, graph: GraphBuffers, z: torch.Tensor,
        state: SplitStructuralState, action: tuple,
        threshold: int = 1,
    ) -> float:
        """Return ensemble std (averaged across dimensions)."""
        _, std = self.predict_learned_ensemble(
            graph, z, state, action, threshold=threshold,
        )
        return float(np.mean(std))

    # Compatibility with exp6_8 recursive_causal_mpc interface.
    def predict_z(self, graph: GraphBuffers, z: torch.Tensor,
                  z_state: np.ndarray, action: tuple,
                  threshold: int = 1) -> np.ndarray:
        """Compatible with ConsequentialStateModel.predict_z."""
        if not self._fitted:
            return z_state.copy()
        state = SplitStructuralState.from_graph(graph)
        if len(z_state) >= LEARNED_STATE_DIM:
            state.learned = LearnedState(
                path_length=float(z_state[-LEARNED_STATE_DIM]),
                efficiency=float(z_state[-LEARNED_STATE_DIM + 1]),
                future_opportunity=float(z_state[-LEARNED_STATE_DIM + 2]),
            )
        learned_pred = self.predict_learned(graph, z, state, action, threshold=threshold)
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

    @property
    def name(self) -> str:
        return "F_ensemble_learned"

    def predict_q_ensemble(
        self, graph: GraphBuffers, z: torch.Tensor,
        state: SplitStructuralState, action: tuple,
        objective, threshold: int = 1,
    ) -> tuple[float, float]:
        """Predict Q value with ensemble, returning (mean_Q, std_Q).

        Each ensemble member predicts the objective-relevant observable
        at t+1, then we evaluate O(S+delta_S) - O(S) for each member.

        Key insight: the ensemble must predict the observable that the
        objective depends on (n_components, min_degree, etc.), not just
        the learned tier. The exact tier is exact for H=1, but for H=2
        the second-step observable depends on which action is taken at
        S_1, which requires prediction.

        For H=1 (single step): Q is exact (no ensemble uncertainty).
        For H=2: Q depends on the best second action, which the ensemble
        predicts with uncertainty.
        """
        if not self._fitted:
            return 0.0, 0.0

        from ..exp6_8.transition_model import exact_transition
        from ..exp6_3.exact_mpc import apply_action_with_status

        # Exact graph transition for first action.
        new_graph, status = exact_transition(graph, action)
        if status != "VALID":
            return 0.0, 0.0

        # For H=1: Q is exact (computed from exact graph).
        # The objective-relevant observable is in the exact/certified tier.
        new_state_exact = SplitStructuralState.from_graph(new_graph)
        current_val = state.get_observable(objective.observable)
        after_val_exact = new_state_exact.get_observable(objective.observable)

        if objective.reward_shape == "threshold":
            if objective.direction == "minimize":
                bonus_after = objective.magnitude if after_val_exact <= objective.threshold else 0.0
                bonus_current = objective.magnitude if current_val <= objective.threshold else 0.0
            else:
                bonus_after = objective.magnitude if after_val_exact >= objective.threshold else 0.0
                bonus_current = objective.magnitude if current_val >= objective.threshold else 0.0
            q_exact_h1 = bonus_after - bonus_current
        else:
            delta = after_val_exact - current_val
            if objective.direction == "minimize":
                q_exact_h1 = -delta * objective.magnitude
            else:
                q_exact_h1 = delta * objective.magnitude

        # For H=2: each ensemble member predicts a different "future
        # opportunity" value at S_1, which affects which second action
        # is best, which affects Q at H=2.
        # We model this as: each member predicts a scalar "future value
        # bonus" that represents the expected second-step improvement.
        # This is the key quantity that the ensemble can be uncertain about.
        q_values = []
        for member in self._members:
            # Member predicts learned state at t+1.
            z_pred = member.predict_learned(graph, z, state, action, threshold=threshold)
            # The "future_opportunity" component (index 2) represents
            # the member's estimate of how much value the second step
            # can extract from S_1.
            future_opportunity = float(z_pred[2]) * objective.magnitude
            # Total Q = H=1 exact + H=2 predicted future opportunity.
            q_total = q_exact_h1 + future_opportunity
            q_values.append(q_total)

        q_arr = np.array(q_values)
        return float(q_arr.mean()), float(q_arr.std())
