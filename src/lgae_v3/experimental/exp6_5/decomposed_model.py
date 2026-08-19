"""Decomposed causal prediction model for exp6.5.

Instead of learning a scalar residual directly, predict intermediate
causal quantities and map them to residual value:

  S,a → [Δconnectivity, Δredundancy, Δhub_load, Δspectral_quality] → R_future

This is an ablation against the scalar model. Both run in parallel.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from .observable_features import extract_observable_features, OBSERVABLE_FEATURE_DIM


class DecomposedModel:
    """Base class for decomposed models."""

    def fit(self, X: np.ndarray, y_residual: np.ndarray,
            y_intermediate: np.ndarray | None = None) -> None:
        raise NotImplementedError

    def predict_residual(self, graph: GraphBuffers, z: torch.Tensor,
                         action: tuple[str, int, int, dict],
                         *, threshold: int = 1, horizon: int = 2) -> float:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "base"


class ScalarMLP(DecomposedModel):
    """Scalar MLP that predicts residual directly (exp6.4 approach).

    This is the baseline — no decomposition.
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

    def fit(self, X, y_residual, y_intermediate=None):
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

    def predict_residual(self, graph, z, action, *, threshold=1, horizon=2):
        if not self._fitted:
            return 0.0
        x = extract_observable_features(graph, z, action, threshold=threshold, horizon=horizon)
        h = np.maximum(0, x @ self._W1 + self._b1)
        return float((h @ self._W2 + self._b2).item())

    @property
    def name(self):
        return "ScalarMLP"


class MultiHeadModel(DecomposedModel):
    """Multi-head model that predicts intermediate causal quantities.

    Predicts:
    - delta_connectivity: change in n_components
    - delta_redundancy: change in min degree
    - delta_hub_load: change in max degree
    - delta_spectral: change in spectral gap

    Then maps these to residual via a linear layer.

    S,a → [Δconn, Δred, Δhub, Δspec] → R_future
    """

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4,
                 n_epochs: int = 300, lr: float = 0.01) -> None:
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_epochs = n_epochs
        self.lr = lr
        self._W1 = None
        self._b1 = None
        self._W_heads = None  # hidden -> n_heads
        self._b_heads = None
        self._W_out = None  # n_heads -> 1
        self._b_out = 0.0
        self._fitted = False

    def _compute_intermediate_targets(self, graph, z, action, utility_fn=None):
        """Compute intermediate causal quantities for training.

        These are derived from the exact utility but are NOT the
        mechanism label. They are observable structural changes.
        """
        from ..exp6_4.structural_features import compute_component_info
        from ..exp6_3.exact_mpc import apply_action

        n = int(graph.num_nodes)
        comp_before = compute_component_info(graph, n)
        next_graph = apply_action(graph, action)
        comp_after = compute_component_info(next_graph, n)

        # Delta connectivity.
        delta_conn = comp_after.n_components - comp_before.n_components

        # Delta redundancy (min degree change).
        from .observable_features import _compute_degree_stats
        _, _, _, _, min_deg_before = _compute_degree_stats(graph, n)
        _, _, _, _, min_deg_after = _compute_degree_stats(next_graph, n)
        delta_redundancy = min_deg_after - min_deg_before

        # Delta hub load (max degree change).
        _, _, _, max_deg_before, _ = _compute_degree_stats(graph, n)
        _, _, _, max_deg_after, _ = _compute_degree_stats(next_graph, n)
        delta_hub = max_deg_after - max_deg_before

        # Delta spectral gap.
        from .observable_features import _compute_spectral_gap
        spec_before = _compute_spectral_gap(graph, n)
        spec_after = _compute_spectral_gap(next_graph, n)
        delta_spec = spec_after - spec_before

        return np.array([delta_conn, delta_redundancy, delta_hub, delta_spec])

    def fit(self, X, y_residual, y_intermediate=None):
        """Fit the multi-head model.

        y_intermediate: (n_samples, 4) array of [delta_conn, delta_red, delta_hub, delta_spec]
        If not provided, skip intermediate supervision and learn heads from residual.
        """
        rng = np.random.RandomState(42)
        n_feat = X.shape[1]

        self._W1 = rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
        self._b1 = np.zeros(self.hidden_dim)
        self._W_heads = rng.randn(self.hidden_dim, self.n_heads) * np.sqrt(2.0 / self.hidden_dim)
        self._b_heads = np.zeros(self.n_heads)
        self._W_out = rng.randn(self.n_heads, 1) * np.sqrt(2.0 / self.n_heads)
        self._b_out = np.zeros(1)

        for epoch in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            heads = h @ self._W_heads + self._b_heads  # (n, n_heads)
            pred = (heads @ self._W_out + self._b_out).flatten()  # (n,)

            # Residual loss.
            err = pred - y_residual
            loss_grad = err.reshape(-1, 1) / len(y_residual)

            # Intermediate supervision (optional).
            if y_intermediate is not None:
                # Add intermediate loss.
                head_err = heads - y_intermediate
                loss_grad += 0.1 * head_err @ self._W_out / len(y_residual)

            grad_W_out = heads.T @ loss_grad
            grad_b_out = loss_grad.sum(axis=0)
            grad_heads = loss_grad @ self._W_out.T
            grad_h = grad_heads @ self._W_heads.T
            grad_h[h <= 0] = 0
            grad_W_heads = h.T @ grad_heads
            grad_b_heads = grad_heads.sum(axis=0)
            grad_W1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)

            self._W1 -= self.lr * grad_W1
            self._b1 -= self.lr * grad_b1
            self._W_heads -= self.lr * grad_W_heads
            self._b_heads -= self.lr * grad_b_heads
            self._W_out -= self.lr * grad_W_out
            self._b_out -= self.lr * grad_b_out

        self._fitted = True

    def predict_residual(self, graph, z, action, *, threshold=1, horizon=2):
        if not self._fitted:
            return 0.0
        x = extract_observable_features(graph, z, action, threshold=threshold, horizon=horizon)
        h = np.maximum(0, x @ self._W1 + self._b1)
        heads = h @ self._W_heads + self._b_heads
        return float((heads @ self._W_out + self._b_out).item())

    def predict_intermediate(self, graph, z, action, *, threshold=1, horizon=2):
        """Predict intermediate causal quantities."""
        if not self._fitted:
            return np.zeros(self.n_heads)
        x = extract_observable_features(graph, z, action, threshold=threshold, horizon=horizon)
        h = np.maximum(0, x @ self._W1 + self._b1)
        heads = h @ self._W_heads + self._b_heads
        return heads

    @property
    def name(self):
        return "MultiHeadMLP"


class EnsembleScalarMLP(DecomposedModel):
    """Ensemble of scalar MLPs for uncertainty estimation."""

    def __init__(self, n_models: int = 5, hidden_dim: int = 32,
                 n_epochs: int = 200) -> None:
        self.n_models = n_models
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self._models: list[ScalarMLP] = []
        self._fitted = False

    def fit(self, X, y_residual, y_intermediate=None):
        self._models = []
        for i in range(self.n_models):
            m = ScalarMLP(hidden_dim=self.hidden_dim, n_epochs=self.n_epochs, lr=0.01)
            rng = np.random.RandomState(42 + i)
            idx = rng.choice(len(X), size=len(X), replace=True)
            m.fit(X[idx], y_residual[idx])
            self._models.append(m)
        self._fitted = True

    def predict_residual(self, graph, z, action, *, threshold=1, horizon=2):
        if not self._fitted:
            return 0.0
        preds = [m.predict_residual(graph, z, action, threshold=threshold, horizon=horizon)
                 for m in self._models]
        return float(np.mean(preds))

    def predict_residual_std(self, graph, z, action, *, threshold=1, horizon=2):
        """Predict residual standard deviation (uncertainty)."""
        if not self._fitted:
            return 0.0
        preds = [m.predict_residual(graph, z, action, threshold=threshold, horizon=horizon)
                 for m in self._models]
        return float(np.std(preds))

    @property
    def name(self):
        return "EnsembleScalarMLP"


def get_decomposed_model_ladder() -> list[DecomposedModel]:
    """Get the decomposed model ladder."""
    return [
        ScalarMLP(hidden_dim=64, n_epochs=300),
        MultiHeadModel(hidden_dim=64, n_heads=4, n_epochs=300),
        EnsembleScalarMLP(n_models=5, hidden_dim=32, n_epochs=200),
    ]
