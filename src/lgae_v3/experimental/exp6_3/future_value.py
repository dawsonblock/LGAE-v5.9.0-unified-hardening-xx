"""Future value model ladder for exp6.3.

V0: zero future value (greedy baseline)
V1: mutation-type mean residual
V2: linear regression
V3: ridge regression
V5: small MLP

The model predicts V(S') — the future value achievable from state S'.
Used as: Q(a) = ΔU_analytical + γ * V(S')
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from .exact_mpc import apply_action


def extract_features(graph: GraphBuffers, z: torch.Tensor) -> np.ndarray:
    """Extract structural features from a graph state."""
    n = int(graph.num_nodes)
    valid = graph.valid.bool()
    n_edges = int(valid.sum().item())
    density = n_edges / max(n * (n - 1) / 2, 1)

    degrees = np.zeros(n)
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1

    src = graph.src[valid]
    dst = graph.dst[valid]
    w = graph.weight[valid]
    if src.numel() > 0:
        d = (z[src] - z[dst]).pow(2).sum(-1)
        u_add = float(-(w * d).sum().item())
        d_mean = float(d.mean().item())
        d_std = float(d.std().item()) if d.numel() > 1 else 0.0
    else:
        u_add = 0.0
        d_mean = d_std = 0.0

    # Count components (non-additive feature).
    from .delayed_tasks import _count_components
    n_comp = _count_components(graph, n)

    return np.array([
        n / 50.0, density, float(np.mean(degrees)) / 10.0,
        float(np.std(degrees)) / 10.0, float(np.max(degrees)) / 20.0,
        u_add / 100.0, d_mean, d_std, n_comp / n,
    ])


class FutureValueModel:
    """Base class for future value models."""
    def predict(self, graph: GraphBuffers, z: torch.Tensor) -> float:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "base"


class V0Zero(FutureValueModel):
    """V0: zero future value. Pure greedy baseline."""
    def predict(self, graph: GraphBuffers, z: torch.Tensor) -> float:
        return 0.0
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
    @property
    def name(self) -> str:
        return "V0_zero"


class V1TypeMean(FutureValueModel):
    """V1: mutation-type mean residual."""
    def __init__(self) -> None:
        self._mean = 0.0
    def predict(self, graph: GraphBuffers, z: torch.Tensor) -> float:
        return self._mean
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._mean = float(np.mean(y)) if len(y) > 0 else 0.0
    @property
    def name(self) -> str:
        return "V1_type_mean"


class V2Linear(FutureValueModel):
    """V2: ordinary least squares linear regression."""
    def __init__(self) -> None:
        self._w: np.ndarray | None = None
        self._b = 0.0
    def predict(self, graph: GraphBuffers, z: torch.Tensor) -> float:
        if self._w is None:
            return 0.0
        x = extract_features(graph, z)
        return float(x @ self._w + self._b)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) == 0:
            return
        try:
            self._w, self._b = np.polyfit(X[:, 0], y, 1) if X.shape[1] == 1 else (np.linalg.lstsq(X, y, rcond=None)[0], 0.0)
            if X.shape[1] > 1:
                X_aug = np.hstack([X, np.ones((len(X), 1))])
                sol = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                self._w = sol[:-1]
                self._b = sol[-1]
        except Exception:
            self._w = np.zeros(X.shape[1])
            self._b = float(np.mean(y))
    @property
    def name(self) -> str:
        return "V2_linear"


class V3Ridge(FutureValueModel):
    """V3: ridge regression with L2 regularization."""
    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._w: np.ndarray | None = None
        self._b = 0.0
    def predict(self, graph: GraphBuffers, z: torch.Tensor) -> float:
        if self._w is None:
            return 0.0
        x = extract_features(graph, z)
        return float(x @ self._w + self._b)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) == 0:
            return
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        n_feat = X_aug.shape[1]
        reg = self._alpha * np.eye(n_feat)
        reg[-1, -1] = 0  # don't regularize bias
        try:
            self._w = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
            self._b = self._w[-1]
            self._w = self._w[:-1]
        except Exception:
            self._w = np.zeros(X.shape[1])
            self._b = float(np.mean(y))
    @property
    def name(self) -> str:
        return "V3_ridge"


class V5MLP(FutureValueModel):
    """V5: small MLP with one hidden layer."""
    def __init__(self, hidden_dim: int = 32, n_epochs: int = 200, lr: float = 0.01, seed: int = 42) -> None:
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed
        self._W1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._W2: np.ndarray | None = None
        self._b2 = 0.0
        self._input_dim = 0
    def predict(self, graph: GraphBuffers, z: torch.Tensor) -> float:
        if self._W1 is None:
            return 0.0
        x = extract_features(graph, z)
        h = np.maximum(0, x @ self._W1 + self._b1)
        return float((h @ self._W2 + self._b2).item())
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) == 0:
            return
        rng = np.random.RandomState(self.seed)
        n_feat = X.shape[1]
        self._input_dim = n_feat
        self._W1 = rng.randn(n_feat, self.hidden_dim) * 0.5
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, 1) * 0.5
        self._b2 = np.zeros(1)

        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            pred = (h @ self._W2 + self._b2).flatten()
            err = pred - y
            grad_out = err.reshape(-1, 1) / len(y)
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
    @property
    def name(self) -> str:
        return "V5_mlp"


def get_model_ladder() -> list[FutureValueModel]:
    """Get the full model ladder V0-V5."""
    return [V0Zero(), V1TypeMean(), V2Linear(), V3Ridge(), V5MLP()]
