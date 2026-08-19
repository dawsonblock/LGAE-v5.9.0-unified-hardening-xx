"""Dynamics models for the lightweight world model.

Learns: z_{t+1} = F_θ(z_t, a_t)

Two variants:
1. LinearDynamics: z_{t+1} = A·z_t + B·a_t + c
   - Simple, interpretable, easy to verify
   - Parameters: (state_dim × state_dim) + (action_dim × state_dim) + state_dim

2. MLPDynamics: z_{t+1} = MLP([z_t, a_t])
   - Small MLP with one hidden layer
   - Still lightweight (hidden_dim default 32)

Both are advisory-only. They never mutate authoritative state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import time
import math
import numpy as np

from .state_encoding import STATE_DIM, ACTION_DIM, state_action_schema_hash


# ---------------------------------------------------------------------------
# Base class.
# ---------------------------------------------------------------------------

class DynamicsModel:
    """Base class for dynamics models.

    All dynamics models implement:
    - predict(z_t, a_t) -> z_{t+1}
    - fit(Z_t, A_t, Z_{t+1}) on train split only
    - get_state() / set_state() for serialization
    - hyperparameters() for provenance
    """

    model_type: str = "dynamics"
    version: str = "v6.0-exp5"
    deterministic: bool = True

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}"

    @property
    def requires_fit(self) -> bool:
        return True

    def predict(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        """Predict next state. Must be implemented by subclasses."""
        raise NotImplementedError

    def predict_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Batch prediction. Shape: (batch, state_dim)."""
        if z_t.ndim == 1:
            z_t = z_t[np.newaxis, :]
        if a_t.ndim == 1:
            a_t = a_t[np.newaxis, :]
        return np.array([self.predict(z_t[i], a_t[i]) for i in range(len(z_t))])

    def fit(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
        z_next: np.ndarray,
        *,
        split: str = "train",
    ) -> None:
        """Fit on train split only."""
        if split != "train":
            raise ValueError(
                f"Dynamics model can only fit on train split, got '{split}'. "
                f"Validation/held-out fitting is forbidden."
            )
        self._fit(z_t, a_t, z_next)

    def _fit(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
        z_next: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def freeze(self) -> None:
        """Freeze the model after training."""
        pass

    def get_state(self) -> dict[str, Any]:
        """Return serializable model state."""
        raise NotImplementedError

    def set_state(self, state: dict[str, Any]) -> None:
        """Load model state from a dict."""
        raise NotImplementedError

    def hyperparameters(self) -> dict[str, Any]:
        """Return hyperparameter configuration."""
        return {
            "model_type": self.model_type,
            "version": self.version,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
        }

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return 0

    @property
    def schema_hash(self) -> str:
        return state_action_schema_hash()


# ---------------------------------------------------------------------------
# Linear dynamics.
# ---------------------------------------------------------------------------

class LinearDynamics(DynamicsModel):
    """Linear dynamics model: z_{t+1} = A·z_t + B·a_t + c.

    Parameters:
        A: (state_dim, state_dim) state transition matrix
        B: (state_dim, action_dim) action input matrix
        c: (state_dim,) bias

    Total parameters: state_dim² + state_dim × action_dim + state_dim
    """

    model_type = "linear_dynamics"
    deterministic = True

    def __init__(
        self,
        *,
        lr: float = 0.01,
        n_epochs: int = 100,
        seed: int = 42,
        regularization: float = 1e-4,
    ) -> None:
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.regularization = float(regularization)
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._c: np.ndarray | None = None
        self._fitted = False
        self._n_samples = 0

    def _fit(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
        z_next: np.ndarray,
    ) -> None:
        """Fit using least-squares closed-form solution."""
        n = len(z_t)
        if n == 0:
            # Degenerate: identity dynamics.
            self._A = np.eye(STATE_DIM)
            self._B = np.zeros((STATE_DIM, ACTION_DIM))
            self._c = np.zeros(STATE_DIM)
            self._fitted = True
            return

        # Build design matrix: [z_t, a_t, 1]
        X = np.hstack([z_t, a_t, np.ones((n, 1))])
        # Ridge regression: W = (X^T X + λI)^{-1} X^T z_next
        lam = self.regularization
        XtX = X.T @ X + lam * np.eye(X.shape[1])
        XtY = X.T @ z_next
        W = np.linalg.solve(XtX, XtY)

        self._A = W[:STATE_DIM, :].T  # (state_dim, state_dim)
        self._B = W[STATE_DIM:STATE_DIM + ACTION_DIM, :].T  # (state_dim, action_dim)
        self._c = W[-1, :]  # (state_dim,)
        self._fitted = True
        self._n_samples = n

    def predict(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return z_t.copy()
        return self._A @ z_t + self._B @ a_t + self._c

    def predict_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        if not self._fitted:
            return z_t.copy()
        # Vectorized: Z_next = Z_t @ A^T + A_t @ B^T + c
        return z_t @ self._A.T + a_t @ self._B.T + self._c

    def get_state(self) -> dict[str, Any]:
        return {
            "A": self._A.tolist() if self._A is not None else None,
            "B": self._B.tolist() if self._B is not None else None,
            "c": self._c.tolist() if self._c is not None else None,
            "fitted": self._fitted,
            "n_samples": self._n_samples,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._A = np.array(state["A"]) if state.get("A") else None
        self._B = np.array(state["B"]) if state.get("B") else None
        self._c = np.array(state["c"]) if state.get("c") else None
        self._fitted = state.get("fitted", False)
        self._n_samples = state.get("n_samples", 0)

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "seed": self.seed,
            "regularization": self.regularization,
        }

    @property
    def n_parameters(self) -> int:
        if self._A is None:
            return 0
        return self._A.size + self._B.size + self._c.size


# ---------------------------------------------------------------------------
# MLP dynamics.
# ---------------------------------------------------------------------------

class MLPDynamics(DynamicsModel):
    """Small MLP dynamics: z_{t+1} = MLP([z_t, a_t]).

    One hidden layer with ReLU activation. Still lightweight.
    """

    model_type = "mlp_dynamics"
    deterministic = False  # depends on seed

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        lr: float = 0.001,
        n_epochs: int = 100,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self._W1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._W2: np.ndarray | None = None
        self._b2: np.ndarray | None = None
        self._fitted = False
        self._n_samples = 0
        self._input_dim = STATE_DIM + ACTION_DIM

    def _init_weights(self) -> None:
        rng = np.random.RandomState(self.seed)
        # He initialization.
        self._W1 = rng.randn(self._input_dim, self.hidden_dim) * math.sqrt(2.0 / self._input_dim)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, STATE_DIM) * math.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(STATE_DIM)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self._W1 + self._b1)  # ReLU
        return h @ self._W2 + self._b2

    def _fit(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
        z_next: np.ndarray,
    ) -> None:
        n = len(z_t)
        if n == 0:
            self._init_weights()
            self._fitted = True
            return

        self._init_weights()
        X = np.hstack([z_t, a_t])  # (n, input_dim)
        Y = z_next  # (n, state_dim)

        for epoch in range(self.n_epochs):
            # Forward pass.
            H = np.maximum(0, X @ self._W1 + self._b1)
            pred = H @ self._W2 + self._b2

            # Backward pass (MSE loss).
            d_pred = 2.0 * (pred - Y) / n  # (n, state_dim)
            d_W2 = H.T @ d_pred  # (hidden, state)
            d_b2 = d_pred.sum(axis=0)
            d_H = d_pred @ self._W2.T  # (n, hidden)
            d_H[H <= 0] = 0  # ReLU gradient
            d_W1 = X.T @ d_H  # (input, hidden)
            d_b1 = d_H.sum(axis=0)

            # Update.
            self._W2 -= self.lr * d_W2
            self._b2 -= self.lr * d_b2
            self._W1 -= self.lr * d_W1
            self._b1 -= self.lr * d_b1

        self._fitted = True
        self._n_samples = n

    def predict(self, z_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return z_t.copy()
        x = np.concatenate([z_t, a_t])
        return self._forward(x)

    def predict_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        if not self._fitted:
            return z_t.copy()
        X = np.hstack([z_t, a_t])
        return self._forward(X)

    def get_state(self) -> dict[str, Any]:
        return {
            "W1": self._W1.tolist() if self._W1 is not None else None,
            "b1": self._b1.tolist() if self._b1 is not None else None,
            "W2": self._W2.tolist() if self._W2 is not None else None,
            "b2": self._b2.tolist() if self._b2 is not None else None,
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "hidden_dim": self.hidden_dim,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._W1 = np.array(state["W1"]) if state.get("W1") else None
        self._b1 = np.array(state["b1"]) if state.get("b1") else None
        self._W2 = np.array(state["W2"]) if state.get("W2") else None
        self._b2 = np.array(state["b2"]) if state.get("b2") else None
        self._fitted = state.get("fitted", False)
        self._n_samples = state.get("n_samples", 0)
        self.hidden_dim = state.get("hidden_dim", self.hidden_dim)

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "seed": self.seed,
        }

    @property
    def n_parameters(self) -> int:
        if self._W1 is None:
            return 0
        return self._W1.size + self._b1.size + self._W2.size + self._b2.size


# ---------------------------------------------------------------------------
# Dynamics metrics.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DynamicsMetrics:
    """Metrics for dynamics model evaluation."""
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    per_dim_rmse: list[float] = field(default_factory=list)
    n_samples: int = 0
    horizon: int = 1

    def to_log(self) -> dict[str, Any]:
        return {
            "rmse": float(self.rmse),
            "mae": float(self.mae),
            "r2": float(self.r2),
            "per_dim_rmse": [float(x) for x in self.per_dim_rmse],
            "n_samples": int(self.n_samples),
            "horizon": int(self.horizon),
        }


def compute_dynamics_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    horizon: int = 1,
) -> DynamicsMetrics:
    """Compute dynamics prediction metrics.

    Args:
        predicted: (n, state_dim) predicted next states.
        actual: (n, state_dim) actual next states.
        horizon: Prediction horizon (1 = single-step).

    Returns:
        DynamicsMetrics with RMSE, MAE, R², per-dimension RMSE.
    """
    if len(predicted) == 0:
        return DynamicsMetrics(horizon=horizon)

    diff = predicted - actual
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    # R².
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((actual - actual.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    # Per-dimension RMSE.
    per_dim = [float(np.sqrt(np.mean(diff[:, j] ** 2))) for j in range(actual.shape[1])]

    return DynamicsMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        per_dim_rmse=per_dim,
        n_samples=len(predicted),
        horizon=horizon,
    )
