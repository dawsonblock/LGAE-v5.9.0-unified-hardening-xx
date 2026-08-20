"""Advantage model ladder for exp6.8.3.

A0: Zero-advantage baseline (always predicts A=0, never overrides)
A1: Linear regression
A2: Ridge regression
A3: Small MLP
A4: Bootstrap MLP ensemble
A5: Quantile MLP (for conformalized quantile regression)

All models predict the signed advantage A = Q_learned - Q_baseline.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class ZeroAdvantageModel:
    """A0: Always predicts A=0. Never overrides."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float32)

    def predict_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.abs(y - self.predict(X))

    @property
    def name(self) -> str:
        return "A0_zero"


class LinearRegressionModel:
    """A1: Ordinary least squares."""

    def __init__(self) -> None:
        self._w = None
        self._b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Add bias column.
        X_b = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
        try:
            w, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
            self._w = w[:-1]
            self._b = w[-1]
        except np.linalg.LinAlgError:
            self._w = np.zeros(X.shape[1], dtype=np.float32)
            self._b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._w is None:
            return np.zeros(len(X), dtype=np.float32)
        return X @ self._w + self._b

    def predict_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.abs(y - self.predict(X))

    @property
    def name(self) -> str:
        return "A1_linear"


class RidgeRegressionModel:
    """A2: Ridge regression with L2 regularization."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._w = None
        self._b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        # Ridge: w = (X^T X + alpha I)^{-1} X^T y
        XtX = X.T @ X + self.alpha * np.eye(d, dtype=np.float32)
        Xty = X.T @ y
        try:
            self._w = np.linalg.solve(XtX, Xty)
            self._b = float(np.mean(y - X @ self._w))
        except np.linalg.LinAlgError:
            self._w = np.zeros(d, dtype=np.float32)
            self._b = float(np.mean(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._w is None:
            return np.zeros(len(X), dtype=np.float32)
        return X @ self._w + self._b

    def predict_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.abs(y - self.predict(X))

    @property
    def name(self) -> str:
        return "A2_ridge"


class MLPModel:
    """A3: Small MLP for advantage regression."""

    def __init__(self, hidden_dim: int = 64, n_epochs: int = 300, lr: float = 0.01, seed: int = 42) -> None:
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed
        self._W1 = None
        self._b1 = None
        self._W2 = None
        self._b2 = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_feat = X.shape[1]
        rng = np.random.RandomState(self.seed)
        self._W1 = rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(1)

        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            pred = (h @ self._W2 + self._b2).flatten()
            err = pred - y
            grad_out = (err / len(y)).reshape(-1, 1)
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X), dtype=np.float32)
        h = np.maximum(0, X @ self._W1 + self._b1)
        return (h @ self._W2 + self._b2).flatten().astype(np.float32)

    def predict_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.abs(y - self.predict(X))

    @property
    def name(self) -> str:
        return "A3_mlp"


class BootstrapMLPEnsemble:
    """A4: Bootstrap MLP ensemble for advantage regression.

    Each member is trained on a bootstrap sample of the training data.
    The ensemble mean is the prediction; the ensemble std provides
    uncertainty that should be better calibrated than naive ensembles.
    """

    def __init__(self, n_members: int = 5, hidden_dim: int = 64, n_epochs: int = 300, lr: float = 0.01) -> None:
        self.n_members = n_members
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self._members: list[MLPModel] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(X)
        rng = np.random.RandomState(42)
        self._members = []

        for i in range(self.n_members):
            # Bootstrap sample.
            idx = rng.randint(0, n, n)
            X_boot = X[idx]
            y_boot = y[idx]

            member = MLPModel(
                hidden_dim=self.hidden_dim,
                n_epochs=self.n_epochs,
                lr=self.lr,
                seed=42 + i * 100,
            )
            member.fit(X_boot, y_boot)
            self._members.append(member)

        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X), dtype=np.float32)
        preds = np.array([m.predict(X) for m in self._members])
        return preds.mean(axis=0).astype(np.float32)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Ensemble std (per-sample)."""
        if not self._fitted:
            return np.zeros(len(X), dtype=np.float32)
        preds = np.array([m.predict(X) for m in self._members])
        return preds.std(axis=0).astype(np.float32)

    def predict_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.abs(y - self.predict(X))

    @property
    def name(self) -> str:
        return "A4_bootstrap_mlp"


class QuantileMLPModel:
    """A5: Quantile MLP for conformalized quantile regression.

    Predicts the 5th, 50th, and 95th percentile of the advantage.
    Uses pinball loss for quantile regression.
    """

    def __init__(self, hidden_dim: int = 64, n_epochs: int = 300, lr: float = 0.01,
                 quantiles: tuple = (0.05, 0.5, 0.95)) -> None:
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.quantiles = quantiles
        self._W1 = None
        self._b1 = None
        self._W2 = None  # (hidden, 3) for 3 quantiles
        self._b2 = None  # (3,)
        self._fitted = False

    def _pinball_loss_grad(self, err: np.ndarray, q: float) -> np.ndarray:
        """Gradient of pinball loss for quantile q."""
        return np.where(err > 0, q, q - 1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_feat = X.shape[1]
        rng = np.random.RandomState(42)
        self._W1 = rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, 3) * np.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(3)

        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            preds = h @ self._W2 + self._b2  # (n, 3)

            for qi, q in enumerate(self.quantiles):
                err = y - preds[:, qi]
                grad_loss = self._pinball_loss_grad(err, q) / len(y)
                grad_out = grad_loss.reshape(-1, 1)
                grad_W2_q = h.T @ grad_out
                grad_b2_q = grad_out.sum(axis=0)

                # Accumulate gradients across quantiles.
                if qi == 0:
                    grad_W2 = grad_W2_q
                    grad_b2 = grad_b2_q
                else:
                    grad_W2 = np.column_stack([grad_W2, grad_W2_q])
                    grad_b2 = np.concatenate([grad_b2, grad_b2_q])

                # Backprop through hidden layer (sum across quantiles).
                if qi == 0:
                    grad_h_total = grad_out @ self._W2[:, qi:qi+1].T
                else:
                    grad_h_total += grad_out @ self._W2[:, qi:qi+1].T

            grad_h_total[grad_h_total > 0] = 0  # ReLU gradient (approximate)
            # Actually: grad_h should be zero where h <= 0.
            grad_h = grad_h_total
            grad_h[h <= 0] = 0
            grad_W1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)

            self._W1 -= self.lr * grad_W1
            self._b1 -= self.lr * grad_b1
            self._W2 -= self.lr * grad_W2
            self._b2 -= self.lr * grad_b2

        self._fitted = True

    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (q05, q50, q95) predictions."""
        if not self._fitted:
            n = len(X)
            return np.zeros(n), np.zeros(n), np.zeros(n)
        h = np.maximum(0, X @ self._W1 + self._b1)
        preds = h @ self._W2 + self._b2
        return preds[:, 0], preds[:, 1], preds[:, 2]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict median."""
        _, q50, _ = self.predict_quantiles(X)
        return q50.astype(np.float32)

    def predict_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.abs(y - self.predict(X))

    @property
    def name(self) -> str:
        return "A5_quantile_mlp"


def get_model_ladder() -> list:
    """Get the advantage model ladder."""
    return [
        ZeroAdvantageModel(),
        LinearRegressionModel(),
        RidgeRegressionModel(alpha=1.0),
        MLPModel(hidden_dim=64, n_epochs=300, lr=0.01),
        BootstrapMLPEnsemble(n_members=5, hidden_dim=64, n_epochs=300, lr=0.01),
        QuantileMLPModel(hidden_dim=64, n_epochs=300, lr=0.01),
    ]
