"""Linear and Ridge regression predictors with residual-variance uncertainty.

For simple models, uncertainty comes from residual variance.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np

from .protocol import Prediction, ClassificationPrediction, ModelLifecycle, config_hash, safe_sigmoid


class LinearRegressionPredictor:
    """Ordinary least squares linear regression with residual uncertainty."""

    model_type = "linear"
    version = "v1"
    requires_fit = True
    deterministic = True

    def __init__(self, lr: float = 0.01, n_epochs: int = 200, seed: int = 42) -> None:
        self.seed = int(seed)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._residual_std: float = 1.0
        self._lifecycle = ModelLifecycle.UNFIT
        self._n_samples = 0
        self._n_features = 0

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{config_hash({'seed': self.seed, 'lr': self.lr, 'epochs': self.n_epochs})}"

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def n_parameters(self) -> int:
        if self._weights is None:
            return 0
        return len(self._weights) + 1  # weights + bias

    def fit(self, X: np.ndarray, y: np.ndarray, *, split: str = "train") -> dict[str, Any]:
        if split != "train":
            raise ValueError(f"Cannot fit on '{split}' split. Use 'train' only.")
        if self._lifecycle == ModelLifecycle.FROZEN:
            raise RuntimeError("Cannot fit frozen model.")
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n, d = X.shape
        self._n_samples = n
        self._n_features = d
        rng = np.random.RandomState(self.seed)
        self._weights = rng.randn(d) * 0.01
        self._bias = 0.0
        for _ in range(self.n_epochs):
            pred = X @ self._weights + self._bias
            grad = (pred - y) / n
            self._weights -= self.lr * (X.T @ grad)
            self._bias -= self.lr * float(np.mean(grad))
        # Residual std for uncertainty.
        residuals = (X @ self._weights + self._bias) - y
        self._residual_std = float(np.std(residuals)) if n > 1 else 1.0
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {"residual_std": self._residual_std, "n_samples": n, "n_features": d}

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        self._lifecycle = ModelLifecycle.FROZEN

    def predict(self, X: np.ndarray) -> list[Prediction]:
        if self._weights is None:
            return [Prediction(mean=0.0, uncertainty=1.0, model_id=self.model_id) for _ in X]
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        means = X @ self._weights + self._bias
        return [Prediction(
            mean=float(m),
            uncertainty=self._residual_std,
            model_id=self.model_id,
            calibration_state=self._lifecycle,
        ) for m in means]


class RidgeRegressionPredictor:
    """Ridge regression (L2-regularized linear regression)."""

    model_type = "ridge"
    version = "v1"
    requires_fit = True
    deterministic = True

    def __init__(self, alpha: float = 1.0, lr: float = 0.01, n_epochs: int = 200, seed: int = 42) -> None:
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._residual_std: float = 1.0
        self._lifecycle = ModelLifecycle.UNFIT
        self._n_samples = 0
        self._n_features = 0

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{config_hash({'seed': self.seed, 'alpha': self.alpha, 'lr': self.lr, 'epochs': self.n_epochs})}"

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def n_parameters(self) -> int:
        if self._weights is None:
            return 0
        return len(self._weights) + 1

    def fit(self, X: np.ndarray, y: np.ndarray, *, split: str = "train") -> dict[str, Any]:
        if split != "train":
            raise ValueError(f"Cannot fit on '{split}' split. Use 'train' only.")
        if self._lifecycle == ModelLifecycle.FROZEN:
            raise RuntimeError("Cannot fit frozen model.")
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n, d = X.shape
        self._n_samples = n
        self._n_features = d
        rng = np.random.RandomState(self.seed)
        self._weights = rng.randn(d) * 0.01
        self._bias = 0.0
        for _ in range(self.n_epochs):
            pred = X @ self._weights + self._bias
            error = pred - y
            grad_w = X.T @ error / n + self.alpha * self._weights / n
            grad_b = float(np.mean(error) / n)
            self._weights -= self.lr * grad_w
            self._bias -= self.lr * grad_b
        residuals = (X @ self._weights + self._bias) - y
        self._residual_std = float(np.std(residuals)) if n > 1 else 1.0
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {"residual_std": self._residual_std, "alpha": self.alpha, "n_samples": n}

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        self._lifecycle = ModelLifecycle.FROZEN

    def predict(self, X: np.ndarray) -> list[Prediction]:
        if self._weights is None:
            return [Prediction(mean=0.0, uncertainty=1.0, model_id=self.model_id) for _ in X]
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        means = X @ self._weights + self._bias
        return [Prediction(
            mean=float(m),
            uncertainty=self._residual_std,
            model_id=self.model_id,
            calibration_state=self._lifecycle,
        ) for m in means]


class LogisticRegressionPredictor:
    """Logistic regression for sign/success classification."""

    model_type = "logistic"
    version = "v1"
    requires_fit = True
    deterministic = True

    def __init__(self, lr: float = 0.01, n_epochs: int = 200, seed: int = 42) -> None:
        self.seed = int(seed)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._lifecycle = ModelLifecycle.UNFIT

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{config_hash({'seed': self.seed, 'lr': self.lr, 'epochs': self.n_epochs})}"

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def fit(self, X: np.ndarray, y: np.ndarray, *, split: str = "train") -> dict[str, Any]:
        if split != "train":
            raise ValueError(f"Cannot fit on '{split}' split. Use 'train' only.")
        if self._lifecycle == ModelLifecycle.FROZEN:
            raise RuntimeError("Cannot fit frozen model.")
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n, d = X.shape
        rng = np.random.RandomState(self.seed)
        self._weights = rng.randn(d) * 0.01
        self._bias = 0.0
        for _ in range(self.n_epochs):
            logits = np.clip(X @ self._weights + self._bias, -30, 30)
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad = (probs - y) / n
            self._weights -= self.lr * (X.T @ grad)
            self._bias -= self.lr * float(np.mean(grad))
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {"n_samples": n, "n_features": d}

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        self._lifecycle = ModelLifecycle.FROZEN

    def predict_proba(self, X: np.ndarray) -> list[ClassificationPrediction]:
        if self._weights is None:
            return [ClassificationPrediction(probability=0.5, predicted_class=0, uncertainty=1.0, model_id=self.model_id) for _ in X]
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        logits = np.clip(X @ self._weights + self._bias, -30, 30)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return [ClassificationPrediction(
            probability=float(p),
            predicted_class=int(p > 0.5),
            uncertainty=float(abs(p - 0.5) * 2),  # lower uncertainty when far from 0.5
            model_id=self.model_id,
            calibration_state=self._lifecycle,
        ) for p in probs]
