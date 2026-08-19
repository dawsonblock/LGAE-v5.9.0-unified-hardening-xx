"""Gradient-boosted tree predictor with quantile uncertainty.

Uses a simple gradient boosting implementation with decision stumps.
Uncertainty comes from quantile predictions.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np

from .protocol import Prediction, ModelLifecycle, config_hash


class DecisionStump:
    """A single decision stump (depth-1 tree)."""

    def __init__(self) -> None:
        self.feature: int = 0
        self.threshold: float = 0.0
        self.left_value: float = 0.0
        self.right_value: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        n, d = X.shape
        best_loss = float("inf")
        for feat in range(d):
            thresholds = np.unique(X[:, feat])
            if len(thresholds) <= 1:
                continue
            for t in thresholds[:-1]:
                left = X[:, feat] <= t
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue
                lv = float(np.average(y[left], weights=weights[left]))
                rv = float(np.average(y[right], weights=weights[right]))
                pred = np.where(left, lv, rv)
                loss = float(np.average((pred - y) ** 2, weights=weights))
                if loss < best_loss:
                    best_loss = loss
                    self.feature = feat
                    self.threshold = float(t)
                    self.left_value = lv
                    self.right_value = rv

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, self.feature] <= self.threshold, self.left_value, self.right_value)


class GradientBoostedTreePredictor:
    """Simple gradient-boosted regression tree with quantile uncertainty.

    Uses decision stumps as weak learners with gradient boosting.
    Uncertainty is estimated from residual quantiles.
    """

    model_type = "tree"
    version = "v1"
    requires_fit = True
    deterministic = True

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 1,
        seed: int = 42,
    ) -> None:
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self._stumps: list[DecisionStump] = []
        self._initial_value: float = 0.0
        self._residual_quantiles: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 10th, 50th, 90th
        self._residual_std: float = 1.0
        self._lifecycle = ModelLifecycle.UNFIT
        self._n_samples = 0
        self._n_features = 0

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{config_hash({'seed': self.seed, 'n_est': self.n_estimators, 'lr': self.learning_rate})}"

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def n_parameters(self) -> int:
        return len(self._stumps) * 4 + 1  # each stump has 4 params + initial value

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
        self._initial_value = float(np.mean(y))
        pred = np.full(n, self._initial_value)
        weights = np.ones(n) / n
        self._stumps = []
        for _ in range(self.n_estimators):
            residuals = y - pred
            stump = DecisionStump()
            stump.fit(X, residuals, weights)
            update = stump.predict(X)
            pred += self.learning_rate * update
            self._stumps.append(stump)
        # Residual quantiles for uncertainty.
        residuals = y - pred
        self._residual_quantiles = (
            float(np.percentile(residuals, 10)),
            float(np.percentile(residuals, 50)),
            float(np.percentile(residuals, 90)),
        )
        self._residual_std = float(np.std(residuals)) if n > 1 else 1.0
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {
            "n_estimators": len(self._stumps),
            "residual_std": self._residual_std,
            "n_samples": n,
        }

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        self._lifecycle = ModelLifecycle.FROZEN

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pred = np.full(len(X), self._initial_value)
        for stump in self._stumps:
            pred += self.learning_rate * stump.predict(X)
        return pred

    def predict(self, X: np.ndarray) -> list[Prediction]:
        means = self._predict_raw(X)
        q10, q50, q90 = self._residual_quantiles
        iqr = q90 - q10
        return [Prediction(
            mean=float(m),
            uncertainty=float(iqr / 2.0),  # half-IQR as uncertainty
            model_id=self.model_id,
            calibration_state=self._lifecycle,
            lower=float(m + q10),
            upper=float(m + q90),
        ) for m in means]

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type, "version": self.version,
            "seed": self.seed, "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate, "max_depth": self.max_depth,
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "stumps": [
                {
                    "feature": int(s.feature),
                    "threshold": float(s.threshold),
                    "left_value": float(s.left_value),
                    "right_value": float(s.right_value),
                }
                for s in self._stumps
            ],
            "initial_value": self._initial_value,
            "residual_quantiles": list(self._residual_quantiles),
            "residual_std": self._residual_std,
            "n_samples": self._n_samples,
            "n_features": self._n_features,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._stumps = []
        for s in state["stumps"]:
            stump = DecisionStump()
            stump.feature = int(s["feature"])
            stump.threshold = float(s["threshold"])
            stump.left_value = float(s["left_value"])
            stump.right_value = float(s["right_value"])
            self._stumps.append(stump)
        self._initial_value = float(state["initial_value"])
        self._residual_quantiles = tuple(state["residual_quantiles"])
        self._residual_std = float(state["residual_std"])
        self._n_samples = int(state["n_samples"])
        self._n_features = int(state["n_features"])
        self._lifecycle = ModelLifecycle.FROZEN
