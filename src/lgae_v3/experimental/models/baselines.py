"""Baseline predictors: global mean, mutation-type mean, nearest experience.

These are deliberately simple baselines that every real model must beat.

If no model beats these on held-out graph families, the structural
transition problem either lacks sufficient predictable signal or the
current state/action representation is incomplete.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np

from .protocol import Prediction, ClassificationPrediction, ModelLifecycle, config_hash


class GlobalMeanPredictor:
    """Predicts the global mean of the training target.

    This is the absolute floor. If a model cannot beat this, there is
    no learnable signal.
    """

    model_type = "global_mean"
    version = "v1"
    requires_fit = True
    deterministic = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self._mean = 0.0
        self._std = 0.0
        self._lifecycle = ModelLifecycle.UNFIT
        self._n_samples = 0

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{config_hash({'seed': self.seed})}"

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def fit(self, X: np.ndarray, y: np.ndarray, *, split: str = "train") -> dict[str, Any]:
        if split != "train":
            raise ValueError(f"Cannot fit on '{split}' split. Use 'train' only.")
        if self._lifecycle == ModelLifecycle.FROZEN:
            raise RuntimeError("Cannot fit frozen model.")
        self._mean = float(np.mean(y)) if len(y) > 0 else 0.0
        self._std = float(np.std(y)) if len(y) > 0 else 0.0
        self._n_samples = len(y)
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {"mean": self._mean, "std": self._std, "n_samples": self._n_samples}

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        self._lifecycle = ModelLifecycle.FROZEN

    def predict(self, X: np.ndarray) -> list[Prediction]:
        n = len(X)
        return [Prediction(
            mean=self._mean,
            uncertainty=self._std,
            model_id=self.model_id,
            calibration_state=self._lifecycle,
        ) for _ in range(n)]


class MutationTypeMeanPredictor:
    """Predicts the mean target conditioned on mutation type.

    Uses the action type one-hot portion of the representation to
    condition the prediction. This is a stronger baseline than global
    mean because it captures the average effect of each mutation type.
    """

    model_type = "mutation_type_mean"
    version = "v1"
    requires_fit = True
    deterministic = True

    def __init__(self, n_action_types: int = 8, action_offset: int = 0, seed: int = 42) -> None:
        self.seed = int(seed)
        self.n_action_types = int(n_action_types)
        self.action_offset = int(action_offset)  # offset in the feature vector
        self._type_means: dict[int, float] = {}
        self._type_stds: dict[int, float] = {}
        self._global_mean = 0.0
        self._global_std = 0.0
        self._lifecycle = ModelLifecycle.UNFIT
        self._n_samples = 0

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{config_hash({'seed': self.seed, 'n_types': self.n_action_types})}"

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def _get_action_type(self, x: np.ndarray) -> int:
        """Extract the action type from a feature vector."""
        if len(x) <= self.action_offset + self.n_action_types:
            return -1
        segment = x[self.action_offset:self.action_offset + self.n_action_types]
        idx = int(np.argmax(segment))
        return idx if segment[idx] > 0.5 else -1

    def fit(self, X: np.ndarray, y: np.ndarray, *, split: str = "train") -> dict[str, Any]:
        if split != "train":
            raise ValueError(f"Cannot fit on '{split}' split. Use 'train' only.")
        if self._lifecycle == ModelLifecycle.FROZEN:
            raise RuntimeError("Cannot fit frozen model.")
        self._global_mean = float(np.mean(y)) if len(y) > 0 else 0.0
        self._global_std = float(np.std(y)) if len(y) > 0 else 0.0
        self._n_samples = len(y)
        # Group by action type.
        type_groups: dict[int, list[float]] = {}
        for i in range(len(X)):
            t = self._get_action_type(X[i])
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(float(y[i]))
        for t, vals in type_groups.items():
            self._type_means[t] = float(np.mean(vals))
            self._type_stds[t] = float(np.std(vals)) if len(vals) > 1 else self._global_std
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {
            "n_types": len(self._type_means),
            "global_mean": self._global_mean,
            "n_samples": self._n_samples,
        }

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        self._lifecycle = ModelLifecycle.FROZEN

    def predict(self, X: np.ndarray) -> list[Prediction]:
        results = []
        for i in range(len(X)):
            t = self._get_action_type(X[i])
            mean = self._type_means.get(t, self._global_mean)
            std = self._type_stds.get(t, self._global_std)
            results.append(Prediction(
                mean=mean,
                uncertainty=std,
                model_id=self.model_id,
                calibration_state=self._lifecycle,
            ))
        return results


class NearestExperiencePredictor:
    """Predicts based on the nearest structural experience in the training set.

    Uses simple Euclidean distance in feature space. This is a k-NN
    baseline with k=1. It captures whether similar structural situations
    have similar outcomes.
    """

    model_type = "nearest_experience"
    version = "v1"
    requires_fit = True
    deterministic = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._global_mean = 0.0
        self._global_std = 0.0
        self._lifecycle = ModelLifecycle.UNFIT

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{config_hash({'seed': self.seed})}"

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def fit(self, X: np.ndarray, y: np.ndarray, *, split: str = "train") -> dict[str, Any]:
        if split != "train":
            raise ValueError(f"Cannot fit on '{split}' split. Use 'train' only.")
        if self._lifecycle == ModelLifecycle.FROZEN:
            raise RuntimeError("Cannot fit frozen model.")
        self._X_train = np.array(X, dtype=np.float64)
        self._y_train = np.array(y, dtype=np.float64)
        self._global_mean = float(np.mean(y)) if len(y) > 0 else 0.0
        self._global_std = float(np.std(y)) if len(y) > 0 else 0.0
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {"n_train": len(y), "global_mean": self._global_mean}

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        self._lifecycle = ModelLifecycle.FROZEN

    def predict(self, X: np.ndarray) -> list[Prediction]:
        if self._X_train is None or self._y_train is None:
            return [Prediction(mean=0.0, uncertainty=1.0, model_id=self.model_id) for _ in X]
        results = []
        for i in range(len(X)):
            # Euclidean distance.
            dists = np.sqrt(np.sum((self._X_train - X[i]) ** 2, axis=1))
            nearest_idx = int(np.argmin(dists))
            nearest_dist = float(dists[nearest_idx])
            # Uncertainty proportional to distance.
            mean = float(self._y_train[nearest_idx])
            uncertainty = nearest_dist * self._global_std if self._global_std > 0 else 1.0
            results.append(Prediction(
                mean=mean,
                uncertainty=uncertainty,
                model_id=self.model_id,
                calibration_state=self._lifecycle,
                metadata={"nearest_distance": nearest_dist},
            ))
        return results
