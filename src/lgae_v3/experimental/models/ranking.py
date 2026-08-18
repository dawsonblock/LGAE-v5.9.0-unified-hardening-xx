"""Ranking models: pointwise score and pairwise ranking.

MPC ultimately cares more about ordering candidates than about getting
the exact utility delta numerically perfect. A predictor can have
mediocre numerical RMSE and still be an excellent planner surrogate if
rank(a_1, a_2, ...) is correct.

Pairwise ranking directly targets the planner's decision problem:
    P(a_i > a_j | S, a_i, a_j)
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np

from .protocol import RankingPrediction, ModelLifecycle, config_hash, safe_sigmoid


class PointwiseRankingModel:
    """Pointwise ranking: trains a regression model on candidate scores,
    then ranks by predicted score.

    This is equivalent to using a regression model for ranking — the
    key insight is that the evaluation metric is ranking quality, not
    absolute error.
    """

    model_type = "pointwise_rank"
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
        if n == 0:
            # Empty dataset: initialize with defaults.
            if self._weights is None:
                rng = np.random.RandomState(self.seed)
                self._weights = rng.randn(max(d, 1)) * 0.01
                self._bias = 0.0
            self._lifecycle = ModelLifecycle.FITTED_TRAIN
            return {"n_samples": 0, "n_features": d, "empty": True}
        rng = np.random.RandomState(self.seed)
        self._weights = rng.randn(d) * 0.01
        self._bias = 0.0
        for _ in range(self.n_epochs):
            pred = X @ self._weights + self._bias
            grad = (pred - y) / n
            self._weights -= self.lr * (X.T @ grad)
            self._bias -= self.lr * float(np.mean(grad))
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {"n_samples": n, "n_features": d}

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        self._lifecycle = ModelLifecycle.FROZEN

    def rank(self, X: np.ndarray) -> RankingPrediction:
        if self._weights is None:
            n = len(X)
            return RankingPrediction(
                scores=tuple(0.0 for _ in range(n)),
                ranked_indices=tuple(range(n)),
                model_id=self.model_id,
            )
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        scores = X @ self._weights + self._bias
        ranked = np.argsort(-scores)  # descending
        return RankingPrediction(
            scores=tuple(float(s) for s in scores),
            ranked_indices=tuple(int(i) for i in ranked),
            model_id=self.model_id,
        )


class PairwiseRankingModel:
    """Pairwise ranking: trains on pairwise comparisons.

    For each pair (i, j) from the same state:
        y = 1[ΔU_i > ΔU_j]
    Train: P(a_i > a_j | S, a_i, a_j)

    This directly targets the planner's decision problem and may
    outperform absolute utility regression even with simpler features.
    """

    model_type = "pairwise_rank"
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

    def fit_pairwise(
        self,
        X: np.ndarray,
        pairs: np.ndarray,
        labels: np.ndarray,
        *,
        split: str = "train",
    ) -> dict[str, Any]:
        """Fit on pairwise examples.

        Args:
            X: (n_candidates, d) feature matrix.
            pairs: (n_pairs, 2) array of (i, j) index pairs.
            labels: (n_pairs,) array of {0, 1} labels (1 if ΔU_i > ΔU_j).
            split: Must be "train".
        """
        if split != "train":
            raise ValueError(f"Cannot fit on '{split}' split. Use 'train' only.")
        if self._lifecycle == ModelLifecycle.FROZEN:
            raise RuntimeError("Cannot fit frozen model.")
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n, d = X.shape
        rng = np.random.RandomState(self.seed)
        self._weights = rng.randn(d) * 0.01
        self._bias = 0.0
        # Pairwise features: x_i - x_j (difference encodes preference).
        for _ in range(self.n_epochs):
            total_grad_w = np.zeros(d)
            total_grad_b = 0.0
            for p in range(len(pairs)):
                i, j = pairs[p]
                diff = X[i] - X[j]
                score = diff @ self._weights + self._bias
                prob = safe_sigmoid(float(score))
                grad = prob - labels[p]
                total_grad_w += grad * diff
                total_grad_b += grad
            n_pairs = len(pairs)
            if n_pairs > 0:
                self._weights -= self.lr * total_grad_w / n_pairs
                self._bias -= self.lr * total_grad_b / n_pairs
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {"n_pairs": len(pairs), "n_features": d}

    def fit(self, X: np.ndarray, y: np.ndarray, *, split: str = "train") -> dict[str, Any]:
        """Fit by generating pairwise examples from absolute scores."""
        from .targets import compute_pairwise_labels
        pairs, labels = compute_pairwise_labels(list(y))
        if len(pairs) == 0:
            # Fallback to pointwise.
            if self._weights is None:
                rng = np.random.RandomState(self.seed)
                self._weights = rng.randn(X.shape[1]) * 0.01
                self._bias = 0.0
            self._lifecycle = ModelLifecycle.FITTED_TRAIN
            return {"n_pairs": 0, "fallback": True}
        return self.fit_pairwise(X, pairs, labels, split=split)

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        self._lifecycle = ModelLifecycle.FROZEN

    def rank(self, X: np.ndarray) -> RankingPrediction:
        if self._weights is None:
            n = len(X)
            return RankingPrediction(
                scores=tuple(0.0 for _ in range(n)),
                ranked_indices=tuple(range(n)),
                model_id=self.model_id,
            )
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        scores = X @ self._weights + self._bias
        ranked = np.argsort(-scores)
        return RankingPrediction(
            scores=tuple(float(s) for s in scores),
            ranked_indices=tuple(int(i) for i in ranked),
            model_id=self.model_id,
        )
