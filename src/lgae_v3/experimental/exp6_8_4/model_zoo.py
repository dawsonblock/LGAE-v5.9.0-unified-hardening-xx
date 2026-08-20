"""Model zoo for exp6.8.4.

Models:
  M1_ridge: Ridge regression (baseline)
  M2_gbt: Gradient-boosted trees (tabular structured prediction)
  M3_mlp: Small MLP
  M4_pairwise: Pairwise comparison model (learned ranking)

For classification targets (sign, ordinal), models adapt to classification.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class RidgeModel:
    """M1: Ridge regression."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._w = None
        self._b = 0.0
        self._is_classifier = False

    def fit(self, X: np.ndarray, y: np.ndarray, is_classification: bool = False) -> None:
        self._is_classifier = is_classification
        n, d = X.shape
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
        return (X @ self._w + self._b).astype(np.float32)

    @property
    def name(self) -> str:
        return "M1_ridge"


class GBTModel:
    """M2: Gradient-boosted trees.

    Simple implementation using decision stumps as weak learners.
    For tabular structured prediction, this often outperforms MLPs.
    """

    def __init__(self, n_estimators: int = 100, lr: float = 0.1, max_depth: int = 3) -> None:
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self._trees: list[dict] = []
        self._init_pred: float = 0.0
        self._is_classifier = False

    def fit(self, X: np.ndarray, y: np.ndarray, is_classification: bool = False) -> None:
        self._is_classifier = is_classification
        self._init_pred = float(np.mean(y))
        residual = y - self._init_pred

        for _ in range(self.n_estimators):
            tree = self._fit_tree(X, residual, depth=0)
            pred = self._predict_tree(X, tree)
            residual -= self.lr * pred
            self._trees.append(tree)

    def _fit_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        """Fit a shallow regression tree."""
        n, d = X.shape
        if depth >= self.max_depth or n < 5:
            return {"leaf": True, "value": float(np.mean(y)) if len(y) > 0 else 0.0}

        # Find best split.
        best_gain = -1e9
        best_feat = 0
        best_thresh = 0.0
        best_left_idx = None
        best_right_idx = None

        for feat in range(d):
            thresholds = np.percentile(X[:, feat], [25, 50, 75])
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = ~left_idx
                if np.sum(left_idx) < 2 or np.sum(right_idx) < 2:
                    continue
                gain = self._variance_gain(y, left_idx, right_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        if best_left_idx is None or best_gain <= 0:
            return {"leaf": True, "value": float(np.mean(y))}

        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": float(best_thresh),
            "left": self._fit_tree(X[best_left_idx], y[best_left_idx], depth + 1),
            "right": self._fit_tree(X[best_right_idx], y[best_right_idx], depth + 1),
        }

    def _variance_gain(self, y: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """Variance reduction from split."""
        n = len(y)
        var_total = np.var(y) * n
        var_left = np.var(y[left]) * np.sum(left)
        var_right = np.var(y[right]) * np.sum(right)
        return float(var_total - var_left - var_right)

    def _predict_tree(self, X: np.ndarray, tree: dict) -> np.ndarray:
        """Predict with a single tree."""
        if tree["leaf"]:
            return np.full(len(X), tree["value"], dtype=np.float32)
        left_idx = X[:, tree["feat"]] <= tree["thresh"]
        preds = np.zeros(len(X), dtype=np.float32)
        preds[left_idx] = self._predict_tree(X[left_idx], tree["left"])
        preds[~left_idx] = self._predict_tree(X[~left_idx], tree["right"])
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.full(len(X), self._init_pred, dtype=np.float32)
        for tree in self._trees:
            preds += self.lr * self._predict_tree(X, tree)
        return preds

    @property
    def name(self) -> str:
        return "M2_gbt"


class MLPModel:
    """M3: Small MLP."""

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
        self._is_classifier = False

    def fit(self, X: np.ndarray, y: np.ndarray, is_classification: bool = False) -> None:
        self._is_classifier = is_classification
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

    @property
    def name(self) -> str:
        return "M3_mlp"


class PairwiseModel:
    """M4: Pairwise comparison model.

    Instead of predicting A* directly, learns to rank:
      P(learned > baseline | features)

    Uses logistic regression on the feature difference.
    """

    def __init__(self, lr: float = 0.01, n_epochs: int = 500, l2: float = 0.01) -> None:
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self._w = None
        self._b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, is_classification: bool = False) -> None:
        # Binary target: 1 if A > 0, 0 otherwise.
        binary = (y > 0).astype(np.float32)
        n, d = X.shape
        self._w = np.zeros(d, dtype=np.float32)
        self._b = 0.0

        for _ in range(self.n_epochs):
            z = X @ self._w + self._b
            pred = 1.0 / (1.0 + np.exp(-z))
            err = pred - binary
            grad_w = X.T @ err / n + self.l2 * self._w
            grad_b = float(np.mean(err))
            self._w -= self.lr * grad_w
            self._b -= self.lr * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probability that learned > baseline."""
        if self._w is None:
            return np.zeros(len(X), dtype=np.float32)
        z = X @ self._w + self._b
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    @property
    def name(self) -> str:
        return "M4_pairwise"


def get_model_zoo() -> dict:
    """Get all models in the zoo."""
    return {
        "M1_ridge": RidgeModel,
        "M2_gbt": GBTModel,
        "M3_mlp": MLPModel,
        "M4_pairwise": PairwiseModel,
    }


def create_model(name: str):
    """Create a model instance by name."""
    zoo = get_model_zoo()
    cls = zoo.get(name, RidgeModel)
    if cls == GBTModel:
        return cls(n_estimators=100, lr=0.1, max_depth=3)
    elif cls == MLPModel:
        return cls(hidden_dim=64, n_epochs=300, lr=0.01)
    elif cls == PairwiseModel:
        return cls(lr=0.01, n_epochs=500, l2=0.01)
    elif cls == RidgeModel:
        return cls(alpha=1.0)
    return cls()
