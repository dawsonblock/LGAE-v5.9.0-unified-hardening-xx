"""Expanded model ladder B0-B6 for exp6.4.

B0: zero bonus (greedy baseline)
B1: logistic threshold classifier
B2: decision tree classifier
B3: gradient-boosted tree
B4: small MLP
B5: ensemble MLP

For the threshold problem, tree models should perform well because
the true decision boundary is structurally simple:
  if components_remaining == 1 and candidate_merges_components:
      future bonus probability high
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from .structural_features import extract_structural_features, compute_component_info


class BonusModel:
    """Base class for bonus prediction models."""

    def fit(self, X: np.ndarray, y_bonus: np.ndarray,
            y_threshold: np.ndarray | None = None,
            y_delta_comp: np.ndarray | None = None) -> None:
        raise NotImplementedError

    def predict_bonus(self, graph: GraphBuffers, z: torch.Tensor,
                      action: tuple[str, int, int, dict] | None = None,
                      *, threshold: int = 1, horizon: int = 2) -> float:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "base"


class B0Zero(BonusModel):
    """B0: zero bonus. Pure greedy baseline."""

    def fit(self, X, y_bonus, y_threshold=None, y_delta_comp=None):
        pass

    def predict_bonus(self, graph, z, action=None, *, threshold=1, horizon=2):
        return 0.0

    @property
    def name(self):
        return "B0_zero"


class B1Logistic(BonusModel):
    """B1: logistic regression threshold classifier.

    Predicts P(threshold reached | S, a) and multiplies by lambda.
    V_bonus = lambda * P(threshold reached)
    """

    def __init__(self, lambda_conn: float = 30.0) -> None:
        self.lambda_conn = lambda_conn
        self._model = None
        self._fitted = False

    def fit(self, X, y_bonus, y_threshold=None, y_delta_comp=None):
        from sklearn.linear_model import LogisticRegression
        if y_threshold is None:
            # Derive threshold labels from bonus: bonus > 0 means threshold reached.
            y_threshold = (y_bonus > 0).astype(int)
        self._model = LogisticRegression(max_iter=1000, C=1.0)
        self._model.fit(X, y_threshold)
        self._fitted = True

    def predict_bonus(self, graph, z, action=None, *, threshold=1, horizon=2):
        if not self._fitted or action is None:
            return 0.0
        x = extract_structural_features(graph, z, action, threshold=threshold, horizon=horizon)
        prob = self._model.predict_proba(x.reshape(1, -1))[0]
        # P(class=1) * lambda
        p_reach = prob[1] if len(prob) > 1 else prob[0]
        return self.lambda_conn * p_reach

    @property
    def name(self):
        return "B1_logistic"


class B2Tree(BonusModel):
    """B2: decision tree classifier.

    Trees can capture the non-linear threshold boundary directly:
    if components_remaining <= 1 and merges_components:
        predict high bonus
    """

    def __init__(self, lambda_conn: float = 30.0, max_depth: int = 8) -> None:
        self.lambda_conn = lambda_conn
        self.max_depth = max_depth
        self._model = None
        self._fitted = False

    def fit(self, X, y_bonus, y_threshold=None, y_delta_comp=None):
        from sklearn.tree import DecisionTreeRegressor
        # Train regression tree on bonus directly.
        self._model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
        self._model.fit(X, y_bonus)
        self._fitted = True

    def predict_bonus(self, graph, z, action=None, *, threshold=1, horizon=2):
        if not self._fitted or action is None:
            return 0.0
        x = extract_structural_features(graph, z, action, threshold=threshold, horizon=horizon)
        return float(self._model.predict(x.reshape(1, -1))[0])

    @property
    def name(self):
        return "B2_tree"


class B3GBT(BonusModel):
    """B3: gradient-boosted tree regressor."""

    def __init__(self, lambda_conn: float = 30.0, n_estimators: int = 100) -> None:
        self.lambda_conn = lambda_conn
        self.n_estimators = n_estimators
        self._model = None
        self._fitted = False

    def fit(self, X, y_bonus, y_threshold=None, y_delta_comp=None):
        from sklearn.ensemble import GradientBoostingRegressor
        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators, max_depth=4,
            learning_rate=0.1, random_state=42,
        )
        self._model.fit(X, y_bonus)
        self._fitted = True

    def predict_bonus(self, graph, z, action=None, *, threshold=1, horizon=2):
        if not self._fitted or action is None:
            return 0.0
        x = extract_structural_features(graph, z, action, threshold=threshold, horizon=horizon)
        return float(self._model.predict(x.reshape(1, -1))[0])

    @property
    def name(self):
        return "B3_gbt"


class B4MLP(BonusModel):
    """B4: small MLP regressor."""

    def __init__(self, lambda_conn: float = 30.0, hidden_dim: int = 64,
                 n_epochs: int = 300, lr: float = 0.01) -> None:
        self.lambda_conn = lambda_conn
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self._W1 = None
        self._b1 = None
        self._W2 = None
        self._b2 = 0.0
        self._fitted = False

    def fit(self, X, y_bonus, y_threshold=None, y_delta_comp=None):
        rng = np.random.RandomState(42)
        n_feat = X.shape[1]
        self._W1 = rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self._b2 = np.zeros(1)

        for _ in range(self.n_epochs):
            h = np.maximum(0, X @ self._W1 + self._b1)
            pred = (h @ self._W2 + self._b2).flatten()
            err = pred - y_bonus
            grad_out = err.reshape(-1, 1) / len(y_bonus)
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

    def predict_bonus(self, graph, z, action=None, *, threshold=1, horizon=2):
        if not self._fitted or action is None:
            return 0.0
        x = extract_structural_features(graph, z, action, threshold=threshold, horizon=horizon)
        h = np.maximum(0, x @ self._W1 + self._b1)
        return float((h @ self._W2 + self._b2).item())

    @property
    def name(self):
        return "B4_mlp"


class B5EnsembleMLP(BonusModel):
    """B5: ensemble of small MLPs for uncertainty estimation."""

    def __init__(self, lambda_conn: float = 30.0, n_models: int = 5,
                 hidden_dim: int = 32, n_epochs: int = 200) -> None:
        self.lambda_conn = lambda_conn
        self.n_models = n_models
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self._models: list[B4MLP] = []
        self._fitted = False

    def fit(self, X, y_bonus, y_threshold=None, y_delta_comp=None):
        self._models = []
        for i in range(self.n_models):
            m = B4MLP(lambda_conn=self.lambda_conn, hidden_dim=self.hidden_dim,
                      n_epochs=self.n_epochs, lr=0.01)
            # Bootstrap sample.
            rng = np.random.RandomState(42 + i)
            idx = rng.choice(len(X), size=len(X), replace=True)
            m.fit(X[idx], y_bonus[idx])
            self._models.append(m)
        self._fitted = True

    def predict_bonus(self, graph, z, action=None, *, threshold=1, horizon=2):
        if not self._fitted or action is None:
            return 0.0
        preds = [m.predict_bonus(graph, z, action, threshold=threshold, horizon=horizon)
                 for m in self._models]
        return float(np.mean(preds))

    def predict_bonus_std(self, graph, z, action=None, *, threshold=1, horizon=2):
        """Predict bonus standard deviation (uncertainty)."""
        if not self._fitted or action is None:
            return 0.0
        preds = [m.predict_bonus(graph, z, action, threshold=threshold, horizon=horizon)
                 for m in self._models]
        return float(np.std(preds))

    @property
    def name(self):
        return "B5_ensemble_mlp"


def get_model_ladder(lambda_conn: float = 30.0) -> list[BonusModel]:
    """Get the full model ladder B0-B5."""
    return [
        B0Zero(),
        B1Logistic(lambda_conn=lambda_conn),
        B2Tree(lambda_conn=lambda_conn),
        B3GBT(lambda_conn=lambda_conn),
        B4MLP(lambda_conn=lambda_conn),
        B5EnsembleMLP(lambda_conn=lambda_conn),
    ]
