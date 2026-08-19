"""Small MLP predictor with deep ensemble uncertainty.

A 5-member ensemble is enough to answer whether epistemic disagreement
is useful. Uses small MLPs with mean + max pooling.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import Prediction, ClassificationPrediction, ModelLifecycle, config_hash


class SmallMLP(nn.Module):
    """A small MLP for regression or classification."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 1, n_layers: int = 2) -> None:
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegressor:
    """Small MLP regression predictor with deep ensemble uncertainty."""

    model_type = "mlp"
    version = "v1"
    requires_fit = True
    deterministic = False  # Neural network (deterministic with fixed seed)

    def __init__(
        self,
        hidden_dim: int = 32,
        n_layers: int = 2,
        n_ensemble: int = 5,
        lr: float = 0.01,
        n_epochs: int = 100,
        seed: int = 42,
    ) -> None:
        self.seed = int(seed)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_ensemble = int(n_ensemble)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self._models: list[SmallMLP] = []
        self._in_dim: int = 0
        self._lifecycle = ModelLifecycle.UNFIT
        self._n_samples = 0

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{config_hash({'seed': self.seed, 'hidden': self.hidden_dim, 'n_ens': self.n_ensemble, 'lr': self.lr, 'epochs': self.n_epochs})}"

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def n_parameters(self) -> int:
        if not self._models:
            return 0
        return sum(sum(p.numel() for p in m.parameters()) for m in self._models)

    def fit(self, X: np.ndarray, y: np.ndarray, *, split: str = "train") -> dict[str, Any]:
        if split != "train":
            raise ValueError(f"Cannot fit on '{split}' split. Use 'train' only.")
        if self._lifecycle == ModelLifecycle.FROZEN:
            raise RuntimeError("Cannot fit frozen model.")
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n, d = X.shape
        self._in_dim = d
        self._n_samples = n
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self._models = []
        final_losses = []
        for i in range(self.n_ensemble):
            torch.manual_seed(self.seed + i)
            model = SmallMLP(d, self.hidden_dim, 1, self.n_layers)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            for epoch in range(self.n_epochs):
                opt.zero_grad()
                pred = model(X_t)
                loss = F.mse_loss(pred, y_t)
                loss.backward()
                opt.step()
            self._models.append(model)
            final_losses.append(float(loss.item()))
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {"n_ensemble": len(self._models), "final_losses": final_losses, "n_samples": n}

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        for m in self._models:
            for p in m.parameters():
                p.requires_grad = False
        self._lifecycle = ModelLifecycle.FROZEN

    def predict(self, X: np.ndarray) -> list[Prediction]:
        if not self._models:
            return [Prediction(mean=0.0, uncertainty=1.0, model_id=self.model_id) for _ in X]
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_t = torch.tensor(X, dtype=torch.float32)
        # Ensemble predictions.
        all_preds = []
        with torch.no_grad():
            for model in self._models:
                preds = model(X_t).numpy().flatten()
                all_preds.append(preds)
        all_preds = np.array(all_preds)  # (n_ensemble, n_samples)
        means = all_preds.mean(axis=0)
        stds = all_preds.std(axis=0)
        return [Prediction(
            mean=float(means[i]),
            uncertainty=float(stds[i]),
            model_id=self.model_id,
            calibration_state=self._lifecycle,
        ) for i in range(len(X))]

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type, "version": self.version,
            "seed": self.seed, "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers, "n_ensemble": self.n_ensemble,
            "lr": self.lr, "n_epochs": self.n_epochs,
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "ensemble_states": [
                m.state_dict() for m in self._models
            ],
            "in_dim": self._in_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._in_dim = int(state["in_dim"])
        self._models = []
        for sd in state["ensemble_states"]:
            model = SmallMLP(
                self._in_dim, int(state["hidden_dim"]),
                1, int(state["n_layers"]),
            )
            model.load_state_dict(sd)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self._models.append(model)
        self._lifecycle = ModelLifecycle.FROZEN


class MLPClassifier:
    """Small MLP classification predictor with ensemble uncertainty."""

    model_type = "mlp_clf"
    version = "v1"
    requires_fit = True
    deterministic = False

    def __init__(
        self,
        hidden_dim: int = 32,
        n_layers: int = 2,
        n_ensemble: int = 5,
        lr: float = 0.01,
        n_epochs: int = 100,
        seed: int = 42,
    ) -> None:
        self.seed = int(seed)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_ensemble = int(n_ensemble)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self._models: list[SmallMLP] = []
        self._in_dim: int = 0
        self._lifecycle = ModelLifecycle.UNFIT

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}-{config_hash({'seed': self.seed, 'hidden': self.hidden_dim, 'n_ens': self.n_ensemble})}"

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
        self._in_dim = d
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self._models = []
        for i in range(self.n_ensemble):
            torch.manual_seed(self.seed + i)
            model = SmallMLP(d, self.hidden_dim, 1, self.n_layers)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            for epoch in range(self.n_epochs):
                opt.zero_grad()
                logits = model(X_t)
                loss = F.binary_cross_entropy_with_logits(logits, y_t)
                loss.backward()
                opt.step()
            self._models.append(model)
        self._lifecycle = ModelLifecycle.FITTED_TRAIN
        return {"n_ensemble": len(self._models), "n_samples": n}

    def freeze(self) -> None:
        if self._lifecycle != ModelLifecycle.FITTED_TRAIN:
            raise RuntimeError("Cannot freeze unfitted model.")
        for m in self._models:
            for p in m.parameters():
                p.requires_grad = False
        self._lifecycle = ModelLifecycle.FROZEN

    def predict_proba(self, X: np.ndarray) -> list[ClassificationPrediction]:
        if not self._models:
            return [ClassificationPrediction(probability=0.5, predicted_class=0, uncertainty=1.0, model_id=self.model_id) for _ in X]
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_t = torch.tensor(X, dtype=torch.float32)
        all_probs = []
        with torch.no_grad():
            for model in self._models:
                logits = model(X_t)
                probs = torch.sigmoid(logits).numpy().flatten()
                all_probs.append(probs)
        all_probs = np.array(all_probs)
        mean_probs = all_probs.mean(axis=0)
        prob_stds = all_probs.std(axis=0)
        return [ClassificationPrediction(
            probability=float(mean_probs[i]),
            predicted_class=int(mean_probs[i] > 0.5),
            uncertainty=float(prob_stds[i]),
            model_id=self.model_id,
            calibration_state=self._lifecycle,
        ) for i in range(len(X))]

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type, "version": self.version,
            "seed": self.seed, "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers, "n_ensemble": self.n_ensemble,
            "lr": self.lr, "n_epochs": self.n_epochs,
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "ensemble_states": [
                {k: v.tolist() for k, v in m.state_dict().items()}
                for m in self._models
            ],
            "in_dim": self._in_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._in_dim = int(state["in_dim"])
        self._models = []
        for sd in state["ensemble_states"]:
            model = SmallMLP(
                self._in_dim, int(state["hidden_dim"]),
                1, int(state["n_layers"]),
            )
            model.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self._models.append(model)
        self._lifecycle = ModelLifecycle.FROZEN
