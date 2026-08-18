"""Uncertainty estimation: bootstrap, ensemble, quantile.

For simple models, uncertainty comes from residual variance.
For MLPs, use small deep ensembles before introducing more exotic
Bayesian machinery. A 5-member ensemble is enough to answer whether
epistemic disagreement is useful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import math
import numpy as np

from .protocol import Prediction


@dataclass(slots=True)
class UncertaintyReport:
    """Summary of uncertainty estimates for a set of predictions."""
    mean_uncertainty: float
    median_uncertainty: float
    max_uncertainty: float
    min_uncertainty: float
    n_predictions: int
    method: str  # "residual", "bootstrap", "ensemble", "quantile"

    def to_log(self) -> dict[str, Any]:
        return {
            "mean_uncertainty": float(self.mean_uncertainty),
            "median_uncertainty": float(self.median_uncertainty),
            "max_uncertainty": float(self.max_uncertainty),
            "min_uncertainty": float(self.min_uncertainty),
            "n_predictions": int(self.n_predictions),
            "method": self.method,
        }


def analyze_uncertainty(predictions: list[Prediction], method: str = "unknown") -> UncertaintyReport:
    """Analyze uncertainty estimates across a set of predictions."""
    if not predictions:
        return UncertaintyReport(0.0, 0.0, 0.0, 0.0, 0, method)
    uncertainties = [p.uncertainty for p in predictions]
    return UncertaintyReport(
        mean_uncertainty=float(np.mean(uncertainties)),
        median_uncertainty=float(np.median(uncertainties)),
        max_uncertainty=float(max(uncertainties)),
        min_uncertainty=float(min(uncertainties)),
        n_predictions=len(predictions),
        method=method,
    )


class BootstrapEnsemble:
    """Bootstrap ensemble for uncertainty estimation.

    Trains n_bootstrap models on resampled data and uses the spread
    of predictions as epistemic uncertainty.
    """

    def __init__(self, base_model_factory, n_bootstrap: int = 10, seed: int = 42) -> None:
        self.base_model_factory = base_model_factory
        self.n_bootstrap = int(n_bootstrap)
        self.seed = int(seed)
        self._models: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray, *, split: str = "train") -> dict[str, Any]:
        if split != "train":
            raise ValueError(f"Cannot fit on '{split}' split. Use 'train' only.")
        rng = np.random.RandomState(self.seed)
        n = len(X)
        self._models = []
        for i in range(self.n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            model = self.base_model_factory()
            model.fit(X_boot, y_boot, split="train")
            self._models.append(model)
        return {"n_bootstrap": len(self._models), "n_samples": n}

    def predict(self, X: np.ndarray) -> list[Prediction]:
        if not self._models:
            return [Prediction(mean=0.0, uncertainty=1.0, model_id="bootstrap") for _ in X]
        all_preds = []
        for model in self._models:
            preds = model.predict(X)
            all_preds.append([p.mean for p in preds])
        all_preds = np.array(all_preds)  # (n_bootstrap, n_samples)
        means = all_preds.mean(axis=0)
        stds = all_preds.std(axis=0)
        return [Prediction(
            mean=float(means[i]),
            uncertainty=float(stds[i]),
            model_id="bootstrap-ensemble",
        ) for i in range(len(X))]


def quantile_uncertainty(predictions: list[float], lower_q: float = 0.1, upper_q: float = 0.9) -> float:
    """Compute uncertainty from quantile spread."""
    if not predictions:
        return 0.0
    lower = float(np.percentile(predictions, lower_q * 100))
    upper = float(np.percentile(predictions, upper_q * 100))
    return (upper - lower) / 2.0
