"""Canonical model protocols and prediction contracts for v6.0-exp4.

Every useful outcome model returns:

    Prediction(
        mean=...,
        uncertainty=...,
        model_id=...,
        calibration_state=...,
    )

The runtime later needs to know when not to trust the world model.
Exp4 is where that trust signal begins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable
import hashlib
import math
import numpy as np


# ---------------------------------------------------------------------------
# Prediction contract.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Prediction:
    """A single prediction with uncertainty.

    Attributes:
        mean: Predicted expected value.
        uncertainty: Epistemic uncertainty estimate (std or IQR).
        model_id: Identifier of the model that produced this prediction.
        calibration_state: Current calibration state ("uncalibrated", "fitted", "frozen").
        lower: Lower bound of prediction interval (optional).
        upper: Upper bound of prediction interval (optional).
        metadata: Additional prediction metadata.
    """
    mean: float
    uncertainty: float
    model_id: str
    calibration_state: str = "uncalibrated"
    lower: float | None = None
    upper: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "uncertainty": float(self.uncertainty),
            "model_id": self.model_id,
            "calibration_state": self.calibration_state,
            "lower": float(self.lower) if self.lower is not None else None,
            "upper": float(self.upper) if self.upper is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ClassificationPrediction:
    """A classification prediction with probability and uncertainty."""
    probability: float
    predicted_class: int
    uncertainty: float
    model_id: str
    calibration_state: str = "uncalibrated"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "probability": float(self.probability),
            "predicted_class": int(self.predicted_class),
            "uncertainty": float(self.uncertainty),
            "model_id": self.model_id,
            "calibration_state": self.calibration_state,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RankingPrediction:
    """A ranking prediction for a set of candidates."""
    scores: tuple[float, ...]
    ranked_indices: tuple[int, ...]
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "scores": list(self.scores),
            "ranked_indices": list(self.ranked_indices),
            "model_id": self.model_id,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Model lifecycle.
# ---------------------------------------------------------------------------

class ModelLifecycle(str):
    UNFIT = "unfit"
    FITTED_TRAIN = "fitted_train"
    FROZEN = "frozen"


# ---------------------------------------------------------------------------
# Protocols.
# ---------------------------------------------------------------------------

@runtime_checkable
class OutcomeModel(Protocol):
    """Protocol for outcome regression models (ΔU, R, C)."""

    @property
    def model_id(self) -> str: ...

    @property
    def model_type(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def requires_fit(self) -> bool: ...

    @property
    def deterministic(self) -> bool: ...

    @property
    def lifecycle(self) -> str: ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        split: str = "train",
    ) -> dict[str, Any]: ...

    def freeze(self) -> None: ...

    def predict(self, X: np.ndarray) -> list[Prediction]: ...


@runtime_checkable
class ClassificationModel(Protocol):
    """Protocol for classification models (sign, success)."""

    @property
    def model_id(self) -> str: ...

    @property
    def model_type(self) -> str: ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        split: str = "train",
    ) -> dict[str, Any]: ...

    def freeze(self) -> None: ...

    def predict_proba(self, X: np.ndarray) -> list[ClassificationPrediction]: ...


@runtime_checkable
class RankingModel(Protocol):
    """Protocol for ranking models."""

    @property
    def model_id(self) -> str: ...

    @property
    def model_type(self) -> str: ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        split: str = "train",
    ) -> dict[str, Any]: ...

    def freeze(self) -> None: ...

    def rank(self, X: np.ndarray) -> RankingPrediction: ...


# ---------------------------------------------------------------------------
# Utility functions.
# ---------------------------------------------------------------------------

def ensure_finite_pred(value: float, default: float = 0.0) -> float:
    """Ensure a prediction value is finite."""
    if not math.isfinite(value):
        return default
    return float(value)


def safe_sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash of a configuration dictionary."""
    import json
    content = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
