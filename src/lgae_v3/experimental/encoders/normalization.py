"""Normalization statistics with train-only fitting and freeze lifecycle.

Lifecycle:
    UNFIT → FITTED_TRAIN → FROZEN

Once frozen:
- validation data cannot modify normalization
- held-out data cannot modify normalization
- attempting to fit on held-out data raises

This prevents held-out statistics from influencing scaling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import hashlib
import math
import numpy as np


class NormalizationState(str):
    UNFIT = "unfit"
    FITTED_TRAIN = "fitted_train"
    FROZEN = "frozen"


class FrozenNormalizationError(Exception):
    """Raised when attempting to modify a frozen normalization."""


class HeldOutFittingError(Exception):
    """Raised when attempting to fit normalization on held-out data."""


@dataclass(slots=True)
class NormalizationStatistics:
    """Train-only normalization statistics.

    Computes mean and standard deviation from training data only.
    Once frozen, these statistics are immutable and applied unchanged
    to validation and held-out data.
    """
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    n_samples: int = 0
    dimension: int = 0
    state: str = NormalizationState.UNFIT
    dataset_hash: str = ""
    feature_schema_hash: str = ""

    @property
    def normalization_hash(self) -> str:
        """Deterministic hash of the normalization parameters."""
        if not self.mean:
            return ""
        content = f"{self.n_samples}:{self.dimension}:{','.join(f'{m:.10f}' for m in self.mean)},{','.join(f'{s:.10f}' for s in self.std)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def is_fit(self) -> bool:
        return self.state in (NormalizationState.FITTED_TRAIN, NormalizationState.FROZEN)

    @property
    def is_frozen(self) -> bool:
        return self.state == NormalizationState.FROZEN

    def fit(
        self,
        features: Sequence[Sequence[float]],
        *,
        split: str = "train",
        dataset_hash: str = "",
        feature_schema_hash: str = "",
    ) -> None:
        """Fit normalization statistics from a set of feature vectors.

        Args:
            features: List of feature vectors (all same dimension).
            split: Which split the data comes from. Must be "train".
            dataset_hash: Hash of the dataset used for fitting.
            feature_schema_hash: Hash of the feature schema.

        Raises:
            HeldOutFittingError: If split is not "train".
            FrozenNormalizationError: If normalization is already frozen.
        """
        if self.is_frozen:
            raise FrozenNormalizationError(
                "Cannot fit frozen normalization. Normalization is already frozen."
            )
        if split != "train":
            raise HeldOutFittingError(
                f"Cannot fit normalization on '{split}' split. "
                "Normalization must be fitted on train data only."
            )
        if not features:
            return
        dim = len(features[0])
        arr = np.array(features, dtype=np.float64)
        # Replace nonfinite with 0 for statistics.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self.mean = tuple(float(x) for x in arr.mean(axis=0))
        self.std = tuple(float(x) for x in arr.std(axis=0))
        self.n_samples = len(features)
        self.dimension = dim
        self.state = NormalizationState.FITTED_TRAIN
        self.dataset_hash = dataset_hash
        self.feature_schema_hash = feature_schema_hash

    def freeze(self) -> None:
        """Freeze normalization statistics. After this, no modifications allowed."""
        if not self.is_fit:
            raise RuntimeError("Cannot freeze unfitted normalization.")
        self.state = NormalizationState.FROZEN

    def transform(self, features: Sequence[float]) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        """Apply normalization to a feature vector.

        Returns (normalized_values, missing_mask).
        If not fitted, returns values as-is with all-false mask.
        """
        if not self.is_fit:
            return tuple(float(v) for v in features), tuple(False for _ in features)
        n = len(features)
        mask = tuple(not math.isfinite(v) for v in features)
        result = []
        for i, v in enumerate(features):
            if mask[i]:
                result.append(0.0)
                continue
            s = self.std[i] if self.std[i] > 1e-10 else 1.0
            result.append((float(v) - self.mean[i]) / s)
        return tuple(result), mask

    def to_log(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "dimension": int(self.dimension),
            "n_samples": int(self.n_samples),
            "normalization_hash": self.normalization_hash,
            "dataset_hash": self.dataset_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "mean": list(self.mean) if self.mean else [],
            "std": list(self.std) if self.std else [],
        }
