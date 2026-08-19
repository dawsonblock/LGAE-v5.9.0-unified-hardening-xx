"""Legacy prediction adapter for backward compatibility.

This module provides a read-only adapter that exposes the deprecated
``predicted_reward`` field for legacy callers without polluting the
canonical ``ModelPrediction`` interface.

Phase 2 (exp4.2): Reward and risk are semantically opposite. This adapter
exists ONLY to support legacy code that has not yet been migrated. It
must NOT be used by exp4.2 or any new code. It will be deleted before exp5.
"""
from __future__ import annotations

from typing import Any
import warnings

from ..world_model import ModelPrediction


class LegacyPredictionAdapter:
    """Read-only adapter that exposes deprecated field names.

    Wraps a ``ModelPrediction`` and provides access to ``predicted_reward``
    as an alias for ``predicted_risk``, emitting a DeprecationWarning.

    This adapter is the ONLY sanctioned way to access the deprecated
    field from legacy code. New code must use ``ModelPrediction`` directly.
    """

    def __init__(self, prediction: ModelPrediction) -> None:
        self._prediction = prediction

    @property
    def predicted_delta_utility(self) -> float | None:
        return self._prediction.predicted_delta_utility

    @property
    def predicted_risk(self) -> float | None:
        return self._prediction.predicted_risk

    @property
    def predicted_cost(self) -> float | None:
        return self._prediction.predicted_cost

    @property
    def predicted_uncertainty(self) -> float | None:
        return self._prediction.predicted_uncertainty

    @property
    def probability_positive(self) -> float | None:
        return self._prediction.probability_positive

    @property
    def predicted_reward(self) -> float | None:
        """Deprecated alias for ``predicted_risk``.

        .. deprecated::
            Use ``predicted_risk`` instead. This will be removed before exp5.
        """
        warnings.warn(
            "LegacyPredictionAdapter.predicted_reward is deprecated. "
            "Use predicted_risk instead. This adapter will be removed "
            "before exp5.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._prediction.predicted_risk

    def to_log(self) -> dict[str, Any]:
        """Return the canonical log (without deprecated fields)."""
        return self._prediction.to_log()

    @property
    def underlying(self) -> ModelPrediction:
        """Access the underlying canonical prediction."""
        return self._prediction
