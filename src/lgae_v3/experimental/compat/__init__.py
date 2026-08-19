"""Legacy compatibility adapters for experimental interfaces.

This package isolates backward-compatibility shims that should NOT be
used by new code or scientific runs (exp4.2+).

Phase 2 (exp4.2): The ``predicted_reward`` alias on ``ModelPrediction``
is semantically incorrect (reward and risk are opposites). It is retained
only for legacy callers and is scheduled for deletion before exp5.
"""
from __future__ import annotations

from .legacy_prediction import LegacyPredictionAdapter

__all__ = ["LegacyPredictionAdapter"]
