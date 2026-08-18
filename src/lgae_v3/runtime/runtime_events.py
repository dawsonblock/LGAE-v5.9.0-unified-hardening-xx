"""Runtime cycle phase and event types.

These names mirror the canonical governed cycle from the v5.10 plan:

    Observation -> Stable Snapshot -> Adaptive Geometry -> Reasoning+Memory
    -> Candidate Generation -> Ranking -> Uncertainty -> IG+Risk
    -> Multi-Step Planning -> Joint Action -> Shadow Transaction
    -> Verification -> Authority Governor -> (Reject|Quarantine|Commit)
    -> Atomic State Update -> Cache Invalidation -> Local Credit
    -> Signed Evidence/Receipt -> Replay/Experience -> Learn
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimePhase(str, Enum):
    OBSERVE = "observe"
    SNAPSHOT = "snapshot"
    GEOMETRY = "geometry"
    REASON = "reason"
    PROPOSE = "propose"
    RANK = "rank"
    UNCERTAINTY = "uncertainty"
    INFORMATION_GAIN = "information_gain"
    PLAN = "plan"
    EVALUATE = "evaluate"
    AUTHORIZE = "authorize"
    COMMIT = "commit"
    CACHE_INVALIDATE = "cache_invalidate"
    CREDIT = "credit"
    EVIDENCE = "evidence"
    LEARN = "learn"


@dataclass(slots=True)
class RuntimeEvent:
    """A structured event emitted at a cycle phase boundary."""
    phase: RuntimePhase
    step: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "step": int(self.step),
            "payload": self.payload,
        }
