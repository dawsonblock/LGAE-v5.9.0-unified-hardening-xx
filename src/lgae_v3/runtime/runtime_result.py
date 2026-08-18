"""Result types for the canonical runtime cycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import MutationDecision
from .runtime_state import RuntimeSnapshot
from .runtime_events import RuntimePhase
from .contracts.learning import LearningResult


@dataclass(slots=True)
class RuntimeStepResult:
    """Complete record of one governed cycle.

    Captures the full provenance of a step: the snapshots before/after, the
    chosen action, the governor decision, whether a commit occurred, the
    emitted evidence/receipt hashes, and per-phase metadata. This is the
    canonical object a replay/experience record (Phase 20) is derived from.
    """
    step: int
    snapshot_before: RuntimeSnapshot
    snapshot_after: RuntimeSnapshot
    chosen_action: str
    governance_decision: str
    executed: bool
    utility_before: float
    utility_after: float
    delta_utility: float
    certification_level: str | None = None
    evidence_hash: str | None = None
    receipt_hash: str | None = None
    phases: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    observation: Any | None = None
    reasoning: Any | None = None
    candidates: Any | None = None
    planning: Any | None = None
    evaluation: Any | None = None
    authorization: Any | None = None
    commit: Any | None = None
    learning: LearningResult | None = None

    @property
    def decision(self) -> str:
        return self.governance_decision

    @property
    def committed(self) -> bool:
        return bool(self.executed) and self.governance_decision == MutationDecision.ACCEPT.value

    def to_log(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "chosen_action": self.chosen_action,
            "governance_decision": self.governance_decision,
            "executed": bool(self.executed),
            "committed": self.committed,
            "utility_before": float(self.utility_before),
            "utility_after": float(self.utility_after),
            "delta_utility": float(self.delta_utility),
            "certification_level": self.certification_level,
            "evidence_hash": self.evidence_hash,
            "receipt_hash": self.receipt_hash,
            "snapshot_before": self.snapshot_before.to_summary(),
            "snapshot_after": self.snapshot_after.to_summary(),
            "phases": self.phases,
        }
