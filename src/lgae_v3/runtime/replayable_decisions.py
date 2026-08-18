"""Replayable decisions (Phase 29).

A replayable decision is a deterministic record of everything needed to
reproduce a runtime step's outcome: the input state, the candidate union,
the chosen action, the governance decision, and the resulting state.

Given the same input state and the same decision record, replaying the
step must produce the same output state. This enables:

  - audit: verify that a past decision was correct
  - debugging: reproduce a failure from a decision record
  - learning: replay decisions for offline RL (Phase 22)

A decision record is distinct from a WAL record (Phase 30): the WAL records
mutations for crash recovery; the decision record records the full reasoning
chain for audit and replay.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .runtime_events import RuntimeEvent, RuntimePhase


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """A complete, replayable record of one runtime step decision."""
    step: int
    state_hash_before: str
    state_hash_after: str
    chosen_action: str
    governance_decision: str
    committed: bool
    candidate_ids: list[str]
    candidate_scores: list[float]
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def record_hash(self) -> str:
        """Deterministic hash of the decision record."""
        payload = json.dumps({
            "step": int(self.step),
            "state_hash_before": self.state_hash_before,
            "state_hash_after": self.state_hash_after,
            "chosen_action": self.chosen_action,
            "governance_decision": self.governance_decision,
            "committed": bool(self.committed),
            "candidate_ids": list(self.candidate_ids),
            "candidate_scores": [float(s) for s in self.candidate_scores],
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_log(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "state_hash_before": self.state_hash_before,
            "state_hash_after": self.state_hash_after,
            "chosen_action": self.chosen_action,
            "governance_decision": self.governance_decision,
            "committed": bool(self.committed),
            "candidate_ids": list(self.candidate_ids),
            "candidate_scores": [float(s) for s in self.candidate_scores],
            "record_hash": self.record_hash,
            "events": list(self.events),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DecisionLedger:
    """An append-only ledger of decision records."""
    records: list[DecisionRecord] = field(default_factory=list)

    def append(self, record: DecisionRecord) -> None:
        self.records.append(record)

    @property
    def latest(self) -> DecisionRecord | None:
        return self.records[-1] if self.records else None

    def by_step(self, step: int) -> DecisionRecord | None:
        for r in self.records:
            if r.step == int(step):
                return r
        return None

    def by_state_hash(self, state_hash: str) -> list[DecisionRecord]:
        return [r for r in self.records if r.state_hash_before == state_hash]

    def committed_records(self) -> list[DecisionRecord]:
        return [r for r in self.records if r.committed]

    def rejected_records(self) -> list[DecisionRecord]:
        return [r for r in self.records if not r.committed]

    def to_log(self) -> dict[str, Any]:
        return {
            "record_count": len(self.records),
            "committed_count": len(self.committed_records()),
            "rejected_count": len(self.rejected_records()),
            "records": [r.to_log() for r in self.records],
        }


def build_decision_record(
    *,
    step: int,
    state_hash_before: str,
    state_hash_after: str,
    chosen_action: str,
    governance_decision: str,
    committed: bool,
    candidate_ids: list[str],
    candidate_scores: list[float],
    events: list[RuntimeEvent] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DecisionRecord:
    """Build a decision record from runtime step outputs."""
    event_logs = []
    if events:
        for ev in events:
            event_logs.append({
                "phase": ev.phase.value,
                "step": int(ev.step),
                "payload": dict(ev.payload),
            })
    return DecisionRecord(
        step=int(step),
        state_hash_before=str(state_hash_before),
        state_hash_after=str(state_hash_after),
        chosen_action=str(chosen_action),
        governance_decision=str(governance_decision),
        committed=bool(committed),
        candidate_ids=[str(cid) for cid in candidate_ids],
        candidate_scores=[float(s) for s in candidate_scores],
        events=event_logs,
        metadata=dict(metadata or {}),
    )


def verify_replay(record: DecisionRecord, *, expected_state_hash_after: str) -> bool:
    """Verify that a replayed decision produces the expected state hash."""
    return record.state_hash_after == expected_state_hash_after
