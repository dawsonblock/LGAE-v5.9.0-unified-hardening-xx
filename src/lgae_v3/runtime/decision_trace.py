"""Runtime decision trace (Phase 41).

A human-readable decision trace that explains *why* a step committed or
rejected. This is the narrative counterpart to the JSONL metrics sink
(Phase 40): the metrics sink is for machines, the decision trace is for
humans debugging or auditing the runtime.

The trace is a structured list of ``TraceEntry`` records, each with a phase,
a human-readable summary, and the machine-readable payload. The trace can be
rendered as plain text or JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, TextIO

from .runtime_events import RuntimeEvent, RuntimePhase


@dataclass(slots=True)
class TraceEntry:
    """One entry in a human-readable decision trace."""
    step: int
    phase: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Render as a single human-readable line."""
        return f"[step {self.step:>3} | {self.phase:>16}] {self.summary}"

    def to_log(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "phase": str(self.phase),
            "summary": self.summary,
            "payload": self.payload,
        }


# Human-readable summaries for each phase.
_PHASE_SUMMARIES: dict[str, str] = {
    RuntimePhase.OBSERVE.value: "Observed authoritative graph state",
    RuntimePhase.SNAPSHOT.value: "Captured immutable runtime snapshot",
    RuntimePhase.GEOMETRY.value: "Ran adaptive geometric diagnostics",
    RuntimePhase.REASON.value: "Reasoning + memory priors applied",
    RuntimePhase.PROPOSE.value: "Generated candidate union",
    RuntimePhase.RANK.value: "Ranked candidates by learned score",
    RuntimePhase.UNCERTAINTY.value: "Estimated epistemic/aleatoric uncertainty",
    RuntimePhase.INFORMATION_GAIN.value: "Computed information gain + risk",
    RuntimePhase.PLAN.value: "Planned via bounded structural MPC",
    RuntimePhase.EVALUATE.value: "Shadow transaction evaluated",
    RuntimePhase.AUTHORIZE.value: "Authority governor decision",
    RuntimePhase.COMMIT.value: "Committed authoritative mutation",
    RuntimePhase.CACHE_INVALIDATE.value: "Invalidated affected caches",
    RuntimePhase.CREDIT.value: "Assigned local structural credit",
    RuntimePhase.EVIDENCE.value: "Generated signed evidence + receipt",
    RuntimePhase.LEARN.value: "Updated learned model from experience",
}


def _summarize_payload(phase: str, payload: dict[str, Any]) -> str:
    """Build a human-readable summary from the phase and payload."""
    base = _PHASE_SUMMARIES.get(phase, phase)
    extras: list[str] = []
    if "decision" in payload:
        extras.append(f"decision={payload['decision']}")
    if "chosen_action" in payload:
        extras.append(f"action={payload['chosen_action']}")
    if "authority_hash_after" in payload:
        extras.append(f"hash={str(payload['authority_hash_after'])[:12]}")
    if "evidence_hash" in payload and payload["evidence_hash"]:
        extras.append(f"evidence={str(payload['evidence_hash'])[:12]}")
    if "receipt_hash" in payload and payload["receipt_hash"]:
        extras.append(f"receipt={str(payload['receipt_hash'])[:12]}")
    if "mutation_impact" in payload:
        impact = payload["mutation_impact"]
        if isinstance(impact, dict):
            changed = [k for k, v in impact.items() if v]
            if changed:
                extras.append(f"impact={'+'.join(changed)}")
    if "invalidated" in payload and payload["invalidated"]:
        extras.append(f"invalidated={','.join(payload['invalidated'])}")
    if "n_candidates" in payload:
        extras.append(f"n_cand={payload['n_candidates']}")
    if extras:
        return f"{base} ({', '.join(extras)})"
    return base


class DecisionTrace:
    """Accumulates a human-readable decision trace for one or more steps."""

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []

    def add_event(self, event: RuntimeEvent) -> None:
        """Add a runtime event to the trace."""
        summary = _summarize_payload(event.phase.value, event.payload)
        self._entries.append(TraceEntry(
            step=int(event.step),
            phase=event.phase.value,
            summary=summary,
            payload=dict(event.payload),
        ))

    def add_events(self, events: Iterable[RuntimeEvent]) -> None:
        for e in events:
            self.add_event(e)

    def add_entry(self, entry: TraceEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[TraceEntry]:
        return list(self._entries)

    def render(self) -> str:
        """Render the full trace as human-readable text."""
        return "\n".join(e.render() for e in self._entries)

    def to_log(self) -> list[dict[str, Any]]:
        return [e.to_log() for e in self._entries]

    def write(self, fp: TextIO) -> None:
        """Write the trace to a file-like object."""
        fp.write(self.render())
        fp.write("\n")

    def write_file(self, path: str) -> None:
        """Write the trace to a file."""
        with open(path, "w", encoding="utf-8") as f:
            self.write(f)

    @property
    def step_count(self) -> int:
        return len(set(e.step for e in self._entries))

    def entries_for_step(self, step: int) -> list[TraceEntry]:
        return [e for e in self._entries if e.step == int(step)]
