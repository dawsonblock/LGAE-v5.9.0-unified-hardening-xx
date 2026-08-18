"""Transition recorder for v6 experimental datasets.

Instruments the v5.11 runtime to produce canonical structural transition
records of the form:

    (S_t, a_t, S_{t+1}, ΔU, C, R)

where:
- S_t: structural state before the transition (graph + fiber + gauge summary).
- a_t: the action taken (chosen candidate + governance decision).
- S_{t+1}: structural state after the transition.
- ΔU: realized utility delta (U_after - U_before).
- C: compute cost of the step (candidate evaluations, shadow executions).
- R: reward signal (realized delta, not predicted).

CRITICAL DESIGN CONSTRAINT:
    The transition recorder is a PASSIVE OBSERVER. It reads state from the
    runtime's RuntimeStepResult but never mutates authoritative state. It
    keeps its own dataset entirely outside the v5.11 authority boundary.

    The recorder does NOT:
    - Call any mutation method on the runtime.
    - Modify the graph, fibers, or gauges.
    - Interfere with the WAL or commit channel.
    - Feed back into the runtime's learning loop.

    It ONLY:
    - Reads the RuntimeStepResult after a step completes.
    - Extracts and serializes state summaries.
    - Appends to an external dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import time

import torch

from ..runtime.runtime_result import RuntimeStepResult
from ..runtime.runtime_state import RuntimeSnapshot


@dataclass(frozen=True, slots=True)
class StructuralTransition:
    """A single canonical structural transition record.

    (S_t, a_t, S_{t+1}, ΔU, C, R)
    """
    # Identity.
    transition_id: str
    step: int
    seed: int

    # State before (S_t).
    state_before_hash: str
    state_before_version: int
    state_before_summary: dict[str, Any]

    # Action (a_t).
    chosen_action: str
    governance_decision: str
    executed: bool
    action_metadata: dict[str, Any]

    # State after (S_{t+1}).
    state_after_hash: str
    state_after_version: int
    state_after_summary: dict[str, Any]

    # Utility delta (ΔU).
    utility_before: float
    utility_after: float
    delta_utility: float

    # Compute cost (C).
    compute_cost: float

    # Reward (R) = realized delta (not predicted).
    reward: float

    # Provenance.
    runtime_version: str
    timestamp: str

    def to_log(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "step": int(self.step),
            "seed": int(self.seed),
            "state_before_hash": self.state_before_hash,
            "state_before_version": int(self.state_before_version),
            "state_before_summary": self.state_before_summary,
            "chosen_action": self.chosen_action,
            "governance_decision": self.governance_decision,
            "executed": bool(self.executed),
            "action_metadata": self.action_metadata,
            "state_after_hash": self.state_after_hash,
            "state_after_version": int(self.state_after_version),
            "state_after_summary": self.state_after_summary,
            "utility_before": float(self.utility_before),
            "utility_after": float(self.utility_after),
            "delta_utility": float(self.delta_utility),
            "compute_cost": float(self.compute_cost),
            "reward": float(self.reward),
            "runtime_version": self.runtime_version,
            "timestamp": self.timestamp,
        }


def _snapshot_summary(snap: RuntimeSnapshot) -> dict[str, Any]:
    """Extract a serializable summary from a RuntimeSnapshot."""
    try:
        return snap.to_summary()
    except Exception:
        return {
            "state_version": getattr(snap, "state_version", -1),
            "authority_hash": getattr(snap, "authority_hash", ""),
        }


def _snapshot_hash(snap: RuntimeSnapshot) -> str:
    """Extract or compute a hash for a snapshot."""
    try:
        h = getattr(snap, "authority_hash", None)
        if h:
            return str(h)
    except Exception:
        pass
    # Fallback: hash the summary.
    summary = _snapshot_summary(snap)
    content = str(sorted(summary.items()))
    return hashlib.sha256(content.encode()).hexdigest()


def _snapshot_version(snap: RuntimeSnapshot) -> int:
    """Extract the state version from a snapshot."""
    try:
        return int(getattr(snap, "generation", -1))
    except Exception:
        return -1


class TransitionRecorder:
    """Records structural transitions from v5.11 runtime steps.

    Usage::

        recorder = TransitionRecorder(seed=42)
        for _ in range(n_steps):
            result = runtime.step(...)
            recorder.record(result)
        dataset = recorder.dataset()

    The recorder is passive: it only reads from RuntimeStepResult and never
    touches the runtime's authoritative state.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self._transitions: list[StructuralTransition] = []
        self._step_counter = 0

    def record(
        self,
        result: RuntimeStepResult,
        *,
        compute_cost: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> StructuralTransition:
        """Record a transition from a RuntimeStepResult.

        Args:
            result: The RuntimeStepResult from a completed runtime step.
            compute_cost: Optional explicit compute cost. If None, estimated
                from the number of candidates evaluated.
            extra_metadata: Optional additional metadata to merge into the
                action metadata.

        Returns:
            The recorded StructuralTransition.
        """
        # Estimate compute cost if not provided.
        if compute_cost is None:
            # Estimate from candidates count and shadow horizons.
            n_candidates = 0
            if result.candidates is not None:
                try:
                    n_candidates = len(result.candidates)
                except TypeError:
                    n_candidates = 1
            compute_cost = float(n_candidates * 10)  # rough estimate

        # Build action metadata.
        action_meta: dict[str, Any] = {}
        if result.planning is not None:
            try:
                action_meta["planning"] = result.planning.to_log()
            except Exception:
                action_meta["planning"] = str(result.planning)
        if result.evaluation is not None:
            try:
                action_meta["evaluation"] = result.evaluation.to_log()
            except Exception:
                action_meta["evaluation"] = str(result.evaluation)
        if extra_metadata:
            action_meta.update(extra_metadata)

        # Compute transition ID deterministically.
        tid_content = f"{self.seed}:{self._step_counter}:{result.step}"
        tid = hashlib.sha256(tid_content.encode()).hexdigest()[:16]

        # Reward = realized delta (not predicted).
        reward = float(result.delta_utility) if result.committed else 0.0

        # Runtime version.
        from ..version import VERSION
        rt_version = VERSION

        transition = StructuralTransition(
            transition_id=tid,
            step=int(self._step_counter),
            seed=self.seed,
            state_before_hash=_snapshot_hash(result.snapshot_before),
            state_before_version=_snapshot_version(result.snapshot_before),
            state_before_summary=_snapshot_summary(result.snapshot_before),
            chosen_action=str(result.chosen_action),
            governance_decision=str(result.governance_decision),
            executed=bool(result.executed),
            action_metadata=action_meta,
            state_after_hash=_snapshot_hash(result.snapshot_after),
            state_after_version=_snapshot_version(result.snapshot_after),
            state_after_summary=_snapshot_summary(result.snapshot_after),
            utility_before=float(result.utility_before),
            utility_after=float(result.utility_after),
            delta_utility=float(result.delta_utility),
            compute_cost=float(compute_cost),
            reward=reward,
            runtime_version=rt_version,
            timestamp=_utc_now(),
        )
        self._transitions.append(transition)
        self._step_counter += 1
        return transition

    def dataset(self) -> list[StructuralTransition]:
        """Return all recorded transitions."""
        return list(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def clear(self) -> None:
        """Clear all recorded transitions."""
        self._transitions.clear()
        self._step_counter = 0

    def to_log(self) -> dict[str, Any]:
        """Summary of the recorder state."""
        return {
            "seed": int(self.seed),
            "n_transitions": len(self._transitions),
            "n_committed": sum(1 for t in self._transitions if t.executed),
            "n_rejected": sum(1 for t in self._transitions if not t.executed),
            "mean_delta_utility": (
                sum(t.delta_utility for t in self._transitions) / max(len(self._transitions), 1)
            ),
            "total_compute_cost": sum(t.compute_cost for t in self._transitions),
        }


def record_runtime_step(
    result: RuntimeStepResult,
    recorder: TransitionRecorder,
    *,
    compute_cost: float | None = None,
) -> StructuralTransition:
    """Convenience function: record a RuntimeStepResult into a TransitionRecorder."""
    return recorder.record(result, compute_cost=compute_cost)


def _utc_now() -> str:
    """Current UTC timestamp in ISO format (deterministic format)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
