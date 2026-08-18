"""Phase 1 contract: ObservationSnapshot.

The immutable snapshot of authoritative state at the start of a cycle.
Every subsequent phase binds to this snapshot's version and hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import PhaseResult


@dataclass(frozen=True, slots=True)
class ObservationSnapshot(PhaseResult):
    """Immutable snapshot of authoritative state.

    Fields:
        snapshot_id: deterministic ID (state_hash + version)
        state_version: monotonic version counter
        state_hash: canonical hash of graph + fibers + gauges + model
        graph_version: engine graph version
        authority_hash: engine authority hash
        task_loss: external task signal
        task_loss_delta: change in task loss
        epistemic_uncertainty: external uncertainty signal
        created_at_step: runtime step index
    """
    graph_version: int = 0
    authority_hash: str = ""
    task_loss: float = 0.0
    task_loss_delta: float = 0.0
    epistemic_uncertainty: float = 0.0
    created_at_step: int = 0

    @property
    def observation_id(self) -> str:
        return self.snapshot_id
