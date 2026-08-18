"""Unified immutable runtime snapshot (Phase 3 foundation).

Every expensive reader operates from an immutable snapshot so no subsystem
silently fetches mutable state halfway through a calculation. The snapshot
binds the authoritative identity (graph + gauge + fiber + governance config)
into one frozen object.

Phase 3 will wire the seqlock/read-coordinator enforcement
(``generation_start == generation_end`` else ``StaleReadError``); this module
establishes the canonical snapshot contract now so the runtime can adopt it
without restructuring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

# Reuse the v5.8 cache-coherence StaleReadError so there is exactly one
# canonical stale-read exception across the runtime and read coordinator.
from ..cache_coherence import StaleReadError  # noqa: F401 (re-exported)


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Immutable authoritative state snapshot.

    ``authority_hash`` is the canonical SHA-256 commitment over graph, gauge,
    fiber, and governance config (delegated to ``LGAEEngine.authority_hash``).
    All other fields are read-only views/copies bound to ``generation``.
    """

    generation: int
    authority_hash: str
    graph_state_hash: str
    graph_version: int
    gauge_state_hash: str | None
    fiber_state_hash: str | None
    memory_version: int = 0
    geometry_version: int = 0
    evidence_root: str | None = None
    cache_epoch: int = 0
    reproducibility_context: dict[str, Any] | None = None

    def assert_generation(self, expected: int) -> None:
        """Fail closed if the snapshot generation no longer matches."""
        if int(self.generation) != int(expected):
            raise StaleReadError(
                f"stale read: snapshot generation {self.generation} != expected {expected}"
            )

    def to_summary(self) -> dict[str, Any]:
        return {
            "generation": int(self.generation),
            "authority_hash": self.authority_hash,
            "graph_state_hash": self.graph_state_hash,
            "graph_version": int(self.graph_version),
            "gauge_state_hash": self.gauge_state_hash,
            "fiber_state_hash": self.fiber_state_hash,
            "memory_version": int(self.memory_version),
            "geometry_version": int(self.geometry_version),
            "evidence_root": self.evidence_root,
            "cache_epoch": int(self.cache_epoch),
        }


def snapshot_from_engine(engine: Any, *, generation: int | None = None) -> RuntimeSnapshot:
    """Build an immutable snapshot from an ``LGAEEngine``'s authoritative state."""
    gen = int(generation) if generation is not None else int(getattr(engine, "step_index", 0))
    graph = engine.graph
    gauge_hash = None if getattr(engine, "gauge_connections", None) is None else engine.gauge_connections.state_hash()
    fiber_hash = engine.fibers.state_hash()
    return RuntimeSnapshot(
        generation=gen,
        authority_hash=engine.authority_hash(),
        graph_state_hash=graph.state_hash(),
        graph_version=int(graph.version),
        gauge_state_hash=gauge_hash,
        fiber_state_hash=fiber_hash,
    )
