"""Phase 3 contract: CandidateSet.

Output of the propose() phase: a deterministically ordered, deduplicated
set of structural candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .common import PhaseResult


@dataclass(frozen=True, slots=True)
class Candidate:
    """A single structural candidate action.

    Attributes:
        candidate_id: SHA-256(state_hash || action_type || canonical_params)
        source_state_hash: hash of the state this candidate was generated from
        source_state_version: version of the source state
        action_type: structural action type (add_edge, prune_edge, etc.)
        parameters: canonical action parameters
        origin: source of the candidate (heuristic, learned, retrieved, etc.)
        expected_utility: predicted utility (from generator)
    """
    candidate_id: str
    source_state_hash: str
    source_state_version: int
    action_type: str
    parameters: Mapping[str, Any]
    origin: str = "heuristic"
    expected_utility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_state_hash": self.source_state_hash,
            "source_state_version": self.source_state_version,
            "action_type": self.action_type,
            "parameters": dict(self.parameters),
            "origin": self.origin,
            "expected_utility": self.expected_utility,
        }


@dataclass(frozen=True, slots=True)
class CandidateSet(PhaseResult):
    """Output of the propose() phase.

    Attributes:
        candidates: deterministically ordered tuple of candidates
        total_generated: count before dedup
        duplicates_removed: count removed by dedup
    """
    candidates: tuple[Candidate, ...] = ()
    total_generated: int = 0
    duplicates_removed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "candidates": [c.to_dict() for c in self.candidates],
            "total_generated": self.total_generated,
            "duplicates_removed": self.duplicates_removed,
        }

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(c.candidate_id for c in self.candidates)
