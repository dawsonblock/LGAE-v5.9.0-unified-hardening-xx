"""Phase 5 contract: EvaluationResult.

Output of the evaluate() phase: shadow-state counterfactual evaluation
of the planned action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import PhaseResult
from .candidates import Candidate


@dataclass(frozen=True, slots=True)
class CounterfactualEvaluation(PhaseResult):
    """Output of the evaluate() phase.

    Attributes:
        candidate: the candidate that was evaluated
        predicted_utility: utility predicted on the shadow state
        invariant_violations: tuple of invariant violation descriptions
        certification_level: certification level (L0-L3 or None)
        certification_reasons: tuple of certification reason strings
        shadow_state_hash: hash of the shadow state after applying candidate
        passed: whether the evaluation passed all checks
    """
    candidate: Candidate | None = None
    predicted_utility: float = 0.0
    invariant_violations: tuple[str, ...] = ()
    certification_level: str | None = None
    certification_reasons: tuple[str, ...] = ()
    shadow_state_hash: str = ""
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "candidate": (
                self.candidate.to_dict()
                if self.candidate is not None else None
            ),
            "predicted_utility": self.predicted_utility,
            "invariant_violations": list(self.invariant_violations),
            "certification_level": self.certification_level,
            "certification_reasons": list(self.certification_reasons),
            "shadow_state_hash": self.shadow_state_hash,
            "passed": self.passed,
        }
