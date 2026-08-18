"""RuntimeStepResult: the aggregate output of one canonical cycle.

Contains all 8 phase results plus the phase execution order for verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import canonical_hash
from .observation import ObservationSnapshot
from .reasoning import ReasoningResult
from .candidates import CandidateSet
from .planning import PlanningResult
from .evaluation import CounterfactualEvaluation
from .authorization import AuthorizationResult
from .commit import CommitResult
from .learning import LearningResult


# The canonical phase order. Every step must execute exactly these phases
# in exactly this order.
CANONICAL_PHASE_ORDER: tuple[str, ...] = (
    "observe",
    "reason",
    "propose",
    "plan",
    "evaluate",
    "authorize",
    "commit",
    "learn",
)


@dataclass(frozen=True, slots=True)
class RuntimeStepResult:
    """Aggregate result of one canonical 8-phase cycle.

    Attributes:
        step: step index
        phase_order: tuple of phase names executed (must match CANONICAL_PHASE_ORDER)
        observation: ObservationSnapshot
        reasoning: ReasoningResult
        candidates: CandidateSet
        planning: PlanningResult
        evaluation: CounterfactualEvaluation
        authorization: AuthorizationResult
        commit: CommitResult
        learning: LearningResult
    """
    step: int
    phase_order: tuple[str, ...] = CANONICAL_PHASE_ORDER
    observation: ObservationSnapshot | None = None
    reasoning: ReasoningResult | None = None
    candidates: CandidateSet | None = None
    planning: PlanningResult | None = None
    evaluation: CounterfactualEvaluation | None = None
    authorization: AuthorizationResult | None = None
    commit: CommitResult | None = None
    learning: LearningResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "phase_order": list(self.phase_order),
            "observation": self.observation.to_dict() if self.observation else None,
            "reasoning": self.reasoning.to_dict() if self.reasoning else None,
            "candidates": self.candidates.to_dict() if self.candidates else None,
            "planning": self.planning.to_dict() if self.planning else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "authorization": self.authorization.to_dict() if self.authorization else None,
            "commit": self.commit.to_dict() if self.commit else None,
            "learning": self.learning.to_dict() if self.learning else None,
        }

    def to_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def executed_all_phases(self) -> bool:
        """Verify that all 8 phases were executed in the correct order."""
        return self.phase_order == CANONICAL_PHASE_ORDER

    @property
    def state_hash_before(self) -> str:
        return self.observation.state_hash if self.observation else ""

    @property
    def state_hash_after(self) -> str:
        return self.commit.new_state_hash if self.commit and self.commit.committed else (
            self.observation.state_hash if self.observation else ""
        )
