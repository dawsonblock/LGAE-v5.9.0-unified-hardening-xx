"""Phase 8 contract: LearningResult.

Output of the learn() phase: the transition record and credit assignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import PhaseResult


@dataclass(frozen=True, slots=True)
class CreditAssignment:
    """Per-subsystem credit for a decision transition.

    v5.11-RC Phase 15: This is per-subsystem scalar credit attribution,
    not hierarchical credit assignment. Credit is decomposed by subsystem
    so the system can diagnose which component contributed to an outcome.
    """
    diagnostic_credit: float = 0.0
    candidate_credit: float = 0.0
    planner_credit: float = 0.0
    action_credit: float = 0.0
    governance_credit: float = 0.0
    outcome_credit: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "diagnostic_credit": self.diagnostic_credit,
            "candidate_credit": self.candidate_credit,
            "planner_credit": self.planner_credit,
            "action_credit": self.action_credit,
            "governance_credit": self.governance_credit,
            "outcome_credit": self.outcome_credit,
        }


@dataclass(frozen=True, slots=True)
class DecisionTransition:
    """A complete decision transition record for learning.

    This is the persistent record that connects a decision to its outcome.
    """
    pre_state_hash: str
    post_state_hash: str
    selected_action: str
    predicted_outcome: float
    realized_outcome: float
    reward: float
    authorization_status: str
    transition_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "selected_action": self.selected_action,
            "predicted_outcome": self.predicted_outcome,
            "realized_outcome": self.realized_outcome,
            "reward": self.reward,
            "authorization_status": self.authorization_status,
            "transition_id": self.transition_id,
        }


@dataclass(frozen=True, slots=True)
class LearningResult(PhaseResult):
    """Output of the learn() phase.

    Attributes:
        transition: the decision transition record
        credit: per-subsystem credit attribution
        replay_buffer_size: current replay buffer size
        calibration_updated: whether calibration was updated
        hard_negatives_added: count of hard negatives added
    """
    transition: DecisionTransition | None = None
    credit: CreditAssignment = field(default_factory=CreditAssignment)
    replay_buffer_size: int = 0
    calibration_updated: bool = False
    hard_negatives_added: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "transition": self.transition.to_dict() if self.transition else None,
            "credit": self.credit.to_dict(),
            "replay_buffer_size": self.replay_buffer_size,
            "calibration_updated": self.calibration_updated,
            "hard_negatives_added": self.hard_negatives_added,
        }
