"""Phase 4 contract: PlanningResult.

Output of the plan() phase: the selected action sequence and its
information-gain / utility / cost / risk decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import PhaseResult
from .candidates import Candidate


@dataclass(frozen=True, slots=True)
class CandidateValue:
    """Decomposed value of a candidate under the planning objective.

    J(a) = E[ΔU|S,a] + nu * IG(S,a) - lambda * C(S,a) - mu * R(S,a) - rho * H(S,a)
    """
    expected_utility: float = 0.0
    information_gain: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    homeostasis_penalty: float = 0.0
    total_score: float = 0.0

    @property
    def expected_delta(self) -> float:
        return self.expected_utility

    def to_dict(self) -> dict[str, float]:
        return {
            "expected_utility": self.expected_utility,
            "expected_delta": self.expected_delta,
            "information_gain": self.information_gain,
            "cost": self.cost,
            "risk": self.risk,
            "homeostasis_penalty": self.homeostasis_penalty,
            "total_score": self.total_score,
        }


@dataclass(frozen=True, slots=True)
class PlanningResult(PhaseResult):
    """Output of the plan() phase.

    Attributes:
        selected_candidate: the chosen candidate (or None for NO_OP/defer)
        candidate_values: per-candidate value decomposition
        horizon: planning horizon (1 = single-step, >1 = MPC)
        mpc_plan: tuple of candidate IDs for multi-step plans
        planner: which planner was used (single_step, mpc, etc.)
    """
    selected_candidate: Candidate | None = None
    candidate_values: tuple[CandidateValue, ...] = ()
    horizon: int = 1
    mpc_plan: tuple[str, ...] = ()
    planner: str = "single_step"

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None else None
            ),
            "candidate_values": [v.to_dict() for v in self.candidate_values],
            "horizon": self.horizon,
            "mpc_plan": list(self.mpc_plan),
            "planner": self.planner,
        }
