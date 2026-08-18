"""Rich structural transition records for v6.0-exp2.

Expands the v6.0-exp1 (S_t, a_t, S_{t+1}, ΔU, C, R) record into a
comprehensive transition record that preserves both successful and
unsuccessful decisions, with full provenance for training and evaluation.

Two record types:

1. **ObservedTransition**: comes from actual runtime steps (committed or
   rejected). These are REALIZED outcomes with full epistemic strength.

2. **CounterfactualTransition**: comes from exact shadow evaluation of
   non-selected candidates. These are COUNTERFACTUAL outcomes — what would
   have happened if an alternative action had been chosen.

Never mix them without an explicit provenance flag. The model must know
REALIZED vs COUNTERFACTUAL because they have different epistemic strength.

Record schema (TransitionRecord):
    run_id, episode_id, step_id, graph_family, split, seed,
    authority_identity_before, authority_identity_after,
    structural_state_before, diagnosis, candidate_set_summary,
    selected_candidate, planner_metadata,
    predicted_delta, predicted_risk, predicted_cost, predicted_ig,
    action, authorization_decision, transaction_id,
    structural_state_after, realized_delta, realized_cost, realized_risk,
    success, rollback/rejection flags, compute_metrics, provenance
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import hashlib
import json


class TransitionProvenance(str, Enum):
    """Epistemic type of a transition record."""
    REALIZED = "realized"            # Actually happened (committed or rejected)
    COUNTERFACTUAL = "counterfactual"  # Shadow evaluation of non-selected candidate
    SHADOW = "shadow"                # Shadow evaluation during planning


class AuthorizationDecision(str, Enum):
    """Governance decision on a candidate."""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    NO_OP = "no_op"


@dataclass(frozen=True, slots=True)
class AuthorityIdentity:
    """Authority state identity binding."""
    state_hash: str
    state_version: int
    authority_hash: str

    def to_log(self) -> dict[str, Any]:
        return {
            "state_hash": self.state_hash,
            "state_version": int(self.state_version),
            "authority_hash": self.authority_hash,
        }


@dataclass(frozen=True, slots=True)
class StructuralStateSummary:
    """Canonical structural state summary.

    This is a serializable summary of the graph + fiber + gauge state,
    not the raw tensors. It captures the information needed for training
    without requiring the model to ingest the complete raw graph.
    """
    n_nodes: int
    n_edges: int
    density: float
    spectral_gap: float
    degree_mean: float
    degree_std: float
    n_components: int
    avg_clustering: float
    fiber_count: int
    fiber_width: int
    gauge_dim: int
    state_hash: str
    graph_version: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "n_nodes": int(self.n_nodes),
            "n_edges": int(self.n_edges),
            "density": float(self.density),
            "spectral_gap": float(self.spectral_gap),
            "degree_mean": float(self.degree_mean),
            "degree_std": float(self.degree_std),
            "n_components": int(self.n_components),
            "avg_clustering": float(self.avg_clustering),
            "fiber_count": int(self.fiber_count),
            "fiber_width": int(self.fiber_width),
            "gauge_dim": int(self.gauge_dim),
            "state_hash": self.state_hash,
            "graph_version": int(self.graph_version),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True, slots=True)
class DiagnosisSummary:
    """Structural diagnosis summary."""
    oversquashing_score: float = 0.0
    bottleneck_score: float = 0.0
    curvature_score: float = 0.0
    epistemic_uncertainty: float = 0.0
    density_score: float = 0.0
    diagnosed_deficits: tuple[str, ...] = ()

    def to_log(self) -> dict[str, Any]:
        return {
            "oversquashing_score": float(self.oversquashing_score),
            "bottleneck_score": float(self.bottleneck_score),
            "curvature_score": float(self.curvature_score),
            "epistemic_uncertainty": float(self.epistemic_uncertainty),
            "density_score": float(self.density_score),
            "diagnosed_deficits": list(self.diagnosed_deficits),
        }


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """Summary of a single candidate action."""
    candidate_id: int
    action_type: str
    target: dict[str, Any]
    predicted_delta: float
    predicted_risk: float
    predicted_cost: float
    predicted_ig: float
    selected: bool = False

    def to_log(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "action_type": self.action_type,
            "target": dict(self.target),
            "predicted_delta": float(self.predicted_delta),
            "predicted_risk": float(self.predicted_risk),
            "predicted_cost": float(self.predicted_cost),
            "predicted_ig": float(self.predicted_ig),
            "selected": bool(self.selected),
        }


@dataclass(frozen=True, slots=True)
class CandidateSetSummary:
    """Summary of the full candidate set for a step."""
    n_candidates: int
    candidates: tuple[CandidateSummary, ...]
    action_distribution: dict[str, int] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "n_candidates": int(self.n_candidates),
            "candidates": [c.to_log() for c in self.candidates],
            "action_distribution": dict(self.action_distribution),
        }


@dataclass(frozen=True, slots=True)
class PlannerMetadata:
    """Planning phase metadata."""
    horizon: int = 1
    ig_weight: float = 0.0
    cost_weight: float = 0.0
    risk_weight: float = 0.0
    planning_score: float = 0.0
    exploration_bonus: float = 0.0
    planner_type: str = "mpc"

    def to_log(self) -> dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "ig_weight": float(self.ig_weight),
            "cost_weight": float(self.cost_weight),
            "risk_weight": float(self.risk_weight),
            "planning_score": float(self.planning_score),
            "exploration_bonus": float(self.exploration_bonus),
            "planner_type": self.planner_type,
        }


@dataclass(frozen=True, slots=True)
class ComputeMetrics:
    """Compute cost metrics for a transition."""
    candidate_evaluations: int = 0
    shadow_executions: int = 0
    wall_clock_seconds: float = 0.0
    flops_estimate: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "candidate_evaluations": int(self.candidate_evaluations),
            "shadow_executions": int(self.shadow_executions),
            "wall_clock_seconds": float(self.wall_clock_seconds),
            "flops_estimate": float(self.flops_estimate),
        }


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """A single rich structural transition record.

    This is the canonical record type for v6.0-exp2 datasets. It captures
    the full provenance of a structural decision, including:

    - State before and after (with authority identity binding).
    - Diagnosis of the pre-state.
    - Full candidate set summary (not just the selected candidate).
    - Planner metadata (horizon, weights, scores).
    - Predicted vs realized outcomes (delta, risk, cost, IG).
    - Authorization decision and transaction ID.
    - Success/rollback/rejection flags.
    - Compute metrics.
    - Provenance (REALIZED vs COUNTERFACTUAL).

    Both successful and unsuccessful decisions are preserved to avoid
    survivorship bias.
    """
    # Identity.
    record_id: str
    run_id: str
    episode_id: str
    step_id: int
    graph_family: str
    split: str  # "train", "validation", "held_out"
    seed: int

    # Authority binding.
    authority_identity_before: AuthorityIdentity
    authority_identity_after: AuthorityIdentity | None

    # State.
    structural_state_before: StructuralStateSummary
    structural_state_after: StructuralStateSummary | None

    # Diagnosis.
    diagnosis: DiagnosisSummary

    # Candidate set.
    candidate_set_summary: CandidateSetSummary

    # Selected candidate.
    selected_candidate: CandidateSummary | None

    # Planner.
    planner_metadata: PlannerMetadata

    # Predictions.
    predicted_delta: float
    predicted_risk: float
    predicted_cost: float
    predicted_ig: float

    # Action and authorization.
    action: str
    action_target: dict[str, Any]
    authorization_decision: AuthorizationDecision
    transaction_id: str | None

    # Realized outcomes.
    realized_delta: float
    realized_cost: float
    realized_risk: float

    # Flags.
    success: bool
    rollback: bool
    rejected: bool

    # Compute.
    compute_metrics: ComputeMetrics

    # Provenance.
    provenance: TransitionProvenance
    base_runtime_version: str
    generator_version: str
    timestamp: str

    # Extended metadata.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "graph_family": self.graph_family,
            "split": self.split,
            "seed": int(self.seed),
            "authority_identity_before": self.authority_identity_before.to_log(),
            "authority_identity_after": self.authority_identity_after.to_log() if self.authority_identity_after else None,
            "structural_state_before": self.structural_state_before.to_log(),
            "structural_state_after": self.structural_state_after.to_log() if self.structural_state_after else None,
            "diagnosis": self.diagnosis.to_log(),
            "candidate_set_summary": self.candidate_set_summary.to_log(),
            "selected_candidate": self.selected_candidate.to_log() if self.selected_candidate else None,
            "planner_metadata": self.planner_metadata.to_log(),
            "predicted_delta": float(self.predicted_delta),
            "predicted_risk": float(self.predicted_risk),
            "predicted_cost": float(self.predicted_cost),
            "predicted_ig": float(self.predicted_ig),
            "action": self.action,
            "action_target": dict(self.action_target),
            "authorization_decision": self.authorization_decision.value,
            "transaction_id": self.transaction_id,
            "realized_delta": float(self.realized_delta),
            "realized_cost": float(self.realized_cost),
            "realized_risk": float(self.realized_risk),
            "success": bool(self.success),
            "rollback": bool(self.rollback),
            "rejected": bool(self.rejected),
            "compute_metrics": self.compute_metrics.to_log(),
            "provenance": self.provenance.value,
            "base_runtime_version": self.base_runtime_version,
            "generator_version": self.generator_version,
            "timestamp": self.timestamp,
            "extra": dict(self.extra),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), sort_keys=True)


def make_record_id(
    run_id: str,
    episode_id: str,
    step_id: int,
    seed: int,
    provenance: TransitionProvenance,
    candidate_id: int | None = None,
) -> str:
    """Deterministic record ID generation."""
    content = f"{run_id}:{episode_id}:{step_id}:{seed}:{provenance.value}"
    if candidate_id is not None:
        content += f":cand{candidate_id}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
