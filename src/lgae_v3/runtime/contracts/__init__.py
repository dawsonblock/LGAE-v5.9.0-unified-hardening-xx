"""Canonical runtime phase contracts (v5.11).

Every phase of the canonical 8-phase cycle emits an immutable, state-bound,
deterministically serializable result. These contracts are the typed boundary
between phases — no phase may mutate the output of another phase.

Phase order:
    observe   -> ObservationSnapshot
    reason    -> ReasoningResult
    propose   -> CandidateSet
    plan      -> PlanningResult
    evaluate  -> CounterfactualEvaluation
    authorize -> AuthorizationResult
    commit    -> CommitResult
    learn     -> LearningResult

Only commit() gets mutation authority. All other phases are read-only
w.r.t. authoritative state.
"""
from __future__ import annotations

from .common import PhaseResult, canonical_json, canonical_hash
from .observation import ObservationSnapshot
from .reasoning import ReasoningResult, StructuralDeficit, DiagnosticBundle
from .candidates import Candidate, CandidateSet
from .planning import PlanningResult, CandidateValue
from .evaluation import CounterfactualEvaluation
from .authorization import AuthorizationResult, AuthorizationStatus, RejectionReason
from .commit import CommitResult
from .learning import LearningResult, DecisionTransition, CreditAssignment
from .step_result import RuntimeStepResult, CANONICAL_PHASE_ORDER

__all__ = [
    "PhaseResult",
    "canonical_json",
    "canonical_hash",
    "ObservationSnapshot",
    "ReasoningResult",
    "StructuralDeficit",
    "DiagnosticBundle",
    "Candidate",
    "CandidateSet",
    "PlanningResult",
    "CandidateValue",
    "CounterfactualEvaluation",
    "AuthorizationResult",
    "AuthorizationStatus",
    "RejectionReason",
    "CommitResult",
    "LearningResult",
    "DecisionTransition",
    "CreditAssignment",
    "RuntimeStepResult",
    "CANONICAL_PHASE_ORDER",
]
