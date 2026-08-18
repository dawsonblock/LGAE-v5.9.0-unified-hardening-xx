"""Phase 6 contract: AuthorizationResult.

Output of the authorize() phase: the governance decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .common import PhaseResult
from .evaluation import CounterfactualEvaluation


class AuthorizationStatus(str, Enum):
    """Possible authorization outcomes."""
    AUTHORIZED = "authorized"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    DEFERRED = "deferred"


class RejectionReason(str, Enum):
    """Reason codes for rejection/quarantine/deferral."""
    STALE_STATE = "stale_state"
    INVARIANT_VIOLATION = "invariant_violation"
    UNCERTAINTY_TOO_HIGH = "uncertainty_too_high"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    QUALIFICATION_MISSING = "qualification_missing"
    SIGNATURE_FAILURE = "signature_failure"
    PERFORMANCE_BUDGET_EXCEEDED = "performance_budget_exceeded"
    NO_OP = "no_op"
    CERTIFICATION_FAILED = "certification_failed"


@dataclass(frozen=True, slots=True)
class AuthorizationResult(PhaseResult):
    """Output of the authorize() phase.

    v5.11 Phase 6: Authorization cryptographically binds to the specific
    transaction it approves. The transaction_hash field is mandatory for
    AUTHORIZED results. Commit requires exact equality:

        authorization.transaction_hash == transaction.transaction_hash

    Attributes:
        status: AUTHORIZED / REJECTED / QUARANTINED / DEFERRED
        reason: reason code if not authorized
        certification_level: certification level from evaluation
        authority_hash_before: state hash before authorization
        transaction_hash: hash of the transaction being authorized (mandatory for AUTHORIZED)
        candidate_id: identifier of the authorized candidate
        evaluation_hash: hash of the evaluation being authorized
    """
    status: AuthorizationStatus = AuthorizationStatus.REJECTED
    reason: RejectionReason = RejectionReason.NO_OP
    certification_level: str | None = None
    authority_hash_before: str = ""
    transaction_hash: str = ""
    candidate_id: str = ""
    evaluation_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "status": self.status.value,
            "reason": self.reason.value,
            "certification_level": self.certification_level,
            "authority_hash_before": self.authority_hash_before,
            "transaction_hash": self.transaction_hash,
            "candidate_id": self.candidate_id,
            "evaluation_hash": self.evaluation_hash,
        }

    @property
    def is_authorized(self) -> bool:
        return self.status == AuthorizationStatus.AUTHORIZED
