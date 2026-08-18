"""Phase 7 contract: CommitResult.

Output of the commit() phase: the atomic state update result.
Only commit() may mutate authoritative state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import PhaseResult


@dataclass(frozen=True, slots=True)
class CommitResult(PhaseResult):
    """Output of the commit() phase.

    Attributes:
        committed: whether a mutation was committed (False for NO_OP)
        new_state_version: state version after commit
        new_state_hash: state hash after commit
        transaction_id: deterministic transaction ID
        receipt_hash: signed receipt hash
        evidence_hash: evidence record hash
        delta_utility: change in utility from the commit
        authority_hash_after: authority hash after commit
    """
    committed: bool = False
    new_state_version: int = 0
    new_state_hash: str = ""
    transaction_id: str = ""
    receipt_hash: str | None = None
    evidence_hash: str | None = None
    delta_utility: float = 0.0
    authority_hash_after: str = ""

    @property
    def pre_identity(self) -> Any:
        from ..state_identity import AuthorityStateIdentity
        return AuthorityStateIdentity(
            version=self.state_version,
            authority_hash=self.state_hash,
        )

    @property
    def post_identity(self) -> Any:
        from ..state_identity import AuthorityStateIdentity
        h = self.authority_hash_after or self.new_state_hash
        return AuthorityStateIdentity(
            version=self.new_state_version,
            authority_hash=h,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "committed": self.committed,
            "new_state_version": self.new_state_version,
            "new_state_hash": self.new_state_hash,
            "transaction_id": self.transaction_id,
            "receipt_hash": self.receipt_hash,
            "evidence_hash": self.evidence_hash,
            "delta_utility": self.delta_utility,
            "authority_hash_after": self.authority_hash_after,
        }
