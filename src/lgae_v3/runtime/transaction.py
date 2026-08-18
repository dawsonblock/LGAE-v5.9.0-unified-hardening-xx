"""Structural transactions: the sole unit of authoritative mutation (v5.11 Phase 4-5).

A StructuralTransaction captures everything needed to atomically transition
authoritative state from S_n to S_{n+1}:

    StructuralTransaction = {
        base_state_version, base_state_hash,
        graph_delta | fiber_delta | gauge_delta | calibration_delta | model_delta,
        candidate_id, plan_id, authorization_id,
        delta_hash  # cryptographic binding of all deltas
    }

The transaction is created during PROPOSE/PLAN, evaluated during EVALUATE
(shadow-only), authorized during AUTHORIZE, and committed during COMMIT
via CommitChannel.commit(transaction, authorization).

Cryptographic binding prevents TOCTOU:
    H(state || candidate || plan || transaction || evaluation || authorization)

CommitChannel.commit() validates:
    1. authorization.status == AUTHORIZED
    2. transaction.authorization_id == authorization.authorization_id
    3. transaction.base_state_hash == current engine state hash
    4. transaction.base_state_version == current engine version
    5. transaction.delta_hash matches recomputed delta hash
    6. WAL is available in production mode
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..types import GraphBuffers, MutationResult, MutationDecision
from .contracts.common import canonical_hash
from .contracts.authorization import AuthorizationResult, AuthorizationStatus


@dataclass(frozen=True, slots=True)
class GraphDelta:
    """Captured graph state delta for atomic commit."""
    shadow_graph: GraphBuffers  # the complete post-mutation graph
    mutation_name: str = ""
    mutation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_hash(self) -> str:
        return canonical_hash({
            "shadow_graph_hash": self.shadow_graph.state_hash(),
            "mutation_name": self.mutation_name,
            "mutation_metadata": str(sorted(self.mutation_metadata.items())),
        })


@dataclass(frozen=True, slots=True)
class FiberDelta:
    """Captured fiber state delta for atomic commit."""
    shadow_fiber_snapshot: Any  # FiberStateSnapshot
    action: str = ""

    def to_hash(self) -> str:
        # v5.11 Phase 4: Never use Python hash() in deterministic paths.
        # FiberStateSnapshot must implement state_hash() for deterministic
        # transaction identity. If it doesn't, raise rather than fall back
        # to nondeterministic hash().
        if hasattr(self.shadow_fiber_snapshot, "state_hash"):
            h = self.shadow_fiber_snapshot.state_hash()
        else:
            from .state.state_errors import DeterminismError
            raise DeterminismError(
                "FiberDelta.to_hash(): shadow_fiber_snapshot must implement "
                "deterministic state_hash(); Python hash() is not allowed in "
                "deterministic runtime paths (defect D11-008)"
            )
        return canonical_hash({"fiber_hash": h, "action": self.action})


@dataclass(frozen=True, slots=True)
class GaugeDelta:
    """Captured gauge state delta for atomic commit."""
    shadow_gauge_raw: Tensor
    action: str = ""

    def to_hash(self) -> str:
        return canonical_hash({
            "gauge_hash": hashlib.sha256(
                self.shadow_gauge_raw.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "action": self.action,
        })


@dataclass(frozen=True, slots=True)
class StructuralTransaction:
    """A complete structural transaction awaiting commit.

    This is the sole unit of authoritative mutation. No state change
    happens without a transaction passing through CommitChannel.commit().

    Attributes:
        transaction_id: deterministic ID (base_hash || delta_hash || step)
        base_state_version: engine state version when transaction was created
        base_state_hash: engine state hash when transaction was created
        graph_delta: graph mutation delta (or None)
        fiber_delta: fiber mutation delta (or None)
        gauge_delta: gauge mutation delta (or None)
        candidate_id: ID of the candidate that generated this transaction
        plan_id: ID of the plan that selected this transaction
        authorization_id: ID of the authorization decision (bound at authorize time)
        delta_hash: cryptographic hash of all deltas
        mutation_result: the governor's evaluation result (for metadata)
    """
    transaction_id: str
    base_state_version: int
    base_state_hash: str
    graph_delta: GraphDelta | None = None
    fiber_delta: FiberDelta | None = None
    gauge_delta: GaugeDelta | None = None
    candidate_id: str | None = None
    plan_id: str | None = None
    authorization_id: str | None = None
    delta_hash: str = ""
    mutation_result: MutationResult | None = None

    @property
    def base_identity(self) -> Any:
        from .state_identity import AuthorityStateIdentity
        return AuthorityStateIdentity(
            version=self.base_state_version,
            authority_hash=self.base_state_hash,
        )

    def compute_delta_hash(self) -> str:
        """Compute the cryptographic hash of all deltas."""
        parts = []
        if self.graph_delta is not None:
            parts.append(("graph", self.graph_delta.to_hash()))
        if self.fiber_delta is not None:
            parts.append(("fiber", self.fiber_delta.to_hash()))
        if self.gauge_delta is not None:
            parts.append(("gauge", self.gauge_delta.to_hash()))
        return canonical_hash({
            "base_state_hash": self.base_state_hash,
            "base_state_version": self.base_state_version,
            "deltas": parts,
        })

    def authorization_binding_hash(self) -> str:
        """Hash that binds authorization to this specific transaction.

        This prevents swapping transactions after authorization.
        """
        return canonical_hash({
            "transaction_id": self.transaction_id,
            "base_state_hash": self.base_state_hash,
            "base_state_version": self.base_state_version,
            "delta_hash": self.delta_hash,
            "candidate_id": self.candidate_id,
            "plan_id": self.plan_id,
        })

    def with_authorization(self, authorization_id: str | None = None) -> StructuralTransaction:
        """Return a copy of this transaction bound with an authorization ID."""
        auth_id = authorization_id or self.authorization_binding_hash()
        return StructuralTransaction(
            transaction_id=self.transaction_id,
            base_state_version=self.base_state_version,
            base_state_hash=self.base_state_hash,
            graph_delta=self.graph_delta,
            fiber_delta=self.fiber_delta,
            gauge_delta=self.gauge_delta,
            candidate_id=self.candidate_id,
            plan_id=self.plan_id,
            authorization_id=auth_id,
            delta_hash=self.delta_hash,
            mutation_result=self.mutation_result,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "base_state_version": self.base_state_version,
            "base_state_hash": self.base_state_hash,
            "has_graph_delta": self.graph_delta is not None,
            "has_fiber_delta": self.fiber_delta is not None,
            "has_gauge_delta": self.gauge_delta is not None,
            "candidate_id": self.candidate_id,
            "plan_id": self.plan_id,
            "authorization_id": self.authorization_id,
            "delta_hash": self.delta_hash,
        }


class TransactionValidationError(RuntimeError):
    """Raised when a transaction fails validation during commit."""


class StaleTransactionError(TransactionValidationError):
    """Raised when a transaction's base state doesn't match current state."""


class AuthorizationBindingError(TransactionValidationError):
    """Raised when authorization doesn't bind to the transaction."""


def make_graph_transaction(
    *,
    base_state_version: int,
    base_state_hash: str,
    shadow_graph: GraphBuffers,
    mutation_result: MutationResult,
    mutation_name: str = "",
    mutation_metadata: dict[str, Any] | None = None,
    candidate_id: str | None = None,
    plan_id: str | None = None,
    step: int = 0,
) -> StructuralTransaction:
    """Create a StructuralTransaction from a shadow graph evaluation."""
    graph_delta = GraphDelta(
        shadow_graph=shadow_graph,
        mutation_name=mutation_name,
        mutation_metadata=mutation_metadata or {},
    )
    txn = StructuralTransaction(
        transaction_id="",  # filled after delta_hash
        base_state_version=base_state_version,
        base_state_hash=base_state_hash,
        graph_delta=graph_delta,
        candidate_id=candidate_id,
        plan_id=plan_id,
        mutation_result=mutation_result,
    )
    delta_hash = txn.compute_delta_hash()
    txn_id = canonical_hash({
        "base_state_hash": base_state_hash,
        "delta_hash": delta_hash,
        "step": step,
    })
    # Reconstruct with the ID and hash (frozen dataclass).
    return StructuralTransaction(
        transaction_id=txn_id,
        base_state_version=base_state_version,
        base_state_hash=base_state_hash,
        graph_delta=graph_delta,
        candidate_id=candidate_id,
        plan_id=plan_id,
        delta_hash=delta_hash,
        mutation_result=mutation_result,
    )


def make_fiber_transaction(
    *,
    base_state_version: int,
    base_state_hash: str,
    shadow_fiber_snapshot: Any,
    action: str = "",
    mutation_result: MutationResult,
    candidate_id: str | None = None,
    plan_id: str | None = None,
    step: int = 0,
) -> StructuralTransaction:
    """Create a StructuralTransaction from a shadow fiber evaluation.

    v5.11-RC Phase 2: Fiber actions are first-class canonical transactions.
    """
    fiber_delta = FiberDelta(
        shadow_fiber_snapshot=shadow_fiber_snapshot,
        action=action,
    )
    txn = StructuralTransaction(
        transaction_id="",
        base_state_version=base_state_version,
        base_state_hash=base_state_hash,
        fiber_delta=fiber_delta,
        candidate_id=candidate_id,
        plan_id=plan_id,
        mutation_result=mutation_result,
    )
    delta_hash = txn.compute_delta_hash()
    txn_id = canonical_hash({
        "base_state_hash": base_state_hash,
        "delta_hash": delta_hash,
        "step": step,
    })
    return StructuralTransaction(
        transaction_id=txn_id,
        base_state_version=base_state_version,
        base_state_hash=base_state_hash,
        fiber_delta=fiber_delta,
        candidate_id=candidate_id,
        plan_id=plan_id,
        delta_hash=delta_hash,
        mutation_result=mutation_result,
    )


def make_gauge_transaction(
    *,
    base_state_version: int,
    base_state_hash: str,
    shadow_gauge_raw: Tensor,
    action: str = "",
    mutation_result: MutationResult,
    candidate_id: str | None = None,
    plan_id: str | None = None,
    step: int = 0,
) -> StructuralTransaction:
    """Create a StructuralTransaction from a shadow gauge evaluation.

    v5.11-RC Phase 2: Gauge actions are first-class canonical transactions.
    """
    gauge_delta = GaugeDelta(
        shadow_gauge_raw=shadow_gauge_raw,
        action=action,
    )
    txn = StructuralTransaction(
        transaction_id="",
        base_state_version=base_state_version,
        base_state_hash=base_state_hash,
        gauge_delta=gauge_delta,
        candidate_id=candidate_id,
        plan_id=plan_id,
        mutation_result=mutation_result,
    )
    delta_hash = txn.compute_delta_hash()
    txn_id = canonical_hash({
        "base_state_hash": base_state_hash,
        "delta_hash": delta_hash,
        "step": step,
    })
    return StructuralTransaction(
        transaction_id=txn_id,
        base_state_version=base_state_version,
        base_state_hash=base_state_hash,
        gauge_delta=gauge_delta,
        candidate_id=candidate_id,
        plan_id=plan_id,
        delta_hash=delta_hash,
        mutation_result=mutation_result,
    )


def make_joint_transaction(
    *,
    base_state_version: int,
    base_state_hash: str,
    shadow_graph: GraphBuffers | None = None,
    shadow_fiber_snapshot: Any = None,
    shadow_gauge_raw: Tensor | None = None,
    graph_action: str = "",
    fiber_action: str = "",
    gauge_action: str = "",
    mutation_result: MutationResult,
    candidate_id: str | None = None,
    plan_id: str | None = None,
    step: int = 0,
) -> StructuralTransaction:
    """Create a joint StructuralTransaction changing graph, fiber, and gauge.

    v5.11-RC Phase 2/20: Joint transactions are first-class canonical
    transactions. One transaction, one authorization, one commit.
    """
    graph_delta = None
    fiber_delta = None
    gauge_delta = None
    if shadow_graph is not None:
        graph_delta = GraphDelta(
            shadow_graph=shadow_graph,
            mutation_name=graph_action,
        )
    if shadow_fiber_snapshot is not None:
        fiber_delta = FiberDelta(
            shadow_fiber_snapshot=shadow_fiber_snapshot,
            action=fiber_action,
        )
    if shadow_gauge_raw is not None:
        gauge_delta = GaugeDelta(
            shadow_gauge_raw=shadow_gauge_raw,
            action=gauge_action,
        )
    txn = StructuralTransaction(
        transaction_id="",
        base_state_version=base_state_version,
        base_state_hash=base_state_hash,
        graph_delta=graph_delta,
        fiber_delta=fiber_delta,
        gauge_delta=gauge_delta,
        candidate_id=candidate_id,
        plan_id=plan_id,
        mutation_result=mutation_result,
    )
    delta_hash = txn.compute_delta_hash()
    txn_id = canonical_hash({
        "base_state_hash": base_state_hash,
        "delta_hash": delta_hash,
        "step": step,
    })
    return StructuralTransaction(
        transaction_id=txn_id,
        base_state_version=base_state_version,
        base_state_hash=base_state_hash,
        graph_delta=graph_delta,
        fiber_delta=fiber_delta,
        gauge_delta=gauge_delta,
        candidate_id=candidate_id,
        plan_id=plan_id,
        delta_hash=delta_hash,
        mutation_result=mutation_result,
    )
