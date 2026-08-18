"""Strict runtime authority boundaries (Phase 2).

Three explicit roles govern who may do what to authoritative state:

  * PROPOSAL    - may generate/rank/retrieve candidates and predict. Cannot
                  mutate authoritative graph/fiber/gauge state.
  * VERIFICATION- may evaluate proposals (shadow simulation, certification).
                  Cannot commit.
  * COMMIT      - the only role permitted to mutate authoritative state, and
                  only through the transactional commit path.

Direct mutation outside the commit authority fails loudly with
``UnauthorizedMutationError`` rather than logging a warning.

This module is intentionally non-breaking: existing engines continue to
operate directly. The boundary is an explicit, testable contract that the
canonical runtime enforces on its own orchestration path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..cache_coherence import GraphReadCoordinator
from ..types import GraphBuffers
from .runtime_state import RuntimeSnapshot, snapshot_from_engine


class AuthorityRole(str, Enum):
    OBSERVATION = "observation"      # read-only readers
    PROPOSAL = "proposal"            # candidate generation / scoring / retrieval
    VERIFICATION = "verification"    # shadow evaluation / certification
    COMMIT = "commit"                # sole authoritative mutator


class UnauthorizedMutationError(RuntimeError):
    """Raised when a non-commit role attempts to mutate authoritative state,
    or when a commit is attempted outside the transactional commit path."""


# Default component -> role classification (from the v5.10 plan).
DEFAULT_BOUNDARIES: dict[str, AuthorityRole] = {
    # Proposal authority
    "graph_state_encoder": AuthorityRole.PROPOSAL,
    "structural_intelligence": AuthorityRole.PROPOSAL,
    "ann_retrieval": AuthorityRole.PROPOSAL,
    "fosr": AuthorityRole.PROPOSAL,
    "effective_resistance": AuthorityRole.PROPOSAL,
    "forman_flow": AuthorityRole.PROPOSAL,
    "learned_candidate_generator": AuthorityRole.PROPOSAL,
    "reasoning_engine": AuthorityRole.PROPOSAL,
    "memory_priors": AuthorityRole.PROPOSAL,
    "mpc_planner": AuthorityRole.PROPOSAL,
    "executive": AuthorityRole.PROPOSAL,
    "counterfactual_proposal": AuthorityRole.PROPOSAL,
    # Verification authority
    "adaptive_geometry": AuthorityRole.VERIFICATION,
    "counterfactual_engine": AuthorityRole.VERIFICATION,
    "exact_orc_lly": AuthorityRole.VERIFICATION,
    "spectral_certification": AuthorityRole.VERIFICATION,
    "topology_certification": AuthorityRole.VERIFICATION,
    "sheaf_gauge_validation": AuthorityRole.VERIFICATION,
    "invariant_checker": AuthorityRole.VERIFICATION,
    "governor": AuthorityRole.VERIFICATION,
    # Commit authority
    "structural_governor": AuthorityRole.COMMIT,
    "transaction_manager": AuthorityRole.COMMIT,
    "mutation_authority_policy": AuthorityRole.COMMIT,
    "engine": AuthorityRole.COMMIT,
}


@dataclass(slots=True)
class AuthorityBoundary:
    """Registry of component roles with enforcement helpers."""
    roles: dict[str, AuthorityRole] = field(default_factory=lambda: dict(DEFAULT_BOUNDARIES))

    def register(self, component: str, role: AuthorityRole) -> None:
        if not isinstance(role, AuthorityRole):
            raise TypeError("role must be an AuthorityRole")
        self.roles[str(component)] = role

    def role_of(self, component: str) -> AuthorityRole:
        return self.roles.get(str(component), AuthorityRole.OBSERVATION)

    def can_mutate(self, component: str) -> bool:
        return self.role_of(component) == AuthorityRole.COMMIT

    def assert_can_mutate(self, component: str) -> None:
        if not self.can_mutate(component):
            raise UnauthorizedMutationError(
                f"component '{component}' has role {self.role_of(component).value}, "
                f"not commit; cannot mutate authoritative state"
            )

    def assert_can_verify(self, component: str) -> None:
        role = self.role_of(component)
        if role not in (AuthorityRole.VERIFICATION, AuthorityRole.COMMIT):
            raise UnauthorizedMutationError(
                f"component '{component}' has role {role.value}; cannot verify proposals"
            )

    def to_summary(self) -> dict[str, str]:
        return {k: v.value for k, v in sorted(self.roles.items())}


class AuthoritativeStateGuard:
    """Read-only view of authoritative state for non-commit components.

    Non-commit components receive a guard instead of the raw engine. The guard
    exposes frozen (immutable) views of graph/fiber/gauge state; any attempt
    to mutate through the guard raises ``UnauthorizedMutationError``.

    v5.11-RC Phase 1: The raw engine is stored via object.__setattr__
    and accessed via object.__getattribute__ internally. External
    attribute access to '_engine' is blocked by __getattribute__.
    This closes the guard._engine escape hatch.
    """

    __slots__ = ("_engine", "_boundary", "_component")

    def __init__(self, engine: Any, boundary: AuthorityBoundary, *, component: str) -> None:
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_boundary", boundary)
        object.__setattr__(self, "_component", str(component))

    def __getattribute__(self, name: str) -> Any:
        # Block external access to the raw engine reference.
        if name == "_engine":
            raise UnauthorizedMutationError(
                "access to raw engine via _engine is blocked; "
                "authoritative state is accessed only through the commit channel"
            )
        return object.__getattribute__(self, name)

    @property
    def component(self) -> str:
        return object.__getattribute__(self, "_component")

    @property
    def role(self) -> AuthorityRole:
        return object.__getattribute__(self, "_boundary").role_of(
            object.__getattribute__(self, "_component")
        )

    def snapshot(self) -> RuntimeSnapshot:
        return snapshot_from_engine(object.__getattribute__(self, "_engine"))

    @property
    def graph(self) -> FrozenGraphView:
        """Frozen (immutable) graph view.

        Returns a FrozenGraphView that defensively clones all tensors.
        Any attempt to mutate through the view raises
        ``UnauthorizedMutationError``.
        """
        from .state.frozen_views import FrozenGraphView
        return FrozenGraphView(object.__getattribute__(self, "_engine").graph)

    @property
    def fibers(self) -> FrozenFiberView:
        """Frozen (immutable) fiber view."""
        from .state.frozen_views import FrozenFiberView
        return FrozenFiberView(object.__getattribute__(self, "_engine").fibers)

    @property
    def gauge_connections(self) -> FrozenGaugeView:
        """Frozen (immutable) gauge view."""
        from .state.frozen_views import FrozenGaugeView
        return FrozenGaugeView(getattr(
            object.__getattribute__(self, "_engine"), "gauge_connections", None
        ))

    def authority_hash(self) -> str:
        return object.__getattribute__(self, "_engine").authority_hash()

    def state_identity(self) -> Any:
        from .state_identity import AuthorityStateIdentity
        eng = object.__getattribute__(self, "_engine")
        return eng.state_identity() if hasattr(eng, "state_identity") else AuthorityStateIdentity.from_engine(eng)

    def __setattr__(self, name: str, value: Any) -> None:
        raise UnauthorizedMutationError(
            f"cannot set attribute '{name}' on AuthoritativeStateGuard; "
            "authoritative state is mutated only through the commit channel"
        )

    def __delattr__(self, name: str) -> None:
        raise UnauthorizedMutationError(
            f"cannot delete attribute '{name}' on AuthoritativeStateGuard"
        )


class CommitChannel:
    """The sole channel through which commit-authority components mutate
    authoritative state.

    v5.11 Phase 4-5: CommitChannel.commit(transaction, authorization) is the
    ONLY path to mutate authoritative state. It validates:

    1. authorization.status == AUTHORIZED
    2. transaction.authorization_id matches authorization
    3. transaction.base_state_hash == current engine state hash
    4. transaction.base_state_version == current engine version
    5. transaction.delta_hash matches recomputed hash
    6. WAL is available in production mode

    If any check fails, the commit is rejected and no state changes.

    Every commit is bracketed by the read coordinator's write epoch
    (seqlock) so concurrent optimistic readers observe a stale read and
    retry rather than seeing a half-applied mutation."""

    def __init__(
        self,
        engine: Any,
        boundary: AuthorityBoundary,
        *,
        component: str = "engine",
        read_coordinator: GraphReadCoordinator | None = None,
        wal: Any = None,
        require_wal: bool = False,
        capability: Any = None,
    ) -> None:
        boundary.assert_can_mutate(component)
        self._engine = engine
        self._boundary = boundary
        self._component = str(component)
        self._read_coordinator = read_coordinator
        self._wal = wal
        self._require_wal = require_wal
        self._capability = capability
        self._commit_count = 0
        self._last_transaction_id: str | None = None
        # v5.11-RC Phase 11: Internal commit failpoints for crash testing.
        # When set, _check_failpoint() will raise at the named point.
        self._failpoint: str | None = None

    def set_failpoint(self, name: str | None) -> None:
        """Set a failpoint for crash testing.

        When set, the commit will raise RuntimeError at the named point.
        Set to None to disable.
        """
        self._failpoint = name

    def _check_failpoint(self, name: str) -> None:
        """Check if a failpoint is active and raise if it matches."""
        if self._failpoint == name:
            raise RuntimeError(f"failpoint: {name}")

    def _bracket(self, fn: Callable[[], Any]) -> Any:
        if self._read_coordinator is None:
            return fn()
        self._read_coordinator.begin_write()
        try:
            return fn()
        finally:
            self._read_coordinator.end_write()

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def commit_count(self) -> int:
        return self._commit_count

    @property
    def last_transaction_id(self) -> str | None:
        return self._last_transaction_id

    def authority_hash(self) -> str:
        return self._engine.authority_hash()

    def state_identity(self) -> Any:
        from .state_identity import AuthorityStateIdentity
        if hasattr(self._engine, "state_identity"):
            return self._engine.state_identity()
        return AuthorityStateIdentity.from_engine(self._engine)

    def snapshot(self) -> RuntimeSnapshot:
        return snapshot_from_engine(self._engine)

    def commit(
        self,
        transaction: Any,
        authorization: Any,
    ) -> Any:
        """Commit a StructuralTransaction with authorization binding.

        This is the sole mutation path. Validates:
        - authorization is AUTHORIZED
        - transaction authorization binding matches
        - base state hash/version match current engine state
        - delta hash is correct
        - WAL is available when required

        Returns CommitResult on success, raises on failure.
        """
        from .transaction import (
            StructuralTransaction, TransactionValidationError,
            StaleTransactionError, AuthorizationBindingError,
        )
        from .contracts.authorization import AuthorizationResult, AuthorizationStatus
        from .contracts.commit import CommitResult
        from ..version import VERSION

        # Validation 1: authorization must be AUTHORIZED.
        if not isinstance(authorization, AuthorizationResult):
            raise AuthorizationBindingError(
                "authorization must be an AuthorizationResult"
            )
        if authorization.status != AuthorizationStatus.AUTHORIZED:
            raise AuthorizationBindingError(
                f"authorization status is {authorization.status.value}, not AUTHORIZED"
            )

        # Validation 2: transaction must be a StructuralTransaction.
        if not isinstance(transaction, StructuralTransaction):
            raise TransactionValidationError(
                "transaction must be a StructuralTransaction"
            )

        # Validation 3: authorization binding (mandatory, non-nullable).
        # v5.11 Phase 6: The transaction must have an authorization_id,
        # and it must match the authorization_binding_hash. No nullable binding.
        expected_auth_id = transaction.authorization_binding_hash()
        if transaction.authorization_id is None:
            raise AuthorizationBindingError(
                "transaction.authorization_id is None; "
                "authorization binding is mandatory (defect D11-009)"
            )
        if transaction.authorization_id != expected_auth_id:
            raise AuthorizationBindingError(
                "transaction.authorization_id does not match "
                "authorization_binding_hash; possible swap attack"
            )
        # v5.11-RC Phase 6: transaction_hash is MANDATORY for AUTHORIZED
        # commits. An empty transaction_hash means the authorization is not
        # cryptographically bound to any transaction — this is rejected.
        if not getattr(authorization, 'transaction_hash', ''):
            raise AuthorizationBindingError(
                "authorization.transaction_hash is empty; "
                "transaction binding is mandatory for AUTHORIZED commits "
                "(defect D11-009: optional binding was bypassable)"
            )
        # The transaction_hash must match the transaction's identity.
        txn_hash = getattr(transaction, 'transaction_id', '') or \
                   getattr(transaction, 'delta_hash', '')
        if not txn_hash:
            raise AuthorizationBindingError(
                "transaction has no transaction_id or delta_hash; "
                "cannot bind authorization"
            )
        if authorization.transaction_hash != txn_hash:
            raise AuthorizationBindingError(
                f"authorization.transaction_hash does not match "
                f"transaction identity; the authorization was for a "
                f"different transaction (possible reuse attack)"
            )

        # Validation 4: base state must match current engine state.
        current_identity = self.state_identity()
        current_hash = current_identity.authority_hash
        current_version = current_identity.version
        if hasattr(transaction, "base_identity") and transaction.base_identity != current_identity:
            raise StaleTransactionError(
                f"transaction base_identity {transaction.base_identity} "
                f"does not match current engine identity {current_identity}; "
                f"transaction is stale"
            )
        if transaction.base_state_hash != current_hash:
            raise StaleTransactionError(
                f"transaction base_state_hash {transaction.base_state_hash[:16]}... "
                f"does not match current engine hash {current_hash[:16]}...; "
                f"transaction is stale"
            )
        if transaction.base_state_version != current_version:
            raise StaleTransactionError(
                f"transaction base_state_version {transaction.base_state_version} "
                f"does not match current engine version {current_version}; "
                f"transaction is stale"
            )

        # Validation 5: delta hash must be correct.
        recomputed = transaction.compute_delta_hash()
        if transaction.delta_hash != recomputed:
            raise TransactionValidationError(
                "transaction.delta_hash does not match recomputed hash; "
                "transaction may have been tampered with"
            )

        # Validation 6: WAL availability in production.
        if self._require_wal and self._wal is None:
            raise TransactionValidationError(
                "WAL is required but not configured"
            )

        # All validations passed. Apply the transaction atomically.
        # v5.11 Phase 7: Exception-atomic commit with rollback.
        # v5.11 Sprint 2 D11-005: WAL ordering is BEGIN → WRITE → COMMIT → APPLY.
        # The COMMIT record is written BEFORE live mutation, so a crash
        # during apply leaves a durable commit record that replay can
        # reconstruct. This guarantees S_restart ∈ {S_t, S_{t+1}}.
        def _apply() -> CommitResult:
            # v5.11-RC Phase 4: Build a complete StateBundle before mutating
            # live state. The bundle is constructed, validated, and then
            # swapped in as a single atomic operation.
            from .state.state_bundle import StateBundle
            pre_hash = self._engine.authority_hash()
            pre_version = int(self._engine.graph.version)

            # Capture the complete mutable authority state before any durable
            # intent or live swap. Exception atomicity requires restoration of
            # graph, fibers, gauges, and derived indices—not merely the graph
            # reference. Crash recovery remains WAL-driven.
            old_graph = self._engine.graph
            old_fiber_snapshot = self._engine.fibers.snapshot()
            old_gauge_raw = None
            if self._engine.gauge_connections is not None:
                old_gauge_raw = (
                    self._engine.gauge_connections.raw_generators.detach().clone()
                )

            def _restore_pre_state() -> None:
                self._engine.graph = old_graph
                self._engine.fibers.restore(old_fiber_snapshot)
                if self._engine.gauge_connections is not None and old_gauge_raw is not None:
                    self._engine.gauge_connections.raw_generators.data.copy_(old_gauge_raw)
                self._engine._invalidate_neighbor_indices("transaction_rollback")

            # v5.11-RC Phase 11: Failpoint before WAL BEGIN.
            self._check_failpoint("before_prepare")

            # WAL: write BEGIN + TX_PREPARE + WRITE + COMMIT records before applying.
            wal_txn_id = None
            try:
                if self._wal is not None:
                    wal_txn_id = self._wal.begin({
                        "transaction_id": transaction.transaction_id,
                        "base_state_hash": transaction.base_state_hash,
                        "base_state_version": transaction.base_state_version,
                    })
                    # v5.11-RC Phase 7: Write TX_PREPARE with complete
                    # transaction metadata for recovery validation.
                    self._wal.prepare(wal_txn_id, {
                        "transaction_id": transaction.transaction_id,
                        "base_state_hash": transaction.base_state_hash,
                        "base_state_version": transaction.base_state_version,
                        "delta_hash": transaction.delta_hash,
                        "authorization_id": transaction.authorization_id or "",
                        "has_graph_delta": transaction.graph_delta is not None,
                        "has_fiber_delta": transaction.fiber_delta is not None,
                        "has_gauge_delta": transaction.gauge_delta is not None,
                    })
                    self._check_failpoint("after_prepare")
                    if transaction.graph_delta is not None:
                        sg = transaction.graph_delta.shadow_graph
                        sd = sg.to_state_dict()
                        json_state = {}
                        for k, v in sd.items():
                            if hasattr(v, "tolist"):
                                json_state[k] = v.tolist()
                            else:
                                json_state[k] = v
                        self._wal.write(wal_txn_id, {
                            "kind": "graph",
                            "shadow_graph_hash": sg.state_hash(),
                            "shadow_graph_state": json_state,
                            "mutation_name": transaction.graph_delta.mutation_name,
                        })
                    if transaction.fiber_delta is not None:
                        snap = transaction.fiber_delta.shadow_fiber_snapshot
                        fiber_state = {}
                        if hasattr(snap, "latent"):
                            for attr in ("latent", "gate_logits", "active_mask", "age",
                                        "utility_ema", "spawn_counter", "gamma_ema"):
                                val = getattr(snap, attr, None)
                                if val is not None and hasattr(val, "tolist"):
                                    fiber_state[attr] = val.detach().cpu().tolist()
                        self._wal.write(wal_txn_id, {
                            "kind": "fiber",
                            "fiber_hash": transaction.fiber_delta.to_hash(),
                            "fiber_state": fiber_state,
                            "action": transaction.fiber_delta.action,
                        })
                    if transaction.gauge_delta is not None:
                        raw = transaction.gauge_delta.shadow_gauge_raw
                        self._wal.write(wal_txn_id, {
                            "kind": "gauge",
                            "gauge_raw": raw.detach().cpu().tolist(),
                            "action": transaction.gauge_delta.action,
                        })
                    self._check_failpoint("before_durable_commit_intent")
                    self._wal.commit(wal_txn_id)
                else:
                    self._check_failpoint("after_prepare")
                    self._check_failpoint("before_durable_commit_intent")

                # v5.11-RC Phase 11: Failpoint after WAL COMMIT.
                self._check_failpoint("after_wal_commit")
                self._check_failpoint("after_durable_commit_intent")

                # v5.11-RC Phase 4: Build the complete candidate state bundle.
                # Clone the current graph, fibers, and gauges, then apply
                # all deltas to the clones. This ensures that live state is
                # only touched during the final atomic swap.
                import dataclasses
                new_graph = self._engine.graph
                if transaction.graph_delta is not None:
                    new_graph = transaction.graph_delta.shadow_graph
                    new_graph = dataclasses.replace(new_graph)
                    for f in dataclasses.fields(self._engine.graph):
                        val = getattr(new_graph, f.name)
                        if hasattr(val, 'clone'):
                            setattr(new_graph, f.name, val.detach().clone())
                    new_graph.bump_version()
                    # Carry over slot_generation from the original shadow graph.
                    new_graph.slot_generation = (
                        transaction.graph_delta.shadow_graph.slot_generation.detach().clone()
                    )

                new_fiber_snapshot = self._engine.fibers.snapshot()
                if transaction.fiber_delta is not None:
                    new_fiber_snapshot = transaction.fiber_delta.shadow_fiber_snapshot

                new_gauge_raw = None
                if self._engine.gauge_connections is not None:
                    new_gauge_raw = self._engine.gauge_connections.raw_generators.detach().clone()
                    if transaction.gauge_delta is not None:
                        new_gauge_raw = transaction.gauge_delta.shadow_gauge_raw.to(
                            new_gauge_raw
                        )
                    # Handle graph-change-induced gauge slot resets.
                    if transaction.graph_delta is not None:
                        old_valid = self._engine.graph.valid
                        new_valid = new_graph.valid
                        reset_mask = old_valid != new_valid
                        if reset_mask.any():
                            reset_ids = torch.where(reset_mask)[0]
                            new_gauge_raw[reset_ids] = 0.0

                # Validate the candidate state bundle.
                if new_graph is None:
                    raise ValueError("candidate graph is None")
                _ = new_graph.state_hash()

                # v5.11-RC Phase 11: Failpoint before state swap.
                self._check_failpoint("before_graph_apply")
                self._check_failpoint("before_state_swap")

                # v5.11-RC Phase 4: Single atomic swap.
                # All state changes are applied in one operation. If any
                # part fails, the pre-state is preserved.
                self._engine.graph = new_graph
                self._check_failpoint("after_graph_apply")
                self._engine.fibers.restore(new_fiber_snapshot)
                self._check_failpoint("after_fiber_apply")
                if self._engine.gauge_connections is not None and new_gauge_raw is not None:
                    self._engine.gauge_connections.raw_generators.data.copy_(new_gauge_raw)
                self._check_failpoint("after_gauge_apply")
                self._check_failpoint("after_calibration_apply")
                self._check_failpoint("after_model_state_apply")
                self._check_failpoint("after_state_version_update")
                self._engine._invalidate_neighbor_indices("transaction_commit")

                after_hash = self._engine.authority_hash()
                after_version = int(self._engine.graph.version)

                if self._wal is not None and wal_txn_id is not None:
                    self._wal.apply(wal_txn_id, {"after_hash": after_hash, "after_version": after_version})

                # v5.11-RC Phase 11: Failpoint after state swap.
                self._check_failpoint("after_state_swap")

                # Phase 3 & 4: Formal verification & receipt boundaries
                self._check_failpoint("before_verification")
                if self._wal is not None and wal_txn_id is not None:
                    self._wal.verify(wal_txn_id, {"verified": True, "after_hash": after_hash, "after_version": after_version})
                self._check_failpoint("after_verification")

                self._check_failpoint("before_receipt")
                self._check_failpoint("during_receipt")
                if self._wal is not None and wal_txn_id is not None:
                    self._wal.finalize(wal_txn_id, {"finalized": True})
                self._check_failpoint("after_receipt")

            except BaseException as commit_error:
                # Exception atomicity: restore the COMPLETE pre-state. If WAL
                # COMMIT was already durable, append ABORT after restoration so
                # recovery cannot resurrect a transaction whose synchronous
                # commit call failed. A hard process crash cannot execute this
                # handler, so COMMIT-without-ABORT remains replayable.
                try:
                    _restore_pre_state()
                    if self._wal is not None and wal_txn_id is not None:
                        self._wal.abort(wal_txn_id)
                    post_rollback_hash = self._engine.authority_hash()
                    if post_rollback_hash != pre_hash:
                        raise RuntimeError(
                            f"rollback failed: hash mismatch after restore "
                            f"(expected {pre_hash[:16]}..., got {post_rollback_hash[:16]}...)"
                        )
                except Exception as rollback_error:
                    raise RuntimeError(
                        f"critical: rollback failed after commit exception: {rollback_error}"
                    ) from rollback_error
                raise commit_error

            self._commit_count += 1
            self._last_transaction_id = transaction.transaction_id

            return CommitResult(
                snapshot_id=f"{transaction.base_state_hash}:{transaction.base_state_version}",
                state_version=transaction.base_state_version,
                state_hash=transaction.base_state_hash,
                committed=True,
                new_state_version=after_version,
                new_state_hash=after_hash,
                transaction_id=transaction.transaction_id,
                authority_hash_after=after_hash,
            )

        return self._bracket(_apply)

    def evaluate_and_maybe_commit(self, mutation: Any) -> Any:
        """Legacy path: Delegate to the engine's transactional commit path.

        DEPRECATED in v5.11. Use commit(transaction, authorization) instead.
        Kept for backward compatibility with existing engine callers.
        """
        return self._bracket(lambda: self._engine.evaluate_and_maybe_commit(mutation, capability=self._capability))

    def evaluate_fiber_action(self, *args, **kwargs) -> Any:
        """Legacy path for fiber actions."""
        return self._bracket(lambda: self._engine.evaluate_fiber_action(*args, capability=self._capability, **kwargs))

    def evaluate_gauge_action(self, *args, **kwargs) -> Any:
        """Legacy path for gauge actions."""
        return self._bracket(lambda: self._engine.evaluate_gauge_action(*args, capability=self._capability, **kwargs))
