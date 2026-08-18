"""v5.11 Phases 6-8: Cryptographic authorization binding + exception atomicity + CAS.

These tests prove that:
1. Authorization binding is mandatory (non-nullable)
2. Transaction with None authorization_id is rejected
3. Authorization with wrong transaction_hash is rejected
4. Authorization cannot be reused for a different transaction
5. Exception during commit leaves exact pre-state (rollback)
6. Compare-and-swap: stale version is rejected
7. Compare-and-swap: stale hash is rejected
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.transaction import (
    StructuralTransaction, make_graph_transaction,
    AuthorizationBindingError, StaleTransactionError,
)
from lgae_v3.runtime.contracts.authorization import (
    AuthorizationResult, AuthorizationStatus,
)


def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


def _make_authorized_txn(rt):
    """Helper: create a properly authorized transaction."""
    from lgae_v3.types import MutationResult
    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * 3.0
    txn = make_graph_transaction(
        base_state_version=int(rt.engine.graph.version),
        base_state_hash=rt.authority_hash,
        shadow_graph=shadow,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    )
    # Create a StructuralTransaction with authorization binding.
    full_txn = StructuralTransaction(
        transaction_id=txn.transaction_id,
        base_state_version=txn.base_state_version,
        base_state_hash=txn.base_state_hash,
        graph_delta=txn.graph_delta,
        authorization_id=txn.authorization_binding_hash(),
        delta_hash=txn.delta_hash,
        mutation_result=txn.mutation_result,
    )
    auth = AuthorizationResult(
        snapshot_id="s1",
        state_version=int(rt.engine.graph.version),
        state_hash=rt.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=txn.transaction_id,
    )
    return full_txn, auth


class TestMandatoryAuthorizationBinding:
    """Prove that authorization binding is mandatory and non-nullable."""

    def test_none_authorization_id_rejected(self):
        """A transaction with None authorization_id is rejected."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 3.0
        txn = make_graph_transaction(
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        # Create a transaction with None authorization_id.
        full_txn = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=txn.base_state_version,
            base_state_hash=txn.base_state_hash,
            graph_delta=txn.graph_delta,
            authorization_id=None,  # None — should be rejected
            delta_hash=txn.delta_hash,
            mutation_result=txn.mutation_result,
        )
        auth = AuthorizationResult(
            snapshot_id="s1",
            state_version=int(rt.engine.graph.version),
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn.transaction_id,
        )
        with pytest.raises(AuthorizationBindingError, match="mandatory"):
            rt.commit_channel.commit(full_txn, auth)

    def test_wrong_transaction_hash_rejected(self):
        """Authorization with wrong transaction_hash is rejected."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        full_txn, _ = _make_authorized_txn(rt)
        # Create authorization with wrong transaction_hash.
        wrong_auth = AuthorizationResult(
            snapshot_id="s1",
            state_version=int(rt.engine.graph.version),
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash="wrong_hash",
        )
        with pytest.raises(AuthorizationBindingError, match="different transaction"):
            rt.commit_channel.commit(full_txn, wrong_auth)

    def test_correct_binding_accepted(self):
        """A correctly bound transaction is accepted."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        full_txn, auth = _make_authorized_txn(rt)
        result = rt.commit_channel.commit(full_txn, auth)
        assert result.committed

    def test_authorization_cannot_be_reused(self):
        """An authorization for transaction T1 cannot be used for T2."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        # Create and commit T1.
        txn1, auth1 = _make_authorized_txn(rt)
        result1 = rt.commit_channel.commit(txn1, auth1)
        assert result1.committed
        # Create T2 with a different mutation.
        from lgae_v3.types import MutationResult
        shadow2 = rt.engine.graph.clone()
        shadow2.weight[1] = shadow2.weight[1] * 5.0
        txn2 = make_graph_transaction(
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow2,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=1,
        )
        full_txn2 = StructuralTransaction(
            transaction_id=txn2.transaction_id,
            base_state_version=txn2.base_state_version,
            base_state_hash=txn2.base_state_hash,
            graph_delta=txn2.graph_delta,
            authorization_id=txn2.authorization_binding_hash(),
            delta_hash=txn2.delta_hash,
            mutation_result=txn2.mutation_result,
        )
        # Try to use auth1 (from T1) for T2 — should fail.
        with pytest.raises((AuthorizationBindingError, StaleTransactionError)):
            rt.commit_channel.commit(full_txn2, auth1)


class TestExceptionAtomicity:
    """Prove that exception during commit leaves exact pre-state."""

    def test_rollback_on_exception(self):
        """If an exception occurs during commit, state is rolled back."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        pre_hash = rt.authority_hash
        # Create a transaction that will cause an exception during apply.
        # We do this by making the fiber_delta reference an invalid snapshot.
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 3.0
        txn = make_graph_transaction(
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        full_txn = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=txn.base_state_version,
            base_state_hash=txn.base_state_hash,
            graph_delta=txn.graph_delta,
            authorization_id=txn.authorization_binding_hash(),
            delta_hash=txn.delta_hash,
            mutation_result=txn.mutation_result,
        )
        auth = AuthorizationResult(
            snapshot_id="s1",
            state_version=int(rt.engine.graph.version),
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn.transaction_id,
        )
        # Monkey-patch the engine to throw during fiber restore.
        original_restore = rt._engine.fibers.restore
        call_count = [0]
        def failing_restore(snap):
            call_count[0] += 1
            if call_count[0] > 0:
                raise RuntimeError("injected failure")
            original_restore(snap)
        # Add a fiber delta to trigger the restore path.
        from lgae_v3.runtime.transaction import FiberDelta
        fiber_snap = rt._engine.fibers.snapshot()
        full_txn_with_fiber = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=txn.base_state_version,
            base_state_hash=txn.base_state_hash,
            graph_delta=txn.graph_delta,
            fiber_delta=FiberDelta(shadow_fiber_snapshot=fiber_snap, action="test"),
            authorization_id=txn.authorization_binding_hash(),
            delta_hash=txn.delta_hash,
            mutation_result=txn.mutation_result,
        )
        # Patch the fiber restore to fail.
        rt._engine.fibers.restore = failing_restore
        try:
            with pytest.raises((RuntimeError, Exception)):
                rt.commit_channel.commit(full_txn_with_fiber, auth)
            # Verify state was rolled back.
            post_hash = rt.authority_hash
            assert post_hash == pre_hash, (
                f"State not rolled back after exception! "
                f"Pre: {pre_hash[:16]}..., Post: {post_hash[:16]}..."
            )
        finally:
            rt._engine.fibers.restore = original_restore


class TestCompareAndSwap:
    """Prove compare-and-swap semantics for commit."""

    def test_stale_version_rejected(self):
        """A transaction with stale base_state_version is rejected."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        # Commit once to advance the version.
        txn1, auth1 = _make_authorized_txn(rt)
        rt.commit_channel.commit(txn1, auth1)
        # Now try to commit a transaction with the old version.
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 2.0
        stale_txn = make_graph_transaction(
            base_state_version=0,  # stale — current is 1
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        full_txn = StructuralTransaction(
            transaction_id=stale_txn.transaction_id,
            base_state_version=stale_txn.base_state_version,
            base_state_hash=stale_txn.base_state_hash,
            graph_delta=stale_txn.graph_delta,
            authorization_id=stale_txn.authorization_binding_hash(),
            delta_hash=stale_txn.delta_hash,
            mutation_result=stale_txn.mutation_result,
        )
        auth = AuthorizationResult(
            snapshot_id="s1",
            state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=stale_txn.transaction_id,
        )
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(full_txn, auth)

    def test_stale_hash_rejected(self):
        """A transaction with stale base_state_hash is rejected."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        pre_hash = rt.authority_hash
        # Commit once to change the hash.
        txn1, auth1 = _make_authorized_txn(rt)
        rt.commit_channel.commit(txn1, auth1)
        # Now try to commit with the old hash.
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 2.0
        stale_txn = make_graph_transaction(
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=pre_hash,  # stale hash
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=1,
        )
        full_txn = StructuralTransaction(
            transaction_id=stale_txn.transaction_id,
            base_state_version=stale_txn.base_state_version,
            base_state_hash=stale_txn.base_state_hash,
            graph_delta=stale_txn.graph_delta,
            authorization_id=stale_txn.authorization_binding_hash(),
            delta_hash=stale_txn.delta_hash,
            mutation_result=stale_txn.mutation_result,
        )
        auth = AuthorizationResult(
            snapshot_id="s1",
            state_version=int(rt.engine.graph.version),
            state_hash=pre_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=stale_txn.transaction_id,
        )
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(full_txn, auth)
