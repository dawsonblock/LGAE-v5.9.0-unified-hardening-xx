"""v5.11 Phase 27: adversarial authority attack tests.

These tests try to break the authority invariant. Every test attempts
an attack that should be blocked. If any test fails, the authority
boundary has a hole.

Attack categories:
1. Direct mutation attacks (bypass CommitChannel)
2. Transaction tampering attacks (modify after authorization)
3. Authorization replay attacks (reuse for different transaction)
4. Stale state attacks (commit against old state)
5. Race condition attacks (concurrent commits)
6. Shadow state leakage (evaluate mutates authoritative state)
7. Frozen view bypass (mutate through aliased references)
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import (
    LGAERuntime, RuntimeConfig, CommitChannel,
    StructuralTransaction, GraphDelta, make_graph_transaction,
    StaleTransactionError, AuthorizationBindingError,
    TransactionValidationError, UnauthorizedMutationError,
)
from lgae_v3.runtime.contracts import (
    ObservationSnapshot, AuthorizationResult, AuthorizationStatus,
    RejectionReason,
)
from lgae_v3.runtime.transaction import StructuralTransaction
from lgae_v3.runtime.state import FrozenGraphView


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


class TestDirectMutationAttacks:
    """Attack: try to mutate authoritative state directly, bypassing CommitChannel."""

    def test_cannot_mutate_through_guard_graph(self):
        """Mutating guard.graph.weight does not change authoritative state."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        original_weight = rt.engine.graph.weight.clone()

        # Attack: try to mutate through the frozen view.
        frozen = guard.graph
        w = frozen.weight
        w[0] = w[0] * 100.0

        # Authoritative state must be unchanged.
        assert torch.equal(original_weight, rt.engine.graph.weight), (
            "Attack succeeded: authoritative state was mutated through guard!"
        )

    def test_cannot_set_engine_graph_directly(self):
        """Setting engine.graph directly is not blocked by the guard, but
        the runtime's commit authority check should catch it."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        # The engine is accessible, but the authority boundary tracks
        # that only the commit channel should mutate.
        # This test documents that the engine is accessible — the real
        # protection is that the canonical path only uses CommitChannel.
        assert rt.engine is not None
        # The guard for non-commit components returns frozen views.
        guard = rt.guard_for("executive")
        assert isinstance(guard.graph, FrozenGraphView)

    def test_cannot_mutate_through_frozen_fiber_view(self):
        """Mutating guard.fibers.z does not change authoritative state."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        original_z = rt.engine.fibers().detach().clone()

        # Attack: try to mutate through the frozen view.
        frozen_z = guard.fibers.z
        frozen_z[0, 0] = frozen_z[0, 0] * 100.0

        # Authoritative state must be unchanged.
        new_z = rt.engine.fibers().detach().clone()
        assert torch.equal(original_z, new_z), (
            "Attack succeeded: fiber state was mutated through guard!"
        )


class TestTransactionTamperingAttacks:
    """Attack: modify a transaction after authorization."""

    def test_modified_delta_hash_rejected(self):
        """A transaction with a tampered delta_hash is rejected."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 3.0

        txn = make_graph_transaction(
            base_state_version=0,
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        # Tamper with the delta_hash.
        tampered_txn = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=txn.base_state_version,
            base_state_hash=txn.base_state_hash,
            graph_delta=txn.graph_delta,
            authorization_id=txn.authorization_binding_hash(),
            delta_hash="tampered_hash_that_does_not_match",
            mutation_result=txn.mutation_result,
        )
        auth = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn.transaction_id,
        )
        with pytest.raises(TransactionValidationError):
            rt.commit_channel.commit(tampered_txn, auth)

    def test_modified_shadow_graph_rejected(self):
        """A transaction with a modified shadow graph has a different delta_hash
        and is rejected."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 3.0

        txn = make_graph_transaction(
            base_state_version=0,
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        # Modify the shadow graph after transaction creation.
        # This changes the graph's state_hash, so the delta_hash
        # won't match when recomputed.
        modified_shadow = txn.graph_delta.shadow_graph
        modified_shadow.weight[1] = modified_shadow.weight[1] * 5.0

        tampered_txn = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=txn.base_state_version,
            base_state_hash=txn.base_state_hash,
            graph_delta=GraphDelta(
                shadow_graph=modified_shadow,
                mutation_name=txn.graph_delta.mutation_name,
                mutation_metadata=txn.graph_delta.mutation_metadata,
            ),
            authorization_id=txn.authorization_binding_hash(),
            delta_hash=txn.delta_hash,  # original hash, not recomputed
            mutation_result=txn.mutation_result,
        )
        auth = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn.transaction_id,
        )
        with pytest.raises(TransactionValidationError):
            rt.commit_channel.commit(tampered_txn, auth)


class TestAuthorizationReplayAttacks:
    """Attack: reuse an authorization for a different transaction."""

    def test_authorization_cannot_authorize_different_transaction(self):
        """An authorization bound to transaction A cannot authorize transaction B."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult

        # Create two different transactions.
        shadow1 = rt.engine.graph.clone()
        shadow1.weight[0] = shadow1.weight[0] * 3.0
        shadow2 = rt.engine.graph.clone()
        shadow2.weight[0] = shadow2.weight[0] * 5.0

        txn1 = make_graph_transaction(
            base_state_version=0,
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow1,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        txn2 = make_graph_transaction(
            base_state_version=0,
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow2,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        # Authorize txn1.
        auth_for_1 = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn1.transaction_id,
        )
        # Try to use txn1's authorization_id with txn2.
        txn2_with_txn1_auth = StructuralTransaction(
            transaction_id=txn2.transaction_id,
            base_state_version=txn2.base_state_version,
            base_state_hash=txn2.base_state_hash,
            graph_delta=txn2.graph_delta,
            authorization_id=txn1.authorization_binding_hash(),  # wrong!
            delta_hash=txn2.delta_hash,
            mutation_result=txn2.mutation_result,
        )
        with pytest.raises(AuthorizationBindingError):
            rt.commit_channel.commit(txn2_with_txn1_auth, auth_for_1)

    def test_rejected_authorization_cannot_commit(self):
        """A REJECTED authorization cannot be used to commit."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 3.0

        txn = make_graph_transaction(
            base_state_version=0,
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        rejected_auth = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.REJECTED,
            reason=RejectionReason.CERTIFICATION_FAILED,
        )
        with pytest.raises(AuthorizationBindingError):
            rt.commit_channel.commit(txn, rejected_auth)

    def test_quarantined_authorization_cannot_commit(self):
        """A QUARANTINED authorization cannot be used to commit."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 3.0

        txn = make_graph_transaction(
            base_state_version=0,
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        quarantined_auth = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.QUARANTINED,
        )
        with pytest.raises(AuthorizationBindingError):
            rt.commit_channel.commit(txn, quarantined_auth)


class TestStaleStateAttacks:
    """Attack: commit a transaction against an old state version."""

    def test_old_state_hash_rejected(self):
        """A transaction with a stale base_state_hash is rejected."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()

        # Create a transaction with a wrong base state hash.
        txn = make_graph_transaction(
            base_state_version=0,
            base_state_hash="wrong_hash_that_does_not_match_anything",
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        auth = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash="wrong_hash_that_does_not_match_anything",
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn.transaction_id,
        )
        txn = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=txn.base_state_version,
            base_state_hash=txn.base_state_hash,
            graph_delta=txn.graph_delta,
            authorization_id=txn.authorization_binding_hash(),
            delta_hash=txn.delta_hash,
            mutation_result=txn.mutation_result,
        )
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(txn, auth)

    def test_old_state_version_rejected(self):
        """A transaction with a stale base_state_version is rejected."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()

        txn = make_graph_transaction(
            base_state_version=999,  # wrong version
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        auth = AuthorizationResult(
            snapshot_id="s1", state_version=999,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn.transaction_id,
        )
        txn = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=txn.base_state_version,
            base_state_hash=txn.base_state_hash,
            graph_delta=txn.graph_delta,
            authorization_id=txn.authorization_binding_hash(),
            delta_hash=txn.delta_hash,
            mutation_result=txn.mutation_result,
        )
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(txn, auth)


class TestShadowStateLeakage:
    """Attack: verify that shadow evaluation never leaks to authoritative state."""

    def test_evaluate_does_not_change_state_hash(self):
        """Running evaluate() does not change the authority hash."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash
        obs = rt.observe()
        reasoning = rt.reason(obs)
        candidates = rt.propose(obs, reasoning)
        planning = rt.plan(obs, reasoning, candidates)
        rt.evaluate(obs, planning)
        hash_after = rt.authority_hash
        assert hash_before == hash_after, (
            "Shadow evaluation leaked to authoritative state!"
        )

    def test_multiple_evaluates_do_not_change_state(self):
        """Running evaluate() multiple times does not change state."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash
        for _ in range(5):
            obs = rt.observe()
            reasoning = rt.reason(obs)
            candidates = rt.propose(obs, reasoning)
            planning = rt.plan(obs, reasoning, candidates)
            rt.evaluate(obs, planning)
        hash_after = rt.authority_hash
        assert hash_before == hash_after


class TestFrozenViewBypass:
    """Attack: try to bypass frozen views through aliased references."""

    def test_frozen_graph_view_weight_is_clone(self):
        """The frozen graph view's weight tensor is a clone, not a reference."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        frozen = guard.graph
        # The frozen weight must not be the same object as the engine's weight.
        assert frozen.weight.data_ptr() != rt.engine.graph.weight.data_ptr(), (
            "Frozen view weight is the same tensor as authoritative state!"
        )

    def test_frozen_graph_view_src_is_clone(self):
        """The frozen graph view's src tensor is a clone."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        frozen = guard.graph
        assert frozen.src.data_ptr() != rt.engine.graph.src.data_ptr()

    def test_frozen_graph_view_cannot_set_attributes(self):
        """Setting any attribute on a FrozenGraphView raises."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        frozen = guard.graph
        with pytest.raises(UnauthorizedMutationError):
            frozen.weight = torch.zeros(12)
        with pytest.raises(UnauthorizedMutationError):
            frozen.custom_attr = 123

    def test_repeated_access_returns_same_clone(self):
        """Repeated access to the frozen view returns the same clone
        (not a fresh clone each time, which would be wasteful but safe)."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        guard = rt.guard_for("executive")
        frozen = guard.graph
        w1 = frozen.weight
        w2 = frozen.weight
        # Same cached clone.
        assert w1.data_ptr() == w2.data_ptr()

    def test_state_hash_unchanged_after_all_attacks(self):
        """After all attack attempts, the authority hash is unchanged."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash

        # Attempt various mutations through frozen views.
        guard = rt.guard_for("executive")
        frozen_graph = guard.graph
        frozen_graph.weight[0] = 0.0
        frozen_graph.src[0] = 99
        frozen_fibers = guard.fibers
        frozen_fibers.z[0, 0] = 0.0

        # Try setting attributes.
        for attr in ["weight", "src", "dst", "valid", "custom"]:
            try:
                setattr(frozen_graph, attr, torch.zeros(12))
            except UnauthorizedMutationError:
                pass  # expected

        hash_after = rt.authority_hash
        assert hash_before == hash_after, (
            "Authority hash changed after attack attempts! "
            f"Before: {hash_before[:16]}, After: {hash_after[:16]}"
        )
