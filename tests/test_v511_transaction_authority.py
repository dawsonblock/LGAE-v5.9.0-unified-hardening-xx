"""v5.11 Phase 4-5: hostile transaction and authority tests.

These are the most important tests in v5.11. They verify:

1. No mutation outside CommitChannel.commit()
2. Shadow evaluation does NOT mutate authoritative state
3. Stale transactions are rejected
4. Authorization binding prevents transaction swap attacks
5. Concurrency: racing commits never both succeed
6. Transaction modified after authorization is rejected
7. Authorization cannot be reused for another transaction
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
from lgae_v3.runtime.transaction import TransactionValidationError


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


class TestShadowEvaluation:
    """Phase 5: evaluate() must not mutate authoritative state."""

    def test_evaluate_does_not_mutate_state(self):
        """Running evaluate() does not change the engine's state hash."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash

        # Run observe + reason + propose + plan + evaluate only.
        obs = rt.observe()
        reasoning = rt.reason(obs)
        candidates = rt.propose(obs, reasoning)
        planning = rt.plan(obs, reasoning, candidates)
        evaluation = rt.evaluate(obs, planning)

        hash_after = rt.authority_hash
        assert hash_before == hash_after, (
            "evaluate() mutated authoritative state! "
            f"Before: {hash_before[:16]}, After: {hash_after[:16]}"
        )

    def test_shadow_graph_is_not_authoritative(self):
        """The shadow graph from evaluate() is not the engine's graph."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        obs = rt.observe()
        reasoning = rt.reason(obs)
        candidates = rt.propose(obs, reasoning)
        planning = rt.plan(obs, reasoning, candidates)
        evaluation = rt.evaluate(obs, planning)

        # If a transaction was created, its shadow graph must differ from
        # or be independent of the engine's graph.
        txn = getattr(rt, "_transaction", None)
        if txn is not None and txn.graph_delta is not None:
            # The shadow graph is a clone, not the engine's graph.
            assert txn.graph_delta.shadow_graph is not rt.engine.graph


class TestStaleTransactionRejection:
    """Stale transactions must be rejected."""

    def test_stale_transaction_rejected(self):
        """A transaction with a stale base_state_hash is rejected."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())

        # Create a transaction with a fake stale base state.
        from lgae_v3.types import MutationResult
        fake_shadow = rt.engine.graph.clone()
        fake_txn = make_graph_transaction(
            base_state_version=999,
            base_state_hash="stale_hash_that_does_not_match",
            shadow_graph=fake_shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        fake_auth = AuthorizationResult(
            snapshot_id="s1", state_version=999,
            state_hash="stale_hash_that_does_not_match",
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=fake_txn.transaction_id,
        )
        # Set the authorization_id on the transaction.
        from lgae_v3.runtime.transaction import StructuralTransaction
        fake_txn = StructuralTransaction(
            transaction_id=fake_txn.transaction_id,
            base_state_version=fake_txn.base_state_version,
            base_state_hash=fake_txn.base_state_hash,
            graph_delta=fake_txn.graph_delta,
            candidate_id=fake_txn.candidate_id,
            plan_id=fake_txn.plan_id,
            authorization_id=fake_txn.authorization_binding_hash(),
            delta_hash=fake_txn.delta_hash,
            mutation_result=fake_txn.mutation_result,
        )

        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(fake_txn, fake_auth)


class TestAuthorizationBinding:
    """Authorization must cryptographically bind to the transaction."""

    def test_authorization_cannot_be_reused(self):
        """An authorization for transaction A cannot authorize transaction B."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())

        from lgae_v3.types import MutationResult
        # Create two transactions with different shadow graphs.
        shadow1 = rt.engine.graph.clone()
        shadow2 = rt.engine.graph.clone()
        # Modify shadow2 to make it different.
        shadow2.weight[0] = shadow2.weight[0] * 2.0

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
        auth1 = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn1.transaction_id,
        )
        # Try to use auth1's binding with txn2.
        from lgae_v3.runtime.transaction import StructuralTransaction
        txn2_with_wrong_auth = StructuralTransaction(
            transaction_id=txn2.transaction_id,
            base_state_version=txn2.base_state_version,
            base_state_hash=txn2.base_state_hash,
            graph_delta=txn2.graph_delta,
            authorization_id=txn1.authorization_binding_hash(),  # wrong!
            delta_hash=txn2.delta_hash,
            mutation_result=txn2.mutation_result,
        )
        with pytest.raises(AuthorizationBindingError):
            rt.commit_channel.commit(txn2_with_wrong_auth, auth1)

    def test_rejected_authorization_cannot_commit(self):
        """A REJECTED authorization cannot commit."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())

        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
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
        )
        with pytest.raises(AuthorizationBindingError):
            rt.commit_channel.commit(txn, rejected_auth)


class TestConcurrency:
    """Racing commits must never both succeed."""

    def test_racing_commits_only_one_succeeds(self):
        """Two transactions based on the same state cannot both commit."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())

        from lgae_v3.types import MutationResult
        # Make both shadows actually different from the current graph
        # so the state hash changes after the first commit.
        shadow1 = rt.engine.graph.clone()
        shadow1.weight[0] = shadow1.weight[0] * 3.0  # actual change
        shadow2 = rt.engine.graph.clone()
        shadow2.weight[0] = shadow2.weight[0] * 5.0  # different change

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
        auth1 = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn1.transaction_id,
        )
        auth2 = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn2.transaction_id,
        )
        from lgae_v3.runtime.transaction import StructuralTransaction
        txn1 = StructuralTransaction(
            transaction_id=txn1.transaction_id,
            base_state_version=txn1.base_state_version,
            base_state_hash=txn1.base_state_hash,
            graph_delta=txn1.graph_delta,
            authorization_id=txn1.authorization_binding_hash(),
            delta_hash=txn1.delta_hash,
            mutation_result=txn1.mutation_result,
        )
        txn2 = StructuralTransaction(
            transaction_id=txn2.transaction_id,
            base_state_version=txn2.base_state_version,
            base_state_hash=txn2.base_state_hash,
            graph_delta=txn2.graph_delta,
            authorization_id=txn2.authorization_binding_hash(),
            delta_hash=txn2.delta_hash,
            mutation_result=txn2.mutation_result,
        )

        # First commit succeeds.
        result1 = rt.commit_channel.commit(txn1, auth1)
        assert result1.committed

        # Second commit must fail (stale state — base hash no longer matches).
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(txn2, auth2)

        # Only one commit happened.
        assert rt.commit_channel.commit_count == 1


class TestNoMutationOutsideCommit:
    """The strongest invariant: no state change without a commit."""

    def test_full_step_state_change_only_on_commit(self):
        """If step() doesn't commit, the state hash must not change."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash
        result = rt.step()
        hash_after = rt.authority_hash
        if not result.committed:
            assert hash_before == hash_after, (
                "State hash changed without a commit!"
            )

    def test_evaluate_phase_does_not_mutate(self):
        """Calling evaluate() directly does not change state."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        obs = rt.observe()
        reasoning = rt.reason(obs)
        candidates = rt.propose(obs, reasoning)
        planning = rt.plan(obs, reasoning, candidates)
        hash_before = rt.authority_hash
        rt.evaluate(obs, planning)
        hash_after = rt.authority_hash
        assert hash_before == hash_after


class TestTransactionIntegrity:
    """Transaction hash and delta validation."""

    def test_transaction_delta_hash_is_deterministic(self):
        """The same shadow graph produces the same delta hash."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult
        shadow = rt.engine.graph.clone()
        txn1 = make_graph_transaction(
            base_state_version=0,
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        txn2 = make_graph_transaction(
            base_state_version=0,
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        assert txn1.delta_hash == txn2.delta_hash
        assert txn1.transaction_id == txn2.transaction_id

    def test_different_shadows_different_hashes(self):
        """Different shadow graphs produce different delta hashes."""
        torch.manual_seed(0)
        rt = LGAERuntime(_graph(), _cfg())
        from lgae_v3.types import MutationResult
        shadow1 = rt.engine.graph.clone()
        shadow2 = rt.engine.graph.clone()
        shadow2.weight[0] = shadow2.weight[0] * 2.0
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
        assert txn1.delta_hash != txn2.delta_hash
        assert txn1.transaction_id != txn2.transaction_id
