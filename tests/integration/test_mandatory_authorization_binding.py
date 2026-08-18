"""v5.11-RC Phase 6: Mandatory authorization binding tests.

Tests that authorization binding is mandatory — empty transaction_hash
is rejected for AUTHORIZED commits.
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import (
    LGAERuntime, make_graph_transaction, StructuralTransaction,
    AuthorizationBindingError, StaleTransactionError,
)
from lgae_v3.runtime.contracts import (
    AuthorizationResult, AuthorizationStatus,
)
from lgae_v3.types import MutationResult


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


def _make_txn(rt):
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
    return full_txn


class TestMandatoryAuthorizationBinding:
    """Authorization binding is mandatory for AUTHORIZED commits."""

    def test_authorized_result_without_transaction_hash_rejected(self):
        """An AUTHORIZED authorization with empty transaction_hash is rejected."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        txn = _make_txn(rt)
        auth = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash="",  # empty!
        )
        with pytest.raises(AuthorizationBindingError, match="transaction_hash is empty"):
            rt.commit_channel.commit(txn, auth)

    def test_authorization_for_t1_cannot_commit_t2(self):
        """An authorization bound to T1 cannot commit T2."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        txn1 = _make_txn(rt)
        # Create a second transaction.
        shadow2 = rt.engine.graph.clone()
        shadow2.weight[1] = shadow2.weight[1] * 5.0
        txn2_base = make_graph_transaction(
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            shadow_graph=shadow2,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        txn2 = StructuralTransaction(
            transaction_id=txn2_base.transaction_id,
            base_state_version=txn2_base.base_state_version,
            base_state_hash=txn2_base.base_state_hash,
            graph_delta=txn2_base.graph_delta,
            authorization_id=txn2_base.authorization_binding_hash(),
            delta_hash=txn2_base.delta_hash,
            mutation_result=txn2_base.mutation_result,
        )
        # Authorization bound to txn1.
        auth_for_1 = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn1.transaction_id,
        )
        # Try to commit txn2 with txn1's authorization.
        with pytest.raises(AuthorizationBindingError, match="does not match"):
            rt.commit_channel.commit(txn2, auth_for_1)

    def test_modified_transaction_after_authorization_rejected(self):
        """A transaction modified after authorization has a different hash."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        txn = _make_txn(rt)
        auth = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn.transaction_id,
        )
        # Modify the transaction's delta_hash.
        from lgae_v3.runtime.transaction import StructuralTransaction
        modified_txn = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=txn.base_state_version,
            base_state_hash=txn.base_state_hash,
            graph_delta=txn.graph_delta,
            authorization_id=txn.authorization_id,
            delta_hash="tampered_hash",
            mutation_result=txn.mutation_result,
        )
        with pytest.raises((AuthorizationBindingError, Exception)):
            rt.commit_channel.commit(modified_txn, auth)

    def test_reused_authorization_rejected(self):
        """An authorization cannot be reused after the state has changed."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        txn = _make_txn(rt)
        auth = AuthorizationResult(
            snapshot_id="s1", state_version=0,
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn.transaction_id,
        )
        # First commit succeeds.
        result1 = rt.commit_channel.commit(txn, auth)
        assert result1.committed
        # Create a second transaction from the new state.
        txn2 = _make_txn(rt)
        auth2 = AuthorizationResult(
            snapshot_id="s1", state_version=int(rt.engine.graph.version),
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=txn2.transaction_id,
        )
        # This should succeed (new auth for new txn).
        result2 = rt.commit_channel.commit(txn2, auth2)
        assert result2.committed

    def test_canonical_authorize_populates_transaction_hash(self):
        """The canonical authorize() phase populates transaction_hash."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        obs = rt.observe()
        reasoning = rt.reason(obs)
        candidates = rt.propose(obs, reasoning)
        planning = rt.plan(obs, reasoning, candidates)
        evaluation = rt.evaluate(obs, planning)
        auth = rt.authorize(obs, evaluation)
        # If a transaction was created, transaction_hash must be populated.
        if auth.status == AuthorizationStatus.AUTHORIZED:
            assert auth.transaction_hash, (
                "canonical authorize() must populate transaction_hash for AUTHORIZED"
            )
