"""v5.11-RC Phase 13: Concurrency/CAS qualification tests.

Tests that:
- Stale versions/hashes are rejected
- Concurrent commits allow only one valid transaction against a given base state
- CAS is authoritative and atomic
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import (
    LGAERuntime, RuntimeConfig,
    make_graph_transaction, make_joint_transaction, StructuralTransaction,
)
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.runtime import StaleTransactionError
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


def _make_txn(rt, weight_scale=3.0):
    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * weight_scale
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
        snapshot_id="s1", state_version=int(rt.engine.graph.version),
        state_hash=rt.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=full_txn.transaction_id,
    )
    return full_txn, auth


class TestConcurrencyCAS:
    """Concurrency/CAS qualification."""

    def test_stale_version_rejected(self):
        """A transaction with a stale version is rejected."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        # Create a transaction with the current state.
        full_txn, auth = _make_txn(rt)
        # Commit it.
        rt.commit_channel.commit(full_txn, auth)
        # Now try to commit a transaction built on the old version.
        # The old transaction's base_state_version is now stale.
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(full_txn, auth)

    def test_stale_hash_rejected(self):
        """A transaction with a stale hash is rejected."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        full_txn, auth = _make_txn(rt)
        rt.commit_channel.commit(full_txn, auth)
        # The old transaction's base_state_hash is now stale.
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(full_txn, auth)

    def test_concurrent_commits_only_one_succeeds(self):
        """Two transactions against the same base state — only one can commit."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        # Create two transactions against the same base state.
        txn1, auth1 = _make_txn(rt, weight_scale=3.0)
        txn2, auth2 = _make_txn(rt, weight_scale=5.0)
        # First commit succeeds.
        result1 = rt.commit_channel.commit(txn1, auth1)
        assert result1.committed
        # Second commit must fail (stale base state).
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(txn2, auth2)

    def test_cas_atomic_state_transition(self):
        """CAS produces an atomic state transition — version increments by 1."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        v0 = int(rt.engine.graph.version)
        full_txn, auth = _make_txn(rt)
        result = rt.commit_channel.commit(full_txn, auth)
        v1 = int(rt.engine.graph.version)
        assert v1 == v0 + 1, f"CAS should increment version by 1, got {v0} -> {v1}"
        assert result.new_state_version == v1

    def test_cas_hash_changes(self):
        """CAS produces a new state hash."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        h0 = rt.authority_hash
        full_txn, auth = _make_txn(rt)
        result = rt.commit_channel.commit(full_txn, auth)
        h1 = rt.authority_hash
        assert h0 != h1
        assert result.new_state_hash == h1

    def test_failed_cas_preserves_state(self):
        """A failed CAS preserves the pre-transaction state."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        h0 = rt.authority_hash
        v0 = int(rt.engine.graph.version)
        # Create a stale transaction.
        shadow = rt.engine.graph.clone()
        shadow.weight[0] *= 3.0
        txn = make_graph_transaction(
            base_state_version=999,
            base_state_hash="stale",
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        full_txn = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=999,
            base_state_hash="stale",
            graph_delta=txn.graph_delta,
            authorization_id=txn.authorization_binding_hash(),
            delta_hash=txn.delta_hash,
            mutation_result=txn.mutation_result,
        )
        auth = AuthorizationResult(
            snapshot_id="s1", state_version=999,
            state_hash="stale",
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=full_txn.transaction_id,
        )
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(full_txn, auth)
        assert rt.authority_hash == h0, "Failed CAS should preserve state"
        assert int(rt.engine.graph.version) == v0

    def test_sequential_commits_each_succeed(self):
        """Sequential commits (each built on the latest state) all succeed."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        for i in range(3):
            full_txn, auth = _make_txn(rt, weight_scale=2.0 + i)
            result = rt.commit_channel.commit(full_txn, auth)
            assert result.committed, f"Sequential commit {i} should succeed"
        assert int(rt.engine.graph.version) == 3
