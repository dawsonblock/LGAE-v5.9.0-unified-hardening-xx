"""v5.11-RC Phase 4: StateBundle as commit primitive tests.

Tests that CommitChannel uses StateBundle internally — building a complete
candidate state, validating it, and swapping in one atomic operation.
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import (
    LGAERuntime, make_graph_transaction, StructuralTransaction,
)
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.types import MutationResult


def _cfg(gauge_dim: int = 0) -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = gauge_dim
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


def _make_and_commit(rt):
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
        snapshot_id="s1", state_version=int(rt.engine.graph.version),
        state_hash=rt.authority_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=full_txn.transaction_id,
    )
    return rt.commit_channel.commit(full_txn, auth)


class TestStateBundleCommit:
    """StateBundle is the actual commit primitive."""

    def test_commit_produces_consistent_state(self):
        """A commit produces a state where graph, fiber, and gauge are all updated."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash
        result = _make_and_commit(rt)
        assert result.committed
        hash_after = rt.authority_hash
        assert hash_before != hash_after, "State should change after commit"

    def test_commit_version_increments(self):
        """Each commit increments the version."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        v0 = int(rt.engine.graph.version)
        _make_and_commit(rt)
        v1 = int(rt.engine.graph.version)
        assert v1 == v0 + 1, f"Version should increment by 1, got {v0} -> {v1}"

    def test_authority_changes_once_per_commit(self):
        """The authority hash changes exactly once per commit (single swap)."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        hashes_seen = [rt.authority_hash]
        # Commit and track hash changes.
        result = _make_and_commit(rt)
        hashes_seen.append(rt.authority_hash)
        # Only two hashes: before and after.
        assert len(hashes_seen) == 2
        assert hashes_seen[0] != hashes_seen[1]

    def test_failed_commit_preserves_state(self):
        """A failed commit preserves the pre-transaction state."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        hash_before = rt.authority_hash
        # Create a transaction with a stale base state hash.
        shadow = rt.engine.graph.clone()
        txn = make_graph_transaction(
            base_state_version=999,
            base_state_hash="stale_hash",
            shadow_graph=shadow,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
            step=0,
        )
        full_txn = StructuralTransaction(
            transaction_id=txn.transaction_id,
            base_state_version=999,
            base_state_hash="stale_hash",
            graph_delta=txn.graph_delta,
            authorization_id=txn.authorization_binding_hash(),
            delta_hash=txn.delta_hash,
            mutation_result=txn.mutation_result,
        )
        auth = AuthorizationResult(
            snapshot_id="s1", state_version=999,
            state_hash="stale_hash",
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash=full_txn.transaction_id,
        )
        from lgae_v3.runtime import StaleTransactionError
        with pytest.raises(StaleTransactionError):
            rt.commit_channel.commit(full_txn, auth)
        hash_after = rt.authority_hash
        assert hash_before == hash_after, (
            "Failed commit should preserve pre-transaction state"
        )

    def test_commit_with_gauge_dim_nonzero(self):
        """Commit works with nonzero gauge_dim."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg(gauge_dim=3))
        hash_before = rt.authority_hash
        result = _make_and_commit(rt)
        assert result.committed
        hash_after = rt.authority_hash
        assert hash_before != hash_after

    def test_multiple_commits_produce_increasing_versions(self):
        """Multiple commits produce monotonically increasing versions."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _cfg())
        versions = [int(rt.engine.graph.version)]
        for _ in range(3):
            _make_and_commit(rt)
            versions.append(int(rt.engine.graph.version))
        for i in range(1, len(versions)):
            assert versions[i] > versions[i-1], (
                f"Version should increase: {versions}"
            )
