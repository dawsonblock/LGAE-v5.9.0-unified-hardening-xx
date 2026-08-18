"""v5.11-RC Phase 9: State-aware idempotent WAL replay tests.

Tests that WAL replay is state-aware and idempotent:
- Replaying twice produces the same state
- Already-applied transactions are skipped
- Checkpoint LSN prevents duplicate apply
"""
from __future__ import annotations

import os

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, RuntimeConfig, make_graph_transaction, StructuralTransaction
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.runtime.wal import replay_committed_transactions
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


def _commit_one(rt):
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


class TestStateAwareIdempotentReplay:
    """WAL replay is state-aware and idempotent."""

    def test_replay_twice_is_idempotent(self, tmp_path):
        """Replaying the same WAL twice produces the same state."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_one(rt)
        post_hash = rt.authority_hash

        # Recover onto a fresh runtime.
        torch.manual_seed(42)
        fresh1 = LGAERuntime(_graph(), _cfg())
        replay_committed_transactions(wal_path, fresh1._engine)
        hash1 = fresh1.authority_hash

        # Recover again onto another fresh runtime.
        torch.manual_seed(42)
        fresh2 = LGAERuntime(_graph(), _cfg())
        replay_committed_transactions(wal_path, fresh2._engine)
        hash2 = fresh2.authority_hash

        assert hash1 == hash2, "Replaying the same WAL should produce the same state"
        assert hash1 == post_hash, "Recovered state should match post-commit state"

    def test_already_applied_transaction_is_skipped(self, tmp_path):
        """If the engine already has the post-transaction state, replay skips."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_one(rt)
        post_hash = rt.authority_hash

        # Replay onto the same engine (which already has the post-state).
        results = replay_committed_transactions(wal_path, rt._engine)
        # The transaction should have been skipped (state-aware).
        skipped = [r for r in results if not r.get("applied", False)]
        assert len(skipped) > 0, "Already-applied transaction should be skipped"
        # State should be unchanged.
        assert rt.authority_hash == post_hash

    def test_replay_onto_pre_state_applies_transaction(self, tmp_path):
        """Replaying onto the pre-state applies the transaction."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        pre_hash = rt.authority_hash
        _commit_one(rt)
        post_hash = rt.authority_hash

        # Recover onto a fresh runtime (pre-state).
        torch.manual_seed(42)
        fresh = LGAERuntime(_graph(), _cfg())
        assert fresh.authority_hash == pre_hash
        results = replay_committed_transactions(wal_path, fresh._engine)
        # The transaction should have been applied.
        applied = [r for r in results if r.get("applied", False)]
        assert len(applied) > 0, "Transaction should be applied to pre-state"
        assert fresh.authority_hash == post_hash

    def test_checkpoint_lsn_prevents_duplicate_apply(self, tmp_path):
        """Checkpoint LSN prevents replaying transactions before the checkpoint."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_one(rt)
        post_hash = rt.authority_hash

        # Recover with a high checkpoint LSN — should skip all transactions.
        torch.manual_seed(42)
        fresh = LGAERuntime(_graph(), _cfg())
        results = replay_committed_transactions(wal_path, fresh._engine, checkpoint_lsn=999)
        # All transactions should be skipped.
        skipped = [r for r in results if not r.get("applied", False)]
        assert len(skipped) > 0, "Transactions before checkpoint should be skipped"
        # State should remain at pre-state.
        torch.manual_seed(42)
        initial = LGAERuntime(_graph(), _cfg())
        assert fresh.authority_hash == initial.authority_hash

    def test_replay_preserves_graph_version(self, tmp_path):
        """Replaying doesn't double-increment the graph version."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_one(rt)
        post_version = int(rt.engine.graph.version)

        # Recover onto a fresh runtime.
        torch.manual_seed(42)
        fresh = LGAERuntime(_graph(), _cfg())
        replay_committed_transactions(wal_path, fresh._engine)
        assert int(fresh.engine.graph.version) == post_version, (
            f"Version should be {post_version}, got {int(fresh.engine.graph.version)}"
        )
