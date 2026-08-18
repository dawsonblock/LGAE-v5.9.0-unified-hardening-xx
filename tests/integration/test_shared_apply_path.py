"""v5.11-RC Phase 10: Normal commit and recovery share one apply path tests.

Tests that:
- Normal commit and recovery replay use the same apply logic
- Both produce identical state for the same transaction
- The shared apply path is deterministic
"""
from __future__ import annotations

import os

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import (
    LGAERuntime, RuntimeConfig,
    make_graph_transaction, make_joint_transaction, StructuralTransaction,
)
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.runtime.wal import replay_committed_transactions, apply_wal_mutation
from lgae_v3.types import MutationResult


def _cfg(gauge_dim: int = 3) -> ResearchConfig:
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


def _commit_joint(rt):
    """Commit a joint graph/fiber/gauge transaction."""
    shadow_graph = rt.engine.graph.clone()
    shadow_graph.weight[0] *= 3.0
    snap = rt.engine.fibers.snapshot()
    if hasattr(snap, 'latent') and snap.latent is not None:
        snap.latent.data.fill_(0.5)
    raw = rt.engine.gauge_connections.raw_generators.detach().clone()
    raw.fill_(0.123)
    txn = make_joint_transaction(
        base_state_version=int(rt.engine.graph.version),
        base_state_hash=rt.authority_hash,
        shadow_graph=shadow_graph,
        shadow_fiber_snapshot=snap,
        shadow_gauge_raw=raw,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    )
    full_txn = StructuralTransaction(
        transaction_id=txn.transaction_id,
        base_state_version=txn.base_state_version,
        base_state_hash=txn.base_state_hash,
        graph_delta=txn.graph_delta,
        fiber_delta=txn.fiber_delta,
        gauge_delta=txn.gauge_delta,
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


class TestSharedApplyPath:
    """Normal commit and recovery share one apply path."""

    def test_normal_and_recovery_produce_same_state(self, tmp_path):
        """Normal commit and WAL recovery produce the same state."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        result = _commit_joint(rt)
        post_hash = rt.authority_hash
        post_graph = rt.engine.graph.state_hash()
        post_fiber = rt.engine.fibers.state_hash()
        post_gauge = rt.engine.gauge_connections.state_hash

        # Recover.
        torch.manual_seed(42)
        fresh = LGAERuntime(_graph(), _cfg())
        replay_committed_transactions(wal_path, fresh._engine)

        assert fresh.authority_hash == post_hash
        assert fresh.engine.graph.state_hash() == post_graph
        assert fresh.engine.fibers.state_hash() == post_fiber
        assert fresh.engine.gauge_connections.state_hash == post_gauge

    def test_apply_wal_mutation_is_deterministic(self, tmp_path):
        """apply_wal_mutation is deterministic — same mutation → same state."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_joint(rt)

        # Read WAL records and extract mutations.
        from lgae_v3.runtime.wal import WriteAheadLog, recover_transactions
        wal = WriteAheadLog(wal_path)
        records = list(wal.iter_records())
        committed = recover_transactions(records)
        mutations = list(committed.values())[0] if committed else []

        # Apply to two fresh engines.
        torch.manual_seed(42)
        fresh1 = LGAERuntime(_graph(), _cfg())
        for m in mutations:
            apply_wal_mutation(fresh1._engine, m)

        torch.manual_seed(42)
        fresh2 = LGAERuntime(_graph(), _cfg())
        for m in mutations:
            apply_wal_mutation(fresh2._engine, m)

        assert fresh1.authority_hash == fresh2.authority_hash, (
            "apply_wal_mutation should be deterministic"
        )

    def test_shared_apply_path_produces_correct_graph(self, tmp_path):
        """The shared apply path produces the correct graph state."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_joint(rt)
        post_graph_weight = rt.engine.graph.weight.clone()

        # Recover.
        torch.manual_seed(42)
        fresh = LGAERuntime(_graph(), _cfg())
        replay_committed_transactions(wal_path, fresh._engine)

        assert torch.allclose(fresh.engine.graph.weight, post_graph_weight), (
            "Recovered graph weight should match committed graph weight"
        )

    def test_shared_apply_path_produces_correct_fiber(self, tmp_path):
        """The shared apply path produces the correct fiber state."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_joint(rt)
        post_fiber_latent = rt.engine.fibers.latent.clone()

        # Recover.
        torch.manual_seed(42)
        fresh = LGAERuntime(_graph(), _cfg())
        replay_committed_transactions(wal_path, fresh._engine)

        assert torch.allclose(fresh.engine.fibers.latent, post_fiber_latent), (
            "Recovered fiber latent should match committed fiber latent"
        )

    def test_shared_apply_path_produces_correct_gauge(self, tmp_path):
        """The shared apply path produces the correct gauge state."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_joint(rt)
        post_gauge_raw = rt.engine.gauge_connections.raw_generators.detach().clone()

        # Recover.
        torch.manual_seed(42)
        fresh = LGAERuntime(_graph(), _cfg())
        replay_committed_transactions(wal_path, fresh._engine)

        assert torch.allclose(
            fresh.engine.gauge_connections.raw_generators.detach(),
            post_gauge_raw,
        ), "Recovered gauge raw should match committed gauge raw"
