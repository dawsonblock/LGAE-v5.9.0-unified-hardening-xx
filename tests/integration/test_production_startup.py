"""v5.11-RC Phase 14: Harden production startup tests.

Tests that:
- Production startup requires WAL
- Recovery uses the canonical authority state and WAL protocol
- Invalid or incomplete persisted state fails closed
- Corrupted WAL is detected and rejected in production
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from lgae_v3 import ResearchConfig, ProductionConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import (
    LGAERuntime, RuntimeConfig,
    make_graph_transaction, StructuralTransaction,
)
from lgae_v3.runtime.runtime_config import production_runtime_config
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.types import MutationResult


def _prod_cfg() -> ResearchConfig:
    cfg = ProductionConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.orc_top_k = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    return cfg


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


def _prod_rc(tmp_path, wal_path=None):
    return production_runtime_config(
        evidence_path=str(tmp_path / "evidence"),
        receipt_path=str(tmp_path / "receipts"),
        signing_key="test_key",
        wal_path=wal_path,
    )


def _commit_one(rt):
    shadow = rt.engine.graph.clone()
    shadow.weight[0] *= 3.0
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


class TestProductionStartup:
    """Production startup hardening."""

    def test_production_requires_wal(self, tmp_path):
        """Production mode without a WAL path raises an error."""
        torch.manual_seed(42)
        with pytest.raises((ValueError, RuntimeError)):
            LGAERuntime(_graph(), _prod_cfg(), runtime_config=_prod_rc(tmp_path, wal_path=None))

    def test_production_recover_from_wal(self, tmp_path):
        """Production recovery uses the canonical WAL protocol."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _prod_cfg(), runtime_config=_prod_rc(tmp_path, wal_path=wal_path))
        _commit_one(rt)
        post_hash = rt.authority_hash

        # Recover.
        torch.manual_seed(42)
        fresh = LGAERuntime(_graph(), _prod_cfg(), runtime_config=_prod_rc(tmp_path, wal_path=wal_path))
        results = fresh.recover_from_wal()
        assert fresh.authority_hash == post_hash, (
            "Recovery should reproduce post-commit state"
        )

    def test_corrupted_wal_fails_closed_in_production(self, tmp_path):
        """Corrupted WAL hash chain fails closed in production."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _prod_cfg(), runtime_config=_prod_rc(tmp_path, wal_path=wal_path))
        _commit_one(rt)

        # Corrupt the WAL.
        with open(wal_path) as f:
            lines = f.readlines()
        if len(lines) >= 2:
            data = json.loads(lines[1])
            data["payload"]["corrupted"] = True
            lines[1] = json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"
        with open(wal_path, "w") as f:
            f.writelines(lines)

        torch.manual_seed(42)
        fresh = LGAERuntime(_graph(), _prod_cfg(), runtime_config=_prod_rc(tmp_path, wal_path=wal_path))
        with pytest.raises(RuntimeError, match="hash chain"):
            fresh.recover_from_wal()

    def test_non_production_no_wal_is_ok(self):
        """Non-production mode without a WAL is acceptable."""
        torch.manual_seed(42)
        rt = LGAERuntime(_graph(), _prod_cfg(), runtime_config=RuntimeConfig(
            mode="research", wal_path=None,
        ))
        results = rt.recover_from_wal()
        assert results == [], "Non-production with no WAL should return empty"

    def test_production_research_config_rejected(self, tmp_path):
        """Production mode with a research config is rejected."""
        torch.manual_seed(42)
        cfg = ResearchConfig()  # research config, not ProductionConfig
        cfg.fiber.d_base = 2
        cfg.fiber.d_max = 6
        cfg.fiber.gauge_dim = 0
        with pytest.raises((ValueError, RuntimeError)):
            LGAERuntime(_graph(), cfg, runtime_config=_prod_rc(tmp_path, wal_path="/tmp/test_wal.jsonl"))
