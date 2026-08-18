"""v5.11-RC Phase 7: WAL complete transaction records (TX_PREPARE) tests.

Tests that:
- WAL writes TX_PREPARE records with complete transaction metadata
- Recovery can reconstruct and validate the complete transaction
- TX_PREPARE includes expected pre-state and post-state hashes
"""
from __future__ import annotations

import os

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import (
    LGAERuntime, RuntimeConfig,
    make_graph_transaction, StructuralTransaction,
)
from lgae_v3.runtime.contracts import AuthorizationResult, AuthorizationStatus
from lgae_v3.runtime.wal import (
    WriteAheadLog, WALRecordType,
    recover_transactions, recover_transaction_metadata,
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
    return rt.commit_channel.commit(full_txn, auth), full_txn


class TestWALCompleteTransactions:
    """WAL complete transaction records (TX_PREPARE)."""

    def test_wal_has_tx_prepare_record(self, tmp_path):
        """WAL writes a TX_PREPARE record with complete transaction metadata."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        result, full_txn = _commit_one(rt)

        wal = WriteAheadLog(wal_path)
        records = list(wal.iter_records())
        prepare_records = [r for r in records if r.record_type == WALRecordType.TX_PREPARE]
        assert len(prepare_records) == 1, "Should have exactly one TX_PREPARE record"

        payload = prepare_records[0].payload
        assert payload["transaction_id"] == full_txn.transaction_id
        assert payload["base_state_hash"] == full_txn.base_state_hash
        assert payload["base_state_version"] == full_txn.base_state_version
        assert payload["delta_hash"] == full_txn.delta_hash
        assert payload["authorization_id"] == full_txn.authorization_id

    def test_recover_transaction_metadata(self, tmp_path):
        """recover_transaction_metadata extracts TX_PREPARE data."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        result, full_txn = _commit_one(rt)

        wal = WriteAheadLog(wal_path)
        records = list(wal.iter_records())
        metadata = recover_transaction_metadata(records)

        assert len(metadata) == 1, "Should have metadata for one committed transaction"
        txn_id = list(metadata.keys())[0]
        meta = metadata[txn_id]
        assert meta["transaction_id"] == full_txn.transaction_id
        assert meta["base_state_hash"] == full_txn.base_state_hash
        assert meta["delta_hash"] == full_txn.delta_hash

    def test_tx_prepare_includes_delta_info(self, tmp_path):
        """TX_PREPARE includes which deltas are present."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        result, full_txn = _commit_one(rt)

        wal = WriteAheadLog(wal_path)
        records = list(wal.iter_records())
        metadata = recover_transaction_metadata(records)
        meta = list(metadata.values())[0]
        assert meta["has_graph_delta"] is True
        assert meta["has_fiber_delta"] is False
        assert meta["has_gauge_delta"] is False

    def test_tx_prepare_hash_chain_valid(self, tmp_path):
        """TX_PREPARE records are part of the hash chain."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_one(rt)

        wal = WriteAheadLog(wal_path)
        assert wal.verify_chain(), "Hash chain should be valid with TX_PREPARE"

    def test_recover_transactions_includes_prepare_data(self, tmp_path):
        """recover_transactions includes TX_PREPARE data in the mutations list."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(_graph(), _cfg(), runtime_config=RuntimeConfig(wal_path=wal_path))
        _commit_one(rt)

        wal = WriteAheadLog(wal_path)
        records = list(wal.iter_records())
        committed = recover_transactions(records)
        assert len(committed) == 1
        mutations = list(committed.values())[0]
        # The first mutation should be the TX_PREPARE data.
        assert mutations[0].get("transaction_id") is not None
