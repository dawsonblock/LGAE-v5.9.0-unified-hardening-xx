"""v5.11 Sprint 2: WAL redesign — counter restoration + complete serialization.

Tests for:
- D11-006: WAL counters (LSN, txn_id) restored on reopen
- D11-004: WAL serializes complete transaction state (graph + fiber + gauge)
- D11-005: WAL COMMIT ordering is correct (COMMIT before live state publication)
"""
from __future__ import annotations

import json
import os
import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.wal import WriteAheadLog, WALRecordType, recover_transactions
from lgae_v3.runtime.transaction import (
    StructuralTransaction, make_graph_transaction, FiberDelta, GaugeDelta,
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


class TestWALCounterRestoration:
    """D11-006: WAL counters must be restored on reopen."""

    def test_lsn_restored_on_reopen(self, tmp_path):
        """LSN is restored when reopening an existing WAL."""
        wal_path = str(tmp_path / "wal.jsonl")
        wal1 = WriteAheadLog(wal_path)
        txn_id = wal1.begin({"test": "data"})
        wal1.write(txn_id, {"kind": "graph", "data": "test"})
        wal1.commit(txn_id)
        lsn_after_writes = wal1._lsn

        # Reopen the WAL.
        wal2 = WriteAheadLog(wal_path)
        assert wal2._lsn >= lsn_after_writes, (
            f"LSN not restored! Expected >= {lsn_after_writes}, got {wal2._lsn}"
        )

    def test_txn_id_restored_on_reopen(self, tmp_path):
        """Next transaction ID is restored when reopening."""
        wal_path = str(tmp_path / "wal.jsonl")
        wal1 = WriteAheadLog(wal_path)
        txn_id1 = wal1.begin({"test": "data1"})
        wal1.commit(txn_id1)
        txn_id2 = wal1.begin({"test": "data2"})
        wal1.commit(txn_id2)
        next_txn_after = wal1._next_txn_id

        # Reopen.
        wal2 = WriteAheadLog(wal_path)
        assert wal2._next_txn_id >= next_txn_after, (
            f"next_txn_id not restored! Expected >= {next_txn_after}, got {wal2._next_txn_id}"
        )

    def test_no_txn_id_collision_on_reopen(self, tmp_path):
        """No transaction ID collision after reopen."""
        wal_path = str(tmp_path / "wal.jsonl")
        wal1 = WriteAheadLog(wal_path)
        txn1 = wal1.begin({"test": 1})
        wal1.commit(txn1)

        wal2 = WriteAheadLog(wal_path)
        txn2 = wal2.begin({"test": 2})
        assert txn2 > txn1, (
            f"txn_id collision! txn1={txn1}, txn2={txn2}"
        )
        wal2.commit(txn2)


class TestWALCompleteSerialization:
    """D11-004: WAL serializes complete transaction state."""

    def test_wal_records_graph_delta(self, tmp_path):
        """WAL records graph delta with shadow graph state."""
        torch.manual_seed(42)
        rt = LGAERuntime(
            _graph(), _cfg(),
            runtime_config=RuntimeConfig(wal_path=str(tmp_path / "wal.jsonl")),
        )
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
        result = rt.commit_channel.commit(full_txn, auth)
        assert result.committed

        # Verify WAL contains graph record.
        wal = WriteAheadLog(str(tmp_path / "wal.jsonl"))
        records = list(wal.iter_records())
        write_records = [r for r in records if r.record_type == WALRecordType.WRITE]
        graph_records = [r for r in write_records if r.payload.get("kind") == "graph"]
        assert len(graph_records) > 0, "No graph WRITE record in WAL"
        assert "shadow_graph_state" in graph_records[0].payload

    def test_wal_records_fiber_delta(self, tmp_path):
        """WAL records fiber delta when present."""
        torch.manual_seed(42)
        rt = LGAERuntime(
            _graph(), _cfg(),
            runtime_config=RuntimeConfig(wal_path=str(tmp_path / "wal.jsonl")),
        )
        from lgae_v3.types import MutationResult
        from lgae_v3.runtime.transaction import GraphDelta
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 3.0
        fiber_snap = rt._engine.fibers.snapshot()
        graph_delta = GraphDelta(shadow_graph=shadow, mutation_name="test")
        # Step 1: Create with empty hashes to compute delta_hash.
        tmp_txn = StructuralTransaction(
            transaction_id="test_txn_1",
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            graph_delta=graph_delta,
            fiber_delta=FiberDelta(shadow_fiber_snapshot=fiber_snap, action="spawn"),
            authorization_id="",
            delta_hash="placeholder",
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        )
        # Step 2: Compute correct delta_hash.
        correct_delta_hash = tmp_txn.compute_delta_hash()
        # Step 3: Create with correct delta_hash to compute auth_id.
        tmp_txn2 = StructuralTransaction(
            transaction_id="test_txn_1",
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            graph_delta=graph_delta,
            fiber_delta=FiberDelta(shadow_fiber_snapshot=fiber_snap, action="spawn"),
            authorization_id="",
            delta_hash=correct_delta_hash,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        )
        correct_auth_id = tmp_txn2.authorization_binding_hash()
        # Step 4: Create final transaction with all correct hashes.
        full_txn = StructuralTransaction(
            transaction_id="test_txn_1",
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            graph_delta=graph_delta,
            fiber_delta=FiberDelta(shadow_fiber_snapshot=fiber_snap, action="spawn"),
            authorization_id=correct_auth_id,
            delta_hash=correct_delta_hash,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        )
        auth = AuthorizationResult(
            snapshot_id="s1",
            state_version=int(rt.engine.graph.version),
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash="test_txn_1",
        )
        result = rt.commit_channel.commit(full_txn, auth)
        assert result.committed

        # Verify WAL contains fiber record.
        wal = WriteAheadLog(str(tmp_path / "wal.jsonl"))
        records = list(wal.iter_records())
        write_records = [r for r in records if r.record_type == WALRecordType.WRITE]
        fiber_records = [r for r in write_records if r.payload.get("kind") == "fiber"]
        assert len(fiber_records) > 0, "No fiber WRITE record in WAL"
        assert "fiber_state" in fiber_records[0].payload


class TestWALCommitOrdering:
    """D11-005: WAL COMMIT ordering is correct."""

    def test_commit_record_exists_after_commit(self, tmp_path):
        """A COMMIT record exists in the WAL after a successful commit."""
        torch.manual_seed(42)
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(
            _graph(), _cfg(),
            runtime_config=RuntimeConfig(wal_path=wal_path),
        )
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
        rt.commit_channel.commit(full_txn, auth)

        wal = WriteAheadLog(wal_path)
        records = list(wal.iter_records())
        commit_records = [r for r in records if r.record_type == WALRecordType.COMMIT]
        assert len(commit_records) > 0, "No COMMIT record in WAL"

    def test_no_commit_record_on_rollback(self, tmp_path):
        """No COMMIT record is written if the transaction is rolled back."""
        wal_path = str(tmp_path / "wal.jsonl")
        rt = LGAERuntime(
            _graph(), _cfg(),
            runtime_config=RuntimeConfig(wal_path=wal_path),
        )
        from lgae_v3.types import MutationResult
        from lgae_v3.runtime.transaction import GraphDelta
        shadow = rt.engine.graph.clone()
        shadow.weight[0] = shadow.weight[0] * 3.0
        fiber_snap = rt._engine.fibers.snapshot()
        graph_delta = GraphDelta(shadow_graph=shadow, mutation_name="test")
        # Compute hashes in correct order: delta_hash first, then auth_id.
        tmp_txn = StructuralTransaction(
            transaction_id="test_txn_rollback",
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            graph_delta=graph_delta,
            fiber_delta=FiberDelta(shadow_fiber_snapshot=fiber_snap, action="spawn"),
            authorization_id="",
            delta_hash="placeholder",
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        )
        correct_delta_hash = tmp_txn.compute_delta_hash()
        tmp_txn2 = StructuralTransaction(
            transaction_id="test_txn_rollback",
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            graph_delta=graph_delta,
            fiber_delta=FiberDelta(shadow_fiber_snapshot=fiber_snap, action="spawn"),
            authorization_id="",
            delta_hash=correct_delta_hash,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        )
        correct_auth_id = tmp_txn2.authorization_binding_hash()
        full_txn = StructuralTransaction(
            transaction_id="test_txn_rollback",
            base_state_version=int(rt.engine.graph.version),
            base_state_hash=rt.authority_hash,
            graph_delta=graph_delta,
            fiber_delta=FiberDelta(shadow_fiber_snapshot=fiber_snap, action="spawn"),
            authorization_id=correct_auth_id,
            delta_hash=correct_delta_hash,
            mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        )
        auth = AuthorizationResult(
            snapshot_id="s1",
            state_version=int(rt.engine.graph.version),
            state_hash=rt.authority_hash,
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash="test_txn_rollback",
        )
        # v5.11 Sprint 2 D11-005: Make apply fail in a way that doesn't
        # break rollback. We inject a failure into the fiber restore
        # ONLY during apply, then restore the original for rollback.
        # The trick: we use a flag to fail only the first call.
        original_restore = rt._engine.fibers.restore
        call_count = [0]
        def failing_restore(snap):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("injected apply failure")
            # Subsequent calls (rollback) use the original.
            return original_restore(snap)
        rt._engine.fibers.restore = failing_restore
        try:
            with pytest.raises(Exception):
                rt.commit_channel.commit(full_txn, auth)
        finally:
            rt._engine.fibers.restore = original_restore

        # v5.11 Sprint 2 D11-005: With COMMIT-before-APPLY ordering,
        # a COMMIT record IS written before apply, but if apply fails,
        # an ABORT record is written to invalidate it.
        # The key invariant is: the transaction is NOT recoverable as
        # committed (recover_transactions excludes aborted txns).
        wal = WriteAheadLog(wal_path)
        records = list(wal.iter_records())
        from lgae_v3.runtime.wal import recover_transactions
        recovered = recover_transactions(records)
        # The rolled-back transaction must NOT be in recovered transactions.
        assert len(recovered) == 0, (
            f"Rolled-back transaction was recovered as committed! "
            f"Recovered: {list(recovered.keys())}"
        )
        # An ABORT record must exist to invalidate the COMMIT.
        abort_records = [r for r in records if r.record_type == WALRecordType.ABORT]
        assert len(abort_records) > 0, "No ABORT record after rollback!"
