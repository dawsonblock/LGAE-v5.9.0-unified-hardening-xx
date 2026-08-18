"""Phase 3 tests: Formal WAL transaction lifecycle state machine."""
from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers, MutationDecision
from lgae_v3.types import MutationResult
from lgae_v3.runtime import (
    LGAERuntime,
    WriteAheadLog,
    WALRecordType,
    make_graph_transaction,
)
from lgae_v3.runtime.wal import (
    TransactionState,
    TransactionLifecycle,
    InvalidStateTransitionError,
    recover_transactions,
    replay_committed_transactions,
)
from lgae_v3.runtime.contracts import (
    AuthorizationResult,
    AuthorizationStatus,
)


def _cfg():
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
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
    return make_graph_buffers(
        6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12,
    )


def test_transaction_lifecycle_happy_path():
    lc = TransactionLifecycle(1)
    assert lc.state == TransactionState.NEW
    
    lc.transition_to(TransactionState.PREPARED)
    assert lc.state == TransactionState.PREPARED
    
    lc.transition_to(TransactionState.COMMIT_INTENT)
    assert lc.state == TransactionState.COMMIT_INTENT
    
    lc.transition_to(TransactionState.APPLIED)
    assert lc.state == TransactionState.APPLIED
    
    lc.transition_to(TransactionState.VERIFIED)
    assert lc.state == TransactionState.VERIFIED
    
    lc.transition_to(TransactionState.FINALIZED)
    assert lc.state == TransactionState.FINALIZED
    
    # Finalized cannot transition further
    with pytest.raises(InvalidStateTransitionError):
        lc.transition_to(TransactionState.ABORTED)


def test_transaction_lifecycle_abort_from_any_active_state():
    for initial in [TransactionState.NEW, TransactionState.PREPARED, TransactionState.COMMIT_INTENT, TransactionState.APPLIED, TransactionState.VERIFIED]:
        lc = TransactionLifecycle(1, initial)
        lc.transition_to(TransactionState.ABORTED)
        assert lc.state == TransactionState.ABORTED
        with pytest.raises(InvalidStateTransitionError):
            lc.transition_to(TransactionState.FINALIZED)


def test_transaction_lifecycle_illegal_jumps():
    lc = TransactionLifecycle(1)
    # Cannot jump straight from NEW to FINALIZED or APPLIED
    with pytest.raises(InvalidStateTransitionError):
        lc.transition_to(TransactionState.FINALIZED)
    with pytest.raises(InvalidStateTransitionError):
        lc.transition_to(TransactionState.APPLIED)
    with pytest.raises(InvalidStateTransitionError):
        lc.transition_to(TransactionState.VERIFIED)


def test_wal_lifecycle_full_execution_records(tmp_path):
    wal_file = tmp_path / "test.wal"
    wal = WriteAheadLog(wal_file)
    
    txn_id = wal.begin({"base_state_hash": "hash0", "base_state_version": 0})
    assert wal.get_lifecycle(txn_id).state == TransactionState.NEW
    
    wal.prepare(txn_id, {"transaction_id": "tx1", "delta_hash": "d1"})
    assert wal.get_lifecycle(txn_id).state == TransactionState.PREPARED
    
    wal.write(txn_id, {"kind": "graph", "shadow_graph_hash": "h1"})
    wal.commit(txn_id)
    assert wal.get_lifecycle(txn_id).state == TransactionState.COMMIT_INTENT
    
    wal.apply(txn_id, {"after_hash": "h1"})
    assert wal.get_lifecycle(txn_id).state == TransactionState.APPLIED
    
    wal.verify(txn_id, {"verified": True})
    assert wal.get_lifecycle(txn_id).state == TransactionState.VERIFIED
    
    wal.finalize(txn_id, {"finalized": True})
    
    # After reopening wal, infer from durable records
    reopened = WriteAheadLog(wal_file)
    records = list(reopened.iter_records())
    inferred = TransactionLifecycle.infer_from_records(txn_id, records)
    assert inferred.state == TransactionState.FINALIZED


def test_wal_recovery_interprets_states(tmp_path):
    wal_file = tmp_path / "test.wal"
    wal = WriteAheadLog(wal_file)
    
    # 1. Unprepared txn
    t1 = wal.begin({"name": "t1"})
    
    # 2. Prepared only
    t2 = wal.begin({"name": "t2"})
    wal.prepare(t2, {"transaction_id": "tx2"})
    
    # 3. Committed (COMMIT_INTENT)
    t3 = wal.begin({"name": "t3"})
    wal.prepare(t3, {"transaction_id": "tx3"})
    wal.write(t3, {"kind": "graph", "mutation_name": "add"})
    wal.commit(t3)
    
    # 4. Aborted
    t4 = wal.begin({"name": "t4"})
    wal.prepare(t4, {"transaction_id": "tx4"})
    wal.write(t4, {"kind": "graph", "mutation_name": "add"})
    wal.commit(t4)
    wal.abort(t4)
    
    records = list(wal.iter_records())
    committed = recover_transactions(records)
    
    # Only t3 is recovered for replay; t1, t2, t4 are excluded
    assert set(committed.keys()) == {t3}


def test_wal_abort_preserves_pre_state_invariant(tmp_path):
    from lgae_v3.runtime import RuntimeConfig
    wal_file = tmp_path / "test.wal"
    rcfg = RuntimeConfig(wal_path=str(wal_file))
    rt = LGAERuntime(_graph(), _cfg(), runtime_config=rcfg)
    wal = rt._wal
    
    before_hash = rt.authority_hash
    before_version = rt.state_identity.version
    
    # Set failpoint after state swap to trigger exception rollback
    rt.commit_channel.set_failpoint("after_state_swap")
    
    shadow = rt.engine.graph.clone()
    shadow.weight[0] = shadow.weight[0] * 3.0
    tx = make_graph_transaction(
        base_state_version=before_version,
        base_state_hash=before_hash,
        shadow_graph=shadow,
        mutation_result=MutationResult(MutationDecision.ACCEPT, []),
        step=0,
    ).with_authorization()
    auth = AuthorizationResult(
        snapshot_id="s",
        state_version=before_version,
        state_hash=before_hash,
        status=AuthorizationStatus.AUTHORIZED,
        transaction_hash=tx.transaction_id,
    )
    with pytest.raises(RuntimeError, match="failpoint"):
        rt.commit_channel.commit(tx, auth)
        
    # Live authority MUST be pre-state
    assert rt.authority_hash == before_hash
    assert rt.state_identity.version == before_version
    
    # WAL must have logged ABORT
    records = list(wal.iter_records())
    assert any(r.record_type == WALRecordType.ABORT for r in records)
    committed = recover_transactions(records)
    assert len(committed) == 0
