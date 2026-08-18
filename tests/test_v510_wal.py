"""v5.10 Phase 30: crash-safe transactions (WAL) tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    WALRecordType, WALRecord, WriteAheadLog, recover_transactions,
)


def test_wal_begin_write_commit(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    txn_id = wal.begin({"state_hash": "abc"})
    wal.write(txn_id, {"action": "add_edge", "u": 0, "v": 1})
    wal.write(txn_id, {"action": "add_edge", "u": 1, "v": 2})
    wal.commit(txn_id)
    records = list(wal.iter_records())
    assert len(records) == 4  # begin + 2 writes + commit


def test_wal_abort_discards_transaction(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    txn_id = wal.begin()
    wal.write(txn_id, {"action": "add_edge"})
    wal.abort(txn_id)
    records = list(wal.iter_records())
    recovered = recover_transactions(records)
    assert len(recovered) == 0  # aborted txn is not recovered


def test_wal_recovery_replays_committed_only(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    txn1 = wal.begin()
    wal.write(txn1, {"action": "add_edge", "u": 0, "v": 1})
    wal.commit(txn1)
    txn2 = wal.begin()
    wal.write(txn2, {"action": "remove_edge", "u": 0, "v": 1})
    # txn2 is NOT committed (simulates crash before commit)
    records = list(wal.iter_records())
    recovered = recover_transactions(records)
    assert txn1 in recovered
    assert txn2 not in recovered
    assert len(recovered[txn1]) == 1


def test_wal_record_serialize_deserialize():
    r = WALRecord(txn_id=0, record_type=WALRecordType.BEGIN, lsn=1, payload={"x": 1})
    s = r.serialize()
    r2 = WALRecord.deserialize(s)
    assert r2.txn_id == 0
    assert r2.record_type == WALRecordType.BEGIN
    assert r2.lsn == 1
    assert r2.payload == {"x": 1}


def test_wal_write_to_nonexistent_txn_raises(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    with pytest.raises(ValueError):
        wal.write(999, {"action": "x"})


def test_wal_commit_to_nonexistent_txn_raises(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    with pytest.raises(ValueError):
        wal.commit(999)


def test_wal_checkpoint(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    wal.checkpoint()
    records = list(wal.iter_records())
    assert len(records) == 1
    assert records[0].record_type == WALRecordType.CHECKPOINT


def test_wal_truncate(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    txn = wal.begin()
    wal.write(txn, {"action": "x"})
    wal.commit(txn)
    assert len(list(wal.iter_records())) > 0
    wal.truncate()
    assert len(list(wal.iter_records())) == 0


def test_wal_multiple_transactions(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    t1 = wal.begin()
    t2 = wal.begin()
    wal.write(t1, {"action": "a"})
    wal.write(t2, {"action": "b"})
    wal.commit(t1)
    wal.commit(t2)
    records = list(wal.iter_records())
    recovered = recover_transactions(records)
    assert len(recovered) == 2
    assert recovered[t1] == [{"action": "a"}]
    assert recovered[t2] == [{"action": "b"}]


def test_wal_lsn_monotonic(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    t = wal.begin()
    wal.write(t, {"action": "x"})
    wal.commit(t)
    records = list(wal.iter_records())
    lsns = [r.lsn for r in records]
    assert lsns == sorted(lsns)
    assert len(set(lsns)) == len(lsns)
