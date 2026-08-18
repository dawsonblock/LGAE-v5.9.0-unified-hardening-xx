"""Crash-safe transactions via write-ahead log (Phase 30).

A WAL ensures that committed transactions survive crashes. The protocol:

  1. BEGIN: write a BEGIN record with transaction ID and state hash
  2. WRITE: write each mutation as a WAL record
  3. COMMIT: write a COMMIT record and fsync
  4. APPLY: apply the mutations to the authoritative state
  5. CHECKPOINT: truncate the WAL after a successful checkpoint

On recovery, the WAL is replayed:
  - If a transaction has COMMIT: re-apply it
  - If a transaction has no COMMIT: discard it (rollback)

This is the standard ARIES-style WAL protocol, simplified for the runtime's
single-writer model.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class WALRecordType(str, Enum):
    BEGIN = "begin"
    TX_PREPARE = "tx_prepare"  # v5.11-RC Phase 7: complete transaction record
    WRITE = "write"
    COMMIT = "commit"
    COMMIT_INTENT = "commit_intent"
    APPLY = "apply"
    VERIFY = "verify"
    FINALIZE = "finalize"
    ABORT = "abort"
    CHECKPOINT = "checkpoint"


class TransactionState(str, Enum):
    """Formal transaction lifecycle states (Phase 3)."""
    NEW = "NEW"
    PREPARED = "PREPARED"
    COMMIT_INTENT = "COMMIT_INTENT"
    APPLIED = "APPLIED"
    VERIFIED = "VERIFIED"
    FINALIZED = "FINALIZED"
    ABORTED = "ABORTED"


class InvalidStateTransitionError(RuntimeError):
    """Raised when an illegal transaction state transition is attempted."""


class TransactionLifecycle:
    """Formal deterministic state machine for WAL transaction lifecycle."""

    ALLOWED_TRANSITIONS: dict[TransactionState, set[TransactionState]] = {
        TransactionState.NEW: {TransactionState.PREPARED, TransactionState.COMMIT_INTENT, TransactionState.ABORTED},
        TransactionState.PREPARED: {TransactionState.COMMIT_INTENT, TransactionState.ABORTED},
        TransactionState.COMMIT_INTENT: {TransactionState.APPLIED, TransactionState.FINALIZED, TransactionState.ABORTED},
        TransactionState.APPLIED: {TransactionState.VERIFIED, TransactionState.FINALIZED, TransactionState.ABORTED},
        TransactionState.VERIFIED: {TransactionState.FINALIZED, TransactionState.ABORTED},
        TransactionState.FINALIZED: set(),
        TransactionState.ABORTED: set(),
    }

    def __init__(self, txn_id: int, initial_state: TransactionState = TransactionState.NEW) -> None:
        self.txn_id = txn_id
        self.state = initial_state
        self.history: list[tuple[TransactionState, float]] = [(initial_state, time.time())]

    def can_transition_to(self, target: TransactionState) -> bool:
        return target in self.ALLOWED_TRANSITIONS.get(self.state, set())

    def transition_to(self, target: TransactionState) -> None:
        if not self.can_transition_to(target):
            raise InvalidStateTransitionError(
                f"illegal transaction lifecycle transition: {self.state.value} -> {target.value} "
                f"for transaction {self.txn_id}"
            )
        self.state = target
        self.history.append((target, time.time()))

    @classmethod
    def infer_from_records(cls, txn_id: int, records: list[WALRecord]) -> "TransactionLifecycle":
        """Infer lifecycle state from durable WAL records."""
        txn_records = [r for r in records if r.txn_id == txn_id]
        if not txn_records:
            return cls(txn_id, TransactionState.NEW)

        if any(r.record_type == WALRecordType.ABORT for r in txn_records):
            lifecycle = cls(txn_id, TransactionState.NEW)
            lifecycle.state = TransactionState.ABORTED
            return lifecycle

        has_finalize = any(r.record_type == WALRecordType.FINALIZE for r in txn_records)
        has_verify = any(r.record_type == WALRecordType.VERIFY for r in txn_records)
        has_apply = any(r.record_type == WALRecordType.APPLY for r in txn_records)
        has_commit = any(r.record_type in (WALRecordType.COMMIT, WALRecordType.COMMIT_INTENT) for r in txn_records)
        has_prepare = any(r.record_type == WALRecordType.TX_PREPARE for r in txn_records)

        lifecycle = cls(txn_id, TransactionState.NEW)
        if has_finalize:
            lifecycle.state = TransactionState.FINALIZED
        elif has_verify:
            lifecycle.state = TransactionState.VERIFIED
        elif has_apply:
            lifecycle.state = TransactionState.APPLIED
        elif has_commit:
            lifecycle.state = TransactionState.COMMIT_INTENT
        elif has_prepare:
            lifecycle.state = TransactionState.PREPARED
        return lifecycle


@dataclass(frozen=True, slots=True)
class WALRecord:
    """One record in the write-ahead log.

    v5.11-RC Phase 8: Records are hash-chained for tamper detection.
    Each record includes:
    - previous_record_hash: hash of the previous record (or "" for the first)
    - record_hash: SHA256(previous_record_hash || canonical(record))
    """
    txn_id: int
    record_type: WALRecordType
    lsn: int  # log sequence number
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    previous_record_hash: str = ""
    record_hash: str = ""

    def _canonical_content(self) -> str:
        """Canonical JSON of the record content (excluding hash fields)."""
        return json.dumps({
            "txn_id": int(self.txn_id),
            "record_type": self.record_type.value,
            "lsn": int(self.lsn),
            "payload": self.payload,
            "timestamp": float(self.timestamp),
        }, sort_keys=True, separators=(",", ":"))

    def compute_hash(self, prev_hash: str) -> str:
        """Compute the record hash given the previous record's hash."""
        h = hashlib.sha256()
        h.update(prev_hash.encode())
        h.update(self._canonical_content().encode())
        return h.hexdigest()

    def serialize(self) -> str:
        return json.dumps({
            "txn_id": int(self.txn_id),
            "record_type": self.record_type.value,
            "lsn": int(self.lsn),
            "payload": self.payload,
            "timestamp": float(self.timestamp),
            "previous_record_hash": self.previous_record_hash,
            "record_hash": self.record_hash,
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def deserialize(cls, line: str) -> "WALRecord":
        data = json.loads(line)
        return cls(
            txn_id=int(data["txn_id"]),
            record_type=WALRecordType(data["record_type"]),
            lsn=int(data["lsn"]),
            payload=data["payload"],
            timestamp=float(data["timestamp"]),
            previous_record_hash=data.get("previous_record_hash", ""),
            record_hash=data.get("record_hash", ""),
        )

    def to_log(self) -> dict[str, Any]:
        return {
            "txn_id": int(self.txn_id),
            "record_type": self.record_type.value,
            "lsn": int(self.lsn),
            "payload": self.payload,
            "timestamp": float(self.timestamp),
            "previous_record_hash": self.previous_record_hash,
            "record_hash": self.record_hash,
        }


@dataclass(slots=True)
class WALTransaction:
    """An in-progress transaction with formal lifecycle state machine."""
    txn_id: int
    records: list[WALRecord] = field(default_factory=list)
    committed: bool = False
    aborted: bool = False
    lifecycle: TransactionLifecycle = field(default_factory=lambda: TransactionLifecycle(0))

    def __post_init__(self) -> None:
        if self.lifecycle.txn_id == 0:
            self.lifecycle = TransactionLifecycle(self.txn_id)


class WriteAheadLog:
    """A crash-safe write-ahead log.

    v5.11 Sprint 2: Counters (LSN, txn_id) are restored from existing
    records on reopen. This ensures monotonicity across restarts.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lsn = 0
        self._next_txn_id = 0
        self._active_txns: dict[int, WALTransaction] = {}
        # v5.11-RC Phase 8: Track the last record hash for chaining.
        self._last_record_hash: str = ""
        # v5.11 D11-006: Restore counters from existing records.
        self._restore_counters()

    def _restore_counters(self) -> None:
        """Restore LSN, next_txn_id, and last_record_hash from existing WAL records.

        D11-006 fix: Without this, reopening a WAL resets counters to 0,
        which can cause txn_id collisions and LSN non-monotonicity.
        v5.11-RC Phase 8: Also restores the hash chain.
        """
        if not self.path.exists():
            return
        max_lsn = 0
        max_txn_id = -1
        last_hash = ""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = WALRecord.deserialize(line)
                        max_lsn = max(max_lsn, int(record.lsn))
                        max_txn_id = max(max_txn_id, int(record.txn_id))
                        if record.record_hash:
                            last_hash = record.record_hash
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except OSError:
            return
        self._lsn = max_lsn
        self._next_txn_id = max_txn_id + 1
        self._last_record_hash = last_hash

    def _append(self, record: WALRecord) -> WALRecord:
        # v5.11-RC Phase 8: Compute hash chain.
        prev_hash = self._last_record_hash
        record_hash = record.compute_hash(prev_hash)
        # Create a new record with the hash fields populated.
        chained = WALRecord(
            txn_id=record.txn_id,
            record_type=record.record_type,
            lsn=record.lsn,
            payload=record.payload,
            timestamp=record.timestamp,
            previous_record_hash=prev_hash,
            record_hash=record_hash,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(chained.serialize() + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._last_record_hash = record_hash
        return chained

    def begin(self, metadata: dict[str, Any] | None = None) -> int:
        """Begin a new transaction. Returns the transaction ID."""
        txn_id = self._next_txn_id
        self._next_txn_id += 1
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.BEGIN, lsn=self._lsn,
            payload=dict(metadata or {}), timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id] = WALTransaction(
            txn_id=txn_id,
            records=[record],
            lifecycle=TransactionLifecycle(txn_id, TransactionState.NEW),
        )
        return txn_id

    def prepare(self, txn_id: int, transaction_data: dict[str, Any]) -> WALRecord:
        """Write a TX_PREPARE record with the complete transaction."""
        if txn_id not in self._active_txns:
            raise ValueError(f"txn {txn_id} is not active")
        self._active_txns[txn_id].lifecycle.transition_to(TransactionState.PREPARED)
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.TX_PREPARE, lsn=self._lsn,
            payload=transaction_data, timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id].records.append(record)
        return record

    def write(self, txn_id: int, mutation: dict[str, Any]) -> WALRecord:
        """Write a mutation within a transaction."""
        if txn_id not in self._active_txns:
            raise ValueError(f"txn {txn_id} is not active")
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.WRITE, lsn=self._lsn,
            payload=mutation, timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id].records.append(record)
        return record

    def commit(self, txn_id: int) -> WALRecord:
        """Commit intent for a transaction (COMMIT-before-APPLY)."""
        if txn_id not in self._active_txns:
            raise ValueError(f"txn {txn_id} is not active")
        self._active_txns[txn_id].lifecycle.transition_to(TransactionState.COMMIT_INTENT)
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.COMMIT, lsn=self._lsn,
            payload={}, timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id].committed = True
        self._active_txns[txn_id].records.append(record)
        return record

    def apply(self, txn_id: int, payload: dict[str, Any] | None = None) -> WALRecord:
        """Record applied state mutation transition."""
        if txn_id not in self._active_txns:
            raise ValueError(f"txn {txn_id} is not active")
        self._active_txns[txn_id].lifecycle.transition_to(TransactionState.APPLIED)
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.APPLY, lsn=self._lsn,
            payload=dict(payload or {}), timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id].records.append(record)
        return record

    def verify(self, txn_id: int, payload: dict[str, Any] | None = None) -> WALRecord:
        """Record post-commit state verification."""
        if txn_id not in self._active_txns:
            raise ValueError(f"txn {txn_id} is not active")
        self._active_txns[txn_id].lifecycle.transition_to(TransactionState.VERIFIED)
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.VERIFY, lsn=self._lsn,
            payload=dict(payload or {}), timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id].records.append(record)
        return record

    def finalize(self, txn_id: int, payload: dict[str, Any] | None = None) -> WALRecord:
        """Finalize transaction and receipt."""
        if txn_id not in self._active_txns:
            raise ValueError(f"txn {txn_id} is not active")
        self._active_txns[txn_id].lifecycle.transition_to(TransactionState.FINALIZED)
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.FINALIZE, lsn=self._lsn,
            payload=dict(payload or {}), timestamp=time.time(),
        )
        self._append(record)
        self._active_txns[txn_id].records.append(record)
        del self._active_txns[txn_id]
        return record

    def abort(self, txn_id: int) -> WALRecord:
        """Abort a transaction (rollback)."""
        self._lsn += 1
        record = WALRecord(
            txn_id=txn_id, record_type=WALRecordType.ABORT, lsn=self._lsn,
            payload={}, timestamp=time.time(),
        )
        self._append(record)
        if txn_id in self._active_txns:
            try:
                self._active_txns[txn_id].lifecycle.transition_to(TransactionState.ABORTED)
            except Exception:
                pass
            self._active_txns[txn_id].aborted = True
            self._active_txns[txn_id].records.append(record)
            del self._active_txns[txn_id]
        return record

    def get_lifecycle(self, txn_id: int) -> TransactionLifecycle:
        """Get lifecycle state machine for an active or recorded transaction."""
        if txn_id in self._active_txns:
            return self._active_txns[txn_id].lifecycle
        records = list(self.iter_records())
        return TransactionLifecycle.infer_from_records(txn_id, records)

    def checkpoint(self) -> WALRecord:
        """Write a checkpoint record and truncate the log."""
        self._lsn += 1
        record = WALRecord(
            txn_id=-1, record_type=WALRecordType.CHECKPOINT, lsn=self._lsn,
            payload={"active_txns": list(self._active_txns.keys())},
            timestamp=time.time(),
        )
        self._append(record)
        return record

    def truncate(self) -> None:
        """Truncate the WAL (after a successful checkpoint)."""
        self.path.write_text("")

    def iter_records(self) -> Iterator[WALRecord]:
        """Iterate over all records in the log."""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield WALRecord.deserialize(line)

    def verify_chain(self) -> bool:
        """Verify the hash chain of all records.

        v5.11-RC Phase 8: Checks that:
        - Each record's previous_record_hash matches the previous record's hash
        - Each record's record_hash matches the recomputed hash
        - LSN is monotonic

        Returns True if the chain is valid, False otherwise.
        """
        prev_hash = ""
        prev_lsn = 0
        for record in self.iter_records():
            # Check LSN monotonicity.
            if record.lsn <= prev_lsn:
                return False
            prev_lsn = record.lsn
            # Check previous_record_hash.
            if record.previous_record_hash != prev_hash:
                return False
            # Check record_hash.
            expected_hash = record.compute_hash(prev_hash)
            if record.record_hash != expected_hash:
                return False
            prev_hash = record.record_hash
        return True


def recover_transactions(records: list[WALRecord]) -> dict[int, list[dict[str, Any]]]:
    """Recover committed transactions from WAL records.

    Returns a dict of {txn_id: [mutations]} for committed transactions only.
    Transactions without a COMMIT record are discarded (rollback).

    This is the core crash-recovery invariant:
        S_restart ∈ { S_n, S_{n+1} }
    Never S_n + partial(Δ).
    """
    txns: dict[int, list[dict[str, Any]]] = {}
    committed: set[int] = set()
    aborted: set[int] = set()
    commit_types = {
        WALRecordType.COMMIT,
        WALRecordType.COMMIT_INTENT,
        WALRecordType.APPLY,
        WALRecordType.VERIFY,
        WALRecordType.FINALIZE,
    }
    for record in records:
        if record.record_type == WALRecordType.BEGIN:
            txns[record.txn_id] = []
        elif record.record_type == WALRecordType.TX_PREPARE:
            # v5.11-RC Phase 7: TX_PREPARE contains complete transaction info.
            if record.txn_id in txns:
                txns[record.txn_id].append(record.payload)
        elif record.record_type == WALRecordType.WRITE:
            if record.txn_id in txns:
                txns[record.txn_id].append(record.payload)
        elif record.record_type in commit_types:
            committed.add(record.txn_id)
        elif record.record_type == WALRecordType.ABORT:
            aborted.add(record.txn_id)
    # Return only committed, non-aborted transactions.
    return {
        txn_id: mutations
        for txn_id, mutations in txns.items()
        if txn_id in committed and txn_id not in aborted
    }


def recover_transaction_metadata(records: list[WALRecord]) -> dict[int, dict[str, Any]]:
    """Recover TX_PREPARE metadata for committed transactions."""
    committed: set[int] = set()
    aborted: set[int] = set()
    prepare_data: dict[int, dict[str, Any]] = {}
    commit_types = {
        WALRecordType.COMMIT,
        WALRecordType.COMMIT_INTENT,
        WALRecordType.APPLY,
        WALRecordType.VERIFY,
        WALRecordType.FINALIZE,
    }
    for record in records:
        if record.record_type == WALRecordType.TX_PREPARE:
            prepare_data[record.txn_id] = record.payload
        elif record.record_type in commit_types:
            committed.add(record.txn_id)
        elif record.record_type == WALRecordType.ABORT:
            aborted.add(record.txn_id)
    return {
        txn_id: metadata
        for txn_id, metadata in prepare_data.items()
        if txn_id in committed and txn_id not in aborted
    }


def replay_committed_transactions(
    wal_path: str | Path,
    engine: Any,
    *,
    checkpoint_lsn: int = 0,
) -> list[dict[str, Any]]:
    """Replay committed WAL transactions onto an engine.

    v5.11-RC Phase 9: State-aware, idempotent replay.

    For each committed transaction, the replay checks the current engine
    state against the transaction's base_state_hash:
    - If current state == base_state_hash → apply the transaction
    - If current state != base_state_hash → skip (already applied or
      unknown state). This makes replay idempotent.

    The checkpoint_lsn parameter allows skipping transactions that were
    already applied before a checkpoint.

    This is the crash-recovery procedure. It:
    1. Reads all WAL records.
    2. Identifies committed transactions (have COMMIT, no ABORT).
    3. For each transaction, checks if it needs to be applied.
    4. Re-applies only transactions that need to be applied.
    5. Returns a list of replay results.

    The central recovery invariant:
        S_restart ∈ { S_n, S_{n+1} }
    Never S_n + partial(Δ).
    """
    from ..types import GraphBuffers
    import torch

    wal = WriteAheadLog(wal_path)
    records = list(wal.iter_records())
    committed = recover_transactions(records)

    # Build a map of txn_id -> base_state_hash from BEGIN records.
    txn_base_hashes: dict[int, str] = {}
    for record in records:
        if record.record_type == WALRecordType.BEGIN:
            txn_base_hashes[record.txn_id] = record.payload.get("base_state_hash", "")

    results: list[dict[str, Any]] = []
    for txn_id, mutations in committed.items():
        # Skip transactions before the checkpoint LSN.
        if checkpoint_lsn > 0:
            txn_lsn = next(
                (r.lsn for r in records
                 if r.txn_id == txn_id and r.record_type == WALRecordType.BEGIN),
                0,
            )
            if txn_lsn <= checkpoint_lsn:
                results.append({
                    "txn_id": txn_id,
                    "kind": "checkpoint_skip",
                    "applied": False,
                    "reason": f"before checkpoint LSN {checkpoint_lsn}",
                })
                continue

        # State-aware replay: check if this transaction needs to be applied.
        base_hash = txn_base_hashes.get(txn_id, "")
        current_hash = engine.authority_hash() if hasattr(engine, "authority_hash") else ""

        if base_hash and current_hash and current_hash != base_hash:
            # Current state doesn't match the base state — this transaction
            # was likely already applied. Skip it (idempotent replay).
            results.append({
                "txn_id": txn_id,
                "kind": "state_aware_skip",
                "applied": False,
                "reason": f"current hash {current_hash[:16]} != base hash {base_hash[:16]}",
                "current_hash": current_hash,
                "base_hash": base_hash,
            })
            continue

        # Apply the transaction using the shared apply path.
        for mutation in mutations:
            # v5.11-RC Phase 7: Skip TX_PREPARE metadata records — they
            # contain transaction metadata, not state mutations.
            if "transaction_id" in mutation and "kind" not in mutation:
                continue
            apply_wal_mutation(engine, mutation)
            results.append({
                "txn_id": txn_id,
                "kind": mutation.get("kind", "unknown"),
                "applied": True,
                "new_hash": engine.authority_hash() if hasattr(engine, "authority_hash") else "",
            })
    return results


def apply_wal_mutation(engine: Any, mutation: dict[str, Any]) -> None:
    """Apply a single WAL mutation to an engine.

    v5.11-RC Phase 10: This is the shared apply path used by both:
    - Normal commit (via CommitChannel._apply)
    - Recovery replay (via replay_committed_transactions)

    Both paths must produce identical state for the same mutation.
    """
    from ..types import GraphBuffers
    import torch

    kind = mutation.get("kind")
    if kind == "graph":
        state = mutation.get("shadow_graph_state")
        if state is not None:
            shadow = GraphBuffers.from_state_dict(state)
            engine.graph = shadow
            engine.graph.bump_version()
            if hasattr(engine, "_invalidate_neighbor_indices"):
                engine._invalidate_neighbor_indices("wal_apply")
    elif kind == "fiber":
        fiber_state = mutation.get("fiber_state", {})
        if fiber_state and hasattr(engine, "fibers"):
            fibers = engine.fibers
            if hasattr(fibers, "latent"):
                for attr in ("latent", "gate_logits", "active_mask", "age",
                            "utility_ema", "spawn_counter", "gamma_ema"):
                    if attr in fiber_state:
                        tensor = getattr(fibers, attr, None)
                        if tensor is not None:
                            restored = torch.tensor(
                                fiber_state[attr],
                                dtype=tensor.dtype,
                                device=tensor.device,
                            )
                            if hasattr(tensor, 'data'):
                                tensor.data.copy_(restored)
                            else:
                                tensor.copy_(restored)
    elif kind == "gauge":
        gauge_raw = mutation.get("gauge_raw")
        if gauge_raw is not None and hasattr(engine, "gauge_connections") \
           and engine.gauge_connections is not None:
            raw = torch.tensor(
                gauge_raw,
                dtype=engine.gauge_connections.raw_generators.dtype,
                device=engine.gauge_connections.raw_generators.device,
            )
            engine.gauge_connections.raw_generators.data.copy_(raw)
