"""Immutable evidence ledger for LGAE structural reasoning.

The ledger is the non-learned ground-truth substrate beneath the mutable
experience-memory graph.  Entries are append-only, canonically serialized and
hash chained.  Derived memory may be rebuilt from this file at any time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .version import VERSION

EVIDENCE_SCHEMA = "LGAE_EVIDENCE_V1"


def _safe(value: Any) -> Any:
    if is_dataclass(value):
        return _safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


@dataclass(slots=True, frozen=True)
class EvidenceRecord:
    """One immutable observation about a reasoning/intervention episode."""

    record_type: str
    graph_hash: str
    payload: dict[str, Any]
    authority_hash: str | None = None
    reasoning_run_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EvidenceLedger:
    """Append-only JSONL evidence ledger with deterministic chain verification."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last(self) -> tuple[int, str | None]:
        if not self.path.exists():
            return -1, None
        idx = -1
        digest: str | None = None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                idx = int(item["index"])
                digest = str(item["sha256"])
        return idx, digest

    def append(self, record: EvidenceRecord) -> dict[str, Any]:
        idx, previous = self._last()
        envelope = {
            "schema": EVIDENCE_SCHEMA,
            "build_version": VERSION,
            "index": idx + 1,
            "previous_hash": previous,
            "record": _safe(record),
        }
        envelope["sha256"] = hashlib.sha256(_canonical(envelope)).hexdigest()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(envelope, sort_keys=True, separators=(",", ":"), default=str) + "\n")
        return envelope

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
        return out

    def verify(self) -> tuple[bool, list[str]]:
        errors: list[str] = []
        expected_previous: str | None = None
        expected_index = 0
        for line_no, item in enumerate(self.records(), start=1):
            if item.get("schema") != EVIDENCE_SCHEMA:
                errors.append(f"line {line_no}: schema mismatch")
            if int(item.get("index", -1)) != expected_index:
                errors.append(f"line {line_no}: index mismatch")
            if item.get("previous_hash") != expected_previous:
                errors.append(f"line {line_no}: previous_hash mismatch")
            stored = item.get("sha256")
            unhashed = {k: v for k, v in item.items() if k != "sha256"}
            computed = hashlib.sha256(_canonical(unhashed)).hexdigest()
            if stored != computed:
                errors.append(f"line {line_no}: sha256 mismatch")
            expected_previous = stored
            expected_index += 1
        return not errors, errors

    @property
    def root_hash(self) -> str | None:
        _, digest = self._last()
        return digest
