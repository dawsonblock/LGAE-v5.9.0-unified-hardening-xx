"""Common types for canonical runtime phase contracts.

Every phase output is:
- immutable (frozen dataclass)
- state-bound (carries source_state_version and source_state_hash)
- deterministically serializable (canonical_json)
- traceable (snapshot_id links to the observation that started the cycle)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def canonical_json(obj: Any) -> str:
    """Canonical JSON serialization for deterministic hashing.

    Keys are sorted, floats are repr'd consistently, no whitespace.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=_default_serializer)


def _default_serializer(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value") and isinstance(obj.value, str):
        return obj.value
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


def canonical_hash(obj: Any) -> str:
    """SHA-256 hash of the canonical JSON representation."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """Base contract for all phase outputs.

    Every phase result binds to the snapshot it was computed from.
    This enables stale-write detection during authorization and commit.
    """
    snapshot_id: str
    state_version: int
    state_hash: str

    @property
    def base_identity(self) -> Any:
        from ..state_identity import AuthorityStateIdentity
        h = getattr(self, "authority_hash", "") or getattr(self, "authority_hash_before", "") or self.state_hash
        return AuthorityStateIdentity(
            version=self.state_version,
            authority_hash=h,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_hash(self) -> str:
        return canonical_hash(self.to_dict())
