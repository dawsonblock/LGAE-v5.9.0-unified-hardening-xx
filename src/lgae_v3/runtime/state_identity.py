"""Authority state identity (Phase 2): unified immutable version + authority hash token.

Eliminates loose string-based state hash and integer version pairs by providing
a first-class immutable identity object that binds Observation, Transaction,
Authorization, CommitResult, StateBundle, WAL, and Receipts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AuthorityStateIdentity:
    """Canonical immutable state identity token.

    Fields:
        version: monotonic state version counter
        authority_hash: canonical SHA-256 state commitment over full authority
    """
    version: int
    authority_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "authority_hash": str(self.authority_hash),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuthorityStateIdentity:
        v = d.get("version", d.get("state_version", d.get("base_state_version", 0)))
        h = d.get("authority_hash", d.get("state_hash", d.get("base_state_hash", "")))
        return cls(version=int(v), authority_hash=str(h))

    @classmethod
    def from_engine(cls, engine: Any) -> AuthorityStateIdentity:
        v = int(engine.graph.version) if hasattr(engine, "graph") and hasattr(engine.graph, "version") else 0
        h = engine.authority_hash() if hasattr(engine, "authority_hash") else ""
        return cls(version=v, authority_hash=str(h))

    def __str__(self) -> str:
        return f"AuthorityStateIdentity(version={self.version}, hash={self.authority_hash[:12]}...)"

    def matches(self, other: Any) -> bool:
        if isinstance(other, AuthorityStateIdentity):
            return self.version == other.version and self.authority_hash == other.authority_hash
        return False
