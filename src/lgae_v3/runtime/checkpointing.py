"""Signed Merkle checkpointing (Phase 31).

A checkpoint binds the full runtime state (graph, authority hash, generation,
receipt chain) to a Merkle root and signs it with Ed25519. This enables:

  - crash recovery: restore from the last signed checkpoint
  - audit: verify that a checkpoint was produced by a known authority
  - chain verification: each checkpoint links to the previous one

A checkpoint is immutable once created. The checkpoint chain is a sequence
of signed Merkle roots, each linking to the previous checkpoint's hash.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .merkle_evidence import MerkleTree


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A signed Merkle checkpoint of the runtime state."""
    checkpoint_index: int
    authority_hash: str
    generation: int
    merkle_root: str
    previous_checkpoint_hash: str | None
    created_at: float
    signature: str | None = None  # Ed25519 signature over (merkle_root || previous_hash)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_hash(self) -> str:
        """Hash of the checkpoint itself (for chaining)."""
        payload = json.dumps({
            "checkpoint_index": int(self.checkpoint_index),
            "authority_hash": self.authority_hash,
            "generation": int(self.generation),
            "merkle_root": self.merkle_root,
            "previous_checkpoint_hash": self.previous_checkpoint_hash,
            "created_at": float(self.created_at),
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_log(self) -> dict[str, Any]:
        return {
            "checkpoint_index": int(self.checkpoint_index),
            "authority_hash": self.authority_hash,
            "generation": int(self.generation),
            "merkle_root": self.merkle_root,
            "previous_checkpoint_hash": self.previous_checkpoint_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "created_at": float(self.created_at),
            "signed": self.signature is not None,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CheckpointChain:
    """An append-only chain of signed checkpoints."""
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def append(
        self,
        *,
        authority_hash: str,
        generation: int,
        evidence_items: list[bytes | str],
        signing_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create and append a new checkpoint to the chain."""
        tree = MerkleTree.build(evidence_items)
        prev_hash = self.checkpoints[-1].checkpoint_hash if self.checkpoints else None
        cp = Checkpoint(
            checkpoint_index=len(self.checkpoints),
            authority_hash=authority_hash,
            generation=int(generation),
            merkle_root=tree.root,
            previous_checkpoint_hash=prev_hash,
            created_at=time.time(),
            signature=None,
            metadata=dict(metadata or {}),
        )
        if signing_key is not None:
            from ..receipts import sign_receipt
            message = (cp.merkle_root + str(cp.previous_checkpoint_hash)).encode()
            cp = Checkpoint(
                checkpoint_index=cp.checkpoint_index,
                authority_hash=cp.authority_hash,
                generation=cp.generation,
                merkle_root=cp.merkle_root,
                previous_checkpoint_hash=cp.previous_checkpoint_hash,
                created_at=cp.created_at,
                signature=sign_receipt(signing_key, message.hex()),
                metadata=cp.metadata,
            )
        self.checkpoints.append(cp)
        return cp

    @property
    def latest(self) -> Checkpoint | None:
        return self.checkpoints[-1] if self.checkpoints else None

    @property
    def root_hash(self) -> str | None:
        return self.latest.checkpoint_hash if self.latest else None

    def verify_chain(self) -> bool:
        """Verify that the checkpoint chain is internally consistent."""
        for i, cp in enumerate(self.checkpoints):
            if cp.checkpoint_index != i:
                return False
            if i == 0:
                if cp.previous_checkpoint_hash is not None:
                    return False
            else:
                if cp.previous_checkpoint_hash != self.checkpoints[i - 1].checkpoint_hash:
                    return False
        return True

    def to_log(self) -> dict[str, Any]:
        return {
            "checkpoint_count": len(self.checkpoints),
            "root_hash": self.root_hash,
            "checkpoints": [cp.to_log() for cp in self.checkpoints],
        }
