"""Cryptographic evidence v2: Merkle tree aggregation (Phase 28).

The existing receipts.py provides hash-chained Ed25519-signed receipts for
individual mutations. Phase 28 adds a Merkle tree for aggregating batches of
evidence into a single root hash, enabling efficient batch verification and
inclusion proofs.

A Merkle tree over N evidence items produces:
  - root: a single hash representing the entire batch
  - proofs: for each item, a list of sibling hashes needed to verify inclusion

This is used for checkpointing (Phase 31) and batch evidence verification.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _hash_leaf(data: bytes | str) -> str:
    """Hash a leaf node (with 0x00 prefix to distinguish from internal)."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(b"\x00" + data).hexdigest()


def _hash_internal(left: str, right: str) -> str:
    """Hash an internal node (with 0x01 prefix)."""
    return hashlib.sha256(b"\x01" + left.encode() + right.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class MerkleProof:
    """Inclusion proof for one leaf."""
    leaf_index: int
    leaf_hash: str
    siblings: list[tuple[str, str]] = field(default_factory=list)  # (hash, direction: "L"|"R")

    def to_log(self) -> dict[str, Any]:
        return {
            "leaf_index": int(self.leaf_index),
            "leaf_hash": self.leaf_hash,
            "siblings": [{"hash": h, "direction": d} for h, d in self.siblings],
        }


@dataclass(slots=True)
class MerkleTree:
    """A Merkle tree over a batch of evidence items."""
    leaves: list[str]  # hashes of leaf data
    levels: list[list[str]]  # levels[0] = leaves, levels[-1] = [root]
    root: str = ""

    @classmethod
    def build(cls, items: list[bytes | str]) -> "MerkleTree":
        """Build a Merkle tree from a list of evidence items."""
        if not items:
            return cls(leaves=[], levels=[[]], root=hashlib.sha256(b"").hexdigest())
        leaves = [_hash_leaf(item) for item in items]
        levels = [leaves]
        current = leaves
        while len(current) > 1:
            next_level: list[str] = []
            for i in range(0, len(current), 2):
                left = current[i]
                right = current[i + 1] if i + 1 < len(current) else current[i]
                next_level.append(_hash_internal(left, right))
            levels.append(next_level)
            current = next_level
        return cls(leaves=leaves, levels=levels, root=current[0])

    def proof(self, index: int) -> MerkleProof:
        """Generate an inclusion proof for the leaf at ``index``."""
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"leaf index {index} out of range [0, {len(self.leaves)})")
        siblings: list[tuple[str, str]] = []
        idx = index
        for level in range(len(self.levels) - 1):
            current = self.levels[level]
            if idx % 2 == 0:
                # Left child; sibling is to the right.
                sib_idx = idx + 1
                if sib_idx < len(current):
                    siblings.append((current[sib_idx], "R"))
                elif sib_idx == idx:
                    # Odd node duplicated; no sibling needed.
                    pass
                else:
                    siblings.append((current[sib_idx - 1], "R"))
            else:
                # Right child; sibling is to the left.
                sib_idx = idx - 1
                siblings.append((current[sib_idx], "L"))
            idx //= 2
        return MerkleProof(leaf_index=index, leaf_hash=self.leaves[index], siblings=siblings)

    def to_log(self) -> dict[str, Any]:
        return {
            "leaf_count": len(self.leaves),
            "root": self.root,
            "depth": len(self.levels),
        }


def verify_proof(leaf_data: bytes | str, proof: MerkleProof, expected_root: str) -> bool:
    """Verify that ``leaf_data`` is included in the tree with root ``expected_root``."""
    current = _hash_leaf(leaf_data)
    if current != proof.leaf_hash:
        return False
    for sib_hash, direction in proof.siblings:
        if direction == "L":
            current = _hash_internal(sib_hash, current)
        else:
            current = _hash_internal(current, sib_hash)
    return current == expected_root


@dataclass(slots=True)
class BatchEvidence:
    """A batch of evidence items with a Merkle root."""
    items: list[bytes | str]
    tree: MerkleTree
    root: str

    @classmethod
    def build(cls, items: list[bytes | str]) -> "BatchEvidence":
        tree = MerkleTree.build(items)
        return cls(items=items, tree=tree, root=tree.root)

    def proof_for(self, index: int) -> MerkleProof:
        return self.tree.proof(index)

    def verify_item(self, index: int) -> bool:
        """Verify that the item at ``index`` is included in the batch."""
        if index < 0 or index >= len(self.items):
            return False
        proof = self.proof_for(index)
        return verify_proof(self.items[index], proof, self.root)

    def to_log(self) -> dict[str, Any]:
        return {
            "item_count": len(self.items),
            "root": self.root,
            "tree": self.tree.to_log(),
        }
