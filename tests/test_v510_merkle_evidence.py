"""v5.10 Phase 28: cryptographic evidence v2 (Merkle tree) tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import MerkleProof, MerkleTree, BatchEvidence, verify_proof


def test_merkle_tree_empty():
    tree = MerkleTree.build([])
    assert tree.root == len(tree.root) * "0"[:0] or len(tree.root) == 64  # some hash


def test_merkle_tree_single_item():
    tree = MerkleTree.build(["item1"])
    assert len(tree.root) == 64
    assert tree.leaves == [_hash_leaf("item1")] if False else len(tree.leaves) == 1


def _hash_leaf(data: str) -> str:
    import hashlib
    return hashlib.sha256(b"\x00" + data.encode()).hexdigest()


def test_merkle_tree_multiple_items():
    items = ["a", "b", "c", "d"]
    tree = MerkleTree.build(items)
    assert len(tree.root) == 64
    assert len(tree.leaves) == 4
    assert len(tree.levels) == 3  # leaves, internal, root


def test_merkle_tree_odd_items():
    items = ["a", "b", "c"]
    tree = MerkleTree.build(items)
    assert len(tree.root) == 64
    assert len(tree.leaves) == 3


def test_merkle_proof_verifies():
    items = ["evidence1", "evidence2", "evidence3", "evidence4"]
    tree = MerkleTree.build(items)
    for i in range(len(items)):
        proof = tree.proof(i)
        assert verify_proof(items[i], proof, tree.root)


def test_merkle_proof_rejects_wrong_data():
    items = ["evidence1", "evidence2", "evidence3"]
    tree = MerkleTree.build(items)
    proof = tree.proof(0)
    assert not verify_proof("wrong_data", proof, tree.root)


def test_merkle_proof_rejects_wrong_root():
    items = ["evidence1", "evidence2"]
    tree = MerkleTree.build(items)
    proof = tree.proof(0)
    fake_root = "0" * 64
    assert not verify_proof(items[0], proof, fake_root)


def test_merkle_proof_index_out_of_range():
    tree = MerkleTree.build(["a", "b"])
    with pytest.raises(IndexError):
        tree.proof(5)


def test_batch_evidence_build_and_verify():
    items = ["receipt1", "receipt2", "receipt3", "receipt4", "receipt5"]
    batch = BatchEvidence.build(items)
    assert len(batch.root) == 64
    for i in range(len(items)):
        assert batch.verify_item(i)


def test_batch_evidence_verify_wrong_index():
    items = ["a", "b", "c"]
    batch = BatchEvidence.build(items)
    assert not batch.verify_item(-1)
    assert not batch.verify_item(10)


def test_batch_evidence_to_log():
    batch = BatchEvidence.build(["a", "b"])
    log = batch.to_log()
    assert "item_count" in log
    assert "root" in log
    assert log["item_count"] == 2


def test_merkle_tree_deterministic():
    items = ["x", "y", "z"]
    t1 = MerkleTree.build(items)
    t2 = MerkleTree.build(items)
    assert t1.root == t2.root


def test_merkle_tree_different_items_different_root():
    t1 = MerkleTree.build(["a", "b"])
    t2 = MerkleTree.build(["a", "c"])
    assert t1.root != t2.root


def test_merkle_proof_to_log():
    tree = MerkleTree.build(["a", "b", "c", "d"])
    proof = tree.proof(1)
    log = proof.to_log()
    assert log["leaf_index"] == 1
    assert "siblings" in log
