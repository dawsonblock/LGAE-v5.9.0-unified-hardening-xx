"""v5.10 Phase 31: signed Merkle checkpointing tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import Checkpoint, CheckpointChain


def test_checkpoint_hash_is_deterministic():
    cp1 = Checkpoint(
        checkpoint_index=0, authority_hash="abc", generation=1,
        merkle_root="root1", previous_checkpoint_hash=None, created_at=1000.0,
    )
    cp2 = Checkpoint(
        checkpoint_index=0, authority_hash="abc", generation=1,
        merkle_root="root1", previous_checkpoint_hash=None, created_at=1000.0,
    )
    assert cp1.checkpoint_hash == cp2.checkpoint_hash


def test_checkpoint_hash_differs_for_different_state():
    cp1 = Checkpoint(
        checkpoint_index=0, authority_hash="abc", generation=1,
        merkle_root="root1", previous_checkpoint_hash=None, created_at=1000.0,
    )
    cp2 = Checkpoint(
        checkpoint_index=0, authority_hash="xyz", generation=1,
        merkle_root="root1", previous_checkpoint_hash=None, created_at=1000.0,
    )
    assert cp1.checkpoint_hash != cp2.checkpoint_hash


def test_checkpoint_chain_append():
    chain = CheckpointChain()
    cp = chain.append(
        authority_hash="abc", generation=0,
        evidence_items=["e1", "e2"],
    )
    assert cp.checkpoint_index == 0
    assert cp.previous_checkpoint_hash is None
    assert len(chain.checkpoints) == 1
    assert chain.latest is cp


def test_checkpoint_chain_links():
    chain = CheckpointChain()
    cp0 = chain.append(authority_hash="abc", generation=0, evidence_items=["a"])
    cp1 = chain.append(authority_hash="abc", generation=1, evidence_items=["b"])
    assert cp1.previous_checkpoint_hash == cp0.checkpoint_hash
    assert cp1.checkpoint_index == 1


def test_checkpoint_chain_verify():
    chain = CheckpointChain()
    for i in range(5):
        chain.append(authority_hash="abc", generation=i, evidence_items=[f"e{i}"])
    assert chain.verify_chain()


def test_checkpoint_chain_verify_fails_on_tampered_link():
    chain = CheckpointChain()
    cp0 = chain.append(authority_hash="abc", generation=0, evidence_items=["a"])
    # Manually append a checkpoint with wrong previous_hash.
    from lgae_v3.runtime import Checkpoint as CP
    chain.checkpoints.append(CP(
        checkpoint_index=1, authority_hash="abc", generation=1,
        merkle_root="root", previous_checkpoint_hash="wrong_hash",
        created_at=2000.0,
    ))
    assert not chain.verify_chain()


def test_checkpoint_to_log():
    cp = Checkpoint(
        checkpoint_index=0, authority_hash="abc", generation=1,
        merkle_root="root1", previous_checkpoint_hash=None, created_at=1000.0,
    )
    log = cp.to_log()
    assert log["checkpoint_index"] == 0
    assert log["signed"] is False
    assert "checkpoint_hash" in log


def test_checkpoint_chain_to_log():
    chain = CheckpointChain()
    chain.append(authority_hash="abc", generation=0, evidence_items=["a"])
    log = chain.to_log()
    assert log["checkpoint_count"] == 1
    assert "root_hash" in log
    assert len(log["checkpoints"]) == 1


def test_checkpoint_chain_root_hash():
    chain = CheckpointChain()
    assert chain.root_hash is None
    chain.append(authority_hash="abc", generation=0, evidence_items=["a"])
    assert chain.root_hash is not None
    assert len(chain.root_hash) == 64


def test_checkpoint_with_metadata():
    chain = CheckpointChain()
    cp = chain.append(
        authority_hash="abc", generation=0, evidence_items=["a"],
        metadata={"phase": "v5.10", "node_count": 100},
    )
    assert cp.metadata["phase"] == "v5.10"
    assert cp.metadata["node_count"] == 100
