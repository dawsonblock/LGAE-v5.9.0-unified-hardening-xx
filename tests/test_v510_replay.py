"""v5.10 Phase 20: replay redesign tests."""
from __future__ import annotations

import random

import pytest

from lgae_v3.runtime import ReplayTransition, ReplayBuffer


def _make_transition(state: str = "s1", action: str = "a1", reward: float = 1.0,
                     next_state: str = "s2", family: str = "ba") -> ReplayTransition:
    return ReplayTransition(
        state_hash=state, action_id=action, reward=reward,
        next_state_hash=next_state, graph_family=family,
    )


def test_replay_buffer_add_and_len():
    buf = ReplayBuffer(capacity=100)
    buf.add(_make_transition())
    assert len(buf) == 1


def test_replay_buffer_dedup():
    buf = ReplayBuffer(capacity=100)
    t = _make_transition()
    assert buf.add(t) is True
    assert buf.add(t) is False  # duplicate
    assert len(buf) == 1


def test_replay_buffer_fifo_eviction():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(_make_transition(state=f"s{i}", next_state=f"s{i+1}"))
    assert len(buf) == 3  # capacity bound
    # First two should have been evicted.
    assert buf.transitions[0].state_hash == "s2"


def test_replay_buffer_sample_uniform():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.add(_make_transition(state=f"s{i}", next_state=f"s{i+1}"))
    rng = random.Random(42)
    sample = buf.sample(5, rng=rng)
    assert len(sample) == 5
    assert all(isinstance(t, ReplayTransition) for t in sample)


def test_replay_buffer_sample_empty():
    buf = ReplayBuffer(capacity=100)
    assert buf.sample(5) == []


def test_replay_buffer_sample_prioritized():
    buf = ReplayBuffer(capacity=100)
    # Add transitions with different priorities.
    for i in range(10):
        buf.add(_make_transition(state=f"s{i}", next_state=f"s{i+1}"),
                priority=10.0 if i < 2 else 1.0)
    rng = random.Random(42)
    sample = buf.sample_prioritized(20, rng=rng)  # request more than available
    assert len(sample) == 10  # can't sample more than available


def test_replay_buffer_sample_stratified():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        family = "ba" if i < 5 else "ws"
        buf.add(_make_transition(state=f"s{i}", next_state=f"s{i+1}", family=family))
    rng = random.Random(42)
    sample = buf.sample_stratified(6, by="graph_family", rng=rng)
    assert len(sample) <= 6
    # Should have transitions from both families.
    families = {t.graph_family for t in sample}
    assert len(families) >= 1


def test_replay_buffer_filter_by():
    buf = ReplayBuffer(capacity=100)
    buf.add(_make_transition(state="s1", family="ba"))
    buf.add(_make_transition(state="s2", family="ws"))
    buf.add(_make_transition(state="s3", family="ba"))
    result = buf.filter_by(graph_family="ba")
    assert len(result) == 2


def test_replay_transition_id_deterministic():
    t1 = _make_transition()
    t2 = _make_transition()
    assert t1.transition_id == t2.transition_id


def test_replay_transition_to_log():
    t = _make_transition(reward=0.5)
    log = t.to_log()
    assert log["reward"] == 0.5
    assert log["action_id"] == "a1"
    assert len(log["transition_id"]) == 16


def test_replay_buffer_to_log():
    buf = ReplayBuffer(capacity=50)
    for i in range(3):
        buf.add(_make_transition(state=f"s{i}"))
    log = buf.to_log()
    assert log["capacity"] == 50
    assert log["size"] == 3
