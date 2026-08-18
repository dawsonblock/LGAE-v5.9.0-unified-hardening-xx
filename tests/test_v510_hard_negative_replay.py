"""v5.10 Phase 21: hard-negative replay tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    ReplayTransition, ReplayBuffer,
    HardNegative, HardNegativeMiner, augment_buffer_with_hard_negatives,
)


def _make_transition(state: str = "s1", action: str = "a1", reward: float = 0.1) -> ReplayTransition:
    return ReplayTransition(state_hash=state, action_id=action, reward=reward, next_state_hash="s2")


def test_hard_negative_gap():
    hn = HardNegative(
        transition=_make_transition(), predicted_utility=0.9,
        realized_utility=0.1, difficulty=0.8,
    )
    assert hn.gap == pytest.approx(0.8)


def test_hard_negative_to_log():
    hn = HardNegative(
        transition=_make_transition(), predicted_utility=0.9,
        realized_utility=0.1, difficulty=0.8,
    )
    log = hn.to_log()
    assert log["predicted_utility"] == 0.9
    assert log["gap"] == pytest.approx(0.8)


def test_hard_negative_miner_finds_overconfident_wrong():
    buf = ReplayBuffer(capacity=100)
    # Transition with high predicted utility but low reward.
    t = _make_transition(reward=0.1)
    buf.add(t)
    miner = HardNegativeMiner(difficulty_threshold=0.05)
    hard_negs = miner.mine(buf, predicted_utilities={t.transition_id: 0.9})
    assert len(hard_negs) == 1
    assert hard_negs[0].gap == pytest.approx(0.8)


def test_hard_negative_miner_skips_easy_transitions():
    buf = ReplayBuffer(capacity=100)
    t = _make_transition(reward=0.8)
    buf.add(t)
    miner = HardNegativeMiner(difficulty_threshold=0.5)
    hard_negs = miner.mine(buf, predicted_utilities={t.transition_id: 0.85})
    assert len(hard_negs) == 0  # gap is only 0.05, below threshold


def test_hard_negative_miner_sorts_by_difficulty():
    buf = ReplayBuffer(capacity=100)
    t1 = _make_transition(state="s1", reward=0.1)
    t2 = _make_transition(state="s2", reward=0.2)
    buf.add(t1)
    buf.add(t2)
    miner = HardNegativeMiner(difficulty_threshold=0.01)
    hard_negs = miner.mine(buf, predicted_utilities={
        t1.transition_id: 0.9,  # gap=0.8, difficulty=0.72
        t2.transition_id: 0.5,  # gap=0.3, difficulty=0.15
    })
    assert len(hard_negs) == 2
    assert hard_negs[0].difficulty > hard_negs[1].difficulty


def test_hard_negative_miner_without_predicted_utilities():
    buf = ReplayBuffer(capacity=100)
    t = _make_transition(reward=0.5)
    buf.add(t)
    miner = HardNegativeMiner(difficulty_threshold=0.01)
    # Without predicted utilities, uses reward as proxy -> gap=0, no hard negs.
    hard_negs = miner.mine(buf)
    assert len(hard_negs) == 0


def test_hard_negative_miner_max_limit():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        t = _make_transition(state=f"s{i}", reward=0.1)
        buf.add(t)
    miner = HardNegativeMiner(difficulty_threshold=0.01, max_hard_negatives=3)
    preds = {t.transition_id: 0.9 for t in buf.transitions}
    hard_negs = miner.mine(buf, predicted_utilities=preds)
    assert len(hard_negs) == 3


def test_augment_buffer_with_hard_negatives():
    buf = ReplayBuffer(capacity=100)
    t = _make_transition(reward=0.1)
    buf.add(t, priority=1.0)
    hn = HardNegative(transition=t, predicted_utility=0.9, realized_utility=0.1, difficulty=0.72)
    count = augment_buffer_with_hard_negatives(buf, [hn], priority_boost=10.0)
    assert count == 1
    # Check that priority was boosted.
    idx = buf.transitions.index(t)
    assert buf._priorities[idx] == 10.0


def test_hard_negative_miner_to_log():
    miner = HardNegativeMiner(difficulty_threshold=0.3, max_hard_negatives=50)
    log = miner.to_log()
    assert log["difficulty_threshold"] == 0.3
    assert log["max_hard_negatives"] == 50
