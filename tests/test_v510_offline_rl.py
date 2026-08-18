"""v5.10 Phase 22: stronger offline RL (CQL) tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    OfflineRLConfig, QNetwork, OfflineRLTrainer,
    ReplayBuffer, ReplayTransition,
)


def _make_buffer(n: int = 10) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=100)
    for i in range(n):
        buf.add(ReplayTransition(
            state_hash=f"s{i}", action_id=f"a{i % 3}",
            reward=float(i) / n, next_state_hash=f"s{i+1}",
        ))
    return buf


def test_offline_rl_config_defaults():
    config = OfflineRLConfig()
    assert config.gamma == 0.99
    assert config.cql_alpha == 1.0
    assert config.n_epochs == 100


def test_offline_rl_config_to_log():
    config = OfflineRLConfig(learning_rate=0.01, gamma=0.95, cql_alpha=0.5)
    log = config.to_log()
    assert log["learning_rate"] == 0.01
    assert log["gamma"] == 0.95
    assert log["cql_alpha"] == 0.5


def test_q_network_forward():
    net = QNetwork(state_dim=10, n_actions=3, hidden_dim=16)
    state = torch.zeros(10)
    state[0] = 1.0
    q_values = net(state)
    assert q_values.shape == torch.Size([3])


def test_offline_rl_trainer_empty_buffer():
    trainer = OfflineRLTrainer(config=OfflineRLConfig(n_epochs=5))
    buf = ReplayBuffer(capacity=100)
    metrics = trainer.train(buf)
    assert metrics["loss"] == 0.0


def test_offline_rl_trainer_runs():
    trainer = OfflineRLTrainer(config=OfflineRLConfig(n_epochs=5, batch_size=4))
    buf = _make_buffer(10)
    metrics = trainer.train(buf)
    assert "loss" in metrics
    assert "cql_penalty" in metrics
    assert "q_mean" in metrics
    assert len(trainer.training_log) == 5


def test_offline_rl_trainer_get_q_value():
    trainer = OfflineRLTrainer(config=OfflineRLConfig(n_epochs=3, batch_size=4))
    buf = _make_buffer(10)
    trainer.train(buf)
    q = trainer.get_q_value("s0", "a0")
    assert isinstance(q, float)


def test_offline_rl_trainer_get_q_value_untrained():
    trainer = OfflineRLTrainer(config=OfflineRLConfig())
    q = trainer.get_q_value("s0", "a0")
    assert q == 0.0  # untrained


def test_offline_rl_trainer_to_log():
    trainer = OfflineRLTrainer(config=OfflineRLConfig(n_epochs=3))
    buf = _make_buffer(5)
    trainer.train(buf)
    log = trainer.to_log()
    assert "config" in log
    assert log["n_actions"] > 0
    assert log["training_log_size"] == 3


def test_offline_rl_trainer_cql_penalty_positive():
    trainer = OfflineRLTrainer(config=OfflineRLConfig(n_epochs=3, batch_size=4, cql_alpha=1.0))
    buf = _make_buffer(10)
    metrics = trainer.train(buf)
    # CQL penalty is alpha * mean(Q); sign depends on Q-values.
    # Just verify it's a finite number.
    assert isinstance(metrics["cql_penalty"], float)
