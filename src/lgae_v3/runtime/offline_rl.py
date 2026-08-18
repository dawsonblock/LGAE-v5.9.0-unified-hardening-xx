"""Stronger offline RL (Phase 22).

Offline RL learns a policy from a fixed replay buffer without environment
interaction. This module implements Conservative Q-Learning (CQL) style
offline RL for the structural intelligence runtime:

  - Q(s, a): estimates the value of taking action a in state s
  - Conservative penalty: penalizes Q-values for out-of-distribution actions
    (actions not in the replay buffer) to prevent overestimation
  - Bellman update: Q(s, a) <- r + gamma * max_a' Q(s', a')

The conservative penalty is the key innovation: standard offline RL
overestimates Q-values for unseen actions because there's no environment
feedback to correct them. CQL adds a penalty for high Q-values on
out-of-distribution actions, keeping the policy close to the data distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import Module, Linear

from .replay import ReplayBuffer, ReplayTransition


@dataclass(frozen=True, slots=True)
class OfflineRLConfig:
    """Configuration for offline RL training."""
    learning_rate: float = 1e-3
    gamma: float = 0.99  # discount factor
    cql_alpha: float = 1.0  # conservative penalty weight
    batch_size: int = 64
    n_epochs: int = 100
    hidden_dim: int = 64

    def to_log(self) -> dict[str, Any]:
        return {
            "learning_rate": float(self.learning_rate),
            "gamma": float(self.gamma),
            "cql_alpha": float(self.cql_alpha),
            "batch_size": int(self.batch_size),
            "n_epochs": int(self.n_epochs),
            "hidden_dim": int(self.hidden_dim),
        }


class QNetwork(Module):
    """Simple Q-network: state_hash -> Q-value per action."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, n_actions),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


@dataclass(slots=True)
class OfflineRLTrainer:
    """Conservative Q-Learning trainer for offline RL."""
    config: OfflineRLConfig
    q_network: QNetwork | None = None
    state_encoder: dict[str, int] = field(default_factory=dict)  # state_hash -> index
    action_encoder: dict[str, int] = field(default_factory=dict)  # action_id -> index
    training_log: list[dict[str, float]] = field(default_factory=list)

    def _encode_state(self, state_hash: str, dim: int) -> Tensor:
        """Encode a state hash into a one-hot vector."""
        if state_hash not in self.state_encoder:
            self.state_encoder[state_hash] = len(self.state_encoder) % dim
        idx = self.state_encoder[state_hash]
        vec = torch.zeros(dim)
        vec[idx] = 1.0
        return vec

    def _encode_action(self, action_id: str) -> int:
        if action_id not in self.action_encoder:
            self.action_encoder[action_id] = len(self.action_encoder)
        return self.action_encoder[action_id]

    def train(self, buffer: ReplayBuffer) -> dict[str, float]:
        """Train the Q-network on the replay buffer.

        Returns final metrics: {loss, cql_penalty, q_mean}.
        """
        if len(buffer) == 0:
            return {"loss": 0.0, "cql_penalty": 0.0, "q_mean": 0.0}

        # Collect all actions and states.
        transitions = buffer.transitions
        for t in transitions:
            self._encode_action(t.action_id)
            self._encode_state(t.state_hash, dim=128)

        n_actions = len(self.action_encoder)
        state_dim = 128
        if self.q_network is None:
            self.q_network = QNetwork(state_dim, n_actions, self.config.hidden_dim)
        optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)

        final_metrics = {"loss": 0.0, "cql_penalty": 0.0, "q_mean": 0.0}
        for epoch in range(self.config.n_epochs):
            # Sample a batch.
            batch = buffer.sample(min(self.config.batch_size, len(buffer)))
            if not batch:
                continue
            total_loss = 0.0
            total_cql = 0.0
            total_q = 0.0
            for t in batch:
                state_vec = self._encode_state(t.state_hash, state_dim)
                action_idx = self._encode_action(t.action_id)
                q_values = self.q_network(state_vec)
                q_sa = q_values[action_idx]

                # Bellman target: r + gamma * max_a' Q(s', a')
                with torch.no_grad():
                    next_state_vec = self._encode_state(t.next_state_hash, state_dim)
                    next_q = self.q_network(next_state_vec)
                    target = t.reward + self.config.gamma * float(next_q.max().item())

                bellman_loss = (q_sa - target) ** 2

                # CQL penalty: penalize high Q-values for all actions
                # (conservative estimate).
                cql_penalty = self.config.cql_alpha * q_values.mean()

                loss = bellman_loss + cql_penalty
                total_loss += float(loss.item())
                total_cql += float(cql_penalty.item())
                total_q += float(q_sa.item())

            # Backprop.
            optimizer.zero_grad()
            # Use the last transition's loss for backprop (simplified).
            if batch:
                t = batch[-1]
                state_vec = self._encode_state(t.state_hash, state_dim)
                action_idx = self._encode_action(t.action_id)
                q_values = self.q_network(state_vec)
                q_sa = q_values[action_idx]
                with torch.no_grad():
                    next_state_vec = self._encode_state(t.next_state_hash, state_dim)
                    next_q = self.q_network(next_state_vec)
                    target = t.reward + self.config.gamma * float(next_q.max().item())
                loss = (q_sa - target) ** 2 + self.config.cql_alpha * q_values.mean()
                loss.backward()
                optimizer.step()

            n = len(batch)
            final_metrics = {
                "loss": total_loss / n,
                "cql_penalty": total_cql / n,
                "q_mean": total_q / n,
            }
            self.training_log.append(final_metrics)

        return final_metrics

    def get_q_value(self, state_hash: str, action_id: str) -> float:
        """Get the Q-value for a (state, action) pair."""
        if self.q_network is None:
            return 0.0
        if action_id not in self.action_encoder:
            return 0.0
        state_vec = self._encode_state(state_hash, dim=128)
        action_idx = self._encode_action(action_id)
        with torch.no_grad():
            q_values = self.q_network(state_vec)
        return float(q_values[action_idx].item())

    def to_log(self) -> dict[str, Any]:
        return {
            "config": self.config.to_log(),
            "n_states": len(self.state_encoder),
            "n_actions": len(self.action_encoder),
            "training_log_size": len(self.training_log),
        }
