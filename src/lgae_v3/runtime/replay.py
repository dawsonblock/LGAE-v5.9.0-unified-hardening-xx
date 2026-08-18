"""Replay redesign (Phase 20).

A replay buffer stores (state, action, reward, next_state) transitions for
offline learning. The redesigned replay buffer supports:

  - prioritized sampling: high-advantage transitions are sampled more often
  - stratified sampling: ensure coverage across graph families and authority levels
  - deduplication: avoid storing identical transitions
  - bounded size with FIFO eviction

The replay buffer is the foundation for offline RL (Phase 22) and
hard-negative replay (Phase 21).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence
import random


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    """One (state, action, reward, next_state) transition."""
    state_hash: str
    action_id: str
    reward: float
    next_state_hash: str
    authority_level: str = ""
    graph_family: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def transition_id(self) -> str:
        payload = json.dumps({
            "state_hash": self.state_hash,
            "action_id": self.action_id,
            "next_state_hash": self.next_state_hash,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_log(self) -> dict[str, Any]:
        return {
            "state_hash": self.state_hash,
            "action_id": self.action_id,
            "reward": float(self.reward),
            "next_state_hash": self.next_state_hash,
            "authority_level": self.authority_level,
            "graph_family": self.graph_family,
            "transition_id": self.transition_id,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ReplayBuffer:
    """Bounded replay buffer with prioritized and stratified sampling."""
    capacity: int = 10000
    transitions: list[ReplayTransition] = field(default_factory=list)
    _seen_ids: set[str] = field(default_factory=set)
    _priorities: list[float] = field(default_factory=list)

    def add(self, transition: ReplayTransition, priority: float = 1.0) -> bool:
        """Add a transition. Returns True if added, False if duplicate."""
        tid = transition.transition_id
        if tid in self._seen_ids:
            return False
        if len(self.transitions) >= self.capacity:
            # FIFO eviction.
            evicted = self.transitions.pop(0)
            self._priorities.pop(0)
            self._seen_ids.discard(evicted.transition_id)
        self.transitions.append(transition)
        self._priorities.append(max(float(priority), 1e-6))
        self._seen_ids.add(tid)
        return True

    def __len__(self) -> int:
        return len(self.transitions)

    def sample(self, n: int, *, rng: random.Random | None = None) -> list[ReplayTransition]:
        """Sample n transitions uniformly."""
        if not self.transitions or n <= 0:
            return []
        rng = rng or random.Random(42)
        n = min(n, len(self.transitions))
        return rng.sample(self.transitions, n)

    def sample_prioritized(self, n: int, *, rng: random.Random | None = None) -> list[ReplayTransition]:
        """Sample n transitions with priority weighting."""
        if not self.transitions or n <= 0:
            return []
        rng = rng or random.Random(42)
        n = min(n, len(self.transitions))
        # Weighted sampling without replacement.
        weights = self._priorities[:]
        selected: list[ReplayTransition] = []
        indices = list(range(len(self.transitions)))
        for _ in range(n):
            if not indices:
                break
            total = sum(weights[i] for i in indices)
            r = rng.random() * total
            cum = 0.0
            chosen_idx = indices[0]
            for idx in indices:
                cum += weights[idx]
                if cum >= r:
                    chosen_idx = idx
                    break
            selected.append(self.transitions[chosen_idx])
            indices.remove(chosen_idx)
        return selected

    def sample_stratified(
        self, n: int, *,
        by: str = "graph_family",
        rng: random.Random | None = None,
    ) -> list[ReplayTransition]:
        """Sample n transitions with stratification by a field."""
        if not self.transitions or n <= 0:
            return []
        rng = rng or random.Random(42)
        # Group by the stratification field.
        groups: dict[str, list[ReplayTransition]] = {}
        for t in self.transitions:
            key = getattr(t, by, "") or "unknown"
            groups.setdefault(key, []).append(t)
        # Sample from each group proportionally.
        n_groups = len(groups)
        per_group = max(1, n // n_groups)
        result: list[ReplayTransition] = []
        for key, group in groups.items():
            sample_size = min(per_group, len(group))
            result.extend(rng.sample(group, sample_size))
        return result[:n]

    def filter_by(self, **kwargs: Any) -> list[ReplayTransition]:
        """Filter transitions by field values."""
        result = []
        for t in self.transitions:
            if all(getattr(t, k, None) == v for k, v in kwargs.items()):
                result.append(t)
        return result

    def to_log(self) -> dict[str, Any]:
        return {
            "capacity": int(self.capacity),
            "size": len(self.transitions),
            "unique_ids": len(self._seen_ids),
        }
