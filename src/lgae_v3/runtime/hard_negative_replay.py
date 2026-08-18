"""Hard-negative replay (Phase 21).

Hard-negative mining for replay: identify transitions where the policy made
confident but wrong decisions, and replay them more frequently during
training. This accelerates learning from mistakes.

A "hard negative" is a transition where:
  - the policy was confident (high predicted utility)
  - the realized utility was low (negative advantage)
  - the gap between predicted and realized is large

Hard-negative replay complements prioritized replay (Phase 20) by
specifically targeting overconfident wrong predictions, which are the
most informative for calibration improvement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .replay import ReplayBuffer, ReplayTransition


@dataclass(frozen=True, slots=True)
class HardNegative:
    """A hard-negative transition with its difficulty score."""
    transition: ReplayTransition
    predicted_utility: float
    realized_utility: float
    difficulty: float  # |predicted - realized| * confidence

    @property
    def gap(self) -> float:
        """Gap between predicted and realized utility."""
        return abs(self.predicted_utility - self.realized_utility)

    def to_log(self) -> dict[str, Any]:
        return {
            "transition": self.transition.to_log(),
            "predicted_utility": float(self.predicted_utility),
            "realized_utility": float(self.realized_utility),
            "gap": float(self.gap),
            "difficulty": float(self.difficulty),
        }


@dataclass(slots=True)
class HardNegativeMiner:
    """Mines hard negatives from a replay buffer."""
    difficulty_threshold: float = 0.3
    max_hard_negatives: int = 100

    def mine(
        self,
        buffer: ReplayBuffer,
        *,
        predicted_utilities: dict[str, float] | None = None,
    ) -> list[HardNegative]:
        """Mine hard negatives from the replay buffer.

        ``predicted_utilities`` maps transition_id -> predicted utility.
        If not provided, uses the transition's reward as a proxy.
        """
        hard_negatives: list[HardNegative] = []
        for t in buffer.transitions:
            pred = predicted_utilities.get(t.transition_id, t.reward) if predicted_utilities else t.reward
            realized = t.reward
            gap = abs(pred - realized)
            # Difficulty: gap * confidence (higher confidence + bigger gap = harder)
            confidence = max(pred, 0.0)  # simple confidence proxy
            difficulty = gap * confidence
            if difficulty >= self.difficulty_threshold:
                hard_negatives.append(HardNegative(
                    transition=t,
                    predicted_utility=float(pred),
                    realized_utility=float(realized),
                    difficulty=float(difficulty),
                ))
            if len(hard_negatives) >= self.max_hard_negatives:
                break
        # Sort by difficulty (hardest first).
        hard_negatives.sort(key=lambda hn: hn.difficulty, reverse=True)
        return hard_negatives

    def to_log(self) -> dict[str, Any]:
        return {
            "difficulty_threshold": float(self.difficulty_threshold),
            "max_hard_negatives": int(self.max_hard_negatives),
        }


def augment_buffer_with_hard_negatives(
    buffer: ReplayBuffer,
    hard_negatives: list[HardNegative],
    *,
    priority_boost: float = 10.0,
) -> int:
    """Re-add hard negatives to the buffer with boosted priority.

    Returns the number of hard negatives re-added.
    """
    count = 0
    for hn in hard_negatives:
        # Re-add with boosted priority (will be deduplicated if already present).
        # We update the priority by removing and re-adding.
        tid = hn.transition.transition_id
        # Find and update priority if already in buffer.
        for i, t in enumerate(buffer.transitions):
            if t.transition_id == tid:
                buffer._priorities[i] = max(buffer._priorities[i], priority_boost)
                count += 1
                break
        else:
            # Not in buffer, add it.
            if buffer.add(hn.transition, priority=priority_boost):
                count += 1
    return count
