"""Structural credit assignment (Phase 18).

When a runtime step produces a utility delta, credit assignment determines
which candidate action was responsible. The challenge is that multiple
actions may be taken in a single step, and the utility change may be delayed.

This module implements:
  - direct credit: the chosen action gets all credit
  - feature-based credit: credit is distributed by feature attribution
  - temporal credit: credit is discounted by temporal distance
  - baseline credit: credit relative to a no-op baseline

Credit is always relative to a baseline. The no-op baseline measures what
would have happened without any action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class CreditAssignment:
    """Credit assigned to one action for a utility delta."""
    action_id: str
    credit: float  # can be positive (good) or negative (bad)
    baseline_utility: float
    realized_utility: float
    method: str  # "direct", "feature", "temporal", "baseline"

    @property
    def advantage(self) -> float:
        """Advantage over baseline."""
        return self.realized_utility - self.baseline_utility

    def to_log(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "credit": float(self.credit),
            "baseline_utility": float(self.baseline_utility),
            "realized_utility": float(self.realized_utility),
            "advantage": float(self.advantage),
            "method": self.method,
        }


def direct_credit(
    *,
    action_id: str,
    baseline_utility: float,
    realized_utility: float,
) -> CreditAssignment:
    """Direct credit: the chosen action gets all the advantage."""
    credit = realized_utility - baseline_utility
    return CreditAssignment(
        action_id=action_id, credit=credit,
        baseline_utility=float(baseline_utility),
        realized_utility=float(realized_utility),
        method="direct",
    )


def feature_based_credit(
    *,
    action_ids: Sequence[str],
    feature_weights: Sequence[float],
    baseline_utility: float,
    realized_utility: float,
) -> list[CreditAssignment]:
    """Distribute credit by feature weights.

    Each action gets a share of the total advantage proportional to its
    feature weight. Weights should be non-negative and sum to 1.
    """
    total_weight = sum(feature_weights)
    if total_weight <= 0:
        # Equal distribution if weights are all zero.
        n = len(action_ids)
        shares = [1.0 / n] * n if n > 0 else []
    else:
        shares = [w / total_weight for w in feature_weights]
    total_advantage = realized_utility - baseline_utility
    return [
        CreditAssignment(
            action_id=aid, credit=share * total_advantage,
            baseline_utility=float(baseline_utility),
            realized_utility=float(realized_utility),
            method="feature",
        )
        for aid, share in zip(action_ids, shares)
    ]


def temporal_credit(
    *,
    action_ids: Sequence[str],
    utilities: Sequence[float],
    baseline_utility: float,
    discount: float = 0.9,
) -> list[CreditAssignment]:
    """Temporal credit assignment with discounting.

    Actions closer to the realized utility get more credit. The discount
    factor gamma < 1 reduces credit for temporally distant actions.
    """
    n = len(action_ids)
    if n == 0:
        return []
    # Discount weights: last action gets gamma^0, second-to-last gets gamma^1, etc.
    weights = [discount ** (n - 1 - i) for i in range(n)]
    total_weight = sum(weights)
    shares = [w / total_weight for w in weights]
    total_advantage = utilities[-1] - baseline_utility
    return [
        CreditAssignment(
            action_id=aid, credit=share * total_advantage,
            baseline_utility=float(baseline_utility),
            realized_utility=float(utilities[-1]),
            method="temporal",
        )
        for aid, share in zip(action_ids, shares)
    ]


def baseline_credit(
    *,
    action_id: str,
    noop_utility: float,  # utility if no action was taken
    action_utility: float,  # utility after the action
) -> CreditAssignment:
    """Credit relative to a no-op baseline.

    This is the simplest and most robust form of credit assignment: the
    action's credit is the difference between acting and not acting.
    """
    return CreditAssignment(
        action_id=action_id, credit=action_utility - noop_utility,
        baseline_utility=float(noop_utility),
        realized_utility=float(action_utility),
        method="baseline",
    )
