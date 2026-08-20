"""Conformal advantage arbitrator for exp6.8.3.

The arbitration rule is:
  LCB_A = A_hat - q_{1-alpha}
  override only if LCB_A > 0
  otherwise: use baseline action

This is the entire authority recommendation logic for the experiment.
No learned decision gains direct authority — exact finalist replay
is still mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..exp6_3.exact_mpc import ActionIdentity


@dataclass
class ConformalArbitrationResult:
    """Result of conformal advantage arbitration."""
    selected_action: tuple = ("", 0, 0)
    selected_action_id: Optional[ActionIdentity] = None
    used_learned: bool = False
    source: str = "baseline"  # "learned" or "baseline"
    # Advantage prediction.
    advantage_hat: float = 0.0
    conformal_quantile: float = 0.0
    lcb_advantage: float = 0.0
    alpha: float = 0.05
    # For diagnostics.
    baseline_action: tuple = ("", 0, 0)
    learned_action: tuple = ("", 0, 0)
    baseline_action_id: Optional[ActionIdentity] = None
    learned_action_id: Optional[ActionIdentity] = None


def conformal_arbitrate(
    baseline_action: tuple,
    learned_action: tuple,
    baseline_action_id: ActionIdentity,
    learned_action_id: ActionIdentity,
    advantage_hat: float,
    conformal_quantile: float,
    alpha: float = 0.05,
) -> ConformalArbitrationResult:
    """Arbitrate between baseline and learned actions using conformal LCB.

    LCB_A = A_hat - q_{1-alpha}
    override only if LCB_A > 0

    This directly asks: "Is there calibrated evidence that the learned
    action is better than baseline?"
    """
    lcb = advantage_hat - conformal_quantile

    if lcb > 0 and learned_action[0]:
        return ConformalArbitrationResult(
            selected_action=learned_action,
            selected_action_id=learned_action_id,
            used_learned=True,
            source="learned",
            advantage_hat=advantage_hat,
            conformal_quantile=conformal_quantile,
            lcb_advantage=lcb,
            alpha=alpha,
            baseline_action=baseline_action,
            learned_action=learned_action,
            baseline_action_id=baseline_action_id,
            learned_action_id=learned_action_id,
        )
    else:
        return ConformalArbitrationResult(
            selected_action=baseline_action,
            selected_action_id=baseline_action_id,
            used_learned=False,
            source="baseline",
            advantage_hat=advantage_hat,
            conformal_quantile=conformal_quantile,
            lcb_advantage=lcb,
            alpha=alpha,
            baseline_action=baseline_action,
            learned_action=learned_action,
            baseline_action_id=baseline_action_id,
            learned_action_id=learned_action_id,
        )


def batch_arbitrate(
    baseline_actions: list[tuple],
    learned_actions: list[tuple],
    baseline_ids: list[ActionIdentity],
    learned_ids: list[ActionIdentity],
    advantage_hats: np.ndarray,
    conformal_quantile: float,
    alpha: float = 0.05,
) -> list[ConformalArbitrationResult]:
    """Arbitrate a batch of decisions."""
    results = []
    for i in range(len(baseline_actions)):
        result = conformal_arbitrate(
            baseline_actions[i],
            learned_actions[i],
            baseline_ids[i],
            learned_ids[i],
            float(advantage_hats[i]),
            conformal_quantile,
            alpha=alpha,
        )
        results.append(result)
    return results
