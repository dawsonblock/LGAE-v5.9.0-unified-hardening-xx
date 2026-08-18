"""Causal structural credit (Phase 19).

Causal credit assignment goes beyond correlation: it identifies which
action *caused* the utility change, not just which action *correlated*
with it. This uses do-calculus style counterfactual reasoning:

  - Observational: P(utility | action)  -- what happened when we acted
  - Interventional: P(utility | do(action))  -- what would happen if we forced it
  - Counterfactual: P(utility | do(action), observed different action)

In the runtime, we approximate causal credit by:
  1. Running counterfactual simulations (shadow transactions)
  2. Comparing the realized utility to the counterfactual utility
  3. Assigning credit only for the *causal* component (interventional vs observational)

This is more robust than Phase 18's correlation-based credit because it
controls for confounders (e.g. the state was already improving before the action).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass(frozen=True, slots=True)
class CausalCreditAssignment:
    """Causal credit for one action."""
    action_id: str
    observational_utility: float  # P(U | A)
    interventional_utility: float  # P(U | do(A))
    counterfactual_utility: float  # P(U | do(A), observed A')
    causal_effect: float  # interventional - observational
    credit: float  # the causal credit assigned

    def to_log(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "observational_utility": float(self.observational_utility),
            "interventional_utility": float(self.interventional_utility),
            "counterfactual_utility": float(self.counterfactual_utility),
            "causal_effect": float(self.causal_effect),
            "credit": float(self.credit),
        }


@dataclass(slots=True)
class CausalCreditAssigner:
    """Assigns causal credit using counterfactual simulation.

    The assigner requires:
    - ``counterfactual_fn``: simulates what would happen under a different action
    - ``observational_utility``: the realized utility from the actual action
    """
    counterfactual_fn: Callable[[str, str], float] | None = None  # (actual_action, counterfactual_action) -> utility

    def assign_credit(
        self,
        *,
        action_id: str,
        observational_utility: float,
        counterfactual_actions: Sequence[str] | None = None,
        baseline_action: str = "noop",
    ) -> CausalCreditAssignment:
        """Assign causal credit to an action.

        The causal effect is the difference between the interventional
        utility (do(action)) and the baseline utility (do(noop)).
        """
        if self.counterfactual_fn is None:
            raise ValueError("no counterfactual function provided")

        # Interventional utility: what happens if we do the action.
        interventional = float(self.counterfactual_fn(baseline_action, action_id))
        # Counterfactual: what would have happened under the baseline.
        counterfactual = float(self.counterfactual_fn(action_id, baseline_action))
        # Causal effect: the difference.
        causal_effect = interventional - counterfactual
        # Credit is the causal effect.
        credit = causal_effect

        return CausalCreditAssignment(
            action_id=action_id,
            observational_utility=float(observational_utility),
            interventional_utility=interventional,
            counterfactual_utility=counterfactual,
            causal_effect=causal_effect,
            credit=credit,
        )

    def assign_credit_batch(
        self,
        *,
        action_ids: Sequence[str],
        observational_utilities: Sequence[float],
        baseline_action: str = "noop",
    ) -> list[CausalCreditAssignment]:
        """Assign causal credit to multiple actions."""
        return [
            self.assign_credit(
                action_id=aid,
                observational_utility=ou,
                baseline_action=baseline_action,
            )
            for aid, ou in zip(action_ids, observational_utilities)
        ]


def average_causal_effect(assignments: list[CausalCreditAssignment]) -> float:
    """Compute the average causal effect across assignments."""
    if not assignments:
        return 0.0
    return sum(a.causal_effect for a in assignments) / len(assignments)


def credit_concentration(assignments: list[CausalCreditAssignment]) -> float:
    """Measure how concentrated credit is (0 = uniform, 1 = single action).

    Uses the Gini coefficient of absolute credit values.
    """
    if len(assignments) <= 1:
        return 0.0
    credits = sorted(abs(a.credit) for a in assignments)
    n = len(credits)
    cum = sum((2 * i - n - 1) * c for i, c in enumerate(credits, 1))
    total = sum(credits)
    if total == 0:
        return 0.0
    return cum / (n * total)
