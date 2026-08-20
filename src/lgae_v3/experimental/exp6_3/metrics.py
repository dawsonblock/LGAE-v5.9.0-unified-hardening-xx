"""Metrics for exp6.3: first-action agreement, planning regret, search savings.

Primary metrics:
- FirstActionAgreement: P(a_model == a_exact_mpc)
- PlanningRegret: Q(a*) - Q(a_model)
- SearchSavings: 1 - nodes_expanded_model / nodes_expanded_exact
- TrajectoryRecall: did model preserve exact-best trajectory in top-M?
- GreedyImprovement: Q(a_model) - Q(a_greedy)
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .exact_mpc import ExactPlan
from .beam_search import BeamSearchResult


def first_action_agreement(exact: ExactPlan, model: BeamSearchResult) -> bool:
    """Check if model picks the same first action as exact MPC."""
    return exact.first_action == model.first_action


def planning_regret(exact: ExactPlan, model: BeamSearchResult) -> float:
    """Planning regret = Q(a*) - Q(a_model).

    Uses exact MPC's first-action values to compute the regret
    of the model's chosen first action. Uses ActionIdentity for
    complete key matching (includes params).
    """
    from .exact_mpc import ActionIdentity
    if exact.first_action_identity is not None:
        exact_key = exact.first_action_identity.key
    else:
        exact_key = f"{exact.first_action[0]}_{exact.first_action[1]}_{exact.first_action[2]}"
    if hasattr(model, 'first_action_identity') and model.first_action_identity is not None:
        model_key = model.first_action_identity.key
    elif model.best_sequence:
        model_key = ActionIdentity.from_action(model.best_sequence[0]).key
    else:
        model_key = f"{model.first_action[0]}_{model.first_action[1]}_{model.first_action[2]}"

    exact_val = exact.all_first_action_values.get(exact_key, exact.total_value)
    model_val = exact.all_first_action_values.get(model_key, model.total_value)

    return float(exact_val - model_val)


def search_savings(exact: ExactPlan, model: BeamSearchResult) -> float:
    """Fraction of search nodes saved by model vs exact MPC."""
    if exact.nodes_expanded == 0:
        return 0.0
    return 1.0 - model.nodes_expanded / exact.nodes_expanded


def trajectory_recall(exact: ExactPlan, model: BeamSearchResult, top_m: int = 1) -> bool:
    """Did the model preserve the exact-best trajectory in its top-M?

    For now, checks if the model's best sequence matches the exact best.
    A proper implementation would check the top-M beam entries.
    """
    if not exact.best_sequence or not model.best_sequence:
        return False
    return exact.best_sequence == model.best_sequence


def greedy_improvement(exact: ExactPlan, greedy: ExactPlan, model: BeamSearchResult) -> float:
    """How much better is the model than greedy?

    Q(a_model) - Q(a_greedy), using exact MPC values.
    """
    model_key = f"{model.first_action[0]}_{model.first_action[1]}_{model.first_action[2]}"
    greedy_key = f"{greedy.first_action[0]}_{greedy.first_action[1]}_{greedy.first_action[2]}"

    model_val = exact.all_first_action_values.get(model_key, model.total_value)
    greedy_val = exact.all_first_action_values.get(greedy_key, greedy.total_value)

    return float(model_val - greedy_val)
