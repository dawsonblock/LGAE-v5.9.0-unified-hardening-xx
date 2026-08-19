"""Causal intermediate prediction targets for exp6.4.

Instead of predicting scalar bonus directly, predict:
1. delta_n_components: how many components does this action merge/split?
2. P(threshold reached within H | S, a): probability of reaching threshold
3. n_components_after: component count after applying action

These are easier to learn because they capture the causal mechanism
(component merging) rather than the downstream scalar consequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from .structural_features import compute_component_info
from ..exp6_3.exact_mpc import apply_action


@dataclass
class CausalTarget:
    """Causal intermediate targets for a (state, action) pair."""
    delta_n_components: int = 0
    n_components_after: int = 0
    threshold_reached: bool = False  # n_comp_after <= threshold
    steps_to_threshold: int = 0  # min steps from S' to threshold
    exact_bonus: float = 0.0  # for verification only
    exact_q_h2: float = 0.0  # for verification only


def compute_causal_targets(
    graph: GraphBuffers,
    z: torch.Tensor,
    action: tuple[str, int, int, dict],
    *,
    lambda_conn: float = 30.0,
    threshold: int = 1,
    horizon: int = 2,
    available_actions: list[tuple[str, int, int, dict]] | None = None,
    utility_fn=None,
) -> CausalTarget:
    """Compute causal intermediate targets for a (state, action) pair.

    Uses exact computation for label generation only.
    These labels are NOT available during search.
    """
    n = int(graph.num_nodes)
    comp_before = compute_component_info(graph, n)

    # Apply action.
    next_graph = apply_action(graph, action)
    comp_after = compute_component_info(next_graph, n)

    delta_n_comp = comp_after.n_components - comp_before.n_components
    threshold_reached = comp_after.n_components <= threshold
    steps_to_threshold = max(0, comp_after.n_components - threshold)

    # Exact bonus of S' (for verification, not for search).
    from ..exp6_3.split_utility import compute_bonus
    exact_bonus = compute_bonus(next_graph, z, lambda_conn, threshold)

    # Exact Q_H2 if utility_fn and actions provided (for verification).
    exact_q_h2 = 0.0
    if utility_fn is not None and available_actions is not None:
        from ..exp6_3.exact_mpc import exact_mpc
        v_h1 = exact_mpc(next_graph, z, available_actions, utility_fn, horizon=1, gamma=0.9)
        from ...runtime.analytical_utility import AnalyticalUtilityOracle
        oracle = AnalyticalUtilityOracle()
        mt, u, v, params = action
        try:
            delta_add = oracle.delta_for_mutation(graph, z, mt, u, v, params)
        except Exception:
            delta_add = 0.0
        exact_q_h2 = delta_add + 0.9 * v_h1.total_value

    return CausalTarget(
        delta_n_components=delta_n_comp,
        n_components_after=comp_after.n_components,
        threshold_reached=threshold_reached,
        steps_to_threshold=steps_to_threshold,
        exact_bonus=exact_bonus,
        exact_q_h2=exact_q_h2,
    )
