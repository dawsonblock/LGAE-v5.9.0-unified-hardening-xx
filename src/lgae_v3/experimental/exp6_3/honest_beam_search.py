"""Honest beam search: additive exact + learned bonus only.

This is the FIXED beam search from the information leakage audit.
The key change: the beam search does NOT have access to utility_fn.
It uses only:
  1. AnalyticalUtilityOracle for exact additive ΔU (O(1) per candidate)
  2. BonusPredictor for learned non-additive bonus prediction

The exact total utility is used ONLY for:
  - Training label generation
  - Finalist replay and verification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from .exact_mpc import apply_action
from .split_utility import (
    compute_additive_utility, compute_bonus, compute_total_utility,
    BonusPredictor, ZeroBonusPredictor,
)
from ...runtime.analytical_utility import AnalyticalUtilityOracle


@dataclass
class HonestBeamResult:
    """Result of honest beam search (no utility_fn leakage)."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    first_action_identity: object = None  # ActionIdentity or None
    best_sequence: list[tuple[str, int, int, dict]] = field(default_factory=list)
    total_value: float = float("-inf")
    nodes_expanded: int = 0
    horizon: int = 0
    beam_width: int = 0
    all_first_action_values: dict[str, float] = field(default_factory=dict)
    used_exact_bonus: bool = False  # Flag: did we cheat?


def honest_beam_search(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    bonus_predictor: BonusPredictor,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    beam_width: int = 5,
    lambda_conn: float = 30.0,
    threshold: int = 1,
) -> HonestBeamResult:
    """Beam search using ONLY additive exact + learned bonus.

    NO access to the total utility function during search.
    The bonus is PREDICTED, not computed.

    Q_hat(S, a) = delta_U_additive(S, a) + gamma * V_bonus_hat(S')

    Where:
      delta_U_additive = exact analytical O(1) per candidate
      V_bonus_hat = learned prediction of future non-additive bonus
    """
    result = HonestBeamResult(horizon=horizon, beam_width=beam_width)

    if horizon == 0 or not available_actions:
        return result

    oracle = AnalyticalUtilityOracle()

    # Beam entries: (graph, cum_val, sequence, first_action_key)
    beam: list[tuple[GraphBuffers, float, list[tuple], str]] = [
        (graph, 0.0, [], "")
    ]

    first_values: dict[str, float] = {}

    for depth in range(horizon):
        candidates: list[tuple[float, GraphBuffers, list[tuple], str]] = []

        for current_graph, cum_val, seq, first_key in beam:
            for action in available_actions:
                mt, u, v, params = action

                # 1. Exact additive ΔU (O(1), no global structure needed).
                try:
                    delta_add = oracle.delta_for_mutation(
                        current_graph, z, mt, u, v, params
                    )
                except Exception:
                    delta_add = 0.0

                # Apply action to get S'.
                next_graph = apply_action(current_graph, action)

                # 2. Learned bonus prediction (NOT exact computation).
                if depth == horizon - 1:
                    # At the horizon boundary, predict future bonus.
                    v_bonus = bonus_predictor.predict(next_graph, z)
                else:
                    # At intermediate steps, predict bonus of S'
                    # to guide pruning toward states with better bonus.
                    v_bonus = bonus_predictor.predict(next_graph, z)

                step_val = (gamma ** depth) * delta_add + (gamma ** (depth + 1)) * v_bonus
                total = cum_val + step_val
                new_seq = seq + [action]
                new_first = first_key if first_key else f"{action[0]}_{action[1]}_{action[2]}"

                if depth == 0:
                    if new_first not in first_values or total > first_values[new_first]:
                        first_values[new_first] = total

                candidates.append((total, next_graph, new_seq, new_first))
                result.nodes_expanded += 1

        # Retain top beam_width.
        candidates.sort(key=lambda x: -x[0])
        beam = [(g, v, s, k) for v, g, s, k in candidates[:beam_width]]

    if beam:
        # beam stores (graph, total, seq, key).
        best_graph, best_val, best_seq, best_key = beam[0]
        result.total_value = best_val
        result.best_sequence = best_seq
        result.all_first_action_values = first_values
        if best_seq:
            a = best_seq[0]
            result.first_action = (a[0], a[1], a[2])
            from .exact_mpc import ActionIdentity
            result.first_action_identity = ActionIdentity.from_action(a)

    return result
