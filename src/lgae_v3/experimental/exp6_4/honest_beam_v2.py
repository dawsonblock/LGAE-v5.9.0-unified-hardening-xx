"""Honest beam search v2: uses causal intermediate predictions.

Architecture (NO information leakage):
    Q_hat(S,a) = delta_U_additive(S,a) + gamma * V_bonus_hat(S')

Where V_bonus_hat comes from a BonusModel that predicts the non-additive
bonus from legitimate online-observable structural features.

The exact utility_fn is NOT accessible during search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_3.exact_mpc import apply_action
from .model_ladder import BonusModel, B0Zero
from .structural_features import extract_structural_features
from ...runtime.analytical_utility import AnalyticalUtilityOracle


@dataclass
class HonestBeamResultV2:
    """Result of honest beam search v2."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    first_action_identity: object = None  # ActionIdentity or None
    best_sequence: list[tuple[str, int, int, dict]] = field(default_factory=list)
    total_value: float = float("-inf")
    nodes_expanded: int = 0
    horizon: int = 0
    beam_width: int = 0
    all_first_action_values: dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    wall_clock_seconds: float = 0.0


def honest_beam_search_v2(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    bonus_model: BonusModel,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    beam_width: int = 5,
    threshold: int = 1,
) -> HonestBeamResultV2:
    """Beam search using ONLY additive exact + learned bonus.

    NO access to utility_fn during search.
    """
    import time
    t_start = time.time()

    result = HonestBeamResultV2(
        horizon=horizon, beam_width=beam_width,
        model_name=bonus_model.name,
    )

    if horizon == 0 or not available_actions:
        result.wall_clock_seconds = time.time() - t_start
        return result

    oracle = AnalyticalUtilityOracle()

    # Beam: (graph, cum_val, sequence, first_action_key)
    beam: list[tuple[GraphBuffers, float, list[tuple], str]] = [
        (graph, 0.0, [], "")
    ]

    first_values: dict[str, float] = {}

    for depth in range(horizon):
        candidates: list[tuple[float, GraphBuffers, list[tuple], str]] = []

        for current_graph, cum_val, seq, first_key in beam:
            for action in available_actions:
                mt, u, v, params = action

                # 1. Exact additive ΔU (O(1), no global structure).
                try:
                    delta_add = oracle.delta_for_mutation(
                        current_graph, z, mt, u, v, params
                    )
                except Exception:
                    delta_add = 0.0

                # Apply action to get S'.
                next_graph = apply_action(current_graph, action)

                # 2. Learned bonus prediction from structural features.
                # Pass (current_graph, action) — the model was trained on
                # (S, a) → bonus(S'), NOT (S', a) → bonus(S').
                v_bonus = bonus_model.predict_bonus(
                    current_graph, z, action,
                    threshold=threshold, horizon=horizon - depth,
                )

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
        # candidates are (total, graph, seq, key); beam needs (graph, total, seq, key)
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
            from ..exp6_3.exact_mpc import ActionIdentity
            result.first_action_identity = ActionIdentity.from_action(a)

    result.wall_clock_seconds = time.time() - t_start
    return result
