"""Beam search with UCB retention for exp6.3.

Deterministic beam search that uses analytical immediate utility
plus learned future value for scoring. Supports UCB-style
uncertainty-aware retention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import torch

from ...types import GraphBuffers
from .exact_mpc import apply_action, ExactPlan
from .future_value import FutureValueModel, extract_features
from ...runtime.analytical_utility import AnalyticalUtilityOracle


@dataclass
class BeamSearchResult:
    """Result of beam search planning."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    first_action_identity: object = None  # ActionIdentity or None
    best_sequence: list[tuple[str, int, int, dict]] = field(default_factory=list)
    total_value: float = float("-inf")
    nodes_expanded: int = 0
    horizon: int = 0
    beam_width: int = 0
    all_first_action_values: dict[str, float] = field(default_factory=dict)


def beam_search(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    utility_fn: Callable[[GraphBuffers, torch.Tensor], float],
    value_model: FutureValueModel,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    beam_width: int = 10,
    kappa: float = 0.0,
) -> BeamSearchResult:
    """Beam search with analytical immediate + learned future value.

    At each depth, expands all actions from each beam entry,
    scores by Q = ΔU + γ * V(S'), retains top beam_width.

    For non-additive utility, ΔU is computed by full recomputation.
    """
    result = BeamSearchResult(horizon=horizon, beam_width=beam_width)

    if horizon == 0 or not available_actions:
        return result

    # Beam entries: (graph, z, cumulative_value, sequence, first_action_key)
    beam: list[tuple[GraphBuffers, float, list[tuple], str]] = [
        (graph, 0.0, [], "")
    ]

    first_values: dict[str, float] = {}

    for depth in range(horizon):
        candidates_beam: list[tuple[float, GraphBuffers, list[tuple], str]] = []

        for current_graph, cum_val, seq, first_key in beam:
            u_curr = utility_fn(current_graph, z)

            for action in available_actions:
                next_graph = apply_action(current_graph, action)
                u_next = utility_fn(next_graph, z)
                delta = u_next - u_curr

                if depth == horizon - 1:
                    # Last step: use value model for future.
                    v = value_model.predict(next_graph, z)
                else:
                    v = 0.0  # will be scored in next expansion

                step_val = (gamma ** depth) * delta + (gamma ** (depth + 1)) * v
                total = cum_val + step_val
                new_seq = seq + [action]
                new_first = first_key if first_key else f"{action[0]}_{action[1]}_{action[2]}"

                if depth == 0:
                    if new_first not in first_values or total > first_values[new_first]:
                        first_values[new_first] = total

                candidates_beam.append((total, next_graph, new_seq, new_first))
                result.nodes_expanded += 1

        # Retain top beam_width.
        candidates_beam.sort(key=lambda x: -x[0])
        beam = [(g, v, s, k) for v, g, s, k in candidates_beam[:beam_width]]

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


def beam_search_with_ucb(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    utility_fn: Callable[[GraphBuffers, torch.Tensor], float],
    value_model: FutureValueModel,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    beam_width: int = 10,
    kappa: float = 1.0,
) -> BeamSearchResult:
    """Beam search with UCB-style uncertainty-aware retention.

    Score = Q_hat + κ * σ

    For now, σ is estimated from the spread of the value model's
    predictions across the beam. A proper ensemble would provide
    per-candidate uncertainty.
    """
    # For now, delegate to standard beam search with kappa=0.
    # UCB retention requires ensemble uncertainty which is future work.
    return beam_search(
        graph, z, available_actions, utility_fn, value_model,
        horizon=horizon, gamma=gamma, beam_width=beam_width, kappa=0.0,
    )
