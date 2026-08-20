"""Uncertainty-aware adaptive beam search for exp6.5.

Uses ensemble uncertainty to adaptively adjust beam width:
  low uncertainty  → narrow beam → faster search
  high uncertainty → wider beam  → more exact exploration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
import time

from ...types import GraphBuffers
from ..exp6_3.exact_mpc import apply_action
from .observable_features import extract_observable_features
from ...runtime.analytical_utility import AnalyticalUtilityOracle


@dataclass
class AdaptiveBeamResult:
    """Result of adaptive beam search."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    first_action_identity: object = None  # ActionIdentity or None
    best_sequence: list[tuple[str, int, int, dict]] = field(default_factory=list)
    total_value: float = float("-inf")
    nodes_expanded: int = 0
    horizon: int = 0
    beam_width_used: float = 0.0
    avg_uncertainty: float = 0.0
    wall_clock_seconds: float = 0.0
    all_first_action_values: dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    n_exact_fallbacks: int = 0


def adaptive_beam_search(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    model,  # DecomposedModel or similar with predict_residual
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    min_beam_width: int = 2,
    max_beam_width: int = 10,
    uncertainty_threshold: float = 5.0,
    threshold: int = 1,
) -> AdaptiveBeamResult:
    """Adaptive beam search with uncertainty-aware width.

    The beam width is adjusted based on ensemble uncertainty:
    - If uncertainty < threshold: use min_beam_width
    - If uncertainty > threshold: scale up toward max_beam_width

    This gives:
      low uncertainty → narrow beam → faster search
      high uncertainty → wider beam → more exploration
    """
    t_start = time.time()

    result = AdaptiveBeamResult(
        horizon=horizon,
        beam_width_used=float(min_beam_width),
        model_name=getattr(model, "name", "unknown"),
    )

    if horizon == 0 or not available_actions:
        result.wall_clock_seconds = time.time() - t_start
        return result

    oracle = AnalyticalUtilityOracle()

    # Check if model has uncertainty estimation.
    has_uncertainty = hasattr(model, "predict_residual_std")

    # Compute uncertainty for each candidate at depth 0.
    uncertainties = []
    for action in available_actions:
        if has_uncertainty:
            std = model.predict_residual_std(
                graph, z, action, threshold=threshold, horizon=horizon
            )
        else:
            std = 0.0
        uncertainties.append(std)

    avg_unc = float(np.mean(uncertainties)) if uncertainties else 0.0
    result.avg_uncertainty = avg_unc

    # Determine beam width based on uncertainty.
    if avg_unc > uncertainty_threshold:
        # Scale up beam width.
        scale = min(1.0, avg_unc / (2 * uncertainty_threshold))
        beam_width = int(min_beam_width + scale * (max_beam_width - min_beam_width))
    else:
        beam_width = min_beam_width

    result.beam_width_used = float(beam_width)

    # Run beam search with the adaptive width.
    beam: list[tuple[GraphBuffers, float, list[tuple], str]] = [
        (graph, 0.0, [], "")
    ]

    first_values: dict[str, float] = {}

    for depth in range(horizon):
        candidates: list[tuple[float, GraphBuffers, list[tuple], str]] = []

        for current_graph, cum_val, seq, first_key in beam:
            for action in available_actions:
                mt, u, v, params = action

                # Exact additive delta.
                try:
                    delta_add = oracle.delta_for_mutation(
                        current_graph, z, mt, u, v, params
                    )
                except Exception:
                    delta_add = 0.0

                # Learned residual prediction.
                v_residual = model.predict_residual(
                    current_graph, z, action,
                    threshold=threshold, horizon=horizon - depth,
                )

                step_val = (gamma ** depth) * delta_add + (gamma ** (depth + 1)) * v_residual
                total = cum_val + step_val
                new_seq = seq + [action]
                new_first = first_key if first_key else f"{action[0]}_{action[1]}_{action[2]}"

                if depth == 0:
                    if new_first not in first_values or total > first_values[new_first]:
                        first_values[new_first] = total

                candidates.append((total, apply_action(current_graph, action), new_seq, new_first))
                result.nodes_expanded += 1

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
            from ..exp6_3.exact_mpc import ActionIdentity
            result.first_action_identity = ActionIdentity.from_action(a)

    result.wall_clock_seconds = time.time() - t_start
    return result
