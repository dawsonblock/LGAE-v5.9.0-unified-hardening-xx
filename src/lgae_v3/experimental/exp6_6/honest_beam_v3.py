"""Honest beam search v3 for exp6.6.

Same no-leakage architecture as v2, but passes the ObjectiveSpec
to the model so objective-conditioned architectures can use it.

No utility_fn access. Only exact additive delta + learned residual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
import time

from ...types import GraphBuffers
from ..exp6_3.exact_mpc import apply_action
from ..exp6_5.observable_features import extract_observable_features
from ...runtime.analytical_utility import AnalyticalUtilityOracle
from .objective_spec import ObjectiveSpec


@dataclass
class HonestBeamResultV3:
    """Result of honest beam search v3."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    best_sequence: list[tuple[str, int, int, dict]] = field(default_factory=list)
    total_value: float = float("-inf")
    nodes_expanded: int = 0
    horizon: int = 0
    beam_width: int = 0
    wall_clock_seconds: float = 0.0
    all_first_action_values: dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    # Candidate-level predictions for calibration analysis.
    candidate_predictions: dict[str, float] = field(default_factory=dict)
    candidate_uncertainties: dict[str, float] = field(default_factory=dict)


def honest_beam_search_v3(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    model,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    beam_width: int = 2,
    threshold: int = 1,
    objective: ObjectiveSpec | None = None,
) -> HonestBeamResultV3:
    """Honest beam search with objective spec support.

    Q_hat = delta_U_additive + gamma * V_residual_hat(S')
    - delta_U_additive: exact analytical O(1)
    - V_residual_hat: learned, may use objective spec
    - NO utility_fn access
    """
    t_start = time.time()

    result = HonestBeamResultV3(
        horizon=horizon,
        beam_width=beam_width,
        model_name=getattr(model, "name", "unknown"),
    )

    if horizon == 0 or not available_actions:
        result.wall_clock_seconds = time.time() - t_start
        return result

    oracle = AnalyticalUtilityOracle()

    # Track candidate-level predictions at depth 0 for calibration.
    candidate_preds: dict[str, float] = {}
    candidate_uncs: dict[str, float] = {}

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
                    objective=objective,
                )

                step_val = (gamma ** depth) * delta_add + (gamma ** (depth + 1)) * v_residual
                total = cum_val + step_val
                new_seq = seq + [action]
                new_first = first_key if first_key else f"{action[0]}_{action[1]}_{action[2]}"

                if depth == 0:
                    if new_first not in first_values or total > first_values[new_first]:
                        first_values[new_first] = total
                    candidate_preds[new_first] = v_residual
                    if hasattr(model, "predict_residual_std"):
                        candidate_uncs[new_first] = model.predict_residual_std(
                            current_graph, z, action,
                            threshold=threshold, horizon=horizon - depth,
                        )

                candidates.append((total, apply_action(current_graph, action), new_seq, new_first))
                result.nodes_expanded += 1

        candidates.sort(key=lambda x: -x[0])
        beam = [(g, v, s, k) for v, g, s, k in candidates[:beam_width]]

    if beam:
        best_val, best_graph, best_seq, best_key = beam[0]
        result.total_value = best_val
        result.best_sequence = best_seq
        result.all_first_action_values = first_values
        result.candidate_predictions = candidate_preds
        result.candidate_uncertainties = candidate_uncs
        if best_seq:
            a = best_seq[0]
            result.first_action = (a[0], a[1], a[2])

    result.wall_clock_seconds = time.time() - t_start
    return result
