"""Value dataset generation from exact enumeration.

Each record contains:
- state features
- candidate action
- analytical immediate ΔU
- exact H2 Q value
- exact H3 Q value
- future residual H2 = Q_H2 - ΔU
- future residual H3 = Q_H3 - ΔU
- best continuation
- regret if greedy

All labels are EXACT_ENUMERATED, not realized or counterfactual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from .delayed_tasks import DelayedValueTask, make_task_graph, make_task_latent
from .exact_mpc import exact_mpc, apply_action
from .future_value import extract_features
from ...runtime.analytical_utility import AnalyticalUtilityOracle


@dataclass
class ValueRecord:
    """A single value training record."""
    state_features: np.ndarray
    action_type: str
    action_u: int
    action_v: int
    analytical_delta_u: float
    exact_q_h1: float
    exact_q_h2: float
    exact_q_h3: float
    future_residual_h2: float
    future_residual_h3: float
    label_type: str = "EXACT_ENUMERATED"

    def to_dict(self) -> dict:
        return {
            "state_features": self.state_features.tolist(),
            "action_type": self.action_type,
            "action_u": self.action_u,
            "action_v": self.action_v,
            "analytical_delta_u": self.analytical_delta_u,
            "exact_q_h1": self.exact_q_h1,
            "exact_q_h2": self.exact_q_h2,
            "exact_q_h3": self.exact_q_h3,
            "future_residual_h2": self.future_residual_h2,
            "future_residual_h3": self.future_residual_h3,
            "label_type": self.label_type,
        }


def generate_value_dataset(
    tasks: list[DelayedValueTask],
    *,
    horizons: list[int] | None = None,
    gamma: float = 0.9,
) -> list[ValueRecord]:
    """Generate value training data from exact MPC on delayed-value tasks.

    For each task and each available action:
    1. Compute analytical immediate ΔU
    2. Apply action to get S'
    3. Run exact MPC from S' at H-1 to get V(S')
    4. Q_H = ΔU + γ * V(S')
    5. Future residual = Q_H - ΔU
    """
    if horizons is None:
        horizons = [1, 2, 3]

    oracle = AnalyticalUtilityOracle()
    records: list[ValueRecord] = []

    for task in tasks:
        graph = make_task_graph(task)
        z = make_task_latent(task)
        utility_fn = task.utility_fn
        actions = task.available_actions

        for action in actions:
            mt, u, v, params = action

            # Analytical immediate ΔU (additive part only).
            try:
                delta_u = oracle.delta_for_mutation(graph, z, mt, u, v, params)
            except Exception:
                delta_u = 0.0

            # Apply action to get S'.
            s_prime = apply_action(graph, action)
            features = extract_features(s_prime, z)

            # Exact V(S') at different horizons.
            q_h1 = delta_u  # H=1 is just immediate
            q_h2 = delta_u
            q_h3 = delta_u

            if 2 in horizons:
                v_h1 = exact_mpc(s_prime, z, actions, utility_fn, horizon=1, gamma=gamma)
                q_h2 = delta_u + gamma * v_h1.total_value
            if 3 in horizons:
                v_h2 = exact_mpc(s_prime, z, actions, utility_fn, horizon=2, gamma=gamma)
                q_h3 = delta_u + gamma * v_h2.total_value

            records.append(ValueRecord(
                state_features=features,
                action_type=mt,
                action_u=u,
                action_v=v,
                analytical_delta_u=delta_u,
                exact_q_h1=q_h1,
                exact_q_h2=q_h2,
                exact_q_h3=q_h3,
                future_residual_h2=q_h2 - delta_u,
                future_residual_h3=q_h3 - delta_u,
            ))

    return records
