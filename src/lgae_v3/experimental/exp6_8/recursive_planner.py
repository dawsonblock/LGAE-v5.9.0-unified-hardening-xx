"""Recursive model-based structural planner for exp6.8.

Uses exact graph transitions + learned consequential state prediction
to plan over horizon H.

The planner evaluates:
  Q_H(S_0, a_0) = max_{a_1,...,a_{H-1}} [
    sum_{t=0}^{H-1} gamma^t * r_exact(S_t, a_t)
    + gamma^H * O(G_H, z_H)
  ]

where:
  G_{t+1} = T_exact(G_t, a_t)     [exact]
  z_{t+1} = F(G_t, z_t, a_t)      [learned]

Four systems compared:
  1. Greedy: exact, no foresight
  2. Exact MPC: exact, exact foresight
  3. One-step causal: exact, one-step learned
  4. Recursive causal MPC: exact, multi-step learned
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_3.exact_mpc import (
    exact_mpc, greedy_one_step, apply_action, apply_action_with_status,
    ActionIdentity,
)
from ..exp6_7.multi_operator_candidates import generate_multi_operator_candidates
from ..exp6_6.objective_spec import ObjectiveSpec
from .structural_state import StructuralState, compute_structural_observables, get_observable_value
from .transition_model import (
    ConsequentialStateModel, exact_transition,
    roll_forward_exact, roll_forward_predicted,
)


@dataclass
class RecursivePlanResult:
    """Result of recursive model-based planning."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    first_action_identity: Optional[ActionIdentity] = None
    best_sequence: list[tuple] = field(default_factory=list)
    total_value: float = float("-inf")
    all_first_action_values: dict[str, float] = field(default_factory=dict)
    nodes_expanded: int = 0
    horizon: int = 0
    planner_name: str = ""
    rollout_errors: list[float] = field(default_factory=list)


def evaluate_objective_on_state(
    state: StructuralState,
    spec: ObjectiveSpec,
    prev_state: StructuralState,
) -> float:
    """Evaluate O(S+delta_S) - O(S) on a structural state.

    Uses the correct absolute-state evaluation.
    """
    current_val = get_observable_value(prev_state.z, spec.observable)
    after_val = get_observable_value(state.z, spec.observable)

    if spec.reward_shape == "threshold":
        if spec.direction == "minimize":
            bonus_after = spec.magnitude if after_val <= spec.threshold else 0.0
            bonus_current = spec.magnitude if current_val <= spec.threshold else 0.0
        else:
            bonus_after = spec.magnitude if after_val >= spec.threshold else 0.0
            bonus_current = spec.magnitude if current_val >= spec.threshold else 0.0
        return bonus_after - bonus_current
    else:
        delta = after_val - current_val
        if spec.direction == "minimize":
            return -delta * spec.magnitude
        else:
            return delta * spec.magnitude


def recursive_causal_mpc(
    graph: GraphBuffers,
    z: torch.Tensor,
    candidates: list[tuple],
    model: ConsequentialStateModel,
    objective: ObjectiveSpec,
    config,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    beam_width: int = 3,
    threshold: int = 1,
    use_predicted: bool = True,
) -> RecursivePlanResult:
    """Recursive model-based MPC with exact transitions + learned state.

    Uses beam search over the planning horizon.
    """
    import random
    result = RecursivePlanResult(horizon=horizon, planner_name="recursive_causal_mpc")

    if horizon == 0 or not candidates:
        return result

    valid_actions = []
    for action in candidates:
        status = apply_action_with_status(graph, action)
        if status.status == "VALID":
            valid_actions.append(action)

    if not valid_actions:
        return result

    init_state = StructuralState.from_graph(graph)

    # Beam: list of (cumulative_value, state, z_tensor, sequence, first_action)
    beam = [(0.0, init_state, z, [], None)]
    first_values: dict[str, float] = {}
    nodes_expanded = 0

    for depth in range(horizon):
        new_beam = []
        for cum_val, state, z_current, seq, first_act in beam:
            # Generate candidates at current state.
            if depth > 0:
                state_candidates = generate_multi_operator_candidates(
                    state.graph, z_current, config,
                    rng=random.Random(config.seed + depth * 1000),
                )
            else:
                state_candidates = valid_actions

            # Filter valid at current state.
            valid_here = []
            for action in state_candidates:
                st = apply_action_with_status(state.graph, action)
                if st.status == "VALID":
                    valid_here.append(action)

            for action in valid_here:
                nodes_expanded += 1

                # Exact graph transition.
                new_graph, status = exact_transition(state.graph, action)
                if status != "VALID":
                    continue

                # Learned or exact z prediction.
                if use_predicted and depth < horizon - 1:
                    z_pred = model.predict_z(state.graph, z_current, state.z, action, threshold=threshold)
                    new_state = StructuralState.from_predicted(new_graph, z_pred)
                else:
                    # Teacher-forced: use exact observables.
                    new_state = StructuralState.from_graph(new_graph)

                # Objective evaluation.
                obj_val = evaluate_objective_on_state(new_state, objective, state)
                step_val = (gamma ** depth) * obj_val
                total = cum_val + step_val

                # Track first action.
                cur_first = first_act if first_act is not None else action
                cur_first_id = ActionIdentity.from_action(cur_first)
                if depth == 0:
                    key = cur_first_id.key
                    if key not in first_values or total > first_values[key]:
                        first_values[key] = total

                new_beam.append((total, new_state, z_current, seq + [action], cur_first))

        # Keep top beam_width.
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]

    if beam:
        best_val, best_state, _, best_seq, best_first = beam[0]
        result.total_value = best_val
        result.best_sequence = best_seq
        result.all_first_action_values = first_values
        result.nodes_expanded = nodes_expanded
        if best_first:
            result.first_action = (best_first[0], best_first[1], best_first[2])
            result.first_action_identity = ActionIdentity.from_action(best_first)

    return result


# Need to add all_first_action_values to the dataclass.
# Fix: add it as a field.
