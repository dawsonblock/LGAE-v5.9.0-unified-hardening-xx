"""Selective hybrid structural planner for exp6.8.1.

Arbitration layer: use the learned planner only when its prediction
is confident enough and the margin over greedy is sufficient.

  use learned action only if:
    sigma < tau_sigma  (uncertainty below threshold)
    AND
    Q_hat(learned) - Q_hat(greedy) > tau_margin  (margin above threshold)

  otherwise: fall back to greedy.

This creates a coverage-vs-risk curve: as tau_sigma decreases,
the learned planner is used on fewer tasks (lower coverage) but
with higher reliability (lower regret).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_3.exact_mpc import (
    apply_action, apply_action_with_status, ActionIdentity,
    greedy_one_step, exact_mpc,
)
from ..exp6_7.multi_operator_candidates import generate_multi_operator_candidates
from ..exp6_6.objective_spec import ObjectiveSpec
from ..exp6_8.transition_model import ConsequentialStateModel
from ..exp6_8.recursive_planner import recursive_causal_mpc, evaluate_objective_on_state
from .split_state import SplitStructuralState, LEARNED_STATE_DIM
from .learned_state_model import LearnedStateModel


@dataclass
class HybridPlanResult:
    """Result of selective hybrid planning."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    first_action_identity: Optional[ActionIdentity] = None
    best_sequence: list[tuple] = field(default_factory=list)
    total_value: float = float("-inf")
    all_first_action_values: dict[str, float] = field(default_factory=dict)
    nodes_expanded: int = 0
    planner_name: str = "hybrid"
    # Which planner was used: "learned" or "greedy_fallback".
    source: str = "greedy_fallback"
    # Uncertainty of the learned prediction.
    uncertainty: float = 0.0
    # Margin of learned over greedy.
    margin: float = 0.0
    # Whether arbitration chose learned.
    used_learned: bool = False


def selective_hybrid_plan(
    graph: GraphBuffers,
    z: torch.Tensor,
    candidates: list[tuple],
    model,  # ConsequentialStateModel or LearnedStateModel
    objective: ObjectiveSpec,
    config,
    utility_fn,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    beam_width: int = 3,
    threshold: int = 1,
    tau_sigma: float = 1.0,   # uncertainty threshold
    tau_margin: float = 0.0,  # margin threshold
) -> HybridPlanResult:
    """Selective hybrid planner with arbitration.

    1. Compute greedy action (exact, no foresight).
    2. Compute recursive causal MPC action (learned foresight).
    3. Estimate uncertainty of the learned prediction.
    4. If uncertainty < tau_sigma and margin > tau_margin: use learned.
    5. Otherwise: fall back to greedy.
    """
    result = HybridPlanResult()

    if not candidates:
        return result

    # Filter valid actions.
    valid_actions = []
    for action in candidates:
        st = apply_action_with_status(graph, action)
        if st.status == "VALID":
            valid_actions.append(action)
    if not valid_actions:
        return result

    # 1. Greedy.
    greedy = greedy_one_step(graph, z, candidates, utility_fn)

    # 2. Recursive causal MPC.
    recursive = recursive_causal_mpc(
        graph, z, candidates, model, objective, config,
        horizon=horizon, gamma=gamma, beam_width=beam_width,
        threshold=threshold, use_predicted=True,
    )

    # 3. Estimate uncertainty: average prediction std over candidates.
    uncertainties = []
    state = SplitStructuralState.from_graph(graph)
    for action in valid_actions[:10]:  # Sample for efficiency.
        if isinstance(model, LearnedStateModel):
            std = model.predict_uncertainty(graph, z, state, action, threshold=threshold)
        else:
            learned_z = state.learned.to_array()
            std = model.predict_z_std(graph, z, learned_z, action, threshold=threshold)
        uncertainties.append(std)
    avg_uncertainty = float(np.mean(uncertainties)) if uncertainties else 1e9

    # 4. Compute margin: Q_hat(learned) - Q_hat(greedy).
    greedy_id = None
    if greedy.first_action[0]:
        # Find greedy's action in best_sequence.
        if greedy.best_sequence:
            greedy_id = ActionIdentity.from_action(greedy.best_sequence[0])
        else:
            greedy_id = ActionIdentity.from_action(
                (greedy.first_action[0], greedy.first_action[1], greedy.first_action[2], {})
            )

    recursive_id = recursive.first_action_identity
    recursive_val = recursive.total_value
    greedy_val = greedy.total_value
    margin = float(recursive_val - greedy_val)

    # 5. Arbitration.
    use_learned = (avg_uncertainty < tau_sigma and margin > tau_margin
                   and recursive_id is not None)

    if use_learned:
        result.first_action = recursive.first_action
        result.first_action_identity = recursive_id
        result.best_sequence = recursive.best_sequence
        result.total_value = recursive.total_value
        result.all_first_action_values = recursive.all_first_action_values
        result.nodes_expanded = recursive.nodes_expanded + greedy.nodes_expanded
        result.source = "learned"
        result.used_learned = True
    else:
        result.first_action = greedy.first_action
        result.first_action_identity = greedy_id
        result.best_sequence = greedy.best_sequence if greedy.best_sequence else []
        result.total_value = greedy.total_value
        result.all_first_action_values = greedy.all_first_action_values
        result.nodes_expanded = recursive.nodes_expanded + greedy.nodes_expanded
        result.source = "greedy_fallback"
        result.used_learned = False

    result.uncertainty = avg_uncertainty
    result.margin = margin

    return result


def run_coverage_sweep(
    graph: GraphBuffers,
    z: torch.Tensor,
    candidates: list[tuple],
    model,  # ConsequentialStateModel or LearnedStateModel
    objective: ObjectiveSpec,
    config,
    utility_fn,
    exact_result,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    beam_width: int = 3,
    threshold: int = 1,
    tau_sigma_values: list[float] = None,
    tau_margin: float = 0.0,
) -> list[dict]:
    """Run the hybrid planner at multiple uncertainty thresholds.

    Returns a coverage-vs-risk curve: for each tau_sigma, records
    coverage (fraction of tasks where learned was used), recovery,
    regret, and savings.
    """
    if tau_sigma_values is None:
        tau_sigma_values = [0.5, 1.0, 2.0, 5.0, 1e9]  # 1e9 = always use learned

    results = []
    for tau_sigma in tau_sigma_values:
        plan = selective_hybrid_plan(
            graph, z, candidates, model, objective, config, utility_fn,
            horizon=horizon, gamma=gamma, beam_width=beam_width,
            threshold=threshold, tau_sigma=tau_sigma, tau_margin=tau_margin,
        )

        # Compute recovery and regret.
        exact_id = exact_result.first_action_identity
        plan_id = plan.first_action_identity
        agree = 1.0 if (exact_id and plan_id and exact_id == plan_id) else 0.0

        exact_val = exact_result.all_first_action_values.get(
            exact_id.key if exact_id else "", exact_result.total_value,
        )
        plan_val = exact_result.all_first_action_values.get(
            plan_id.key if plan_id else "", plan.total_value,
        )
        regret = float(abs(exact_val - plan_val))
        norm_regret = regret / (abs(exact_val) + 1e-6)

        results.append({
            "tau_sigma": tau_sigma,
            "used_learned": plan.used_learned,
            "recovery": agree,
            "regret": regret,
            "normalized_regret": norm_regret,
            "savings": 1.0 - plan.nodes_expanded / max(exact_result.nodes_expanded, 1),
            "uncertainty": plan.uncertainty,
            "margin": plan.margin,
        })

    return results
