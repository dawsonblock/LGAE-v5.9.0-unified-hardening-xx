"""LCB-margin arbitration for exp6.8.2.

Instead of separate sigma/margin thresholds, use a single principled rule:

  margin = Q_hat(learned) - Q_hat(greedy)
  LCB(margin) = mu_margin - kappa * sigma_margin
  use learned only if LCB(margin) > 0

This directly asks: "Is the learned action still predicted to beat
greedy under uncertainty?"

The kappa parameter is chosen on a held-out calibration split,
not on the test set.
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
from ..exp6_8.recursive_planner import recursive_causal_mpc
from ..exp6_8_1.split_state import SplitStructuralState
from .ensemble_model import EnsembleLearnedModel


@dataclass
class LCBPlanResult:
    """Result of LCB-margin selective planning."""
    first_action: tuple[str, int, int] = ("", 0, 0)
    first_action_identity: Optional[ActionIdentity] = None
    best_sequence: list[tuple] = field(default_factory=list)
    total_value: float = float("-inf")
    all_first_action_values: dict[str, float] = field(default_factory=dict)
    nodes_expanded: int = 0
    planner_name: str = "lcb_hybrid"
    source: str = "greedy_fallback"
    used_learned: bool = False
    # Ensemble Q predictions.
    q_learned_mean: float = 0.0
    q_learned_std: float = 0.0
    q_greedy: float = 0.0
    margin_mean: float = 0.0
    margin_std: float = 0.0
    lcb_margin: float = 0.0
    kappa: float = 1.0


def lcb_hybrid_plan(
    graph: GraphBuffers,
    z: torch.Tensor,
    candidates: list[tuple],
    model: EnsembleLearnedModel,
    objective: ObjectiveSpec,
    config,
    utility_fn,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
    beam_width: int = 3,
    threshold: int = 1,
    kappa: float = 1.0,  # LCB confidence parameter
) -> LCBPlanResult:
    """LCB-margin selective hybrid planner.

    1. Compute greedy action and its Q value.
    2. Compute recursive causal MPC action.
    3. For both actions, compute ensemble Q predictions.
    4. margin = Q_learned - Q_greedy
    5. LCB(margin) = mu_margin - kappa * sigma_margin
    6. Use learned only if LCB(margin) > 0.
    """
    result = LCBPlanResult(kappa=kappa)

    if not candidates:
        return result

    valid_actions = []
    for action in candidates:
        st = apply_action_with_status(graph, action)
        if st.status == "VALID":
            valid_actions.append(action)
    if not valid_actions:
        return result

    # 1. Greedy.
    greedy = greedy_one_step(graph, z, candidates, utility_fn)

    # 2. Recursive causal MPC (uses ensemble mean via predict_z).
    recursive = recursive_causal_mpc(
        graph, z, candidates, model, objective, config,
        horizon=horizon, gamma=gamma, beam_width=beam_width,
        threshold=threshold, use_predicted=True,
    )

    # 3. Ensemble Q predictions for learned and greedy first actions.
    state = SplitStructuralState.from_graph(graph)

    # Q for learned action.
    learned_action = None
    if recursive.first_action_identity:
        for action in valid_actions:
            aid = ActionIdentity.from_action(action)
            if aid == recursive.first_action_identity:
                learned_action = action
                break

    if learned_action:
        q_learned_mean, q_learned_std = model.predict_q_ensemble(
            graph, z, state, learned_action, objective, threshold=threshold,
        )
    else:
        q_learned_mean, q_learned_std = recursive.total_value, 0.0

    # Q for greedy action.
    greedy_action = None
    if greedy.first_action[0]:
        for action in valid_actions:
            if (action[0] == greedy.first_action[0]
                    and action[1] == greedy.first_action[1]
                    and action[2] == greedy.first_action[2]):
                greedy_action = action
                break

    if greedy_action:
        q_greedy_mean, q_greedy_std = model.predict_q_ensemble(
            graph, z, state, greedy_action, objective, threshold=threshold,
        )
    else:
        q_greedy_mean, q_greedy_std = greedy.total_value, 0.0

    # 4. Margin.
    margin_mean = q_learned_mean - q_greedy_mean
    margin_std = float(np.sqrt(q_learned_std**2 + q_greedy_std**2))

    # 5. LCB.
    lcb = margin_mean - kappa * margin_std

    # 6. Arbitration.
    use_learned = (lcb > 0 and recursive.first_action_identity is not None)

    # Get action identities.
    greedy_id = None
    if greedy.first_action[0]:
        if greedy.best_sequence:
            greedy_id = ActionIdentity.from_action(greedy.best_sequence[0])
        else:
            greedy_id = ActionIdentity.from_action(
                (greedy.first_action[0], greedy.first_action[1], greedy.first_action[2], {})
            )

    if use_learned:
        result.first_action = recursive.first_action
        result.first_action_identity = recursive.first_action_identity
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

    result.q_learned_mean = q_learned_mean
    result.q_learned_std = q_learned_std
    result.q_greedy = q_greedy_mean
    result.margin_mean = margin_mean
    result.margin_std = margin_std
    result.lcb_margin = lcb

    return result


def calibrate_kappa(
    calibration_tasks: list[dict],
    kappa_candidates: list[float] = None,
) -> tuple[float, dict]:
    """Choose kappa on a held-out calibration split.

    For each kappa candidate, compute the hybrid planner's performance
    on the calibration set. Choose the kappa that minimizes CVaR95 of
    normalized regret while maintaining recovery > greedy.

    Returns (best_kappa, calibration_metrics).
    """
    from ..exp6_8_1.risk_metrics import compute_risk_metrics

    if kappa_candidates is None:
        kappa_candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    best_kappa = 1.0
    best_cvar95 = float("inf")
    calibration_metrics = {}

    for kappa in kappa_candidates:
        regrets = []
        recoveries = []
        used_learned_count = 0

        for task in calibration_tasks:
            plan = task["plans"].get(kappa)
            if plan is None:
                continue

            exact_val = task["exact_val"]
            plan_val = task["plan_val"]
            regret = abs(exact_val - plan_val)
            norm_regret = regret / (abs(exact_val) + 1e-6)
            regrets.append(norm_regret)

            recoveries.append(task["recovery"].get(kappa, 0.0))
            if plan.get("used_learned", False):
                used_learned_count += 1

        if not regrets:
            continue

        regrets_arr = np.array(regrets)
        risk = compute_risk_metrics(regrets_arr)
        cvar95 = _compute_cvar(regrets_arr, 95)

        # Recovery must be > greedy recovery.
        greedy_recovery = np.mean(task["greedy_recovery"] for task in calibration_tasks if "greedy_recovery" in task) if calibration_tasks else 0.0
        avg_recovery = float(np.mean(recoveries)) if recoveries else 0.0

        # Objective: minimize CVaR95 while recovery >= greedy.
        # If recovery < greedy, penalize heavily.
        if avg_recovery < greedy_recovery:
            score = cvar95 + 1000.0  # penalty
        else:
            score = cvar95

        calibration_metrics[kappa] = {
            "cvar95": cvar95,
            "mean_regret": risk["mean_regret"],
            "median_regret": risk["median_regret"],
            "p95_regret": risk["p95_regret"],
            "avg_recovery": avg_recovery,
            "greedy_recovery": greedy_recovery,
            "coverage": used_learned_count / max(len(calibration_tasks), 1),
        }

        if score < best_cvar95:
            best_cvar95 = score
            best_kappa = kappa

    return best_kappa, calibration_metrics


def _compute_cvar(regrets: np.ndarray, percentile: float) -> float:
    """Compute Conditional Value at Risk.

    CVaR_p = E[regret | regret >= P_p]

    This is the expected regret in the worst (100-p)% of cases.
    """
    if len(regrets) == 0:
        return 0.0
    threshold = np.percentile(regrets, percentile)
    tail = regrets[regrets >= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(np.mean(tail))
