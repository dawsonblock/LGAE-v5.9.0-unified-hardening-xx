"""Main experiment runner for v6.0-exp6.

Tests the full adaptive model-assisted MPC pipeline:

1. Train global prior on training families (graphlet representation)
2. For each TEST-C family:
   a. Run calibration acquisition (1, 2, 3, 5, 8, 10 samples)
   b. Assess trust
   c. Run candidate prefilter with UCB pruning
   d. Measure oracle recall, regret, exact evaluations saved
3. Compare against baselines (random, heuristic)
4. Report adaptation sample efficiency
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import time
import math
import numpy as np

from ..exp5_2.state_encoding import (
    encode_normalized_state, encode_normalized_action,
    NORM_STATE_DIM, NORM_ACTION_DIM,
)
from ..exp5_2.dynamics import DeltaDynamicsModel
from ..exp5_3.representations import (
    REPRESENTATION_LADDER, extract_representation,
    INVARIANT_INDICES, RepresentationConfig,
)
from ..exp5_3.experiment_runner import is_realized, extract_realized_data
from ..exp5_3.dynamics_ood import compute_dynamics_ood_distance
from .calibration import TopologyCalibration, fit_calibration, identity_calibration
from .calibration_controller import (
    CalibrationConfig, CalibrationResult, run_calibration_acquisition,
)
from .trust import assess_trust, TrustPolicyState, TrustReport
from .prefilter import (
    Candidate, PrefilterResult, score_candidates, prefilter_candidates,
    compute_oracle_recall,
)


@dataclass
class FamilyMPCResult:
    """Result of the full MPC pipeline for one family."""
    family: str
    # Calibration.
    calibration_state: str = ""
    sample_efficiency: int = -1
    validation_delta_r2: float = 0.0
    n_calibration: int = 0
    # Trust.
    trust_state: str = ""
    max_horizon: int = 0
    # Prefilter.
    n_candidates: int = 0
    n_retained: int = 0
    oracle_recall_at_25: float = 0.0
    oracle_recall_at_50: float = 0.0
    oracle_recall_at_100: float = 0.0
    exact_evaluations_saved: float = 0.0
    # Regret.
    learned_regret: float = 0.0
    random_regret: float = 0.0
    heuristic_regret: float = 0.0
    # Calibration parameters.
    scale: list[float] = field(default_factory=list)
    offset: list[float] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "calibration_state": self.calibration_state,
            "sample_efficiency": int(self.sample_efficiency),
            "validation_delta_r2": float(self.validation_delta_r2),
            "n_calibration": int(self.n_calibration),
            "trust_state": self.trust_state,
            "max_horizon": int(self.max_horizon),
            "n_candidates": int(self.n_candidates),
            "n_retained": int(self.n_retained),
            "oracle_recall_at_25": float(self.oracle_recall_at_25),
            "oracle_recall_at_50": float(self.oracle_recall_at_50),
            "oracle_recall_at_100": float(self.oracle_recall_at_100),
            "exact_evaluations_saved": float(self.exact_evaluations_saved),
            "learned_regret": float(self.learned_regret),
            "random_regret": float(self.random_regret),
            "heuristic_regret": float(self.heuristic_regret),
            "scale": [float(x) for x in self.scale],
            "offset": [float(x) for x in self.offset],
        }


def run_family_mpc(
    family: str,
    family_records: list[Any],
    global_model: DeltaDynamicsModel,
    rep_config: RepresentationConfig,
    *,
    calibration_config: CalibrationConfig | None = None,
    kappa: float = 1.0,
    k_values: list[int] | None = None,
) -> FamilyMPCResult:
    """Run the full MPC pipeline for one family.

    1. Extract realized data
    2. Run calibration acquisition
    3. Assess trust
    4. Generate candidates and score them
    5. Run prefilter with UCB
    6. Compute oracle recall and regret
    """
    if k_values is None:
        k_values = [10, 25, 50, 100]

    result = FamilyMPCResult(family=family)

    # Extract realized data.
    z_t, a_t, z_next, _ = extract_realized_data(family_records)
    if len(z_t) < 5:
        return result

    # Extract representation.
    rep_z = extract_representation(z_t, rep_config)
    rep_z_next = extract_representation(z_next, rep_config)
    actual_deltas = rep_z_next - rep_z

    # Compute dynamics-OOD for this family.
    # (Using training data as reference — in practice this would be precomputed.)
    base_preds = global_model.predict_raw_batch(rep_z, a_t)

    # --- Calibration ---
    cal_result = run_calibration_acquisition(
        global_model,
        rep_z, a_t, rep_z_next,
        topology_signature=family,
        config=calibration_config,
    )

    result.calibration_state = cal_result.state.value
    result.sample_efficiency = cal_result.sample_efficiency
    result.validation_delta_r2 = cal_result.validation_delta_r2
    result.n_calibration = cal_result.n_samples_collected
    result.scale = list(cal_result.calibration.scale)
    result.offset = list(cal_result.calibration.offset)

    # --- Trust assessment ---
    trust = assess_trust(
        delta_r2=cal_result.validation_delta_r2,
        calibration_corr=cal_result.calibration.uncertainty_error_corr,
        dynamics_ood=cal_result.calibration.dynamics_ood_score,
        calibration_samples=cal_result.n_samples_collected,
    )
    result.trust_state = trust.state.value
    result.max_horizon = trust.max_horizon

    # --- Candidate prefilter ---
    # Generate synthetic candidates by combining states with different actions.
    # In a real system, these would come from the structural generator.
    # Here we create candidates by pairing each state with multiple action types.
    n_cand = len(rep_z)
    # Expand: for each state, create candidates with different action encodings.
    # This simulates the candidate generation process.
    expanded_z = []
    expanded_a = []
    expanded_deltas = []
    expanded_exact_util = []

    # Use each state with all available actions (cross product).
    for i in range(n_cand):
        for j in range(n_cand):
            expanded_z.append(rep_z[i])
            expanded_a.append(a_t[j])
            # Oracle: actual delta for state i (approximation).
            expanded_deltas.append(actual_deltas[i])
            expanded_exact_util.append(float(np.mean(actual_deltas[i])))

    n_expanded = len(expanded_z)
    result.n_candidates = n_expanded

    # Create candidates.
    candidates = []
    for i in range(n_expanded):
        cand = Candidate(
            action_type="",
            action_target={},
            z_t=expanded_z[i],
            a_t=expanded_a[i],
            # Oracle: actual utility.
            exact_utility=expanded_exact_util[i],
            exact_delta=expanded_deltas[i],
        )
        candidates.append(cand)

    # Score candidates with calibrated model.
    score_candidates(candidates, global_model, cal_result.calibration, kappa=kappa)

    # Prefilter.
    prefilter_result = prefilter_candidates(candidates, k_values=k_values)

    result.n_retained = prefilter_result.n_retained
    result.oracle_recall_at_25 = prefilter_result.recall_at_k.get(25, 0.0)
    result.oracle_recall_at_50 = prefilter_result.recall_at_k.get(50, 0.0)
    result.oracle_recall_at_100 = prefilter_result.recall_at_k.get(100, 0.0)
    result.exact_evaluations_saved = prefilter_result.exact_evaluations_saved

    # --- Regret ---
    # Oracle best utility.
    oracle_best = max(c.exact_utility for c in candidates) if candidates else 0.0

    # Learned: best retained by UCB (top-K).
    retained_by_ucb = sorted(candidates, key=lambda c: c.ucb_score, reverse=True)
    k_for_regret = min(25, len(retained_by_ucb))
    learned_best = max(c.exact_utility for c in retained_by_ucb[:k_for_regret]) if retained_by_ucb else 0.0
    result.learned_regret = float(oracle_best - learned_best)

    # Random: best of K random selections.
    rng = np.random.RandomState(42)
    random_indices = rng.choice(n_expanded, size=min(k_for_regret, n_expanded), replace=False)
    random_best = max(candidates[i].exact_utility for i in random_indices)
    result.random_regret = float(oracle_best - random_best)

    # Heuristic: pick by highest structural change magnitude.
    heuristic_scores = [float(np.sum(np.abs(d))) for d in expanded_deltas]
    heuristic_sorted = sorted(range(n_expanded), key=lambda i: heuristic_scores[i], reverse=True)
    heuristic_best = max(candidates[i].exact_utility for i in heuristic_sorted[:k_for_regret])
    result.heuristic_regret = float(oracle_best - heuristic_best)

    return result


def run_adaptation_curve(
    family: str,
    family_records: list[Any],
    global_model: DeltaDynamicsModel,
    rep_config: RepresentationConfig,
    *,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Run adaptation curve: R²(k) for k = 0, 1, 2, 3, 5, 8, 10."""
    if k_values is None:
        k_values = [0, 1, 2, 3, 5, 8, 10]

    z_t, a_t, z_next, _ = extract_realized_data(family_records)
    if len(z_t) < 5:
        return {"family": family, "curve": []}

    rep_z = extract_representation(z_t, rep_config)
    rep_z_next = extract_representation(z_next, rep_config)
    actual_deltas = rep_z_next - rep_z

    # 0-shot: no calibration.
    base_preds = global_model.predict_raw_batch(rep_z, a_t)
    base_diff = base_preds - actual_deltas
    base_ss_res = float(np.sum(base_diff ** 2))
    base_ss_tot = float(np.sum((actual_deltas - actual_deltas.mean(axis=0)) ** 2))
    r2_0 = max(-10.0, min(1.0, 1.0 - base_ss_res / max(base_ss_tot, 1e-10)))

    curve = [{"k": 0, "delta_r2": r2_0}]

    # k-shot: use first k samples for calibration, rest for evaluation.
    n = len(rep_z)
    n_eval = max(5, n // 2)
    eval_z = rep_z[n_eval:]
    eval_a = a_t[n_eval:]
    eval_zn = rep_z_next[n_eval:]
    eval_deltas = eval_zn - eval_z

    for k in k_values:
        if k == 0:
            continue
        k_actual = min(k, n_eval)
        adapt_z = rep_z[:k_actual]
        adapt_a = a_t[:k_actual]
        adapt_zn = rep_z_next[:k_actual]
        adapt_deltas = adapt_zn - adapt_z

        if k_actual < 2:
            continue

        # Fit calibration.
        adapt_base_preds = global_model.predict_raw_batch(adapt_z, adapt_a)
        cal = fit_calibration(
            adapt_base_preds, adapt_deltas,
            topology_signature=family,
            regularization=1.0,
        )

        # Evaluate on held-out.
        eval_base_preds = global_model.predict_raw_batch(eval_z, eval_a)
        eval_adapted = cal.apply_batch(eval_base_preds)
        eval_diff = eval_adapted - eval_deltas
        eval_ss_res = float(np.sum(eval_diff ** 2))
        eval_ss_tot = float(np.sum((eval_deltas - eval_deltas.mean(axis=0)) ** 2))
        r2_k = max(-10.0, min(1.0, 1.0 - eval_ss_res / max(eval_ss_tot, 1e-10)))

        curve.append({"k": k, "delta_r2": r2_k})

    # Find k* = min k where R² > 0.
    k_star = -1
    for entry in curve:
        if entry["delta_r2"] > 0:
            k_star = entry["k"]
            break

    return {
        "family": family,
        "curve": curve,
        "k_star": k_star,
    }
