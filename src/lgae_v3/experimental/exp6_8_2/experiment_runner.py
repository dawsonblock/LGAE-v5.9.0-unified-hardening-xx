"""Experiment runner for v6.0-exp6.8.2.

Calibrated selective planning with:
  - Ensemble-based uncertainty (M=5 models)
  - LCB-margin arbitration: use learned only if LCB(Q_learned - Q_greedy) > 0
  - Calibration split: choose kappa on calibration, evaluate on locked test
  - CVaR95 and tail risk metrics
  - Monotonic risk-by-uncertainty decile check
  - Uncertainty-error correlation

Gates:
  1. Calibration chosen only on calibration split (by design)
  2. Connectivity: median regret < greedy, P95 < greedy, recovery > greedy
  3. Redundancy: median regret < greedy, P95 <= greedy, CVaR95 <= greedy
  4. Spectral: no regression vs certified baseline
  5. Hub load: no regression vs baseline
  6. Uncertainty-error correlation > 0
  7. Monotonic risk-by-uncertainty deciles
  8. Learned coverage > 10% on at least 2 mechanisms
  9. Search savings > 50%
  10. Exact replay + release qualification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import sys
import os
import time
import random
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_3.exact_mpc import (
    exact_mpc, greedy_one_step, apply_action, apply_action_with_status,
    ActionIdentity,
)
from ..exp6_5.multi_mechanism_data import (
    MECHANISM_NAMES,
    generate_mechanism_task_configs, MechanismTaskConfig, _make_graph_from_config,
)
from ..exp6_6.objective_spec import get_objective_spec, ObjectiveSpec
from ..exp6_7.multi_operator_candidates import generate_multi_operator_candidates
from ..exp6_8.recursive_planner import recursive_causal_mpc
from ..exp6_8_1.split_state import SplitStructuralState, LEARNED_STATE_DIM
from ..exp6_8_1.risk_metrics import compute_risk_metrics
from .ensemble_model import EnsembleLearnedModel
from .lcb_planner import lcb_hybrid_plan, calibrate_kappa, _compute_cvar
from .extended_risk_metrics import (
    compute_cvar, compute_extended_risk_metrics,
    compute_uncertainty_error_correlation,
    compute_risk_by_uncertainty_deciles,
)


@dataclass
class LOMOResult682:
    held_out_mechanism: str
    n_train_samples: int = 0
    n_calibration: int = 0
    n_test: int = 0
    calibrated_kappa: float = 1.0
    calibration_metrics: dict = field(default_factory=dict)
    arch_results: dict[str, dict] = field(default_factory=dict)
    paired_cis: dict[str, dict] = field(default_factory=dict)
    uncertainty_correlation: dict = field(default_factory=dict)
    risk_deciles: dict = field(default_factory=dict)


@dataclass
class Exp682Result:
    audit_note: str = ""
    lomo_results: list[LOMOResult682] = field(default_factory=list)
    gates: dict[str, dict] = field(default_factory=dict)
    all_gates_passed: bool = False
    summary: dict[str, Any] = field(default_factory=dict)
    training_info: dict[str, Any] = field(default_factory=dict)
    manifest_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "audit_note": self.audit_note,
            "lomo_results": [
                {
                    "held_out_mechanism": r.held_out_mechanism,
                    "n_train_samples": r.n_train_samples,
                    "n_calibration": r.n_calibration,
                    "n_test": r.n_test,
                    "calibrated_kappa": r.calibrated_kappa,
                    "calibration_metrics": r.calibration_metrics,
                    "arch_results": r.arch_results,
                    "paired_cis": r.paired_cis,
                    "uncertainty_correlation": r.uncertainty_correlation,
                    "risk_deciles": r.risk_deciles,
                }
                for r in self.lomo_results
            ],
            "gates": self.gates,
            "all_gates_passed": self.all_gates_passed,
            "summary": self.summary,
            "training_info": self.training_info,
            "manifest_evidence": self.manifest_evidence,
        }


def _generate_training_data(
    mechanisms: list[str],
    n_tasks_per_mechanism: int = 200,
    seed: int = 42,
) -> dict:
    """Generate training data for the ensemble.

    The learned tier's future_opportunity (index 2) is trained on the
    actual second-step improvement: the best objective gain achievable
    from S_1 by any valid action. This is the quantity that the ensemble
    can be uncertain about and that affects H=2 planning.
    """
    from ..exp6_7.multi_operator_features import extract_multi_operator_features
    from ..exp6_4.test_f import make_test_f_utility
    from ..exp6_8.recursive_planner import evaluate_objective_on_state
    from ..exp6_8.transition_model import exact_transition
    from ..exp6_8.structural_state import StructuralState as SS68

    all_X = []
    all_y = []
    all_mechanism = []

    for mech_idx, mechanism in enumerate(mechanisms):
        obj_spec = get_objective_spec(mechanism)
        configs = generate_mechanism_task_configs(
            mechanism=mechanism,
            n_tasks=n_tasks_per_mechanism,
            seed=seed + mech_idx * 1000,
        )
        for config in configs:
            graph, z = _make_graph_from_config(config)
            candidates = generate_multi_operator_candidates(
                graph, z, config, rng=random.Random(config.seed),
            )
            if len(candidates) < 4:
                continue

            state = SplitStructuralState.from_graph(graph)
            exact_z = state.exact.to_array()
            certified_z = state.certified.to_array()
            learned_z = state.learned.to_array()

            utility_fn = make_test_f_utility(
                mechanism, config.lambda_bonus, int(obj_spec.threshold),
            )

            for action in candidates:
                status = apply_action_with_status(graph, action)
                if status.status != "VALID":
                    continue

                x_action = extract_multi_operator_features(
                    graph, z, action, threshold=config.threshold, horizon=2,
                )
                x_full = np.concatenate([x_action, exact_z, certified_z, learned_z])

                # Exact transition.
                new_graph = apply_action(graph, action)
                new_state = SplitStructuralState.from_graph(new_graph)

                # Compute actual future opportunity: best second-step gain.
                future_candidates = generate_multi_operator_candidates(
                    new_graph, z, config, rng=random.Random(config.seed + 1),
                )
                best_second_gain = 0.0
                state_0 = SS68.from_graph(graph)
                state_1_exact = SS68.from_graph(new_graph)

                # First-step gain (exact).
                first_gain = 0.0
                current_val = state.get_observable(obj_spec.observable)
                after_val = new_state.get_observable(obj_spec.observable)
                if obj_spec.reward_shape == "threshold":
                    if obj_spec.direction == "minimize":
                        b_after = obj_spec.magnitude if after_val <= obj_spec.threshold else 0.0
                        b_current = obj_spec.magnitude if current_val <= obj_spec.threshold else 0.0
                    else:
                        b_after = obj_spec.magnitude if after_val >= obj_spec.threshold else 0.0
                        b_current = obj_spec.magnitude if current_val >= obj_spec.threshold else 0.0
                    first_gain = b_after - b_current
                else:
                    delta = after_val - current_val
                    first_gain = (-delta if obj_spec.direction == "minimize" else delta) * obj_spec.magnitude

                # Best second-step gain.
                for second_action in future_candidates[:20]:  # Sample for efficiency.
                    st2 = apply_action_with_status(new_graph, second_action)
                    if st2.status != "VALID":
                        continue
                    g2 = apply_action(new_graph, second_action)
                    state_2 = SplitStructuralState.from_graph(g2)
                    after_val_2 = state_2.get_observable(obj_spec.observable)
                    if obj_spec.reward_shape == "threshold":
                        if obj_spec.direction == "minimize":
                            b2 = obj_spec.magnitude if after_val_2 <= obj_spec.threshold else 0.0
                            b1 = obj_spec.magnitude if after_val <= obj_spec.threshold else 0.0
                        else:
                            b2 = obj_spec.magnitude if after_val_2 >= obj_spec.threshold else 0.0
                            b1 = obj_spec.magnitude if after_val >= obj_spec.threshold else 0.0
                        gain2 = b2 - b1
                    else:
                        delta2 = after_val_2 - after_val
                        gain2 = (-delta2 if obj_spec.direction == "minimize" else delta2) * obj_spec.magnitude
                    best_second_gain = max(best_second_gain, gain2)

                # Label: [path_length, efficiency, future_opportunity]
                # future_opportunity = best second-step gain (normalized).
                y = new_state.learned.to_array()
                y[2] = best_second_gain / max(obj_spec.magnitude, 1.0)  # normalize

                all_X.append(x_full)
                all_y.append(y)
                all_mechanism.append(mechanism)

    return {
        "X": np.array(all_X, dtype=np.float32),
        "y": np.array(all_y, dtype=np.float32),
        "mechanism": np.array(all_mechanism),
    }


def _generate_eval_tasks(
    mechanism: str, n_target: int, seed: int, max_attempts: int = 800,
) -> list[MechanismTaskConfig]:
    """Generate eval tasks, filtering for greedy-suboptimal cases."""
    obj_spec = get_objective_spec(mechanism)
    mech_threshold = int(obj_spec.threshold)

    configs = generate_mechanism_task_configs(
        mechanism=mechanism, n_tasks=max_attempts, seed=seed,
        n_nodes_range=(15, 30), n_components_range=(3, 6),
        lambda_range=(30.0, 50.0),
        threshold_range=(mech_threshold, mech_threshold),
    )

    from ..exp6_4.test_f import make_test_f_utility
    suboptimal = []
    for config in configs:
        if len(suboptimal) >= n_target:
            break
        graph, z = _make_graph_from_config(config)
        candidates = generate_multi_operator_candidates(
            graph, z, config, rng=random.Random(config.seed),
        )
        if len(candidates) < 4:
            continue
        utility_fn = make_test_f_utility(mechanism, config.lambda_bonus, mech_threshold)
        exact = exact_mpc(
            graph, z, candidates, utility_fn, horizon=2, gamma=0.9,
            regenerate_candidates=True,
            candidate_generator=lambda g, z2, **kw: generate_multi_operator_candidates(
                g, z2, config, rng=random.Random(config.seed + 100),
            ),
        )
        greedy = greedy_one_step(graph, z, candidates, utility_fn)
        exact_id = exact.first_action_identity
        greedy_id = ActionIdentity.from_action(
            (greedy.first_action[0], greedy.first_action[1], greedy.first_action[2], {})
        ) if greedy.first_action[0] else None
        if exact_id and greedy_id and exact_id != greedy_id:
            suboptimal.append(config)

    return suboptimal


def _paired_bootstrap_ci(
    a_vals: list[float], b_vals: list[float], n_boot: int = 2000, ci: float = 0.95,
) -> tuple[float, list[float]]:
    if not a_vals:
        return 0.0, [0.0, 0.0]
    diffs = np.array([a - b for a, b in zip(a_vals, b_vals)])
    rng = np.random.RandomState(42)
    boot_means = []
    n = len(diffs)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_means.append(float(np.mean(diffs[idx])))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(np.mean(diffs)), [lo, hi]


def _check_manifest_evidence() -> dict:
    import subprocess
    evidence = {
        "manifest_exists": False, "manifest_valid": False,
        "release_mode": False, "test_count": 0, "test_passed": 0, "test_failed": 0,
    }
    manifest_path = os.path.join(os.getcwd(), "MANIFEST.sha256.json")
    if os.path.exists(manifest_path):
        evidence["manifest_exists"] = True
        try:
            result = subprocess.run(
                [sys.executable, "scripts/generate_manifest.py", "--check"],
                capture_output=True, text=True, timeout=30, cwd=os.getcwd(),
            )
            if result.returncode == 0:
                evidence["manifest_valid"] = True
        except Exception:
            pass

    qual_path = os.path.join(os.getcwd(), "qualification_summary.json")
    if os.path.exists(qual_path):
        try:
            with open(qual_path) as f:
                qual = json.load(f)
            evidence["release_mode"] = qual.get("status") == "QUALIFIED"
            tr = qual.get("test_results", {})
            evidence["test_count"] = tr.get("collected", 0)
            evidence["test_passed"] = tr.get("passed", 0)
            evidence["test_failed"] = tr.get("failed", 0)
        except Exception:
            pass
    return evidence


def run_exp6_8_2(
    *,
    n_train_per_mechanism: int = 200,
    n_calibration: int = 50,
    n_test: int = 50,
    gamma: float = 0.9,
    horizon: int = 2,
    beam_width: int = 3,
    n_ensemble: int = 5,
) -> Exp682Result:
    """Run the v6.0-exp6.8.2 experiment."""
    result = Exp682Result(
        audit_note=(
            "Calibrated selective planning. Ensemble uncertainty (M=5). "
            "LCB-margin arbitration: use learned only if LCB(Q_learned - Q_greedy) > 0. "
            "Kappa chosen on calibration split, evaluated on locked test. "
            "CVaR95 and tail risk. Monotonic risk-by-uncertainty deciles."
        )
    )

    mechanisms = MECHANISM_NAMES
    print(f"\n  Mechanisms: {mechanisms}")
    print(f"  Ensemble size: {n_ensemble}")
    print(f"  Arbitration: LCB(margin) > 0, kappa chosen on calibration split")
    print(f"  Horizon: {horizon}, Beam width: {beam_width}")
    print(f"  Calibration: {n_calibration} tasks, Test: {n_test} tasks")

    # === Phase 1: Training data ===
    print("\n=== Phase 1: Generating training data ===")
    t0 = time.time()
    train_data = _generate_training_data(
        mechanisms=mechanisms,
        n_tasks_per_mechanism=n_train_per_mechanism,
        seed=42,
    )
    X_train = train_data["X"]
    y_train = train_data["y"]
    mech_labels = train_data["mechanism"]
    print(f"  Generated {len(X_train)} samples in {time.time()-t0:.1f}s")
    print(f"  Feature dim: {X_train.shape[1]}, Target dim: {y_train.shape[1]}")
    for m in mechanisms:
        n = int(np.sum(mech_labels == m))
        print(f"    {m}: {n} samples")

    result.training_info = {
        "n_train_total": len(X_train),
        "feature_dim": X_train.shape[1],
        "target_dim": y_train.shape[1],
        "n_ensemble": n_ensemble,
    }

    # === Phase 2: LOMO with calibration/test split ===
    print("\n=== Phase 2: LOMO with calibration/test split ===")

    for held_out in mechanisms:
        print(f"\n  --- LOMO: holding out {held_out} ---")

        train_mask = mech_labels != held_out
        X_lo = X_train[train_mask]
        y_lo = y_train[train_mask]
        print(f"  Train: {len(X_lo)} samples")

        # Train ensemble.
        model = EnsembleLearnedModel(n_members=n_ensemble, hidden_dim=128, n_epochs=500, lr=0.01)
        model.fit(X_lo, y_lo)

        obj_spec = get_objective_spec(held_out)

        # Generate eval tasks (more than needed, then split).
        print(f"  Generating eval tasks (target {n_calibration + n_test})...")
        eval_configs = _generate_eval_tasks(
            held_out, n_calibration + n_test, seed=777, max_attempts=1200,
        )
        print(f"  Got {len(eval_configs)} suboptimal eval tasks")

        # Split into calibration and test (no sharing).
        random.Random(123).shuffle(eval_configs)
        cal_configs = eval_configs[:n_calibration]
        test_configs = eval_configs[n_calibration:]
        print(f"  Calibration: {len(cal_configs)}, Test: {len(test_configs)}")

        lomo = LOMOResult682(
            held_out_mechanism=held_out,
            n_train_samples=len(X_lo),
            n_calibration=len(cal_configs),
            n_test=len(test_configs),
        )

        from ..exp6_4.test_f import make_test_f_utility

        # === Calibration phase: choose kappa ===
        print(f"  Calibrating kappa...")
        kappa_candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        cal_results_by_kappa: dict[float, list] = {k: [] for k in kappa_candidates}

        for config in cal_configs:
            graph, z = _make_graph_from_config(config)
            candidates = generate_multi_operator_candidates(
                graph, z, config, rng=random.Random(config.seed),
            )
            if len(candidates) < 4:
                continue

            utility_fn = make_test_f_utility(
                held_out, config.lambda_bonus, int(obj_spec.threshold),
            )

            exact = exact_mpc(
                graph, z, candidates, utility_fn, horizon=horizon, gamma=gamma,
                regenerate_candidates=True,
                candidate_generator=lambda g, z2, **kw: generate_multi_operator_candidates(
                    g, z2, config, rng=random.Random(config.seed + 100),
                ),
            )
            greedy = greedy_one_step(graph, z, candidates, utility_fn)

            exact_id = exact.first_action_identity
            exact_val = exact.all_first_action_values.get(
                exact_id.key if exact_id else "", exact.total_value,
            )
            greedy_id = ActionIdentity.from_action(
                (greedy.first_action[0], greedy.first_action[1], greedy.first_action[2], {})
            ) if greedy.first_action[0] else None
            greedy_val = exact.all_first_action_values.get(
                greedy_id.key if greedy_id else "", greedy.total_value,
            )
            greedy_recovery = 1.0 if (exact_id and greedy_id and exact_id == greedy_id) else 0.0
            greedy_regret = abs(exact_val - greedy_val)

            for kappa in kappa_candidates:
                plan = lcb_hybrid_plan(
                    graph, z, candidates, model, obj_spec, config, utility_fn,
                    horizon=horizon, gamma=gamma, beam_width=beam_width,
                    threshold=int(obj_spec.threshold), kappa=kappa,
                )
                plan_id = plan.first_action_identity
                plan_val = exact.all_first_action_values.get(
                    plan_id.key if plan_id else "", plan.total_value,
                )
                plan_regret = abs(exact_val - plan_val)
                plan_recovery = 1.0 if (exact_id and plan_id and exact_id == plan_id) else 0.0

                cal_results_by_kappa[kappa].append({
                    "regret": plan_regret,
                    "norm_regret": plan_regret / (abs(exact_val) + 1e-6),
                    "recovery": plan_recovery,
                    "used_learned": plan.used_learned,
                    "greedy_recovery": greedy_recovery,
                    "greedy_regret": greedy_regret,
                })

        # Choose kappa: minimize CVaR95 of norm_regret while recovery >= greedy.
        best_kappa = 1.0
        best_score = float("inf")
        cal_metrics = {}

        for kappa in kappa_candidates:
            results = cal_results_by_kappa[kappa]
            if not results:
                continue

            norm_regrets = np.array([r["norm_regret"] for r in results])
            recoveries = [r["recovery"] for r in results]
            greedy_rec = [r["greedy_recovery"] for r in results]

            avg_recovery = float(np.mean(recoveries))
            avg_greedy_rec = float(np.mean(greedy_rec))
            cvar95 = compute_cvar(norm_regrets, 95)

            # Score: minimize CVaR95, but penalize if recovery < greedy.
            if avg_recovery < avg_greedy_rec:
                score = cvar95 + 100.0
            else:
                score = cvar95

            cal_metrics[kappa] = {
                "cvar95": round(cvar95, 4),
                "mean_norm_regret": round(float(np.mean(norm_regrets)), 4),
                "avg_recovery": round(avg_recovery, 4),
                "avg_greedy_recovery": round(avg_greedy_rec, 4),
                "coverage": round(float(np.mean([r["used_learned"] for r in results])), 4),
            }

            if score < best_score:
                best_score = score
                best_kappa = kappa

        lomo.calibrated_kappa = best_kappa
        lomo.calibration_metrics = cal_metrics
        print(f"  Best kappa: {best_kappa} (CVaR95={cal_metrics.get(best_kappa, {}).get('cvar95', 'N/A')})")

        # === Test phase: evaluate with locked kappa ===
        print(f"  Evaluating on test set with kappa={best_kappa}...")

        recovery = {"greedy": [], "recursive": [], "hybrid": []}
        all_regrets = {"greedy": [], "recursive": [], "hybrid": []}
        all_norm_regrets = {"greedy": [], "recursive": [], "hybrid": []}
        savings = {"recursive": [], "hybrid": []}
        uncertainties = []
        errors = []
        used_learned_count = 0

        for config in test_configs:
            graph, z = _make_graph_from_config(config)
            candidates = generate_multi_operator_candidates(
                graph, z, config, rng=random.Random(config.seed),
            )
            if len(candidates) < 4:
                continue

            utility_fn = make_test_f_utility(
                held_out, config.lambda_bonus, int(obj_spec.threshold),
            )

            exact = exact_mpc(
                graph, z, candidates, utility_fn, horizon=horizon, gamma=gamma,
                regenerate_candidates=True,
                candidate_generator=lambda g, z2, **kw: generate_multi_operator_candidates(
                    g, z2, config, rng=random.Random(config.seed + 100),
                ),
            )
            greedy = greedy_one_step(graph, z, candidates, utility_fn)
            recursive = recursive_causal_mpc(
                graph, z, candidates, model, obj_spec, config,
                horizon=horizon, gamma=gamma, beam_width=beam_width,
                threshold=int(obj_spec.threshold), use_predicted=True,
            )
            hybrid = lcb_hybrid_plan(
                graph, z, candidates, model, obj_spec, config, utility_fn,
                horizon=horizon, gamma=gamma, beam_width=beam_width,
                threshold=int(obj_spec.threshold), kappa=best_kappa,
            )

            if hybrid.used_learned:
                used_learned_count += 1

            exact_id = exact.first_action_identity
            exact_val = exact.all_first_action_values.get(
                exact_id.key if exact_id else "", exact.total_value,
            )

            for name, plan in [("greedy", greedy), ("recursive", recursive), ("hybrid", hybrid)]:
                plan_id = plan.first_action_identity if hasattr(plan, 'first_action_identity') else None
                if plan_id is None and plan.first_action[0]:
                    plan_id = ActionIdentity.from_action(
                        (plan.first_action[0], plan.first_action[1], plan.first_action[2], {})
                    )
                agree = 1.0 if (exact_id and plan_id and exact_id == plan_id) else 0.0
                recovery[name].append(agree)

                plan_val = exact.all_first_action_values.get(
                    plan_id.key if plan_id else "", plan.total_value,
                )
                regret = abs(exact_val - plan_val)
                norm_regret = regret / (abs(exact_val) + 1e-6)
                all_regrets[name].append(regret)
                all_norm_regrets[name].append(norm_regret)

            for name in ["recursive", "hybrid"]:
                plan = recursive if name == "recursive" else hybrid
                s = 1.0 - plan.nodes_expanded / max(exact.nodes_expanded, 1)
                savings[name].append(s)

            # Collect uncertainty and error for correlation analysis.
            if hasattr(hybrid, 'margin_std'):
                uncertainties.append(hybrid.margin_std)
                plan_val = exact.all_first_action_values.get(
                    hybrid.first_action_identity.key if hybrid.first_action_identity else "",
                    hybrid.total_value,
                )
                errors.append(abs(exact_val - plan_val))

        n_test_actual = len(test_configs)
        for name in ["greedy", "recursive", "hybrid"]:
            rec_rate = float(np.mean(recovery[name])) if recovery[name] else 0.0
            regrets_arr = np.array(all_regrets[name])
            risk = compute_extended_risk_metrics(regrets_arr)
            avg_norm_reg = float(np.mean(all_norm_regrets[name])) if all_norm_regrets[name] else 0.0
            avg_savings = float(np.mean(savings[name])) if name in savings and savings[name] else 0.0

            lomo.arch_results[name] = {
                "recovery_rate": round(rec_rate, 4),
                "mean_norm_regret": round(avg_norm_reg, 4),
                "mean_regret": round(risk["mean_regret"], 4),
                "median_regret": round(risk["median_regret"], 4),
                "p95_regret": round(risk["p95_regret"], 4),
                "p99_regret": round(risk["p99_regret"], 4),
                "cvar95": round(risk["cvar95"], 4),
                "avg_savings": round(avg_savings, 4),
                "n_test": n_test_actual,
            }

        # Coverage.
        lomo.arch_results["hybrid"]["coverage"] = round(used_learned_count / max(n_test_actual, 1), 4)

        # Uncertainty-error correlation.
        lomo.uncertainty_correlation = compute_uncertainty_error_correlation(
            uncertainties, errors,
        )

        # Risk by uncertainty deciles.
        lomo.risk_deciles = compute_risk_by_uncertainty_deciles(
            uncertainties, all_norm_regrets["hybrid"],
        )

        # Paired CIs.
        for cmp_name, cmp_vals in [("hybrid", recovery["hybrid"]), ("recursive", recovery["recursive"])]:
            mean_diff, ci = _paired_bootstrap_ci(cmp_vals, recovery["greedy"])
            lomo.paired_cis[f"recovery_{cmp_name}_minus_greedy"] = {
                "mean_diff": round(mean_diff, 4),
                "ci_95": [round(ci[0], 4), round(ci[1], 4)],
            }

        print(f"  Test results ({n_test_actual} tasks):")
        for name in ["greedy", "recursive", "hybrid"]:
            ar = lomo.arch_results[name]
            print(f"    {name}: recovery={ar['recovery_rate']:.0%}, "
                  f"median_reg={ar['median_regret']:.2f}, "
                  f"p95_reg={ar['p95_regret']:.2f}, "
                  f"cvar95={ar['cvar95']:.2f}")
        print(f"    hybrid coverage: {lomo.arch_results['hybrid'].get('coverage', 0):.0%}")
        print(f"    uncertainty-error corr: {lomo.uncertainty_correlation.get('correlation', 0):.3f}")
        print(f"    risk deciles monotonic: {lomo.risk_deciles.get('is_monotonic', 'N/A')}")

        result.lomo_results.append(lomo)

    # === Phase 3: Manifest ===
    print("\n=== Phase 3: Checking manifest/release evidence ===")
    manifest_evidence = _check_manifest_evidence()
    result.manifest_evidence = manifest_evidence
    print(f"  Manifest valid: {manifest_evidence['manifest_valid']}")
    print(f"  Tests: {manifest_evidence['test_passed']}/{manifest_evidence['test_count']}")

    # === Phase 4: Gates ===
    print("\n=== Phase 4: Checking success gates ===")

    valid_lomo = [r for r in result.lomo_results if r.n_test >= 30]

    def get_arch(mechanism, arch_name, field_name):
        for r in valid_lomo:
            if r.held_out_mechanism == mechanism:
                return r.arch_results.get(arch_name, {}).get(field_name, 0.0)
        return 0.0

    # Gate 1: Calibration on calibration split (by design).
    gate_1 = True

    # Gate 2: Connectivity median < greedy, P95 < greedy, recovery > greedy.
    conn_h = get_arch("connectivity_threshold", "hybrid", "recovery_rate")
    conn_g = get_arch("connectivity_threshold", "greedy", "recovery_rate")
    conn_h_med = get_arch("connectivity_threshold", "hybrid", "median_regret")
    conn_g_med = get_arch("connectivity_threshold", "greedy", "median_regret")
    conn_h_p95 = get_arch("connectivity_threshold", "hybrid", "p95_regret")
    conn_g_p95 = get_arch("connectivity_threshold", "greedy", "p95_regret")
    gate_2 = (conn_h > conn_g and conn_h_med < conn_g_med and conn_h_p95 < conn_g_p95)

    # Gate 3: Redundancy median < greedy, P95 <= greedy, CVaR95 <= greedy.
    red_h_med = get_arch("redundancy_threshold", "hybrid", "median_regret")
    red_g_med = get_arch("redundancy_threshold", "greedy", "median_regret")
    red_h_p95 = get_arch("redundancy_threshold", "hybrid", "p95_regret")
    red_g_p95 = get_arch("redundancy_threshold", "greedy", "p95_regret")
    red_h_cvar = get_arch("redundancy_threshold", "hybrid", "cvar95")
    red_g_cvar = get_arch("redundancy_threshold", "greedy", "cvar95")
    gate_3 = (red_h_med < red_g_med and red_h_p95 <= red_g_p95 and red_h_cvar <= red_g_cvar)

    # Gate 4: Spectral no regression.
    spec_h = get_arch("spectral_gap_threshold", "hybrid", "recovery_rate")
    spec_g = get_arch("spectral_gap_threshold", "greedy", "recovery_rate")
    gate_4 = spec_h >= spec_g

    # Gate 5: Hub load no regression.
    hub_h = get_arch("hub_load_threshold", "hybrid", "recovery_rate")
    hub_g = get_arch("hub_load_threshold", "greedy", "recovery_rate")
    gate_5 = hub_h >= hub_g

    # Gate 6: Uncertainty-error correlation > 0.
    all_corrs = [r.uncertainty_correlation.get("correlation", 0.0) for r in valid_lomo]
    avg_corr = float(np.mean(all_corrs)) if all_corrs else 0.0
    gate_6 = avg_corr > 0.0

    # Gate 7: Monotonic risk-by-uncertainty deciles.
    monotonic_count = sum(1 for r in valid_lomo if r.risk_deciles.get("is_monotonic", False))
    gate_7 = monotonic_count >= len(valid_lomo) / 2 if valid_lomo else False

    # Gate 8: Coverage > 10% on at least 2 mechanisms.
    coverage_count = sum(
        1 for r in valid_lomo
        if r.arch_results.get("hybrid", {}).get("coverage", 0) > 0.10
    )
    gate_8 = coverage_count >= 2

    # Gate 9: Search savings > 50%.
    all_savings = [r.arch_results.get("hybrid", {}).get("avg_savings", 0) for r in valid_lomo]
    avg_savings = float(np.mean(all_savings)) if all_savings else 0.0
    gate_9 = avg_savings > 0.5

    # Gate 10: Exact replay + release qualification.
    gate_10 = manifest_evidence["manifest_valid"] and manifest_evidence["test_failed"] == 0

    gates = {
        "1_calibration_split": {"passed": gate_1, "description": "by design"},
        "2_connectivity_full": {
            "passed": gate_2,
            "description": f"rec={conn_h:.0%}>{conn_g:.0%}, med={conn_h_med:.2f}<{conn_g_med:.2f}, p95={conn_h_p95:.2f}<{conn_g_p95:.2f}",
        },
        "3_redundancy_tail": {
            "passed": gate_3,
            "description": f"med={red_h_med:.2f}<{red_g_med:.2f}, p95={red_h_p95:.2f}<={red_g_p95:.2f}, cvar95={red_h_cvar:.2f}<={red_g_cvar:.2f}",
        },
        "4_spectral_no_regression": {
            "passed": gate_4,
            "description": f"hybrid={spec_h:.0%} >= greedy={spec_g:.0%}",
        },
        "5_hub_load_no_regression": {
            "passed": gate_5,
            "description": f"hybrid={hub_h:.0%} >= greedy={hub_g:.0%}",
        },
        "6_uncertainty_error_corr": {
            "passed": gate_6,
            "description": f"avg corr: {avg_corr:.3f}",
        },
        "7_monotonic_risk_deciles": {
            "passed": gate_7,
            "description": f"{monotonic_count}/{len(valid_lomo)} monotonic",
        },
        "8_coverage_gt_10pct_2mechanisms": {
            "passed": gate_8,
            "description": f"{coverage_count} mechanisms > 10% coverage",
        },
        "9_search_savings": {
            "passed": gate_9,
            "description": f"avg savings: {avg_savings:.1%}",
        },
        "10_qualification": {
            "passed": gate_10,
            "description": f"manifest_valid={manifest_evidence['manifest_valid']}, failures={manifest_evidence['test_failed']}",
        },
    }

    result.gates = gates
    result.all_gates_passed = all(g["passed"] for g in gates.values())

    result.summary = {
        "n_mechanisms": len(mechanisms),
        "n_valid_lomo": len(valid_lomo),
        "conn_hybrid_recovery": conn_h,
        "red_hybrid_recovery": get_arch("redundancy_threshold", "hybrid", "recovery_rate"),
        "spec_hybrid_recovery": spec_h,
        "avg_uncertainty_corr": avg_corr,
        "monotonic_count": monotonic_count,
        "coverage_count": coverage_count,
        "avg_savings": avg_savings,
    }

    print(f"\n  Overall: {'ALL GATES PASSED' if result.all_gates_passed else 'GATES NOT ALL MET'}")
    for name, gate in gates.items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"    {name}: {status} — {gate['description']}")

    return result
