"""Experiment runner for v6.0-exp6.8.5.

Full Structural Advantage Features.

Narrow scope:
  - Store graph adjacency in records (enables F4 features)
  - Test only best model/target: GBT x normalized
  - Compare F1 vs F4 at N = 250, 500, 1000, 2000, 5000
  - Learning curve test: does Spearman improve with N under F4?

Hard stop condition:
  - If F4 materially improves the Pareto frontier, integrate and freeze.
  - If F4 plateaus like F1, freeze anyway and move to exp7.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json
import sys
import os
import time
import random
import numpy as np
import torch

from ..exp6_3.exact_mpc import ActionIdentity
from ..exp6_5.multi_mechanism_data import (
    MECHANISM_NAMES, generate_mechanism_task_configs,
    MechanismTaskConfig, _make_graph_from_config,
)
from ..exp6_6.objective_spec import get_objective_spec
from ..exp6_7.multi_operator_candidates import generate_multi_operator_candidates
from ..exp6_8_3.conformal_calibration import calibrate_conformal
from ..exp6_8_4.model_zoo import GBTModel, RidgeModel
from ..exp6_8_4.target_transforms import apply_target_transform, is_classification_target
from ..exp6_8_4.downside_metrics import (
    compute_spearman_correlation, compute_downside_probability,
    compute_cvar_negative, compute_risk_adjusted_score,
)

from .graph_records import (
    GraphAdvantageRecord, generate_graph_advantage_records,
    build_features_for_records,
)


@dataclass
class LearningCurvePoint:
    """One point on a learning curve."""
    n_train: int
    feature_level: str
    mechanism: str
    spearman: float = 0.0
    override_precision: float = 0.0
    coverage: float = 0.0
    mean_override_advantage: float = 0.0
    downside_prob: float = 0.0
    cvar_neg_5: float = 0.0
    risk_adjusted_score: float = 0.0
    p95_regret_hybrid: float = 0.0
    p95_regret_baseline: float = 0.0
    cvar95_hybrid: float = 0.0
    cvar95_baseline: float = 0.0
    n_test: int = 0
    n_overrides: int = 0
    rmse: float = 0.0


@dataclass
class Exp685Result:
    audit_note: str = ""
    learning_curves: list[LearningCurvePoint] = field(default_factory=list)
    f1_vs_f4_comparison: dict = field(default_factory=dict)
    decision: str = ""
    gates: dict[str, dict] = field(default_factory=dict)
    all_gates_passed: bool = False
    summary: dict[str, Any] = field(default_factory=dict)
    manifest_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "audit_note": self.audit_note,
            "learning_curves": [
                {
                    "n_train": p.n_train, "feature_level": p.feature_level,
                    "mechanism": p.mechanism, "spearman": p.spearman,
                    "override_precision": p.override_precision,
                    "coverage": p.coverage,
                    "mean_override_advantage": p.mean_override_advantage,
                    "downside_prob": p.downside_prob,
                    "cvar_neg_5": p.cvar_neg_5,
                    "risk_adjusted_score": p.risk_adjusted_score,
                    "p95_regret_hybrid": p.p95_regret_hybrid,
                    "p95_regret_baseline": p.p95_regret_baseline,
                    "cvar95_hybrid": p.cvar95_hybrid,
                    "cvar95_baseline": p.cvar95_baseline,
                    "n_test": p.n_test, "n_overrides": p.n_overrides,
                    "rmse": p.rmse,
                }
                for p in self.learning_curves
            ],
            "f1_vs_f4_comparison": self.f1_vs_f4_comparison,
            "decision": self.decision,
            "gates": self.gates,
            "all_gates_passed": self.all_gates_passed,
            "summary": self.summary,
            "manifest_evidence": self.manifest_evidence,
        }


def _evaluate_point(
    feature_level: str,
    n_train: int,
    train_records: list[GraphAdvantageRecord],
    cal_records: list[GraphAdvantageRecord],
    test_records: list[GraphAdvantageRecord],
    mechanism: str,
    target_name: str = "T2_normalized",
    model_name: str = "M2_gbt",
) -> LearningCurvePoint:
    """Evaluate one (feature_level, n_train) point."""
    # Subsample training data.
    if len(train_records) > n_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(train_records), n_train, replace=False)
        train_subset = [train_records[i] for i in idx]
    else:
        train_subset = train_records

    # Build features.
    X_train, y_train_raw, bq_train = build_features_for_records(train_subset, feature_level)
    X_cal, y_cal_raw, bq_cal = build_features_for_records(cal_records, feature_level)
    X_test, y_test_raw, bq_test = build_features_for_records(test_records, feature_level)

    # Transform target.
    is_cls = is_classification_target(target_name)
    y_train = apply_target_transform(target_name, y_train_raw, bq_train)

    # Train model.
    if model_name == "M2_gbt":
        model = GBTModel(n_estimators=100, lr=0.1, max_depth=3)
    else:
        model = RidgeModel(alpha=1.0)
    model.fit(X_train, y_train, is_classification=is_cls)

    # Predict on calibration.
    y_hat_cal = model.predict(X_cal)

    # Convert predictions back to raw advantage scale for conformal.
    if target_name == "T2_normalized":
        y_hat_cal_raw = y_hat_cal * (np.abs(bq_cal) + 1e-6)
    else:
        y_hat_cal_raw = y_hat_cal

    # Conformal calibration at alpha=0.20.
    conformal_qs = calibrate_conformal(y_cal_raw, y_hat_cal_raw, alphas=[0.20, 0.10, 0.05])
    q = conformal_qs.get(0.20, float("inf"))

    # Predict on test.
    y_hat_test = model.predict(X_test)
    if target_name == "T2_normalized":
        y_hat_test_raw = y_hat_test * (np.abs(bq_test) + 1e-6)
    else:
        y_hat_test_raw = y_hat_test

    # Arbitrate.
    lcbs = y_hat_test_raw - q
    used_learned = [lcb > 0 for lcb in lcbs]
    true_advs = y_test_raw.tolist()

    # Metrics.
    from ..exp6_8_3.risk_metrics import compute_regret_metrics

    if is_cls:
        spearman = compute_spearman_correlation(y_hat_test, (y_test_raw > 0).astype(float))
    else:
        spearman = compute_spearman_correlation(y_hat_test, y_test_raw)

    # RMSE.
    rmse = float(np.sqrt(np.mean((y_hat_test_raw - y_test_raw) ** 2)))

    overrides = [a for a, u in zip(true_advs, used_learned) if u]
    n_overrides = len(overrides)
    if n_overrides > 0:
        precision = sum(1 for a in overrides if a > 0) / n_overrides
        mean_adv = float(np.mean(overrides))
        downside_prob = sum(1 for a in overrides if a < 0) / n_overrides
        cvar_neg = compute_cvar_negative(true_advs, used_learned, 5.0)
        risk_adj = compute_risk_adjusted_score(true_advs, used_learned)
    else:
        precision = 0.0
        mean_adv = 0.0
        downside_prob = 0.0
        cvar_neg = 0.0
        risk_adj = 0.0

    coverage = n_overrides / max(len(test_records), 1)

    # Regret.
    hybrid_regrets = []
    baseline_regrets = []
    for i, rec in enumerate(test_records):
        q_star = max(rec.learned_q, rec.baseline_q)
        q_selected = rec.learned_q if used_learned[i] else rec.baseline_q
        hybrid_regrets.append(abs(q_star - q_selected))
        baseline_regrets.append(abs(q_star - rec.baseline_q))

    hybrid_reg = compute_regret_metrics(np.array(hybrid_regrets))
    baseline_reg = compute_regret_metrics(np.array(baseline_regrets))

    return LearningCurvePoint(
        n_train=len(train_subset),
        feature_level=feature_level,
        mechanism=mechanism,
        spearman=round(spearman, 4),
        override_precision=round(precision, 4),
        coverage=round(coverage, 4),
        mean_override_advantage=round(mean_adv, 4),
        downside_prob=round(downside_prob, 4),
        cvar_neg_5=round(cvar_neg, 4),
        risk_adjusted_score=round(risk_adj, 4),
        p95_regret_hybrid=round(hybrid_reg["p95"], 4),
        p95_regret_baseline=round(baseline_reg["p95"], 4),
        cvar95_hybrid=round(hybrid_reg["cvar95"], 4),
        cvar95_baseline=round(baseline_reg["cvar95"], 4),
        n_test=len(test_records),
        n_overrides=n_overrides,
        rmse=round(rmse, 4),
    )


def _check_manifest_evidence() -> dict:
    import subprocess
    evidence = {
        "manifest_exists": False, "manifest_valid": False,
        "test_count": 0, "test_passed": 0, "test_failed": 0,
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
            tr = qual.get("test_results", {})
            evidence["test_count"] = tr.get("collected", 0)
            evidence["test_passed"] = tr.get("passed", 0)
            evidence["test_failed"] = tr.get("failed", 0)
        except Exception:
            pass
    return evidence


def run_exp6_8_5(
    *,
    n_train_per_mechanism: int = 2000,
    n_calibration: int = 80,
    n_test: int = 80,
    data_sizes: list[int] = None,
    feature_levels: list[str] = None,
    mechanisms: list[str] = None,
    target_name: str = "T2_normalized",
    model_name: str = "M2_gbt",
) -> Exp685Result:
    """Run the v6.0-exp6.8.5 full structural advantage features experiment."""
    result = Exp685Result(
        audit_note=(
            "Full Structural Advantage Features. "
            "Test whether F4 (rich structural features) breaks the F1 ceiling. "
            f"Model: {model_name}, Target: {target_name}. "
            "Hard stop: if F4 doesn't improve, freeze and move to exp7."
        )
    )

    if data_sizes is None:
        data_sizes = [250, 500, 1000, 2000]
    if feature_levels is None:
        feature_levels = ["F1_current", "F4_full"]
    if mechanisms is None:
        mechanisms = ["connectivity_threshold", "redundancy_threshold"]

    print(f"\n  Mechanisms: {mechanisms}")
    print(f"  Model: {model_name}, Target: {target_name}")
    print(f"  Feature levels: {feature_levels}")
    print(f"  Data sizes: {data_sizes}")

    # === Phase 1: Generate datasets with stored graphs ===
    print("\n=== Phase 1: Generating graph-storing advantage datasets ===")
    t0 = time.time()

    all_train = {}
    cal_by_mech = {}
    test_by_mech = {}

    for mechanism in mechanisms:
        print(f"  Generating {mechanism}...")
        train = generate_graph_advantage_records(
            mechanism, n_train_per_mechanism, seed=42, split="train",
        )
        cal = generate_graph_advantage_records(
            mechanism, n_calibration, seed=777, split="calibration",
        )
        test = generate_graph_advantage_records(
            mechanism, n_test, seed=888, split="test",
        )
        all_train[mechanism] = train
        cal_by_mech[mechanism] = cal
        test_by_mech[mechanism] = test
        print(f"    train={len(train)}, cal={len(cal)}, test={len(test)}")

    print(f"  Total generation time: {time.time()-t0:.1f}s")

    # === Phase 2: Evaluate learning curves ===
    print("\n=== Phase 2: Evaluating learning curves (F1 vs F4) ===")

    total_points = len(feature_levels) * len(data_sizes) * len(mechanisms)
    point_count = 0

    for mechanism in mechanisms:
        train_recs = all_train.get(mechanism, [])
        cal_recs = cal_by_mech.get(mechanism, [])
        test_recs = test_by_mech.get(mechanism, [])

        if len(train_recs) < 50 or len(cal_recs) < 10 or len(test_recs) < 10:
            print(f"  Skipping {mechanism}: insufficient data")
            continue

        for feat_level in feature_levels:
            for n_train in data_sizes:
                if n_train > len(train_recs):
                    continue

                point_count += 1
                print(f"  [{point_count}/{total_points}] {feat_level} x N={n_train} x {mechanism}")

                t0 = time.time()
                point = _evaluate_point(
                    feat_level, n_train,
                    train_recs, cal_recs, test_recs,
                    mechanism, target_name, model_name,
                )
                elapsed = time.time() - t0
                result.learning_curves.append(point)

                print(f"    Spearman={point.spearman:.4f}, "
                      f"precision={point.override_precision:.1%}, "
                      f"coverage={point.coverage:.1%}, "
                      f"P95={point.p95_regret_hybrid:.2f} vs {point.p95_regret_baseline:.2f}, "
                      f"RMSE={point.rmse:.2f} ({elapsed:.1f}s)")

    # === Phase 3: Compare F1 vs F4 ===
    print("\n=== Phase 3: Comparing F1 vs F4 ===")

    f1_vs_f4 = {}
    for mechanism in mechanisms:
        mech_data = {"F1_current": [], "F4_full": []}
        for point in result.learning_curves:
            if point.mechanism != mechanism:
                continue
            if point.feature_level in mech_data:
                mech_data[point.feature_level].append({
                    "n_train": point.n_train,
                    "spearman": point.spearman,
                    "coverage": point.coverage,
                    "p95_regret_hybrid": point.p95_regret_hybrid,
                    "rmse": point.rmse,
                })
        f1_vs_f4[mechanism] = mech_data

    result.f1_vs_f4_comparison = f1_vs_f4

    # Print comparison.
    for mechanism in mechanisms:
        print(f"\n  {mechanism}:")
        for level in feature_levels:
            curve = f1_vs_f4.get(mechanism, {}).get(level, [])
            if not curve:
                continue
            spearmans = [p["spearman"] for p in curve]
            ns = [p["n_train"] for p in curve]
            print(f"    {level}: Spearman = {spearmans} at N = {ns}")
            if len(spearmans) >= 2:
                improvement = spearmans[-1] - spearmans[0]
                print(f"      Improvement: {improvement:+.4f}")

    # === Phase 4: Decision ===
    print("\n=== Phase 4: Making freeze decision ===")

    # Check if F4 improves with data.
    f4_improves = False
    f4_spearman_final = 0.0
    f1_spearman_final = 0.0

    for mechanism in mechanisms:
        f4_curve = f1_vs_f4.get(mechanism, {}).get("F4_full", [])
        f1_curve = f1_vs_f4.get(mechanism, {}).get("F1_current", [])

        if len(f4_curve) >= 2:
            f4_first = f4_curve[0]["spearman"]
            f4_last = f4_curve[-1]["spearman"]
            f4_spearman_final = max(f4_spearman_final, f4_last)
            if f4_last > f4_first + 0.02:  # material improvement
                f4_improves = True

        if len(f1_curve) >= 2:
            f1_spearman_final = max(f1_spearman_final, f1_curve[-1]["spearman"])

    f4_beats_f1 = f4_spearman_final > f1_spearman_final + 0.02

    if f4_improves and f4_beats_f1:
        decision = "F4_MATERIALLY_IMPROVES"
        print(f"  F4 shows material improvement: Spearman improves with N")
        print(f"  F4 final Spearman: {f4_spearman_final:.4f} > F1: {f1_spearman_final:.4f}")
        print(f"  Decision: Integrate F4 into conformal arbitrator and freeze planner")
    elif f4_beats_f1:
        decision = "F4_BEATS_F1_BUT_NO_LEARNING_CURVE"
        print(f"  F4 beats F1 but doesn't show a learning curve")
        print(f"  F4 final Spearman: {f4_spearman_final:.4f} > F1: {f1_spearman_final:.4f}")
        print(f"  Decision: Use F4 features, freeze planner, move to exp7")
    else:
        decision = "F4_DOES_NOT_IMPROVE"
        print(f"  F4 does not materially improve over F1")
        print(f"  F4 final Spearman: {f4_spearman_final:.4f}, F1: {f1_spearman_final:.4f}")
        print(f"  Decision: Freeze planner anyway, move to exp7")

    result.decision = decision

    # === Phase 5: Gates ===
    print("\n=== Phase 5: Checking gates ===")

    manifest_evidence = _check_manifest_evidence()
    result.manifest_evidence = manifest_evidence

    # Gate 1: F4 Spearman > F1 Spearman at largest N.
    gate_1 = f4_beats_f1

    # Gate 2: F4 shows learning curve (Spearman improves with N).
    gate_2 = f4_improves

    # Gate 3: Best F4 point has positive mean override advantage.
    best_f4 = max(
        (p for p in result.learning_curves if p.feature_level == "F4_full"),
        key=lambda p: p.risk_adjusted_score,
        default=None,
    )
    gate_3 = best_f4 is not None and best_f4.mean_override_advantage > 0

    # Gate 4: Best F4 point has P95 regret <= baseline.
    gate_4 = best_f4 is not None and best_f4.p95_regret_hybrid <= best_f4.p95_regret_baseline

    # Gate 5: Best F4 point has coverage > 5%.
    gate_5 = best_f4 is not None and best_f4.coverage > 0.05

    # Gate 6: Release qualification.
    gate_6 = manifest_evidence["manifest_valid"] and manifest_evidence["test_failed"] == 0

    gates = {
        "1_f4_beats_f1": {
            "passed": gate_1,
            "description": f"F4={f4_spearman_final:.4f} vs F1={f1_spearman_final:.4f}",
        },
        "2_f4_learning_curve": {
            "passed": gate_2,
            "description": f"F4 improves with N: {f4_improves}",
        },
        "3_positive_mean_advantage": {
            "passed": gate_3,
            "description": f"mean_adv={best_f4.mean_override_advantage:.4f}" if best_f4 else "no data",
        },
        "4_p95_le_baseline": {
            "passed": gate_4,
            "description": f"P95={best_f4.p95_regret_hybrid:.2f} vs {best_f4.p95_regret_baseline:.2f}" if best_f4 else "no data",
        },
        "5_coverage_gt_5pct": {
            "passed": gate_5,
            "description": f"coverage={best_f4.coverage:.1%}" if best_f4 else "no data",
        },
        "6_qualification": {
            "passed": gate_6,
            "description": f"manifest_valid={manifest_evidence['manifest_valid']}, failures={manifest_evidence['test_failed']}",
        },
    }

    result.gates = gates
    result.all_gates_passed = all(g["passed"] for g in gates.values())

    result.summary = {
        "decision": decision,
        "f4_spearman_final": f4_spearman_final,
        "f1_spearman_final": f1_spearman_final,
        "f4_improves_with_data": f4_improves,
        "f4_beats_f1": f4_beats_f1,
        "best_f4_spearman": best_f4.spearman if best_f4 else 0.0,
        "best_f4_coverage": best_f4.coverage if best_f4 else 0.0,
        "best_f4_p95_regret": best_f4.p95_regret_hybrid if best_f4 else 0.0,
        "best_f4_risk_adjusted_score": best_f4.risk_adjusted_score if best_f4 else 0.0,
    }

    print(f"\n  Overall: {'ALL GATES PASSED' if result.all_gates_passed else 'GATES NOT ALL MET'}")
    print(f"  Decision: {decision}")
    for name, gate in gates.items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"    {name}: {status} — {gate['description']}")

    return result
