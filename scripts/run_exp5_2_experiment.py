#!/usr/bin/env python
"""Run the v6.0-exp5.2 cross-family generalization experiment.

Experiment matrix:
1. Representation ablation (raw vs normalized vs delta)
2. Leave-one-family-out cross-validation
3. Family-bootstrap ensemble evaluation
4. OOD distance analysis
5. Adaptation curves (0-shot, 5-shot, 10-shot, 25-shot, 50-shot)
6. Extended rollout horizons (h=1,2,3,5,10)

Outputs:
- reports/v6_exp5_2/SCIENTIFIC_REPORT.md
- reports/v6_exp5_2/RESULTS.json
- reports/v6_exp5_2/GENERALIZATION_MATRIX.csv
- reports/v6_exp5_2/ADAPTATION_CURVES.csv
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from lgae_v3.experimental.dataset_generator import DatasetGenerator
from lgae_v3.experimental.graph_families import (
    FrozenGraphFamilyRegistry,
    FROZEN_SPLIT,
    FROZEN_SPLIT_V5_1,
    FROZEN_TEST_B_FAMILIES,
)
from lgae_v3.experimental.exp5_2 import (
    run_representation_ablation,
    run_leave_one_family_out,
    run_family_bootstrap_ensemble,
    run_ood_analysis,
    run_adaptation_curves,
    run_extended_rollout,
    extract_normalized_data,
    DeltaDynamicsModel,
    FamilyBootstrapEnsemble,
)


def main() -> int:
    print("=" * 72)
    print("LGAE v6.0-exp5.2 — Cross-Family Generalization")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Generate datasets.
    # ------------------------------------------------------------------
    print("\n[1/7] Generating datasets...")
    t0 = time.time()

    # Original split (train/val/test-a).
    registry = FrozenGraphFamilyRegistry(FROZEN_SPLIT)
    generator = DatasetGenerator(seed=42, registry=registry, n_negative_samples=3)
    datasets = generator.generate_all_splits(n_steps=5, n_episodes=1)
    all_records = (
        list(datasets["train"].records)
        + list(datasets["validation"].records)
        + list(datasets["held_out"].records)
    )
    print(f"  Train: {datasets['train'].n_records}  Val: {datasets['validation'].n_records}  Test-A: {datasets['held_out'].n_records}")

    # TEST-B (fresh, untouched).
    registry_b = FrozenGraphFamilyRegistry(FROZEN_SPLIT_V5_1)
    generator_b = DatasetGenerator(seed=99, registry=registry_b, n_negative_samples=3)
    datasets_b = generator_b.generate_all_splits(n_steps=5, n_episodes=1)
    test_b_records = list(datasets_b["held_out"].records)
    print(f"  TEST-B: {len(test_b_records)} records")
    print(f"  TEST-B families: {[f.value for f in FROZEN_TEST_B_FAMILIES]}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Representation ablation.
    # ------------------------------------------------------------------
    print("\n[2/7] Running representation ablation...")
    t0 = time.time()
    rep_results = run_representation_ablation(all_records, test_b_records)

    print(f"\n  {'Representation':15s} {'Mode':10s} {'Train R²':>10s} {'TEST-B R²':>10s} {'Δ R²':>10s} {'RMSE':>10s} {'Spearman':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in rep_results:
        print(f"  {r.representation:15s} {r.mode:10s} {r.train_r2:10.4f} {r.test_b_r2:10.4f} {r.test_b_delta_r2:10.4f} {r.test_b_rmse:10.4f} {r.test_b_spearman:10.4f}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Leave-one-family-out cross-validation.
    # ------------------------------------------------------------------
    print("\n[3/7] Running leave-one-family-out cross-validation...")
    t0 = time.time()
    loo_results = run_leave_one_family_out(all_records)

    print(f"\n  {'Held-out family':20s} {'R²':>10s} {'Δ R²':>10s} {'RMSE':>10s} {'Spearman':>10s} {'N_train':>8s} {'N_test':>8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in loo_results:
        print(f"  {r.held_out_family:20s} {r.r2:10.4f} {r.delta_r2:10.4f} {r.rmse:10.4f} {r.spearman:10.4f} {r.n_train:8d} {r.n_test:8d}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Family-bootstrap ensemble.
    # ------------------------------------------------------------------
    print("\n[4/7] Running family-bootstrap ensemble...")
    t0 = time.time()
    ensemble_result = run_family_bootstrap_ensemble(all_records, test_b_records)

    print(f"\n  TEST-B R²:        {ensemble_result.get('test_b_r2', 0):.4f}")
    print(f"  TEST-B Δ R²:      {ensemble_result.get('test_b_delta_r2', 0):.4f}")
    print(f"  TEST-B RMSE:      {ensemble_result.get('test_b_rmse', 0):.4f}")
    print(f"  TEST-B Spearman:  {ensemble_result.get('test_b_spearman', 0):.4f}")
    print(f"  Calibration corr: {ensemble_result.get('calibration_corr', 0):.4f}")
    print(f"  Calibration ρ:    {ensemble_result.get('calibration_spearman', 0):.4f}")
    print(f"  Mean uncertainty: {ensemble_result.get('mean_uncertainty', 0):.6f}")
    print(f"  N members:        {ensemble_result.get('n_members', 0)}")
    print(f"  N parameters:     {ensemble_result.get('n_parameters', 0)}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: OOD distance analysis.
    # ------------------------------------------------------------------
    print("\n[5/7] Running OOD distance analysis...")
    t0 = time.time()

    # Build ensemble for OOD analysis.
    z_t, a_t, z_next, families = extract_normalized_data(all_records, split="train")
    ensemble = FamilyBootstrapEnsemble(mode="delta", n_members=8, regularization=1e-3, seed=42)
    ensemble.fit_with_family_split(z_t, a_t, z_next, families, split="train")

    ood_result = run_ood_analysis(all_records, test_b_records, ensemble=ensemble)

    print(f"\n  Family OOD distances:")
    for fam, dist in sorted(ood_result.get("family_distances", {}).items()):
        print(f"    {fam:20s}: {dist:.4f}")
    if "ood_error_corr" in ood_result:
        print(f"\n  OOD distance vs error:        corr={ood_result['ood_error_corr']['corr']:.4f}  ρ={ood_result['ood_error_corr']['spearman']:.4f}")
    if "ood_uncertainty_corr" in ood_result:
        print(f"  OOD distance vs uncertainty:  corr={ood_result['ood_uncertainty_corr']['corr']:.4f}  ρ={ood_result['ood_uncertainty_corr']['spearman']:.4f}")
    print(f"  Mean OOD distance: {ood_result.get('mean_ood_distance', 0):.4f}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 6: Adaptation curves.
    # ------------------------------------------------------------------
    print("\n[6/7] Running adaptation curves...")
    t0 = time.time()
    adapt_results = run_adaptation_curves(all_records, test_b_records)

    print(f"\n  {'Family':20s} {'k':>5s} {'R² before':>10s} {'R² after':>10s} {'RMSE before':>12s} {'RMSE after':>12s}")
    print(f"  {'-'*20} {'-'*5} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    for r in adapt_results:
        print(f"  {r.family:20s} {r.k_shots:5d} {r.r2_before:10.4f} {r.r2_after:10.4f} {r.rmse_before:12.4f} {r.rmse_after:12.4f}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 7: Extended rollout.
    # ------------------------------------------------------------------
    print("\n[7/7] Running extended rollout evaluation...")
    t0 = time.time()

    # Use the delta model for rollout.
    rollout_model = DeltaDynamicsModel(mode="delta", regularization=1e-3, seed=42)
    rollout_model.fit(z_t, a_t, z_next, split="train")

    rollout_val = run_extended_rollout(rollout_model, all_records, split="validation", max_horizon=10)
    rollout_test_b = run_extended_rollout(rollout_model, test_b_records, split="held_out", max_horizon=10)

    print(f"\n  Validation rollout (normalized delta):")
    for h_key in sorted(rollout_val.keys()):
        m = rollout_val[h_key]
        print(f"    {h_key}: NRMSE={m['nrmse']:.4f}  R²={m['r2']:.4f}  n={m['n_samples']}")

    print(f"\n  TEST-B rollout (normalized delta):")
    for h_key in sorted(rollout_test_b.keys()):
        m = rollout_test_b[h_key]
        print(f"    {h_key}: NRMSE={m['nrmse']:.4f}  R²={m['r2']:.4f}  n={m['n_samples']}")

    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Save reports.
    # ------------------------------------------------------------------
    print("\nSaving reports...")
    report_dir = project_root / "reports" / "v6_exp5_2"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Full results JSON.
    full_results = {
        "experiment": "v6.0-exp5.2",
        "description": "Cross-family generalization program",
        "baseline": {
            "test_b_r2": -2.6443,
            "calibration_corr": 0.4213,
            "trust": 0.0,
            "commit": "693cc2f",
        },
        "representation_ablation": [r.to_log() for r in rep_results],
        "leave_one_family_out": [r.to_log() for r in loo_results],
        "family_bootstrap_ensemble": ensemble_result,
        "ood_analysis": ood_result,
        "adaptation_curves": [r.to_log() for r in adapt_results],
        "rollout_validation": rollout_val,
        "rollout_test_b": rollout_test_b,
    }
    with open(report_dir / "RESULTS.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    # Generalization matrix CSV.
    with open(report_dir / "GENERALIZATION_MATRIX.csv", "w") as f:
        f.write("held_out_family,r2,delta_r2,rmse,spearman,n_train,n_test\n")
        for r in loo_results:
            f.write(f"{r.held_out_family},{r.r2:.6f},{r.delta_r2:.6f},{r.rmse:.6f},{r.spearman:.6f},{r.n_train},{r.n_test}\n")

    # Adaptation curves CSV.
    with open(report_dir / "ADAPTATION_CURVES.csv", "w") as f:
        f.write("family,k_shots,r2_before,r2_after,rmse_before,rmse_after\n")
        for r in adapt_results:
            f.write(f"{r.family},{r.k_shots},{r.r2_before:.6f},{r.r2_after:.6f},{r.rmse_before:.6f},{r.rmse_after:.6f}\n")

    # Scientific report.
    best_rep = max(rep_results, key=lambda r: r.test_b_r2) if rep_results else None
    best_loo = max(loo_results, key=lambda r: r.r2) if loo_results else None
    best_adapt = max(adapt_results, key=lambda r: r.r2_after) if adapt_results else None

    report_md = f"""# v6.0-exp5.2 — Cross-Family Generalization

## 1. Purpose

Determine whether structural dynamics can be represented in a
topology-invariant way that transfers across unseen graph families.

## 2. Baseline (from exp5.1, commit 693cc2f)

- TEST-B one-step R² = -2.6443 (negative = worse than mean predictor)
- Calibration corr = 0.4213 (ensemble uncertainty useful)
- Trust = 0.0

## 3. Representation Ablation

| Representation | Mode | Train R² | TEST-B R² | Δ R² | RMSE | Spearman |
|---------------|------|----------|-----------|------|------|----------|
"""
    for r in rep_results:
        report_md += f"| {r.representation} | {r.mode} | {r.train_r2:.4f} | {r.test_b_r2:.4f} | {r.test_b_delta_r2:.4f} | {r.test_b_rmse:.4f} | {r.test_b_spearman:.4f} |\n"

    report_md += f"""
## 4. Leave-One-Family-Out Generalization Matrix

| Held-out family | R² | Δ R² | RMSE | Spearman | N_train | N_test |
|-----------------|-----|------|------|----------|---------|--------|
"""
    for r in loo_results:
        report_md += f"| {r.held_out_family} | {r.r2:.4f} | {r.delta_r2:.4f} | {r.rmse:.4f} | {r.spearman:.4f} | {r.n_train} | {r.n_test} |\n"

    report_md += f"""
## 5. Family-Bootstrap Ensemble (TEST-B)

- TEST-B R²: {ensemble_result.get('test_b_r2', 0):.4f}
- TEST-B Δ R²: {ensemble_result.get('test_b_delta_r2', 0):.4f}
- Calibration corr: {ensemble_result.get('calibration_corr', 0):.4f}
- Calibration ρ: {ensemble_result.get('calibration_spearman', 0):.4f}
- N members: {ensemble_result.get('n_members', 0)}

## 6. OOD Distance Analysis

"""
    for fam, dist in sorted(ood_result.get("family_distances", {}).items()):
        report_md += f"- {fam}: {dist:.4f}\n"
    if "ood_error_corr" in ood_result:
        report_md += f"\nOOD distance vs error: corr={ood_result['ood_error_corr']['corr']:.4f}\n"
    if "ood_uncertainty_corr" in ood_result:
        report_md += f"OOD distance vs uncertainty: corr={ood_result['ood_uncertainty_corr']['corr']:.4f}\n"

    report_md += f"""
## 7. Adaptation Curves

| Family | k | R² before | R² after | RMSE before | RMSE after |
|--------|---|-----------|----------|-------------|------------|
"""
    for r in adapt_results:
        report_md += f"| {r.family} | {r.k_shots} | {r.r2_before:.4f} | {r.r2_after:.4f} | {r.rmse_before:.4f} | {r.rmse_after:.4f} |\n"

    report_md += f"""
## 8. Extended Rollout (Normalized Delta)

### Validation

| Horizon | NRMSE | R² | N samples |
|---------|-------|-----|-----------|
"""
    for h_key in sorted(rollout_val.keys()):
        m = rollout_val[h_key]
        report_md += f"| {h_key} | {m['nrmse']:.4f} | {m['r2']:.4f} | {m['n_samples']} |\n"

    report_md += f"\n### TEST-B\n\n| Horizon | NRMSE | R² | N samples |\n|---------|-------|-----|-----------|\n"
    for h_key in sorted(rollout_test_b.keys()):
        m = rollout_test_b[h_key]
        report_md += f"| {h_key} | {m['nrmse']:.4f} | {m['r2']:.4f} | {m['n_samples']} |\n"

    # Conclusion.
    report_md += f"\n## 9. Conclusion\n\n"
    if best_rep and best_rep.test_b_r2 > 0:
        report_md += f"Best representation: **{best_rep.representation}** ({best_rep.mode}) with TEST-B R²={best_rep.test_b_r2:.4f}\n"
    else:
        report_md += f"No representation achieves positive TEST-B R².\n"
        report_md += f"Best: {best_rep.representation if best_rep else 'none'} with R²={best_rep.test_b_r2 if best_rep else 0:.4f}\n"

    if best_adapt and best_adapt.r2_after > 0:
        report_md += f"\nBest adaptation: {best_adapt.family} at k={best_adapt.k_shots} achieves R²={best_adapt.r2_after:.4f}\n"
    else:
        report_md += f"\nNo adaptation achieves positive R².\n"

    report_md += f"\n## 10. exp6 Authorization\n\n"
    exp6_ready = (
        best_rep and best_rep.test_b_r2 > 0
        and ensemble_result.get("calibration_corr", 0) > 0.3
    )
    if exp6_ready:
        report_md += "**CONDITIONALLY READY** for exp6 MPC as a candidate filter.\n"
    else:
        report_md += "**NOT READY** for exp6 MPC.\n"
        report_md += "Gates not met: TEST-B R² must be > 0, calibration > 0.3.\n"

    report_md += f"\n## 11. Authority Boundary\n\n"
    report_md += "The world model is **advisory-only**. The v5.11 CommitChannel remains\nthe sole authority boundary.\n"

    with open(report_dir / "SCIENTIFIC_REPORT.md", "w") as f:
        f.write(report_md)

    print(f"\n{'='*72}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*72}")
    print(f"\nReports: {report_dir}")
    if best_rep:
        print(f"Best representation: {best_rep.representation} ({best_rep.mode}) R²={best_rep.test_b_r2:.4f}")
    if best_adapt:
        print(f"Best adaptation: {best_adapt.family} k={best_adapt.k_shots} R²={best_adapt.r2_after:.4f}")
    print(f"Ensemble calibration: {ensemble_result.get('calibration_corr', 0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
