#!/usr/bin/env python
"""Run the v6.0-exp5 lightweight latent world model experiment.

Trains and evaluates the joint world model (dynamics + outcome head)
on the real exp2 dataset. Compares Linear vs MLP dynamics variants.

Outputs:
- reports/v6_exp5/SCIENTIFIC_REPORT.md
- reports/v6_exp5/RESULTS.json
- reports/v6_exp5/MODEL_LINEAR.json (serialized linear model)
- reports/v6_exp5/MODEL_MLP.json (serialized MLP model)
"""
from __future__ import annotations

import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from lgae_v3.experimental.dataset_generator import DatasetGenerator
from lgae_v3.experimental.graph_families import FrozenGraphFamilyRegistry
from lgae_v3.experimental.exp5 import (
    TrainingConfig,
    train_world_model,
    evaluate_world_model,
    rollout_evaluation,
    LightweightWorldModel,
    WorldModelTrustReport,
    JointWorldModel,
)


def main() -> int:
    print("=" * 72)
    print("LGAE v6.0-exp5 — Lightweight Latent World Model")
    print("Architecture: lightweight_latent_dynamics (authorized by exp4.2)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Generate dataset (same frozen exp2 dataset).
    # ------------------------------------------------------------------
    print("\n[1/5] Generating exp2 dataset...")
    t0 = time.time()
    registry = FrozenGraphFamilyRegistry()
    generator = DatasetGenerator(seed=42, registry=registry, n_negative_samples=3)
    datasets = generator.generate_all_splits(n_steps=5, n_episodes=1)
    all_records = (
        list(datasets["train"].records)
        + list(datasets["validation"].records)
        + list(datasets["held_out"].records)
    )
    print(f"  Train: {datasets['train'].n_records}  Val: {datasets['validation'].n_records}  Held: {datasets['held_out'].n_records}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Train both variants.
    # ------------------------------------------------------------------
    results = {}
    for variant_name, dyn_type, hidden_dim, n_epochs in [
        ("linear", "linear", 0, 200),
        ("mlp", "mlp", 32, 100),
    ]:
        print(f"\n[2/5] Training {variant_name} dynamics model...")
        config = TrainingConfig(
            dynamics_type=dyn_type,
            hidden_dim=hidden_dim,
            n_epochs=n_epochs,
            lr=0.01 if dyn_type == "linear" else 0.005,
            seed=42,
            regularization=1e-3,
        )
        t0 = time.time()
        result = train_world_model(all_records, config)
        elapsed = time.time() - t0
        print(f"  Trained on {result.n_train} records in {elapsed:.1f}s")
        print(f"  Parameters: {result.model.n_parameters}")
        print(f"  Train dynamics RMSE: {result.train_metrics.dynamics.rmse:.6f}")
        print(f"  Train dynamics R²:   {result.train_metrics.dynamics.r2:.4f}")
        print(f"  Train outcome RMSE:  {result.train_metrics.outcome_rmse:.6f}")
        print(f"  Train outcome R²:    {result.train_metrics.outcome_r2:.4f}")
        results[variant_name] = result

    # ------------------------------------------------------------------
    # Step 3: Evaluate on all splits.
    # ------------------------------------------------------------------
    print("\n[3/5] Evaluating on validation and held-out...")
    eval_results = {}
    for variant_name, result in results.items():
        print(f"\n  {variant_name.upper()}:")
        eval_result = evaluate_world_model(result.model, all_records)
        eval_results[variant_name] = eval_result

        for split_name, metrics in [
            ("Train", eval_result.train_metrics),
            ("Validation", eval_result.validation_metrics),
            ("Held-out", eval_result.heldout_metrics),
        ]:
            if metrics.n_samples > 0:
                print(f"    {split_name:12s}: dynamics RMSE={metrics.dynamics.rmse:.6f}  "
                      f"R²={metrics.dynamics.r2:.4f}  "
                      f"outcome RMSE={metrics.outcome_rmse:.6f}  "
                      f"R²={metrics.outcome_r2:.4f}")

        # Rollout.
        rollout = rollout_evaluation(result.model, all_records, split="validation", max_horizon=3)
        print(f"    Rollout (val):  ", end="")
        for h, rmse in zip(rollout.horizons, rollout.rmse_by_horizon):
            print(f"h{h}={rmse:.6f}  ", end="")
        print(f"  (n_traj={rollout.n_trajectories})")

        rollout_held = rollout_evaluation(result.model, all_records, split="held_out", max_horizon=3)
        print(f"    Rollout (held): ", end="")
        for h, rmse in zip(rollout_held.horizons, rollout_held.rmse_by_horizon):
            print(f"h{h}={rmse:.6f}  ", end="")
        print()

    # ------------------------------------------------------------------
    # Step 4: Compare and decide.
    # ------------------------------------------------------------------
    print("\n[4/5] Comparing variants...")

    # Pick the better model based on held-out dynamics RMSE.
    linear_held_rmse = eval_results["linear"].heldout_metrics.dynamics.rmse
    mlp_held_rmse = eval_results["mlp"].heldout_metrics.dynamics.rmse
    linear_held_r2 = eval_results["linear"].heldout_metrics.dynamics.r2
    mlp_held_r2 = eval_results["mlp"].heldout_metrics.dynamics.r2

    print(f"  Linear held-out: RMSE={linear_held_rmse:.6f}  R²={linear_held_r2:.4f}  params={results['linear'].model.n_parameters}")
    print(f"  MLP held-out:    RMSE={mlp_held_rmse:.6f}  R²={mlp_held_r2:.4f}  params={results['mlp'].model.n_parameters}")

    if linear_held_rmse <= mlp_held_rmse:
        best_variant = "linear"
        print(f"\n  → LINEAR wins (simpler, equal or better held-out RMSE)")
    else:
        best_variant = "mlp"
        improvement = (linear_held_rmse - mlp_held_rmse) / linear_held_rmse * 100
        print(f"\n  → MLP wins ({improvement:.1f}% RMSE improvement over linear)")

    best_model = results[best_variant].model
    best_eval = eval_results[best_variant]

    # Set trust report.
    trust = WorldModelTrustReport(
        mean_prediction_error=best_eval.heldout_metrics.dynamics.rmse,
        trust_score=max(0.0, min(1.0, 0.5 + best_eval.heldout_metrics.dynamics.r2 / 2.0)),
        recommended_horizon=1 if best_eval.heldout_metrics.dynamics.r2 < 0.5 else 2,
        recommended_exact_verification_fraction=1.0,  # always verify with exact shadow
        metadata={
            "variant": best_variant,
            "heldout_r2": best_eval.heldout_metrics.dynamics.r2,
        },
    )

    wm = LightweightWorldModel(joint_model=best_model)
    wm.set_trust_report(trust)
    trust_report = wm.trust_report()
    print(f"\n  Trust score: {trust_report.trust_score:.4f}")
    print(f"  Recommended horizon: {trust_report.recommended_horizon}")
    print(f"  Exact verification fraction: {trust_report.recommended_exact_verification_fraction}")

    # ------------------------------------------------------------------
    # Step 5: Save reports.
    # ------------------------------------------------------------------
    print("\n[5/5] Saving reports...")
    report_dir = project_root / "reports" / "v6_exp5"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save serialized models.
    for variant_name, result in results.items():
        model_path = report_dir / f"MODEL_{variant_name.upper()}.json"
        with open(model_path, "w") as f:
            json.dump(result.model.get_state(), f, indent=2)
        print(f"  Saved {model_path.name}")

    # Save full results.
    full_results = {
        "experiment": "v6.0-exp5",
        "architecture": "lightweight_latent_dynamics",
        "authorized_by": "v6.0-exp4.2 (QUALIFIED_SIMPLE)",
        "best_variant": best_variant,
        "variants": {
            name: {
                "config": result.config.to_log(),
                "n_parameters": result.model.n_parameters,
                "n_train": result.n_train,
                "elapsed_seconds": result.elapsed_seconds,
                "train_metrics": result.train_metrics.to_log(),
                "evaluation": eval_results[name].to_log(),
            }
            for name, result in results.items()
        },
        "trust_report": {
            "trust_score": trust_report.trust_score,
            "recommended_horizon": trust_report.recommended_horizon,
            "recommended_exact_verification_fraction": trust_report.recommended_exact_verification_fraction,
            "mean_prediction_error": trust_report.mean_prediction_error,
        },
    }
    with open(report_dir / "RESULTS.json", "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"  Saved RESULTS.json")

    # Save scientific report.
    report_md = f"""# v6.0-exp5 — Lightweight Latent World Model

**Architecture:** `lightweight_latent_dynamics`
**Authorized by:** v6.0-exp4.2 (QUALIFIED_SIMPLE)

## 1. Question

Can a lightweight latent dynamics model predict the next structural
state and outcome of a mutation, generalizing to unseen graph families?

## 2. Models

### Linear Dynamics
- Parameters: {results['linear'].model.n_parameters}
- Architecture: z_{{t+1}} = A·z_t + B·a_t + c

### MLP Dynamics
- Parameters: {results['mlp'].model.n_parameters}
- Architecture: z_{{t+1}} = MLP([z_t, a_t]) with hidden_dim=32

## 3. Results

### Held-Out Performance

| Variant | Dynamics RMSE | Dynamics R² | Outcome RMSE | Outcome R² |
|---------|--------------|-------------|--------------|------------|
| Linear  | {linear_held_rmse:.6f} | {linear_held_r2:.4f} | {eval_results['linear'].heldout_metrics.outcome_rmse:.6f} | {eval_results['linear'].heldout_metrics.outcome_r2:.4f} |
| MLP     | {mlp_held_rmse:.6f} | {mlp_held_r2:.4f} | {eval_results['mlp'].heldout_metrics.outcome_rmse:.6f} | {eval_results['mlp'].heldout_metrics.outcome_r2:.4f} |

### Multi-Step Rollout (Validation)

| Horizon | Linear RMSE | MLP RMSE |
|---------|-------------|----------|
"""
    for h in range(3):
        lin_r = eval_results["linear"].rollout_report.get("rmse_by_horizon", [0,0,0])[h]
        mlp_r = eval_results["mlp"].rollout_report.get("rmse_by_horizon", [0,0,0])[h]
        report_md += f"| h={h+1} | {lin_r:.6f} | {mlp_r:.6f} |\n"

    report_md += f"""
## 4. Decision

**Best variant:** `{best_variant}`

**Trust score:** {trust_report.trust_score:.4f}
**Recommended planning horizon:** {trust_report.recommended_horizon}
**Exact verification fraction:** {trust_report.recommended_exact_verification_fraction}

## 5. Authority Boundary

The world model is **advisory-only**. It does not mutate authoritative
runtime state. The v5.11 CommitChannel remains the sole authority
boundary. All world model predictions used for planning must still
pass through governance and exact shadow execution.

## 6. Next Steps

The trained world model can now be used as a proposal/prediction
layer in future MPC planning experiments (exp6+). The trust report
informs how much to rely on model predictions vs exact verification.
"""
    with open(report_dir / "SCIENTIFIC_REPORT.md", "w") as f:
        f.write(report_md)
    print(f"  Saved SCIENTIFIC_REPORT.md")

    print(f"\n{'='*72}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*72}")
    print(f"\nBest variant: {best_variant}")
    print(f"Held-out dynamics R²: {best_eval.heldout_metrics.dynamics.r2:.4f}")
    print(f"Held-out outcome R²: {best_eval.heldout_metrics.outcome_r2:.4f}")
    print(f"Trust score: {trust_report.trust_score:.4f}")
    print(f"Reports: {report_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
