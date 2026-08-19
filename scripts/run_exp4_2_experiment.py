#!/usr/bin/env python
"""Run the v6.0-exp4.2 held-out structural prediction study.

This script executes the full scientific experiment:
1. Generate the exp2 dataset (deterministic, frozen seed)
2. Freeze dataset hashes
3. Train all encoder × predictor × target combinations on TRAIN
4. Select on VALIDATION
5. Lock finalists
6. Open held-out ONCE
7. Generate final scientific report
8. Write machine-readable conclusion

The experiment is designed to answer:
    Can LGAE predict which structural mutation will work best on an unseen graph?

Usage:
    python scripts/run_exp4_2_experiment.py
"""
from __future__ import annotations

import sys
import os
import json
import time
import hashlib
from pathlib import Path

# Add src to path.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from lgae_v3.experimental.dataset_generator import (
    DatasetGenerator,
    DATASET_SCHEMA_VERSION,
    GENERATOR_VERSION,
)
from lgae_v3.experimental.graph_families import FrozenGraphFamilyRegistry
from lgae_v3.experimental.exp4_2 import (
    ScientificRunner,
    ScientificResult,
    ScientificConclusion,
    authorize_exp5,
    generate_scientific_report,
    generate_machine_readable_conclusion,
    DatasetFreeze,
    freeze_dataset,
)
from lgae_v3.experimental.exp4_2.experiment_config import default_experiment_config


def main() -> int:
    print("=" * 72)
    print("LGAE v6.0-exp4.2 — Held-Out Structural Prediction Study")
    print("Experiment ID: LGAE_V6_EXP4_2_STRUCTURAL_PREDICTION_STUDY_001")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Generate the exp2 dataset.
    # ------------------------------------------------------------------
    print("\n[1/8] Generating exp2 dataset...")
    t0 = time.time()

    registry = FrozenGraphFamilyRegistry()
    # Compute graph family registry hash.
    gf_log = registry.to_log()
    gf_hash = hashlib.sha256(
        json.dumps(gf_log, sort_keys=True).encode()
    ).hexdigest()[:16]

    generator = DatasetGenerator(seed=42, registry=registry, n_negative_samples=3)
    datasets = generator.generate_all_splits(n_steps=5, n_episodes=1)

    n_train = datasets["train"].n_records
    n_val = datasets["validation"].n_records
    n_held = datasets["held_out"].n_records
    print(f"  Train:       {n_train} records")
    print(f"  Validation:  {n_val} records")
    print(f"  Held-out:    {n_held} records")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Freeze dataset hashes.
    # ------------------------------------------------------------------
    print("\n[2/8] Freezing dataset hashes...")
    dataset_schema_hash = hashlib.sha256(
        f"{DATASET_SCHEMA_VERSION}:{GENERATOR_VERSION}".encode()
    ).hexdigest()[:16]
    feature_schema_hash = hashlib.sha256(
        "exp4.2-global-features-24d".encode()
    ).hexdigest()[:16]

    freeze = freeze_dataset(
        datasets,
        dataset_schema_hash=dataset_schema_hash,
        feature_schema_hash=feature_schema_hash,
        graph_family_registry_hash=gf_hash,
        seed=42,
    )
    print(f"  Freeze hash: {freeze.freeze_hash}")
    print(f"  Train hash:  {freeze.train_split_hash[:16]}...")
    print(f"  Val hash:    {freeze.validation_split_hash[:16]}...")
    print(f"  Held hash:   {freeze.heldout_split_hash[:16]}...")

    # Save freeze manifests.
    freeze_dir = project_root / "reports" / "v6_exp4_2" / "dataset_freeze"
    freeze.save(freeze_dir)
    print(f"  Saved to: {freeze_dir}")

    # ------------------------------------------------------------------
    # Step 3: Configure the experiment.
    # ------------------------------------------------------------------
    print("\n[3/8] Configuring experiment...")
    config = default_experiment_config()
    # Use fewer seeds for speed — still multi-seed.
    config.seeds = [42, 123]
    config.n_epochs = 30
    config.n_ensemble = 2

    # Prune to the scientifically meaningful matrix.
    # Per the spec: don't blindly run every combination.
    # Use the minimum encoder set and minimum predictor set,
    # skipping invalid combinations (e.g., classification predictors
    # on regression targets).
    from lgae_v3.experimental.exp4_2.experiment_config import EncoderConfig, PredictorConfig
    from lgae_v3.experimental.encoders import EncoderRegistry
    from lgae_v3.experimental.models.model_registry import ModelRegistry

    # Minimum encoder set (per spec).
    focused_encoders = [
        "minimal-control", "global", "global-local", "semantic-action",
        "geometric", "spectral", "learned-graph", "hybrid",
    ]
    config.encoders = []
    for enc_id in focused_encoders:
        info = EncoderRegistry.encoder_info(enc_id)
        config.encoders.append(EncoderConfig(
            encoder_id=enc_id,
            version=info.get("version", ""),
            dimension=info.get("dimension", 0),
            schema_hash=info.get("schema_hash", ""),
            requires_fit=info.get("requires_fit", True),
        ))

    # Minimum predictor set (per spec).
    focused_predictors = [
        "global_mean", "mutation_type_mean", "nearest_experience",
        "linear", "ridge", "tree", "mlp",
        "pointwise_rank", "pairwise_rank",
    ]
    config.predictors = []
    for pred_id in focused_predictors:
        info = ModelRegistry.model_info(pred_id)
        config.predictors.append(PredictorConfig(
            predictor_id=pred_id,
            model_type=info.get("model_type", ""),
            version=info.get("version", ""),
            deterministic=info.get("deterministic", True),
        ))

    # Focus on the primary target first.
    config.targets = ["realized_delta", "sign_delta"]

    print(f"  Encoders: {len(config.encoders)} ({focused_encoders})")
    print(f"  Predictors: {len(config.predictors)} ({focused_predictors})")
    print(f"  Targets: {config.targets}")
    print(f"  Seeds: {config.seeds}")
    total_combos = len(config.encoders) * len(config.predictors) * len(config.targets) * len(config.seeds)
    print(f"  Total training runs: ~{total_combos}")

    # ------------------------------------------------------------------
    # Step 4: Run the scientific runner.
    # ------------------------------------------------------------------
    print("\n[4/8] Running scientific runner (train → validate → lock → heldout)...")
    t0 = time.time()

    runner = ScientificRunner(config=config)

    # Prepare (freezes dataset, transitions to TRAINING).
    runner.prepare(
        datasets,
        dataset_schema_hash=dataset_schema_hash,
        feature_schema_hash=feature_schema_hash,
        graph_family_registry_hash=gf_hash,
    )
    print(f"  Dataset frozen. State: {runner.state.state}")

    # Collect all records.
    all_records = (
        list(datasets["train"].records)
        + list(datasets["validation"].records)
        + list(datasets["held_out"].records)
    )

    # Train all combinations.
    print(f"  Training {len(config.encoders)} × {len(config.predictors)} × {len(config.targets)} combinations...")
    train_results = runner.train(all_records)
    print(f"  Trained {len(train_results)} combinations. State: {runner.state.state}")

    # Validate.
    val_results = runner.validate()
    print(f"  Validated. State: {runner.state.state}")

    # ------------------------------------------------------------------
    # Step 5: Lock finalists.
    # ------------------------------------------------------------------
    print("\n[5/8] Locking finalists based on validation performance...")
    lock = runner.lock_finalists()
    print(f"  Locked {len(lock.finalists)} finalists.")
    print(f"  Config hash: {lock.config_hash}")
    print(f"  State: {runner.state.state}")

    # ------------------------------------------------------------------
    # Step 6: Open held-out ONCE.
    # ------------------------------------------------------------------
    print("\n[6/8] Opening held-out (ONE-SHOT, no retraining)...")
    heldout_results = runner.open_heldout()
    print(f"  Evaluated {len(heldout_results)} finalists on held-out.")
    print(f"  State: {runner.state.state}")

    # ------------------------------------------------------------------
    # Step 7: Finalize and generate conclusion.
    # ------------------------------------------------------------------
    print("\n[7/8] Finalizing scientific conclusion...")
    conclusion = runner.finalize()
    print(f"  Status: {conclusion.scientific_status}")
    print(f"  Structural signal detected: {conclusion.structural_signal_detected}")
    print(f"  Generalizes to held-out: {conclusion.generalizes_to_heldout}")
    print(f"  Best encoder: {conclusion.best_encoder}")
    print(f"  Best predictor: {conclusion.best_model}")
    print(f"  Best held-out Spearman: {conclusion.best_heldout_spearman:.4f}")
    print(f"  Best held-out regret: {conclusion.best_heldout_regret:.4f}")
    print(f"  CF→Real transfer OK: {conclusion.cf_real_transfer_ok}")
    print(f"  Uncertainty useful: {conclusion.uncertainty_useful}")
    print(f"  exp5 authorized: {conclusion.exp5_authorized}")
    if conclusion.recommended_exp5_architecture:
        print(f"  Recommended exp5 architecture: {conclusion.recommended_exp5_architecture}")
    if conclusion.limitations:
        print(f"  Limitations:")
        for lim in conclusion.limitations:
            print(f"    - {lim}")

    exp5_authorized = authorize_exp5(conclusion)
    print(f"\n  authorize_exp5() = {exp5_authorized}")

    # ------------------------------------------------------------------
    # Step 8: Generate the full scientific report.
    # ------------------------------------------------------------------
    print("\n[8/8] Generating scientific report...")
    report_dir = project_root / "reports" / "v6_exp4_2"
    report_dir.mkdir(parents=True, exist_ok=True)

    generate_scientific_report(
        heldout_results,
        conclusion,
        report_dir,
        dataset_freeze_log=freeze.to_log(),
        experiment_config_log=config.to_log(),
        finalist_lock_log=lock.to_log(),
    )

    # Save conclusion separately.
    generate_machine_readable_conclusion(conclusion, report_dir / "CONCLUSION.json")

    # Print the competition table.
    print("\n" + "=" * 72)
    print("HELD-OUT COMPETITION TABLE")
    print("=" * 72)
    print(f"{'Encoder':<20} {'Predictor':<20} {'Target':<15} "
          f"{'Val Score':<12} {'Held ρ':<10} {'Regret':<10} {'CF Gap':<10}")
    print("-" * 100)
    for r in heldout_results:
        val_score = r.mean_validation_score
        held_sp = r.heldout_metrics.get("spearman", 0.0)
        regret = r.regret.get("mean_regret", 0.0) if r.regret else 0.0
        cf_gap = r.cf_real.get("gap_cf_to_real_spearman", 0.0) if r.cf_real else 0.0
        print(f"{r.encoder_id:<20} {r.predictor_id:<20} {r.target:<15} "
              f"{val_score:<12.4f} {held_sp:<10.4f} {regret:<10.4f} {cf_gap:<10.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal experiment elapsed: {elapsed:.1f}s")
    print(f"Report saved to: {report_dir}")
    print(f"\nScientific status: {conclusion.scientific_status}")
    print(f"exp5 authorized: {exp5_authorized}")

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)

    return 0 if conclusion.exp5_authorized else 1


if __name__ == "__main__":
    sys.exit(main())
