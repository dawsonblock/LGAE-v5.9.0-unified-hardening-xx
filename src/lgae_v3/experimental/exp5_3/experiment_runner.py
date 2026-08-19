"""Main experiment runner for v6.0-exp5.3.

Runs:
1. Representation ladder (R0-R7) with fixed delta predictor
2. Leave-one-family-out with each representation
3. Component-wise adaptation (bias-only, scale+offset, low-rank, full)
4. Dynamics-OOD distance analysis
5. Parametric family evaluation
6. TEST-C generator extrapolation
7. Extended rollout with proper delta metrics

All evaluation uses REALIZED records only.
Primary metric: delta R² on invariant dimensions.
Always compared against zero-delta baseline.
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
from .representations import (
    REPRESENTATION_LADDER, RepresentationConfig,
    extract_representation, extract_invariant,
    INVARIANT_INDICES, CONTEXT_INDICES, DERIVED_INDICES,
    RepresentationMetrics, compute_representation_metrics,
)
from .adaptation import (
    ComponentAdapter, BiasOnlyAdapter, ScaleOffsetAdapter,
    LowRankAdapter, FullRetrainAdapter, create_adapter,
)
from .dynamics_ood import (
    compute_dynamics_ood_distance, compute_family_dynamics_ood,
    correlate_dynamics_ood_with_error,
    correlate_dynamics_ood_with_uncertainty,
)


def is_realized(record: Any) -> bool:
    """Check if a record is a realized (not counterfactual) transition."""
    p = getattr(record, "provenance", None)
    if p is not None and hasattr(p, "value"):
        return "realized" in p.value.lower()
    return False


def extract_realized_data(
    records: list[Any],
    *,
    split: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract REALIZED records only.

    Returns:
        z_t, a_t, z_next, families
    """
    z_t_list, a_t_list, z_next_list, families = [], [], [], []

    for r in records:
        if split is not None and getattr(r, "split", "") != split:
            continue
        if r.structural_state_after is None:
            continue
        if not is_realized(r):
            continue

        state_before = r.structural_state_before
        state_after = r.structural_state_after

        sv_before = encode_normalized_state(state_before)
        sv_after = encode_normalized_state(state_after)

        n_nodes = int(getattr(state_before, "n_nodes", 20))
        degree_mean = float(getattr(state_before, "degree_mean", 2.0))

        av = encode_normalized_action(
            r.action, r.action_target,
            n_nodes=n_nodes, degree_mean=degree_mean,
        )

        z_t_list.append(sv_before.vector)
        a_t_list.append(av.vector)
        z_next_list.append(sv_after.vector)
        families.append(getattr(r, "graph_family", "unknown"))

    if not z_t_list:
        return (
            np.zeros((0, NORM_STATE_DIM)),
            np.zeros((0, NORM_ACTION_DIM)),
            np.zeros((0, NORM_STATE_DIM)),
            [],
        )

    return (
        np.array(z_t_list, dtype=np.float64),
        np.array(a_t_list, dtype=np.float64),
        np.array(z_next_list, dtype=np.float64),
        families,
    )


def run_representation_ladder(
    train_records: list[Any],
    test_records: list[Any],
    *,
    test_name: str = "TEST-B",
) -> dict[str, RepresentationMetrics]:
    """Run the representation ladder with a fixed delta predictor.

    For each representation R0-R7:
    1. Extract sub-representation from normalized vectors
    2. Train a delta predictor on realized train data
    3. Evaluate on realized test data
    4. Compute delta R², invariant delta R², zero-delta baseline
    """
    # Extract realized data.
    train_z, train_a, train_zn, train_fam = extract_realized_data(train_records, split="train")
    test_z, test_a, test_zn, test_fam = extract_realized_data(test_records, split="held_out")

    print(f"  Train realized: {len(train_z)}  Test realized: {len(test_z)}")

    if len(train_z) < 10 or len(test_z) < 5:
        print("  WARNING: Insufficient realized data")
        return {}

    # Compute actual deltas.
    train_delta = train_zn - train_z
    test_delta = test_zn - test_z

    # Invariant mask for the full 20-dim vector.
    inv_mask = np.array([i in INVARIANT_INDICES for i in range(NORM_STATE_DIM)])

    results: dict[str, RepresentationMetrics] = {}

    for rep_name, rep_config in REPRESENTATION_LADDER.items():
        # Extract sub-representation.
        rep_train_z = extract_representation(train_z, rep_config)
        rep_train_a = train_a  # action encoding stays the same
        rep_test_z = extract_representation(test_z, rep_config)
        rep_test_a = test_a

        # For delta, we need to extract the same dims from z_next.
        rep_train_zn = extract_representation(train_zn, rep_config)
        rep_test_zn = extract_representation(test_zn, rep_config)
        rep_train_delta = rep_train_zn - rep_train_z
        rep_test_delta = rep_test_zn - rep_test_z

        # Invariant mask for this representation.
        rep_inv_mask = np.array([
            i in INVARIANT_INDICES for i in rep_config.indices
        ])

        # Train delta model.
        model = DeltaDynamicsModel(
            mode="delta", regularization=1e-3, seed=42,
            state_dim=rep_config.dim, action_dim=NORM_ACTION_DIM,
        )
        model.fit(rep_train_z, rep_train_a, rep_train_zn, split="train")

        # Predict.
        pred_delta = model.predict_raw_batch(rep_test_z, rep_test_a)
        pred_state = model.predict_batch(rep_test_z, rep_test_a)

        # Compute metrics.
        m = compute_representation_metrics(
            pred_delta, rep_test_delta,
            pred_state=pred_state, actual_state=rep_test_zn,
            invariant_mask=rep_inv_mask if rep_inv_mask.any() else None,
            representation=rep_name,
            n_train=len(train_z),
        )
        results[rep_name] = m

        print(f"  {rep_name:25s} dim={rep_config.dim:2d}  "
              f"ΔR²={m.delta_r2:8.4f}  ΔR²_inv={m.delta_r2_invariant:8.4f}  "
              f"zero_ΔR²={m.zero_delta_r2:8.4f}  beats={m.beats_zero_delta}")

    return results


def run_loo_with_representations(
    all_records: list[Any],
    rep_config: RepresentationConfig,
) -> dict[str, RepresentationMetrics]:
    """Run leave-one-family-out with a specific representation."""
    # Get all training families.
    train_records = [r for r in all_records if getattr(r, "split", "") == "train"]
    all_families = sorted(set(
        getattr(r, "graph_family", "unknown")
        for r in train_records if is_realized(r) and r.structural_state_after is not None
    ))

    results: dict[str, RepresentationMetrics] = {}

    for held_out in all_families:
        # Split.
        train_recs = [r for r in train_records
                      if is_realized(r) and r.structural_state_after is not None
                      and getattr(r, "graph_family", "") != held_out]
        test_recs = [r for r in train_records
                     if is_realized(r) and r.structural_state_after is not None
                     and getattr(r, "graph_family", "") == held_out]

        if len(train_recs) < 10 or len(test_recs) < 3:
            continue

        z_t, a_t, z_next, _ = extract_realized_data(train_recs)
        tz_t, ta_t, tz_next, _ = extract_realized_data(test_recs)

        # Extract representation.
        rz_t = extract_representation(z_t, rep_config)
        rz_next = extract_representation(z_next, rep_config)
        rtz_t = extract_representation(tz_t, rep_config)
        rtz_next = extract_representation(tz_next, rep_config)

        model = DeltaDynamicsModel(
            mode="delta", regularization=1e-3, seed=42,
            state_dim=rep_config.dim, action_dim=NORM_ACTION_DIM,
        )
        model.fit(rz_t, a_t, rz_next, split="train")

        pred_delta = model.predict_raw_batch(rtz_t, ta_t)
        actual_delta = rtz_next - rtz_t

        rep_inv_mask = np.array([
            i in INVARIANT_INDICES for i in rep_config.indices
        ])

        m = compute_representation_metrics(
            pred_delta, actual_delta,
            invariant_mask=rep_inv_mask if rep_inv_mask.any() else None,
            representation=f"LOO_{held_out}",
            n_train=len(rz_t),
        )
        results[held_out] = m

    return results


def run_adaptation_study(
    train_records: list[Any],
    test_records: list[Any],
    *,
    rep_config: RepresentationConfig,
    k_shots: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run component-wise adaptation study.

    For each TEST-B family and each k:
    1. Train global model on train realized data
    2. Evaluate 0-shot
    3. Adapt with k samples using each adaptation strategy
    4. Evaluate adapted model
    """
    if k_shots is None:
        k_shots = [0, 5, 10, 25, 50]

    # Extract global training data.
    train_z, train_a, train_zn, _ = extract_realized_data(train_records, split="train")
    rep_train_z = extract_representation(train_z, rep_config)
    rep_train_zn = extract_representation(train_zn, rep_config)

    # Train global model.
    global_model = DeltaDynamicsModel(
        mode="delta", regularization=1e-3, seed=42,
        state_dim=rep_config.dim, action_dim=NORM_ACTION_DIM,
    )
    global_model.fit(rep_train_z, train_a, rep_train_zn, split="train")

    # Get TEST-B families.
    test_realized = [r for r in test_records
                     if is_realized(r) and r.structural_state_after is not None
                     and getattr(r, "split", "") == "held_out"]
    test_families = sorted(set(getattr(r, "graph_family", "") for r in test_realized))

    adaptation_types = ["none", "bias_only", "scale_offset", "low_rank_r2", "full_retrain"]
    results = []

    for family in test_families:
        fam_records = [r for r in test_realized if getattr(r, "graph_family", "") == family]
        if len(fam_records) < 10:
            continue

        n_total = len(fam_records)
        n_eval = max(5, n_total // 2)
        eval_records = fam_records[n_eval:]
        adapt_records = fam_records[:n_eval]

        # Extract eval data.
        eval_z, eval_a, eval_zn, _ = extract_realized_data(eval_records)
        rep_eval_z = extract_representation(eval_z, rep_config)
        rep_eval_zn = extract_representation(eval_zn, rep_config)
        actual_delta = rep_eval_zn - rep_eval_z

        rep_inv_mask = np.array([
            i in INVARIANT_INDICES for i in rep_config.indices
        ])

        for k in k_shots:
            k_actual = min(k, len(adapt_records))
            adapt_subset = adapt_records[:k_actual]

            for adapt_type in adaptation_types:
                if k == 0 and adapt_type != "none":
                    continue  # 0-shot only has "none"

                if k > 0 and len(adapt_subset) == 0:
                    continue

                # Get adaptation data.
                if k > 0:
                    ad_z, ad_a, ad_zn, _ = extract_realized_data(adapt_subset)
                    rep_ad_z = extract_representation(ad_z, rep_config)
                    rep_ad_zn = extract_representation(ad_zn, rep_config)
                else:
                    ad_z = ad_a = ad_zn = np.zeros((0, rep_config.dim))
                    rep_ad_z = rep_ad_zn = np.zeros((0, rep_config.dim))

                # Create adapter.
                if adapt_type == "none":
                    # No adaptation — just use global model.
                    pred_delta = global_model.predict_raw_batch(rep_eval_z, eval_a)
                    m = compute_representation_metrics(
                        pred_delta, actual_delta,
                        invariant_mask=rep_inv_mask if rep_inv_mask.any() else None,
                        representation=f"{family}_k{k}_{adapt_type}",
                        n_train=len(rep_train_z),
                    )
                else:
                    try:
                        adapter = create_adapter(
                            adapt_type, global_model,
                            global_z_t=rep_train_z, global_a_t=train_a,
                            global_z_next=rep_train_zn, rank=2,
                        )
                        if k > 0:
                            adapter.fit(rep_ad_z, ad_a, rep_ad_zn)
                        pred_delta = adapter.predict_delta(rep_eval_z, eval_a)
                        m = compute_representation_metrics(
                            pred_delta, actual_delta,
                            invariant_mask=rep_inv_mask if rep_inv_mask.any() else None,
                            representation=f"{family}_k{k}_{adapt_type}",
                            n_train=len(rep_train_z),
                        )
                    except Exception as e:
                        m = RepresentationMetrics(
                            representation=f"{family}_k{k}_{adapt_type}",
                            n_samples=len(eval_z),
                        )

                results.append({
                    "family": family,
                    "k_shots": k,
                    "adaptation_type": adapt_type,
                    "delta_r2": m.delta_r2,
                    "delta_r2_invariant": m.delta_r2_invariant,
                    "zero_delta_r2": m.zero_delta_r2,
                    "beats_zero_delta": m.beats_zero_delta,
                    "rmse": m.delta_rmse,
                    "n_eval": m.n_samples,
                })

    return results


def run_dynamics_ood_analysis(
    train_records: list[Any],
    test_records: list[Any],
    *,
    rep_config: RepresentationConfig,
) -> dict[str, Any]:
    """Run dynamics-OOD distance analysis."""
    train_z, train_a, train_zn, train_fam = extract_realized_data(train_records, split="train")
    test_z, test_a, test_zn, test_fam = extract_realized_data(test_records, split="held_out")

    if len(train_z) == 0 or len(test_z) == 0:
        return {"error": "insufficient data"}

    rep_train_z = extract_representation(train_z, rep_config)
    rep_test_z = extract_representation(test_z, rep_config)
    train_delta = train_zn - train_z
    test_delta = test_zn - test_z
    rep_train_delta = extract_representation(train_delta, rep_config)
    rep_test_delta = extract_representation(test_delta, rep_config)

    # Compute dynamics-OOD.
    ood = compute_family_dynamics_ood(
        rep_test_z, test_a, rep_test_delta, test_fam,
        rep_train_z, train_a, rep_train_delta,
    )

    # Train a model to get errors.
    model = DeltaDynamicsModel(
        mode="delta", regularization=1e-3, seed=42,
        state_dim=rep_config.dim, action_dim=NORM_ACTION_DIM,
    )
    model.fit(rep_train_z, train_a, extract_representation(train_zn, rep_config), split="train")

    pred_delta = model.predict_raw_batch(rep_test_z, test_a)
    errors = np.sqrt(np.sum((pred_delta - rep_test_delta) ** 2, axis=1))

    ood_error = correlate_dynamics_ood_with_error(
        np.array(ood["distances"]), errors,
    )

    return {
        "family_distances": ood["family_distances"],
        "mean_distance": ood["mean_distance"],
        "ood_error_corr": ood_error,
    }
