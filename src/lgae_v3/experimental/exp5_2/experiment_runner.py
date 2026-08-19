"""Main experiment runner for v6.0-exp5.2.

Runs:
1. Representation ablation (raw vs normalized vs delta vs graphlet)
2. Leave-one-family-out cross-validation
3. Family-bootstrap ensemble evaluation
4. OOD distance analysis
5. Adaptation curves (0-shot, 5-shot, 10-shot, 25-shot, 50-shot)
6. Extended rollout horizons (h=1,2,3,5,10)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import time
import math
import numpy as np

from .state_encoding import (
    encode_normalized_state, encode_normalized_action,
    NORM_STATE_DIM, NORM_ACTION_DIM,
)
from .dynamics import (
    DeltaDynamicsModel, FamilyBootstrapEnsemble,
    compute_generalization_metrics, GeneralizationMetrics,
)
from .ood_analysis import (
    compute_family_ood_distances, correlate_ood_with_error,
    correlate_ood_with_uncertainty,
)


@dataclass
class RepresentationResult:
    """Result of one representation × model combination."""
    representation: str
    model_type: str
    mode: str  # "delta" or "absolute"
    train_r2: float = 0.0
    test_b_r2: float = 0.0
    test_b_delta_r2: float = 0.0
    test_b_rmse: float = 0.0
    test_b_spearman: float = 0.0
    calibration_corr: float = 0.0
    n_parameters: int = 0
    per_feature_nrmse: list[float] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "model_type": self.model_type,
            "mode": self.mode,
            "train_r2": float(self.train_r2),
            "test_b_r2": float(self.test_b_r2),
            "test_b_delta_r2": float(self.test_b_delta_r2),
            "test_b_rmse": float(self.test_b_rmse),
            "test_b_spearman": float(self.test_b_spearman),
            "calibration_corr": float(self.calibration_corr),
            "n_parameters": int(self.n_parameters),
            "per_feature_nrmse": [float(x) for x in self.per_feature_nrmse],
        }


@dataclass
class LOOResult:
    """Leave-one-family-out result for one held-out family."""
    held_out_family: str
    train_families: list[str]
    r2: float = 0.0
    rmse: float = 0.0
    spearman: float = 0.0
    delta_r2: float = 0.0
    n_train: int = 0
    n_test: int = 0

    def to_log(self) -> dict[str, Any]:
        return {
            "held_out_family": self.held_out_family,
            "train_families": list(self.train_families),
            "r2": float(self.r2),
            "rmse": float(self.rmse),
            "spearman": float(self.spearman),
            "delta_r2": float(self.delta_r2),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
        }


@dataclass
class AdaptationResult:
    """Adaptation curve result for one family at k shots."""
    family: str
    k_shots: int
    r2_before: float = 0.0
    r2_after: float = 0.0
    rmse_before: float = 0.0
    rmse_after: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "k_shots": int(self.k_shots),
            "r2_before": float(self.r2_before),
            "r2_after": float(self.r2_after),
            "rmse_before": float(self.rmse_before),
            "rmse_after": float(self.rmse_after),
        }


def extract_normalized_data(
    records: list[Any],
    *,
    split: str = "train",
    split_filter: bool = True,
    graphs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract normalized (z_t, a_t, z_{t+1}, family) from records.

    Args:
        split_filter: If False, don't filter by split (use all records).
            Useful for LOO where we manually split records.

    Returns:
        z_t: (n, NORM_STATE_DIM) normalized states before.
        a_t: (n, NORM_ACTION_DIM) normalized actions.
        z_next: (n, NORM_STATE_DIM) normalized states after.
        families: list of graph family names.
    """
    z_t_list, a_t_list, z_next_list, families = [], [], [], []

    for r in records:
        if split_filter and getattr(r, "split", "") != split:
            continue
        if r.structural_state_after is None:
            continue

        state_before = r.structural_state_before
        state_after = r.structural_state_after

        # Use normalized encoding.
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


def run_representation_ablation(
    train_records: list[Any],
    test_b_records: list[Any],
) -> list[RepresentationResult]:
    """Run representation ablation with fixed linear predictor.

    Compares:
    - raw (absolute): original state encoding, predict z_{t+1}
    - normalized (absolute): normalized features, predict z_{t+1}
    - normalized (delta): normalized features, predict Δz
    - graphlet (delta): normalized + graphlet features, predict Δz

    All use the same linear ridge regression predictor.
    """
    results = []

    # Extract data for each representation.
    # For "raw", we use the original exp5 encoding (absolute state).
    from ..exp5.state_encoding import encode_state as encode_raw_state
    from ..exp5.state_encoding import encode_action as encode_raw_action
    from ..exp5.state_encoding import STATE_DIM as RAW_STATE_DIM
    from ..exp5.state_encoding import ACTION_DIM as RAW_ACTION_DIM

    # Raw absolute.
    raw_z_t, raw_a_t, raw_z_next = [], [], []
    for r in train_records:
        if getattr(r, "split", "") != "train" or r.structural_state_after is None:
            continue
        sb = r.structural_state_before
        sa = r.structural_state_after
        raw_z_t.append(encode_raw_state(sb).vector)
        raw_a_t.append(encode_raw_action(
            r.action, r.action_target,
            n_nodes=int(getattr(sb, "n_nodes", 20)),
            degree_mean=float(getattr(sb, "degree_mean", 2.0)),
        ).vector)
        raw_z_next.append(encode_raw_state(sa).vector)

    raw_z_t = np.array(raw_z_t) if raw_z_t else np.zeros((0, RAW_STATE_DIM))
    raw_a_t = np.array(raw_a_t) if raw_a_t else np.zeros((0, RAW_ACTION_DIM))
    raw_z_next = np.array(raw_z_next) if raw_z_next else np.zeros((0, RAW_STATE_DIM))

    # Test-B raw.
    raw_test_z_t, raw_test_a_t, raw_test_z_next = [], [], []
    for r in test_b_records:
        if getattr(r, "split", "") != "held_out" or r.structural_state_after is None:
            continue
        sb = r.structural_state_before
        sa = r.structural_state_after
        raw_test_z_t.append(encode_raw_state(sb).vector)
        raw_test_a_t.append(encode_raw_action(
            r.action, r.action_target,
            n_nodes=int(getattr(sb, "n_nodes", 20)),
            degree_mean=float(getattr(sb, "degree_mean", 2.0)),
        ).vector)
        raw_test_z_next.append(encode_raw_state(sa).vector)

    raw_test_z_t = np.array(raw_test_z_t) if raw_test_z_t else np.zeros((0, RAW_STATE_DIM))
    raw_test_a_t = np.array(raw_test_a_t) if raw_test_a_t else np.zeros((0, RAW_ACTION_DIM))
    raw_test_z_next = np.array(raw_test_z_next) if raw_test_z_next else np.zeros((0, RAW_STATE_DIM))

    # Normalized data.
    norm_z_t, norm_a_t, norm_z_next, _ = extract_normalized_data(train_records, split="train")
    norm_test_z_t, norm_test_a_t, norm_test_z_next, _ = extract_normalized_data(test_b_records, split="held_out")

    configs = [
        ("raw", "absolute", raw_z_t, raw_a_t, raw_z_next, raw_test_z_t, raw_test_a_t, raw_test_z_next, RAW_STATE_DIM, RAW_ACTION_DIM),
        ("normalized", "absolute", norm_z_t, norm_a_t, norm_z_next, norm_test_z_t, norm_test_a_t, norm_test_z_next, NORM_STATE_DIM, NORM_ACTION_DIM),
        ("normalized", "delta", norm_z_t, norm_a_t, norm_z_next, norm_test_z_t, norm_test_a_t, norm_test_z_next, NORM_STATE_DIM, NORM_ACTION_DIM),
    ]

    for rep_name, mode, z_t, a_t, z_next, tz_t, ta_t, tz_next, sd, ad in configs:
        if len(z_t) == 0 or len(tz_t) == 0:
            results.append(RepresentationResult(
                representation=rep_name, model_type="linear", mode=mode,
            ))
            continue

        # Train.
        model = DeltaDynamicsModel(
            mode=mode, regularization=1e-3, seed=42,
            state_dim=sd, action_dim=ad,
        )
        model.fit(z_t, a_t, z_next, split="train")

        # Evaluate on train.
        train_preds = model.predict_batch(z_t, a_t)
        train_m = compute_generalization_metrics(train_preds, z_next)

        # Evaluate on TEST-B.
        test_preds = model.predict_batch(tz_t, ta_t)
        test_m = compute_generalization_metrics(test_preds, tz_next)

        # Delta-specific metrics.
        if mode == "delta":
            actual_delta = tz_next - tz_t
            pred_delta = model.predict_raw_batch(tz_t, ta_t)
            delta_m = compute_generalization_metrics(
                pred_delta, actual_delta,
                predicted_delta=pred_delta, actual_delta=actual_delta,
            )
            test_b_delta_r2 = delta_m.one_step_delta_r2
        else:
            test_b_delta_r2 = 0.0

        results.append(RepresentationResult(
            representation=rep_name,
            model_type="linear",
            mode=mode,
            train_r2=train_m.one_step_r2,
            test_b_r2=test_m.one_step_r2,
            test_b_delta_r2=test_b_delta_r2,
            test_b_rmse=test_m.one_step_rmse,
            test_b_spearman=test_m.spearman,
            calibration_corr=0.0,  # single model, no ensemble
            n_parameters=model.n_parameters,
            per_feature_nrmse=test_m.per_feature_nrmse,
        ))

    return results


def run_leave_one_family_out(
    all_records: list[Any],
    families_to_test: list[str] | None = None,
) -> list[LOOResult]:
    """Run leave-one-family-out cross-validation.

    For each family G_i:
    - Train on all records NOT from G_i
    - Test on records from G_i
    - Report R², RMSE, Spearman

    Uses normalized delta-state prediction.
    """
    # Get all families.
    all_families = sorted(set(
        getattr(r, "graph_family", "unknown")
        for r in all_records
        if getattr(r, "split", "") == "train"
    ))
    if families_to_test is None:
        families_to_test = all_families

    results = []
    for held_out in families_to_test:
        # Split: train on all except held_out, test on held_out.
        train_recs = [r for r in all_records
                      if getattr(r, "split", "") == "train"
                      and getattr(r, "graph_family", "") != held_out
                      and r.structural_state_after is not None]
        test_recs = [r for r in all_records
                     if getattr(r, "split", "") == "train"
                     and getattr(r, "graph_family", "") == held_out
                     and r.structural_state_after is not None]

        if len(train_recs) < 10 or len(test_recs) < 5:
            results.append(LOOResult(
                held_out_family=held_out,
                train_families=[f for f in all_families if f != held_out],
                n_train=len(train_recs),
                n_test=len(test_recs),
            ))
            continue

        z_t, a_t, z_next, _ = extract_normalized_data(train_recs, split_filter=False)
        tz_t, ta_t, tz_next, _ = extract_normalized_data(test_recs, split_filter=False)

        if len(z_t) == 0 or len(tz_t) == 0:
            continue

        model = DeltaDynamicsModel(mode="delta", regularization=1e-3, seed=42)
        model.fit(z_t, a_t, z_next, split="train")

        test_preds = model.predict_batch(tz_t, ta_t)
        m = compute_generalization_metrics(test_preds, tz_next)

        # Delta R².
        actual_delta = tz_next - tz_t
        pred_delta = model.predict_raw_batch(tz_t, ta_t)
        delta_m = compute_generalization_metrics(pred_delta, actual_delta)

        results.append(LOOResult(
            held_out_family=held_out,
            train_families=[f for f in all_families if f != held_out],
            r2=m.one_step_r2,
            rmse=m.one_step_rmse,
            spearman=m.spearman,
            delta_r2=delta_m.one_step_delta_r2,
            n_train=len(train_recs),
            n_test=len(test_recs),
        ))

    return results


def run_family_bootstrap_ensemble(
    train_records: list[Any],
    test_b_records: list[Any],
) -> dict[str, Any]:
    """Run family-bootstrap ensemble on TEST-B."""
    z_t, a_t, z_next, families = extract_normalized_data(train_records, split="train")
    tz_t, ta_t, tz_next, _ = extract_normalized_data(test_b_records, split="held_out")

    if len(z_t) == 0 or len(tz_t) == 0:
        return {"error": "insufficient data"}

    ensemble = FamilyBootstrapEnsemble(
        mode="delta", n_members=8, regularization=1e-3, seed=42,
    )
    ensemble.fit_with_family_split(z_t, a_t, z_next, families, split="train")

    # Predict on TEST-B.
    test_preds = ensemble.predict_batch(tz_t, ta_t)
    test_uncs = ensemble.predict_uncertainty_batch(tz_t, ta_t)

    m = compute_generalization_metrics(
        test_preds, tz_next,
        uncertainties=test_uncs,
    )

    # Delta metrics.
    actual_delta = tz_next - tz_t
    pred_deltas = np.array([
        member.predict_raw_batch(tz_t, ta_t)
        for member in ensemble._members
    ]).mean(axis=0)
    delta_m = compute_generalization_metrics(pred_deltas, actual_delta)

    return {
        "test_b_r2": m.one_step_r2,
        "test_b_rmse": m.one_step_rmse,
        "test_b_delta_r2": delta_m.one_step_delta_r2,
        "test_b_spearman": m.spearman,
        "calibration_corr": m.calibration_corr,
        "calibration_spearman": m.calibration_spearman,
        "mean_uncertainty": m.mean_uncertainty,
        "n_parameters": ensemble.n_parameters,
        "n_members": len(ensemble._members),
        "per_feature_nrmse": m.per_feature_nrmse,
    }


def run_ood_analysis(
    train_records: list[Any],
    test_b_records: list[Any],
    ensemble: FamilyBootstrapEnsemble | None = None,
) -> dict[str, Any]:
    """Run OOD distance analysis."""
    ood = compute_family_ood_distances(test_b_records, train_records)

    # If ensemble provided, compute uncertainties and errors.
    if ensemble is not None:
        tz_t, ta_t, tz_next, _ = extract_normalized_data(test_b_records, split="held_out")
        if len(tz_t) > 0:
            test_preds = ensemble.predict_batch(tz_t, ta_t)
            test_uncs = ensemble.predict_uncertainty_batch(tz_t, ta_t)
            errors = np.sqrt(np.sum((test_preds - tz_next) ** 2, axis=1)).tolist()

            ood_error = correlate_ood_with_error(ood["distances"], errors)
            ood_unc = correlate_ood_with_uncertainty(ood["distances"], test_uncs.tolist())

            return {
                "family_distances": ood["family_distances"],
                "ood_error_corr": ood_error,
                "ood_uncertainty_corr": ood_unc,
                "mean_ood_distance": float(np.mean(ood["distances"])),
            }

    return {
        "family_distances": ood["family_distances"],
        "mean_ood_distance": float(np.mean(ood["distances"])),
    }


def run_adaptation_curves(
    train_records: list[Any],
    test_b_records: list[Any],
    k_shots: list[int] | None = None,
) -> list[AdaptationResult]:
    """Run adaptation curves.

    For each TEST-B family and each k:
    1. Train on train data (global)
    2. Evaluate 0-shot on TEST-B family
    3. Fine-tune with k samples from TEST-B family
    4. Evaluate adapted model on remaining TEST-B family samples
    """
    if k_shots is None:
        k_shots = [0, 5, 10, 25, 50]

    # Get TEST-B families.
    test_b_families = sorted(set(
        getattr(r, "graph_family", "unknown")
        for r in test_b_records
        if getattr(r, "split", "") == "held_out"
    ))

    # Global training data.
    global_z_t, global_a_t, global_z_next, _ = extract_normalized_data(
        train_records, split="train",
    )

    results = []

    for family in test_b_families:
        # Get family-specific records.
        fam_records = [r for r in test_b_records
                       if getattr(r, "split", "") == "held_out"
                       and getattr(r, "graph_family", "") == family
                       and r.structural_state_after is not None]

        if len(fam_records) < 10:
            continue

        # Split family records into adaptation set and evaluation set.
        n_total = len(fam_records)
        n_eval = max(5, n_total // 2)

        eval_records = fam_records[n_eval:]
        adapt_records = fam_records[:n_eval]

        # Extract eval data.
        eval_z_t, eval_a_t, eval_z_next, _ = extract_normalized_data(
            eval_records, split_filter=False,
        )

        if len(eval_z_t) == 0:
            continue

        for k in k_shots:
            # Train global model (or fine-tune).
            if k == 0:
                # 0-shot: global model only.
                model = DeltaDynamicsModel(mode="delta", regularization=1e-3, seed=42)
                model.fit(global_z_t, global_a_t, global_z_next, split="train")
            else:
                # k-shot: fine-tune global model with k adaptation samples.
                k_actual = min(k, len(adapt_records))
                adapt_subset = adapt_records[:k_actual]

                # Combine global + adaptation data.
                adapt_z_t, adapt_a_t, adapt_z_next, _ = extract_normalized_data(
                    adapt_subset, split_filter=False,
                )

                combined_z_t = np.vstack([global_z_t, adapt_z_t])
                combined_a_t = np.vstack([global_a_t, adapt_a_t])
                combined_z_next = np.vstack([global_z_next, adapt_z_next])

                model = DeltaDynamicsModel(mode="delta", regularization=1e-3, seed=42)
                model.fit(combined_z_t, combined_a_t, combined_z_next, split="train")

            # Evaluate.
            eval_preds = model.predict_batch(eval_z_t, eval_a_t)
            m = compute_generalization_metrics(eval_preds, eval_z_next)

            # 0-shot baseline for comparison.
            if k == 0:
                r2_before = m.one_step_r2
                rmse_before = m.one_step_rmse
            else:
                # Re-evaluate 0-shot for comparison.
                base_model = DeltaDynamicsModel(mode="delta", regularization=1e-3, seed=42)
                base_model.fit(global_z_t, global_a_t, global_z_next, split="train")
                base_preds = base_model.predict_batch(eval_z_t, eval_a_t)
                base_m = compute_generalization_metrics(base_preds, eval_z_next)
                r2_before = base_m.one_step_r2
                rmse_before = base_m.one_step_rmse

            results.append(AdaptationResult(
                family=family,
                k_shots=k,
                r2_before=r2_before,
                r2_after=m.one_step_r2,
                rmse_before=rmse_before,
                rmse_after=m.one_step_rmse,
            ))

    return results


def run_extended_rollout(
    model: DeltaDynamicsModel | FamilyBootstrapEnsemble,
    records: list[Any],
    *,
    split: str = "held_out",
    max_horizon: int = 10,
) -> dict[str, Any]:
    """Run extended rollout evaluation with per-feature NRMSE.

    Horizons: h=1,2,3,5,10
    """
    from ..transition_record import TransitionProvenance

    # Group realized records by episode.
    episodes: dict[str, list[Any]] = {}
    for r in records:
        if getattr(r, "split", "") != split:
            continue
        if r.structural_state_after is None:
            continue
        prov = getattr(r, "provenance", None)
        if prov is not None and hasattr(prov, "value"):
            if "counterfactual" in prov.value.lower():
                continue
        ep = getattr(r, "episode_id", "unknown")
        episodes.setdefault(ep, []).append(r)

    # Sort and deduplicate.
    for ep in episodes:
        episodes[ep].sort(key=lambda r: getattr(r, "step_id", 0))
        seen: set[int] = set()
        unique = []
        for r in episodes[ep]:
            step = getattr(r, "step_id", 0)
            if step not in seen:
                seen.add(step)
                unique.append(r)
        episodes[ep] = unique

    # Compute normalization scales.
    all_states = []
    for ep_recs in episodes.values():
        for r in ep_recs:
            all_states.append(encode_normalized_state(r.structural_state_before).vector)
    if all_states:
        all_arr = np.array(all_states)
        feat_std = np.std(all_arr, axis=0)
        feat_std[feat_std < 1e-8] = 1.0
    else:
        feat_std = np.ones(NORM_STATE_DIM)

    horizons = [h for h in [1, 2, 3, 5, 10] if h <= max_horizon]
    results = {}

    for h in horizons:
        all_preds, all_actuals = [], []

        for ep_recs in episodes.values():
            if len(ep_recs) < h + 1:
                continue

            for start in range(len(ep_recs) - h):
                r0 = ep_recs[start]
                state = r0.structural_state_before
                z = encode_normalized_state(state).vector.copy()

                for step in range(h):
                    r = ep_recs[start + step]
                    a = encode_normalized_action(
                        r.action, r.action_target,
                        n_nodes=int(getattr(state, "n_nodes", 20)),
                        degree_mean=float(getattr(state, "degree_mean", 2.0)),
                    )
                    if isinstance(model, FamilyBootstrapEnsemble):
                        z = model.predict(z, a.vector)
                    else:
                        z = model.predict(z, a.vector)

                r_actual = ep_recs[start + h]
                z_actual = encode_normalized_state(r_actual.structural_state_before).vector

                all_preds.append(z)
                all_actuals.append(z_actual)

        if all_preds:
            preds = np.array(all_preds)
            actuals = np.array(all_actuals)
            diff = preds - actuals
            norm_diff = diff / feat_std
            nrmse = float(np.sqrt(np.mean(norm_diff ** 2)))
            ss_res = float(np.sum(diff ** 2))
            ss_tot = float(np.sum((actuals - actuals.mean(axis=0)) ** 2))
            r2 = max(-10.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-10))) if ss_tot > 1e-10 else 0.0
            per_feat = [float(np.sqrt(np.mean(norm_diff[:, j] ** 2)))
                        for j in range(actuals.shape[1])]
        else:
            nrmse = 0.0
            r2 = 0.0
            per_feat = []

        results[f"h={h}"] = {
            "nrmse": nrmse,
            "r2": r2,
            "n_samples": len(all_preds),
            "per_feature_nrmse": per_feat,
        }

    return results
