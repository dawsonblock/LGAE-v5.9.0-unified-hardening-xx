"""Evaluation for the exp5 world model.

Evaluates:
- Single-step next-state prediction accuracy
- Multi-step rollout accuracy (horizon 1, 2, 3)
- Outcome prediction quality
- Authority boundary preservation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import math

from .state_encoding import encode_state, encode_action, STATE_DIM, ACTION_DIM
from .joint_model import JointWorldModel, JointModelMetrics
from .dynamics import DynamicsMetrics, compute_dynamics_metrics
from .training import extract_training_data, _evaluate_joint


@dataclass(slots=True)
class EvaluationResult:
    """Full evaluation result for a world model."""
    train_metrics: JointModelMetrics = field(default_factory=JointModelMetrics)
    validation_metrics: JointModelMetrics = field(default_factory=JointModelMetrics)
    heldout_metrics: JointModelMetrics = field(default_factory=JointModelMetrics)
    rollout_report: dict[str, Any] = field(default_factory=dict)
    n_train: int = 0
    n_validation: int = 0
    n_heldout: int = 0
    # v6.0-exp5.1: TEST-B metrics (fresh, untouched external split).
    test_b_metrics: JointModelMetrics | None = None

    def to_log(self) -> dict[str, Any]:
        return {
            "train_metrics": self.train_metrics.to_log(),
            "validation_metrics": self.validation_metrics.to_log(),
            "heldout_metrics": self.heldout_metrics.to_log(),
            "rollout_report": dict(self.rollout_report),
            "n_train": int(self.n_train),
            "n_validation": int(self.n_validation),
            "n_heldout": int(self.n_heldout),
            "test_b_metrics": self.test_b_metrics.to_log() if self.test_b_metrics else None,
        }


@dataclass(slots=True)
class RolloutReport:
    """Multi-step rollout evaluation report."""
    horizons: list[int] = field(default_factory=list)
    rmse_by_horizon: list[float] = field(default_factory=list)
    mae_by_horizon: list[float] = field(default_factory=list)
    r2_by_horizon: list[float] = field(default_factory=list)
    n_trajectories: int = 0

    def to_log(self) -> dict[str, Any]:
        return {
            "horizons": list(self.horizons),
            "rmse_by_horizon": [float(x) for x in self.rmse_by_horizon],
            "mae_by_horizon": [float(x) for x in self.mae_by_horizon],
            "r2_by_horizon": [float(x) for x in self.r2_by_horizon],
            "n_trajectories": int(self.n_trajectories),
        }


def evaluate_world_model(
    model: JointWorldModel,
    records: list[Any],
) -> EvaluationResult:
    """Evaluate a trained world model on all splits.

    Args:
        model: A fitted JointWorldModel.
        records: All transition records (train + validation + held_out).

    Returns:
        EvaluationResult with per-split metrics and rollout report.
    """
    result = EvaluationResult()

    for split_name, metrics_attr in [
        ("train", "train_metrics"),
        ("validation", "validation_metrics"),
        ("held_out", "heldout_metrics"),
    ]:
        z_t, a_t, z_next, y_outcome = extract_training_data(records, split=split_name)
        if len(z_t) > 0:
            metrics = _evaluate_joint(model, z_t, a_t, z_next, y_outcome)
            setattr(result, metrics_attr, metrics)
            setattr(result, f"n_{split_name.replace('held_out', 'heldout')}", len(z_t))

    # Rollout evaluation on validation.
    result.rollout_report = rollout_evaluation(model, records, split="validation").to_log()

    return result


def rollout_evaluation(
    model: JointWorldModel,
    records: list[Any],
    *,
    split: str = "validation",
    max_horizon: int = 3,
) -> RolloutReport:
    """Evaluate multi-step rollout accuracy.

    Groups records by episode and evaluates how prediction error
    grows with rollout horizon. Only uses REALIZED records (not
    counterfactuals) to form trajectories, since counterfactuals
    share the same episode_id and step_id but represent alternative
    actions.

    Uses per-feature normalized RMSE to avoid domination by
    high-variance dimensions, and clips R² to a sane range to
    avoid absurd values when feature variance is near-zero.

    Args:
        model: A fitted JointWorldModel.
        records: Transition records.
        split: Which split to evaluate on.
        max_horizon: Maximum rollout horizon.

    Returns:
        RolloutReport with metrics per horizon.
    """
    # Group REALIZED records by episode (exclude counterfactuals).
    from ..transition_record import TransitionProvenance
    episodes: dict[str, list[Any]] = {}
    for r in records:
        if getattr(r, "split", "") != split:
            continue
        if r.structural_state_after is None:
            continue
        # Only use REALIZED records for trajectory construction.
        prov = getattr(r, "provenance", None)
        if prov is not None and hasattr(prov, "value"):
            if "counterfactual" in prov.value.lower():
                continue
        elif isinstance(prov, str) and "counterfactual" in prov.lower():
            continue
        ep = getattr(r, "episode_id", "unknown")
        episodes.setdefault(ep, []).append(r)

    # Sort each episode by step_id and deduplicate by step.
    for ep in episodes:
        episodes[ep].sort(key=lambda r: getattr(r, "step_id", 0))
        # Keep only one record per step_id (the first/realized one).
        seen_steps: set[int] = set()
        unique: list[Any] = []
        for r in episodes[ep]:
            step = getattr(r, "step_id", 0)
            if step not in seen_steps:
                seen_steps.add(step)
                unique.append(r)
        episodes[ep] = unique

    report = RolloutReport(
        horizons=list(range(1, max_horizon + 1)),
        n_trajectories=len(episodes),
    )

    # Compute per-feature normalization scales from the actual states.
    all_actual_states: list[np.ndarray] = []
    for ep_records in episodes.values():
        for r in ep_records:
            all_actual_states.append(encode_state(r.structural_state_before).vector)
    if all_actual_states:
        all_states_arr = np.array(all_actual_states)
        feat_std = np.std(all_states_arr, axis=0)
        feat_std[feat_std < 1e-8] = 1.0  # avoid division by zero
    else:
        feat_std = np.ones(STATE_DIM)

    for horizon in range(1, max_horizon + 1):
        all_preds = []
        all_actuals = []

        for ep_id, ep_records in episodes.items():
            if len(ep_records) < horizon + 1:
                continue

            for start_idx in range(len(ep_records) - horizon):
                # Start from the actual state.
                r0 = ep_records[start_idx]
                state = r0.structural_state_before
                z = encode_state(state).vector.copy()

                # Roll forward using model predictions.
                for step in range(horizon):
                    r = ep_records[start_idx + step]
                    a = encode_action(
                        r.action,
                        r.action_target,
                        n_nodes=int(getattr(state, "n_nodes", 20)),
                        degree_mean=float(getattr(state, "degree_mean", 2.0)),
                    )
                    z = model.predict_dynamics_batch(
                        z[np.newaxis, :], a.vector[np.newaxis, :]
                    )[0]

                # Compare with actual state at the target step.
                # ep_records[start_idx + horizon].structural_state_before
                # is the state AFTER applying the action at step
                # start_idx + horizon - 1, which is what we want.
                r_actual = ep_records[start_idx + horizon]
                z_actual = encode_state(r_actual.structural_state_before).vector

                all_preds.append(z)
                all_actuals.append(z_actual)

        if all_preds:
            preds = np.array(all_preds)
            actuals = np.array(all_actuals)
            # Normalized RMSE (per-feature, then averaged).
            norm_diff = (preds - actuals) / feat_std
            norm_rmse = float(np.sqrt(np.mean(norm_diff ** 2)))
            norm_mae = float(np.mean(np.abs(norm_diff)))
            # R² with clipping to avoid absurd values.
            diff = preds - actuals
            ss_res = float(np.sum(diff ** 2))
            ss_tot = float(np.sum((actuals - actuals.mean(axis=0)) ** 2))
            if ss_tot < 1e-10:
                r2 = 0.0  # near-constant target → R² undefined
            else:
                r2 = max(-10.0, min(1.0, 1.0 - ss_res / ss_tot))
            report.rmse_by_horizon.append(norm_rmse)
            report.mae_by_horizon.append(norm_mae)
            report.r2_by_horizon.append(r2)
        else:
            report.rmse_by_horizon.append(0.0)
            report.mae_by_horizon.append(0.0)
            report.r2_by_horizon.append(0.0)

    return report


def evaluate_calibration(
    model: JointWorldModel,
    records: list[Any],
    *,
    split: str = "held_out",
) -> dict[str, Any]:
    """Evaluate uncertainty calibration.

    Computes the correlation between per-prediction uncertainty
    (ensemble disagreement) and actual prediction error. A positive
    correlation means uncertainty is useful for abstention.

    Returns:
        dict with:
        - corr_uncertainty_error: Pearson correlation
        - spearman_uncertainty_error: Spearman correlation
        - n_samples: number of evaluated records
        - mean_uncertainty: average uncertainty
        - mean_error: average absolute error
    """
    from .dynamics import EnsembleDynamics

    # Extract data for this split.
    z_t, a_t, z_next, y = extract_training_data(records, split=split)
    if len(z_t) == 0:
        return {
            "corr_uncertainty_error": 0.0,
            "spearman_uncertainty_error": 0.0,
            "n_samples": 0,
            "mean_uncertainty": 0.0,
            "mean_error": 0.0,
        }

    # Get predictions.
    preds = model.predict_dynamics_batch(z_t, a_t)
    errors = np.sqrt(np.sum((preds - z_next) ** 2, axis=1))

    # Get per-prediction uncertainty.
    if isinstance(model._dynamics, EnsembleDynamics):
        uncs = model._dynamics.predict_uncertainty_batch(z_t, a_t)
    else:
        # Constant uncertainty — correlation will be ~0.
        uncs = np.full(len(z_t), model._residual_std)

    # Pearson correlation.
    if len(uncs) > 1 and np.std(uncs) > 1e-10 and np.std(errors) > 1e-10:
        corr = float(np.corrcoef(uncs, errors)[0, 1])
    else:
        corr = 0.0

    # Spearman correlation (rank-based).
    if len(uncs) > 1:
        from scipy.stats import spearmanr
        try:
            sp, _ = spearmanr(uncs, errors)
            sp = float(sp)
        except Exception:
            # Manual Spearman.
            rank_u = np.argsort(np.argsort(uncs)).astype(float)
            rank_e = np.argsort(np.argsort(errors)).astype(float)
            if np.std(rank_u) > 1e-10 and np.std(rank_e) > 1e-10:
                sp = float(np.corrcoef(rank_u, rank_e)[0, 1])
            else:
                sp = 0.0
    else:
        sp = 0.0

    return {
        "corr_uncertainty_error": corr,
        "spearman_uncertainty_error": sp,
        "n_samples": int(len(z_t)),
        "mean_uncertainty": float(np.mean(uncs)),
        "mean_error": float(np.mean(errors)),
    }
