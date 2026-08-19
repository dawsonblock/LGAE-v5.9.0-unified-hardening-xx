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

    def to_log(self) -> dict[str, Any]:
        return {
            "train_metrics": self.train_metrics.to_log(),
            "validation_metrics": self.validation_metrics.to_log(),
            "heldout_metrics": self.heldout_metrics.to_log(),
            "rollout_report": dict(self.rollout_report),
            "n_train": int(self.n_train),
            "n_validation": int(self.n_validation),
            "n_heldout": int(self.n_heldout),
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
    grows with rollout horizon.

    Args:
        model: A fitted JointWorldModel.
        records: Transition records.
        split: Which split to evaluate on.
        max_horizon: Maximum rollout horizon.

    Returns:
        RolloutReport with metrics per horizon.
    """
    # Group records by episode.
    episodes: dict[str, list[Any]] = {}
    for r in records:
        if getattr(r, "split", "") != split:
            continue
        if r.structural_state_after is None:
            continue
        ep = getattr(r, "episode_id", "unknown")
        episodes.setdefault(ep, []).append(r)

    # Sort each episode by step_id.
    for ep in episodes:
        episodes[ep].sort(key=lambda r: getattr(r, "step_id", 0))

    report = RolloutReport(
        horizons=list(range(1, max_horizon + 1)),
        n_trajectories=len(episodes),
    )

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

                # Compare with actual state after horizon steps.
                r_actual = ep_records[start_idx + horizon]
                z_actual = encode_state(r_actual.structural_state_before).vector

                all_preds.append(z)
                all_actuals.append(z_actual)

        if all_preds:
            preds = np.array(all_preds)
            actuals = np.array(all_actuals)
            metrics = compute_dynamics_metrics(preds, actuals, horizon=horizon)
            report.rmse_by_horizon.append(metrics.rmse)
            report.mae_by_horizon.append(metrics.mae)
            report.r2_by_horizon.append(metrics.r2)
        else:
            report.rmse_by_horizon.append(0.0)
            report.mae_by_horizon.append(0.0)
            report.r2_by_horizon.append(0.0)

    return report
