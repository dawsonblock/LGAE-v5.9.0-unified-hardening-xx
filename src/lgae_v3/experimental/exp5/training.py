"""Training pipeline for the exp5 world model.

Extracts (z_t, a_t, z_{t+1}, outcome) tuples from exp2 transition
records, trains the joint world model on the train split only, and
returns training metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import time

from .state_encoding import encode_state, encode_action, STATE_DIM, ACTION_DIM
from .joint_model import JointWorldModel, JointModelConfig, JointModelMetrics, WorldModelPrediction
from .dynamics import compute_dynamics_metrics


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for world model training (frozen)."""
    dynamics_type: str = "linear"
    hidden_dim: int = 32
    lr: float = 0.01
    n_epochs: int = 100
    seed: int = 42
    regularization: float = 1e-4

    def to_log(self) -> dict[str, Any]:
        return {
            "dynamics_type": self.dynamics_type,
            "hidden_dim": int(self.hidden_dim),
            "lr": float(self.lr),
            "n_epochs": int(self.n_epochs),
            "seed": int(self.seed),
            "regularization": float(self.regularization),
        }


@dataclass(slots=True)
class TrainingResult:
    """Result of training a world model."""
    model: JointWorldModel
    train_metrics: JointModelMetrics = field(default_factory=JointModelMetrics)
    n_train: int = 0
    elapsed_seconds: float = 0.0
    config: TrainingConfig = field(default_factory=TrainingConfig)

    def to_log(self) -> dict[str, Any]:
        return {
            "train_metrics": self.train_metrics.to_log(),
            "n_train": int(self.n_train),
            "elapsed_seconds": float(self.elapsed_seconds),
            "config": self.config.to_log(),
            "n_parameters": self.model.n_parameters,
        }


def extract_training_data(
    records: list[Any],
    *,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (z_t, a_t, z_next, y_outcome) from transition records.

    Args:
        records: List of TransitionRecord objects.
        split: Which split to extract ("train", "validation", "held_out").

    Returns:
        Tuple of:
        - z_t: (n, STATE_DIM) encoded states before
        - a_t: (n, ACTION_DIM) encoded actions
        - z_next: (n, STATE_DIM) encoded states after
        - y_outcome: (n, 3) [delta_utility, risk, cost]
    """
    z_t_list, a_t_list, z_next_list, y_list = [], [], [], []

    for r in records:
        if getattr(r, "split", "") != split:
            continue
        state_before = r.structural_state_before
        state_after = r.structural_state_after
        if state_after is None:
            continue

        # Encode state before.
        sv = encode_state(state_before)
        # Encode action.
        av = encode_action(
            r.action,
            r.action_target,
            n_nodes=int(getattr(state_before, "n_nodes", 20)),
            degree_mean=float(getattr(state_before, "degree_mean", 2.0)),
        )
        # Encode state after.
        sv_next = encode_state(state_after)

        z_t_list.append(sv.vector)
        a_t_list.append(av.vector)
        z_next_list.append(sv_next.vector)
        y_list.append([
            float(getattr(r, "realized_delta", 0.0)),
            float(getattr(r, "realized_risk", 0.0)),
            float(getattr(r, "realized_cost", 0.0)),
        ])

    if not z_t_list:
        return (
            np.zeros((0, STATE_DIM)),
            np.zeros((0, ACTION_DIM)),
            np.zeros((0, STATE_DIM)),
            np.zeros((0, 3)),
        )

    return (
        np.array(z_t_list, dtype=np.float64),
        np.array(a_t_list, dtype=np.float64),
        np.array(z_next_list, dtype=np.float64),
        np.array(y_list, dtype=np.float64),
    )


def train_world_model(
    records: list[Any],
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train a joint world model on the train split.

    Args:
        records: List of TransitionRecord objects.
        config: Training configuration.

    Returns:
        TrainingResult with the trained model and training metrics.
    """
    config = config or TrainingConfig()
    t0 = time.time()

    # Extract training data.
    z_t, a_t, z_next, y_outcome = extract_training_data(records, split="train")

    # Create model.
    joint_config = JointModelConfig(
        dynamics_type=config.dynamics_type,
        hidden_dim=config.hidden_dim,
        lr=config.lr,
        n_epochs=config.n_epochs,
        seed=config.seed,
        regularization=config.regularization,
    )
    model = JointWorldModel(config=joint_config)

    # Fit on train only.
    model.fit(z_t, a_t, z_next, y_outcome, split="train")
    model.freeze()

    # Compute training metrics.
    train_metrics = _evaluate_joint(model, z_t, a_t, z_next, y_outcome)

    return TrainingResult(
        model=model,
        train_metrics=train_metrics,
        n_train=len(z_t),
        elapsed_seconds=time.time() - t0,
        config=config,
    )


def _evaluate_joint(
    model: JointWorldModel,
    z_t: np.ndarray,
    a_t: np.ndarray,
    z_next: np.ndarray,
    y_outcome: np.ndarray,
) -> JointModelMetrics:
    """Evaluate the joint model on a dataset."""
    if len(z_t) == 0:
        return JointModelMetrics()

    # Dynamics metrics.
    pred_next = model.predict_dynamics_batch(z_t, a_t)
    dyn_metrics = compute_dynamics_metrics(pred_next, z_next, horizon=1)

    # Outcome metrics.
    pred_outcome = model.predict_outcome_batch(z_t, a_t)
    outcome_diff = pred_outcome - y_outcome
    outcome_rmse = float(np.sqrt(np.mean(outcome_diff[:, 0] ** 2)))
    outcome_mae = float(np.mean(np.abs(outcome_diff[:, 0])))
    outcome_ss_res = float(np.sum(outcome_diff[:, 0] ** 2))
    outcome_ss_tot = float(np.sum((y_outcome[:, 0] - y_outcome[:, 0].mean()) ** 2))
    outcome_r2 = 1.0 - outcome_ss_res / max(outcome_ss_tot, 1e-10)

    risk_rmse = float(np.sqrt(np.mean(outcome_diff[:, 1] ** 2)))
    cost_rmse = float(np.sqrt(np.mean(outcome_diff[:, 2] ** 2)))

    return JointModelMetrics(
        dynamics=dyn_metrics,
        outcome_rmse=outcome_rmse,
        outcome_mae=outcome_mae,
        outcome_r2=outcome_r2,
        risk_rmse=risk_rmse,
        cost_rmse=cost_rmse,
        n_samples=len(z_t),
    )
