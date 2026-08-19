"""Joint world model: dynamics + outcome prediction.

Combines:
    z_{t+1} = F_θ(z_t, a_t)         (dynamics)
    (ΔÛ, R̂, Ĉ, σ) = g_φ(z_t, a_t)  (outcomes)

The outcome head is a simple linear or small MLP model on top of
the same (z_t, a_t) input. It predicts:
- ΔÛ: utility delta
- R̂: risk
- Ĉ: cost
- σ: uncertainty (from residual std)

This is the core exp5 model. It implements the WorldModelInterface
contract via the wrapper in world_model_impl.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import math
import numpy as np

from .state_encoding import STATE_DIM, ACTION_DIM, state_action_schema_hash
from .dynamics import (
    DynamicsModel, LinearDynamics, MLPDynamics, EnsembleDynamics,
    DynamicsMetrics, compute_dynamics_metrics,
)


@dataclass(frozen=True, slots=True)
class JointModelConfig:
    """Configuration for the joint world model (frozen)."""
    dynamics_type: str = "linear"  # "linear", "mlp", or "ensemble"
    outcome_type: str = "linear"   # "linear" or "mlp"
    hidden_dim: int = 32
    lr: float = 0.01
    n_epochs: int = 100
    seed: int = 42
    regularization: float = 1e-4
    n_ensemble_members: int = 5  # for ensemble type

    def to_log(self) -> dict[str, Any]:
        return {
            "dynamics_type": self.dynamics_type,
            "outcome_type": self.outcome_type,
            "hidden_dim": int(self.hidden_dim),
            "lr": float(self.lr),
            "n_epochs": int(self.n_epochs),
            "seed": int(self.seed),
            "regularization": float(self.regularization),
            "n_ensemble_members": int(self.n_ensemble_members),
        }


@dataclass(slots=True)
class WorldModelPrediction:
    """A prediction from the joint world model."""
    predicted_next_state: np.ndarray
    predicted_delta_utility: float
    predicted_risk: float
    predicted_cost: float
    predicted_uncertainty: float
    probability_positive: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "predicted_next_state": [float(x) for x in self.predicted_next_state],
            "predicted_delta_utility": float(self.predicted_delta_utility),
            "predicted_risk": float(self.predicted_risk),
            "predicted_cost": float(self.predicted_cost),
            "predicted_uncertainty": float(self.predicted_uncertainty),
            "probability_positive": self.probability_positive,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class JointModelMetrics:
    """Combined metrics for the joint model."""
    dynamics: DynamicsMetrics = field(default_factory=DynamicsMetrics)
    outcome_rmse: float = 0.0
    outcome_mae: float = 0.0
    outcome_r2: float = 0.0
    risk_rmse: float = 0.0
    cost_rmse: float = 0.0
    n_samples: int = 0

    def to_log(self) -> dict[str, Any]:
        return {
            "dynamics": self.dynamics.to_log(),
            "outcome_rmse": float(self.outcome_rmse),
            "outcome_mae": float(self.outcome_mae),
            "outcome_r2": float(self.outcome_r2),
            "risk_rmse": float(self.risk_rmse),
            "cost_rmse": float(self.cost_rmse),
            "n_samples": int(self.n_samples),
        }


class JointWorldModel:
    """Joint dynamics + outcome world model.

    Predicts both the next structural state and the outcome
    (utility, risk, cost, uncertainty) from (z_t, a_t).

    Advisory-only. Never mutates authoritative state.
    """

    model_type = "joint_world_model"
    version = "v6.0-exp5"

    def __init__(self, config: JointModelConfig | None = None) -> None:
        self.config = config or JointModelConfig()
        self._dynamics: DynamicsModel = self._create_dynamics()
        self._outcome_W: np.ndarray | None = None  # outcome weights
        self._outcome_b: np.ndarray | None = None  # outcome bias
        self._residual_std: float = 0.0  # for uncertainty
        self._fitted = False
        self._n_samples = 0

    def _create_dynamics(self) -> DynamicsModel:
        if self.config.dynamics_type == "ensemble":
            return EnsembleDynamics(
                base_type="linear",
                n_members=self.config.n_ensemble_members,
                lr=self.config.lr,
                n_epochs=self.config.n_epochs,
                seed=self.config.seed,
                regularization=self.config.regularization,
            )
        if self.config.dynamics_type == "mlp":
            return MLPDynamics(
                hidden_dim=self.config.hidden_dim,
                lr=self.config.lr,
                n_epochs=self.config.n_epochs,
                seed=self.config.seed,
            )
        return LinearDynamics(
            lr=self.config.lr,
            n_epochs=self.config.n_epochs,
            seed=self.config.seed,
            regularization=self.config.regularization,
        )

    @property
    def model_id(self) -> str:
        return f"{self.model_type}-{self.version}"

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def deterministic(self) -> bool:
        return self._dynamics.deterministic

    @property
    def schema_hash(self) -> str:
        return state_action_schema_hash()

    @property
    def n_parameters(self) -> int:
        dyn_params = self._dynamics.n_parameters
        if self._outcome_W is not None:
            return dyn_params + self._outcome_W.size + self._outcome_b.size
        return dyn_params

    def fit(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
        z_next: np.ndarray,
        y_outcome: np.ndarray,
        *,
        split: str = "train",
    ) -> None:
        """Fit the joint model on train split only.

        Args:
            z_t: (n, state_dim) encoded states before.
            a_t: (n, action_dim) encoded actions.
            z_next: (n, state_dim) encoded states after.
            y_outcome: (n, 3) outcome targets [delta_utility, risk, cost].
            split: Must be "train".
        """
        if split != "train":
            raise ValueError(
                f"Joint world model can only fit on train split, got '{split}'."
            )

        # Fit dynamics.
        self._dynamics.fit(z_t, a_t, z_next, split="train")

        # Fit outcome head (linear regression with ridge).
        n = len(z_t)
        if n == 0:
            self._outcome_W = np.zeros((STATE_DIM + ACTION_DIM, 3))
            self._outcome_b = np.zeros(3)
            self._fitted = True
            return

        X = np.hstack([z_t, a_t])  # (n, state_dim + action_dim)
        X_aug = np.hstack([X, np.ones((n, 1))])  # (n, input+1)
        lam = self.config.regularization
        XtX = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
        XtY = X_aug.T @ y_outcome
        W = np.linalg.solve(XtX, XtY)

        input_dim = STATE_DIM + ACTION_DIM
        self._outcome_W = W[:input_dim, :]  # (input, 3)
        self._outcome_b = W[-1, :]  # (3,)

        # Compute residual std for uncertainty.
        preds = X_aug @ W
        residuals = y_outcome - preds
        self._residual_std = float(np.std(residuals[:, 0])) if n > 1 else 0.0

        self._fitted = True
        self._n_samples = n

    def freeze(self) -> None:
        self._dynamics.freeze()

    def predict(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> WorldModelPrediction:
        """Predict next state and outcomes."""
        if not self._fitted:
            return WorldModelPrediction(
                predicted_next_state=z_t.copy(),
                predicted_delta_utility=0.0,
                predicted_risk=0.0,
                predicted_cost=0.0,
                predicted_uncertainty=0.0,
            )

        z_next = self._dynamics.predict(z_t, a_t)
        x = np.concatenate([z_t, a_t])
        outcome = x @ self._outcome_W + self._outcome_b

        delta_u = float(outcome[0])
        risk = float(outcome[1])
        cost = float(outcome[2])

        # Per-prediction uncertainty from ensemble disagreement.
        if isinstance(self._dynamics, EnsembleDynamics):
            unc_arr = self._dynamics.predict_uncertainty_batch(
                z_t[np.newaxis, :], a_t[np.newaxis, :]
            )
            unc = float(unc_arr[0]) if len(unc_arr) > 0 else self._residual_std
        else:
            unc = self._residual_std

        prob_pos = 1.0 / (1.0 + math.exp(-delta_u)) if abs(delta_u) < 50 else (1.0 if delta_u > 0 else 0.0)

        return WorldModelPrediction(
            predicted_next_state=z_next,
            predicted_delta_utility=delta_u,
            predicted_risk=risk,
            predicted_cost=cost,
            predicted_uncertainty=unc,
            probability_positive=prob_pos,
        )

    def predict_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> list[WorldModelPrediction]:
        """Batch prediction."""
        if z_t.ndim == 1:
            z_t = z_t[np.newaxis, :]
        if a_t.ndim == 1:
            a_t = a_t[np.newaxis, :]
        return [self.predict(z_t[i], a_t[i]) for i in range(len(z_t))]

    def predict_dynamics_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Predict only the dynamics (next state)."""
        return self._dynamics.predict_batch(z_t, a_t)

    def predict_outcome_batch(
        self,
        z_t: np.ndarray,
        a_t: np.ndarray,
    ) -> np.ndarray:
        """Predict only outcomes (n, 3): [delta_u, risk, cost]."""
        if not self._fitted:
            return np.zeros((len(z_t), 3))
        X = np.hstack([z_t, a_t])
        return X @ self._outcome_W + self._outcome_b

    def get_state(self) -> dict[str, Any]:
        return {
            "config": self.config.to_log(),
            "dynamics_state": self._dynamics.get_state(),
            "outcome_W": self._outcome_W.tolist() if self._outcome_W is not None else None,
            "outcome_b": self._outcome_b.tolist() if self._outcome_b is not None else None,
            "residual_std": float(self._residual_std),
            "fitted": self._fitted,
            "n_samples": self._n_samples,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._outcome_W = np.array(state["outcome_W"]) if state.get("outcome_W") else None
        self._outcome_b = np.array(state["outcome_b"]) if state.get("outcome_b") else None
        self._residual_std = state.get("residual_std", 0.0)
        self._fitted = state.get("fitted", False)
        self._n_samples = state.get("n_samples", 0)
        self._dynamics.set_state(state.get("dynamics_state", {}))

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "config": self.config.to_log(),
            "dynamics_hyperparams": self._dynamics.hyperparameters(),
        }
