"""LightweightWorldModel — implements WorldModelInterface.

This is the adapter that connects the exp5 JointWorldModel to the
abstract WorldModelInterface defined in world_model.py. It allows
the world model to be used by future MPC and planning components
while remaining advisory-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import math

from ..world_model import (
    ModelPrediction,
    ModelTrustReport,
    WorldModelInterface,
)
from .state_encoding import encode_state, encode_action, StateVector, ActionVector, STATE_DIM
from .joint_model import JointWorldModel, JointModelConfig


@dataclass
class WorldModelTrustReport:
    """Trust report for the lightweight world model."""
    model_name: str = "lightweight_world_model"
    mean_prediction_error: float = 0.0
    ood_distance: float = 0.0
    calibration_correlation: float = 0.0
    trust_score: float = 0.0
    recommended_horizon: int = 1
    recommended_exact_verification_fraction: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_model_trust_report(self) -> ModelTrustReport:
        return ModelTrustReport(
            model_name=self.model_name,
            mean_prediction_error=float(self.mean_prediction_error),
            ood_distance=float(self.ood_distance),
            calibration_correlation=float(self.calibration_correlation),
            trust_score=float(self.trust_score),
            recommended_horizon=int(self.recommended_horizon),
            recommended_exact_verification_fraction=float(self.recommended_exact_verification_fraction),
            metadata=dict(self.metadata),
        )


class LightweightWorldModel(WorldModelInterface):
    """WorldModelInterface implementation for the exp5 world model.

    Wraps a JointWorldModel and exposes it through the abstract
    WorldModelInterface contract.

    CRITICAL: This is advisory-only. It receives NO mutation authority.
    The v5.11 CommitChannel is the permanent authority boundary.
    """

    def __init__(self, joint_model: JointWorldModel | None = None) -> None:
        self._model = joint_model or JointWorldModel()
        self._trust_report = WorldModelTrustReport()

    @property
    def name(self) -> str:
        return "lightweight_world_model"

    @property
    def model(self) -> JointWorldModel:
        return self._model

    def set_trust_report(self, report: WorldModelTrustReport) -> None:
        self._trust_report = report

    def predict_next_state(self, state: Any, action: Any) -> ModelPrediction:
        """Predict the next structural state.

        Args:
            state: A StructuralStateSummary or StateVector.
            action: A tuple (action_type, action_target, n_nodes, degree_mean)
                or an ActionVector.

        Returns:
            ModelPrediction with predicted_next_state (as numpy array
            encoded in metadata) and uncertainty.
        """
        # Encode state.
        if isinstance(state, StateVector):
            z_t = state.vector
        else:
            z_t = encode_state(state).vector

        # Encode action.
        if isinstance(action, ActionVector):
            a_t = action.vector
        elif isinstance(action, tuple) and len(action) >= 2:
            action_type, action_target = action[0], action[1]
            n_nodes = action[2] if len(action) > 2 else 20
            degree_mean = action[3] if len(action) > 3 else 2.0
            a_t = encode_action(action_type, action_target, n_nodes=n_nodes, degree_mean=degree_mean).vector
        else:
            a_t = np.zeros(ACTION_DIM if 'ACTION_DIM' in dir() else 12)

        # Predict.
        pred = self._model.predict(z_t, a_t)

        # Compute state hash for the predicted next state.
        import hashlib
        state_bytes = pred.predicted_next_state.tobytes()
        state_hash = hashlib.sha256(state_bytes).hexdigest()[:16]

        return ModelPrediction(
            predicted_delta_utility=pred.predicted_delta_utility,
            predicted_risk=pred.predicted_risk,
            predicted_cost=pred.predicted_cost,
            predicted_uncertainty=pred.predicted_uncertainty,
            probability_positive=pred.probability_positive,
            predicted_next_state_hash=state_hash,
            metadata={
                "predicted_next_state_vector": [float(x) for x in pred.predicted_next_state],
                "model_id": self._model.model_id,
            },
        )

    def rollout(
        self,
        initial_state: Any,
        actions: list[Any],
    ) -> list[ModelPrediction]:
        """Rollout a trajectory of actions from an initial state.

        Args:
            initial_state: Starting structural state.
            actions: Sequence of actions to apply.

        Returns:
            List of ModelPrediction, one per step.
        """
        # Encode initial state.
        if isinstance(initial_state, StateVector):
            z = initial_state.vector.copy()
        else:
            z = encode_state(initial_state).vector.copy()

        predictions = []
        for action in actions:
            # Encode action.
            if isinstance(action, ActionVector):
                a_t = action.vector
            elif isinstance(action, tuple) and len(action) >= 2:
                action_type, action_target = action[0], action[1]
                n_nodes = action[2] if len(action) > 2 else 20
                degree_mean = action[3] if len(action) > 3 else 2.0
                a_t = encode_action(action_type, action_target, n_nodes=n_nodes, degree_mean=degree_mean).vector
            else:
                a_t = np.zeros(12)

            # Predict next state.
            pred = self._model.predict(z, a_t)

            import hashlib
            state_hash = hashlib.sha256(pred.predicted_next_state.tobytes()).hexdigest()[:16]

            predictions.append(ModelPrediction(
                predicted_delta_utility=pred.predicted_delta_utility,
                predicted_risk=pred.predicted_risk,
                predicted_cost=pred.predicted_cost,
                predicted_uncertainty=pred.predicted_uncertainty,
                probability_positive=pred.probability_positive,
                predicted_next_state_hash=state_hash,
                metadata={
                    "predicted_next_state_vector": [float(x) for x in pred.predicted_next_state],
                    "model_id": self._model.model_id,
                    "step": len(predictions),
                },
            ))

            # Update z for next step.
            z = pred.predicted_next_state.copy()

        return predictions

    def update(self, state: Any, action: Any, next_state: Any) -> None:
        """Update the model with a realized transition.

        This is the online learning hook. For the lightweight model,
        this is a no-op — online updates are deferred to exp6+.
        """
        pass

    def trust_report(self) -> ModelTrustReport:
        """Produce a trust report for this model."""
        return self._trust_report.to_model_trust_report()
