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
    """Trust report for the lightweight world model.

    v6.0-exp5.1: Multi-factor trust score. No longer uses the
    simplistic 0.5 + R²/2 formula. Instead combines:
    - one_step_error: held-out one-step prediction error
    - rollout_error: multi-step rollout degradation
    - calibration: uncertainty-error correlation
    - ood_distance: distance to training distribution
    - tail_regret: worst-case regret
    - failure_rate: fraction of predictions with catastrophic error

    The trust score is conservative: it is bounded by the weakest
    factor. If any factor is critically weak, trust is low regardless
    of other factors being strong.
    """
    model_name: str = "lightweight_world_model"
    mean_prediction_error: float = 0.0
    ood_distance: float = 0.0
    calibration_correlation: float = 0.0
    trust_score: float = 0.0
    recommended_horizon: int = 1
    recommended_exact_verification_fraction: float = 1.0
    # v6.0-exp5.1: Multi-factor components.
    one_step_r2: float = 0.0
    rollout_r2: float = 0.0
    rollout_degradation: float = 0.0  # how much worse rollout is vs one-step
    tail_regret: float = 0.0
    failure_rate: float = 0.0
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


def compute_multi_factor_trust(
    *,
    one_step_r2: float,
    rollout_r2: float,
    rollout_degradation: float,
    calibration_correlation: float,
    tail_regret: float,
    failure_rate: float,
    ood_distance: float = 0.0,
) -> WorldModelTrustReport:
    """Compute a multi-factor trust score.

    The trust score is the minimum of several factor scores,
    ensuring that no single strong factor can mask a weak one.

    Factors:
    - one_step_quality: f(one_step_r2) — one-step prediction quality
    - rollout_quality: f(rollout_r2, degradation) — multi-step stability
    - calibration_quality: f(calibration_correlation) — uncertainty usefulness
    - tail_safety: f(tail_regret, failure_rate) — worst-case behavior
    - ood_safety: f(ood_distance) — distribution shift robustness

    The final trust score is:
        trust = min(one_step_quality, rollout_quality, calibration_quality, tail_safety, ood_safety)

    All factors are bounded to [0, 1].
    """
    # One-step quality: R² mapped to [0, 1].
    one_step_quality = max(0.0, min(1.0, one_step_r2))

    # Rollout quality: penalized by degradation.
    rollout_quality = max(0.0, min(1.0, rollout_r2 - rollout_degradation * 0.5))

    # Calibration quality: correlation mapped to [0, 1].
    calibration_quality = max(0.0, min(1.0, calibration_correlation))

    # Tail safety: penalized by tail regret and failure rate.
    tail_safety = max(0.0, min(1.0, 1.0 - tail_regret - failure_rate * 2.0))

    # OOD safety: penalized by distance.
    ood_safety = max(0.0, min(1.0, 1.0 - ood_distance))

    # Trust is the minimum factor — conservative.
    trust = min(one_step_quality, rollout_quality, calibration_quality, tail_safety, ood_safety)

    # Recommended horizon: 1 if rollout is poor, 2 if moderate, 3 if good.
    if rollout_quality < 0.3:
        recommended_horizon = 1
    elif rollout_quality < 0.6:
        recommended_horizon = 2
    else:
        recommended_horizon = 3

    # Exact verification: always 1.0 until trust is very high.
    if trust > 0.8:
        exact_frac = 0.9  # even high trust keeps 90% verification
    else:
        exact_frac = 1.0

    return WorldModelTrustReport(
        trust_score=trust,
        recommended_horizon=recommended_horizon,
        recommended_exact_verification_fraction=exact_frac,
        one_step_r2=one_step_r2,
        rollout_r2=rollout_r2,
        rollout_degradation=rollout_degradation,
        calibration_correlation=calibration_correlation,
        tail_regret=tail_regret,
        failure_rate=failure_rate,
        ood_distance=ood_distance,
        mean_prediction_error=0.0,  # set by caller if available
        metadata={
            "one_step_quality": one_step_quality,
            "rollout_quality": rollout_quality,
            "calibration_quality": calibration_quality,
            "tail_safety": tail_safety,
            "ood_safety": ood_safety,
            "trust_formula": "min(factors)",
        },
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
