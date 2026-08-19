"""v6.0-exp5: Lightweight structural latent world model.

Authorized by exp4.2 held-out study (QUALIFIED_SIMPLE).
The recommended architecture is ``lightweight_latent_dynamics``:
a simple model that predicts the next structural state from the
canonical state/action representation, without requiring a large
graph neural network.

Architecture:
    z_{t+1} = F_θ(z_t, a_t)         (dynamics)
    (ΔÛ, R̂, Ĉ, σ) = g_φ(z_t, a_t)  (outcomes)

where z_t is the canonical structural state vector and a_t is the
encoded action vector.

CRITICAL: This is advisory-only. It receives NO mutation authority.
The v5.11 CommitChannel is the permanent authority boundary.
"""
from __future__ import annotations

from .state_encoding import (
    STATE_DIM,
    ACTION_DIM,
    MUTATION_TYPES,
    encode_state,
    decode_state,
    encode_action,
    StateVector,
    ActionVector,
)
from .dynamics import (
    DynamicsModel,
    LinearDynamics,
    MLPDynamics,
    DynamicsMetrics,
    compute_dynamics_metrics,
)
from .joint_model import (
    JointWorldModel,
    JointModelConfig,
    JointModelMetrics,
    WorldModelPrediction,
)
from .training import (
    TrainingConfig,
    TrainingResult,
    train_world_model,
)
from .evaluation import (
    EvaluationResult,
    evaluate_world_model,
    rollout_evaluation,
    RolloutReport,
)
from .world_model_impl import (
    LightweightWorldModel,
    WorldModelTrustReport,
)

__all__ = [
    "STATE_DIM",
    "ACTION_DIM",
    "MUTATION_TYPES",
    "encode_state",
    "decode_state",
    "encode_action",
    "StateVector",
    "ActionVector",
    "DynamicsModel",
    "LinearDynamics",
    "MLPDynamics",
    "DynamicsMetrics",
    "compute_dynamics_metrics",
    "JointWorldModel",
    "JointModelConfig",
    "JointModelMetrics",
    "WorldModelPrediction",
    "TrainingConfig",
    "TrainingResult",
    "train_world_model",
    "EvaluationResult",
    "evaluate_world_model",
    "rollout_evaluation",
    "RolloutReport",
    "LightweightWorldModel",
    "WorldModelTrustReport",
]
