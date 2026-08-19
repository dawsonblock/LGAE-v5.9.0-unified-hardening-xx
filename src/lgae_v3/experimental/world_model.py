"""Abstract world-model and outcome-model interfaces for v6.

These are INTERFACES ONLY. No neural implementation is provided in
v6.0-exp1. The interfaces define the contracts that future learned models
must satisfy.

Three interfaces:

1. ``StructuralStateEncoderInterface``: encodes a graph state into a
   fixed-dimensional representation vector. This captures topology, local
   geometry, spectral state, fibers/gauges, diagnosis, uncertainty, recent
   structural history, and task context.

2. ``OutcomeModelInterface``: the narrower learned model:
       f_θ(S, a) → (ΔÛ, R̂, Ĉ, σ)
   Predicts utility delta, reward, cost, and uncertainty for a single
   (state, action) pair. This is easier to validate and immediately useful
   for candidate pruning.

3. ``WorldModelInterface``: the full world model:
       F_θ(S_t, a_t) → Ŝ_{t+1}
   Predicts the next structural state. It remains advisory: it receives
   NO mutation authority. The v5.11 CommitChannel is the permanent
   authority boundary.

All interfaces are advisory-only. They propose and predict; they never
mutate authoritative state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import abc

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """A prediction from a learned model.

    For outcome models: (ΔÛ, R̂, Ĉ, σ, P(ΔU > 0))
    For world models: (Ŝ_{t+1}, σ)

    This unified container supports both use cases.

    Fix 4: Unified with exp4 outcome terminology.
    - ``predicted_reward`` → ``predicted_risk`` (exp4 uses risk, not reward)
    - Added ``probability_positive`` for sign/success classification
    - ``predicted_reward`` retained as deprecated alias for backward compat
    """
    predicted_delta_utility: float | None = None
    predicted_risk: float | None = None
    predicted_cost: float | None = None
    predicted_uncertainty: float | None = None
    probability_positive: float | None = None
    predicted_next_state: Tensor | None = None
    predicted_next_state_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Deprecated alias for backward compatibility.
    # Phase 2 (exp4.2): reward and risk are semantically opposite.
    # This alias emits a DeprecationWarning and is scheduled for deletion
    # before exp5. It must NOT be used by any new code or exp4.2 runner.
    @property
    def predicted_reward(self) -> float | None:
        """Deprecated. Use ``predicted_risk`` instead.

        .. deprecated::
            ``predicted_reward`` is semantically incorrect as an alias for
            ``predicted_risk`` (reward and risk are opposites). It is retained
            only for backward compatibility with legacy callers and will be
            removed before exp5. Use ``predicted_risk`` directly.
        """
        import warnings
        warnings.warn(
            "ModelPrediction.predicted_reward is deprecated and semantically "
            "incorrect (reward != risk). Use predicted_risk instead. "
            "This alias will be removed before exp5.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.predicted_risk

    def to_log(self) -> dict[str, Any]:
        return {
            "predicted_delta_utility": self.predicted_delta_utility,
            "predicted_risk": self.predicted_risk,
            "predicted_cost": self.predicted_cost,
            "predicted_uncertainty": self.predicted_uncertainty,
            "probability_positive": self.probability_positive,
            "predicted_next_state_hash": self.predicted_next_state_hash,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class ModelTrustReport:
    """Report on model trustworthiness.

    Used by the trust calibration layer (v6.0-exp7) to decide whether to
    use the learned model for cheap evaluation or fall back to exact
    shadow execution.
    """
    model_name: str
    mean_prediction_error: float
    ood_distance: float
    calibration_correlation: float
    trust_score: float  # 0.0 = no trust, 1.0 = full trust
    recommended_horizon: int  # suggested MPC horizon based on trust
    recommended_exact_verification_fraction: float  # 0.0-1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "mean_prediction_error": float(self.mean_prediction_error),
            "ood_distance": float(self.ood_distance),
            "calibration_correlation": float(self.calibration_correlation),
            "trust_score": float(self.trust_score),
            "recommended_horizon": int(self.recommended_horizon),
            "recommended_exact_verification_fraction": float(self.recommended_exact_verification_fraction),
            "metadata": self.metadata,
        }


class StructuralStateEncoderInterface(abc.ABC):
    """Abstract interface for structural state encoders.

    A state encoder captures the full structural state into a fixed-dimensional
    representation vector. This includes:
    - Graph topology (adjacency, degree distribution, spectral features).
    - Local geometry (latent embeddings, fiber states, gauge states).
    - Spectral state (eigenvalues, spectral gap, curvature).
    - Diagnosis (structural deficits, bottlenecks, oversquashing).
    - Uncertainty (epistemic, aleatoric).
    - Recent structural history (last N mutations and their outcomes).
    - Task context (loss, loss delta, target metrics).

    The encoder does NOT need to ingest the complete raw graph for every
    prediction. It produces a compact summary that downstream models can use.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def encoding_dim(self) -> int: ...

    @abc.abstractmethod
    def encode(self, state: Any) -> Tensor:
        """Encode a structural state into a fixed-dimensional vector.

        Args:
            state: The structural state (graph, fibers, gauges, diagnosis,
                history, task context). The exact type is implementation-
                specific but must be serializable.

        Returns:
            A tensor of shape (encoding_dim,) representing the state.
        """
        ...

    @abc.abstractmethod
    def encode_batch(self, states: list[Any]) -> Tensor:
        """Encode a batch of states.

        Args:
            states: List of structural states.

        Returns:
            A tensor of shape (batch_size, encoding_dim).
        """
        ...


class OutcomeModelInterface(abc.ABC):
    """Abstract interface for learned outcome models.

    The narrower problem:
        f_θ(S, a) → (ΔÛ, R̂, Ĉ, σ, P(ΔU > 0))

    Predicts:
    - ΔÛ: predicted utility delta.
    - R̂: predicted risk (instability, constraint margin, OOD, fragmentation, rollback).
    - Ĉ: predicted compute cost.
    - σ: predicted uncertainty (epistemic + aleatoric).
    - P(ΔU > 0): probability that the action improves utility (sign/success).

    Fix 4: Unified with exp4 outcome terminology.
    - ``reward`` → ``risk`` (exp4 uses risk components, not a scalar reward)
    - Added ``probability_positive`` for sign/success classification

    This is advisory-only. It is used for candidate pruning and cheap
    evaluation. It NEVER mutates authoritative state.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def predict(self, state: Any, action: Any) -> ModelPrediction:
        """Predict the outcome of applying an action to a state.

        Args:
            state: The structural state (or its encoding).
            action: The candidate action.

        Returns:
            A ModelPrediction with (ΔÛ, R̂, Ĉ, σ).
        """
        ...

    @abc.abstractmethod
    def predict_batch(self, state: Any, actions: list[Any]) -> list[ModelPrediction]:
        """Predict outcomes for multiple actions on the same state.

        Args:
            state: The structural state (or its encoding).
            actions: List of candidate actions.

        Returns:
            List of ModelPrediction, one per action.
        """
        ...

    @abc.abstractmethod
    def update(self, state: Any, action: Any, realized: ModelPrediction) -> None:
        """Update the model with a realized outcome.

        This is the online learning hook. The model can update its internal
        state based on the difference between predicted and realized outcomes.

        Args:
            state: The state the action was applied to.
            action: The action that was taken.
            realized: The realized outcome (with actual ΔU, R, C).
        """
        ...

    @abc.abstractmethod
    def trust_report(self) -> ModelTrustReport:
        """Produce a trust report for this model.

        Returns:
            A ModelTrustReport with calibration metrics and recommended
            usage parameters.
        """
        ...


class WorldModelInterface(abc.ABC):
    """Abstract interface for learned world models.

    The full world model:
        F_θ(S_t, a_t) → Ŝ_{t+1}

    Predicts the next structural state given the current state and an action.

    CRITICAL: This is advisory-only. It receives NO mutation authority.
    The v5.11 CommitChannel is the permanent authority boundary. The world
    model is used for:
    - Cheap trajectory expansion in hybrid MPC.
    - Candidate pruning (eliminate actions predicted to fail).
    - Trust-calibrated planning (shorten horizons when trust is low).

    It is NEVER used to:
    - Mutate authoritative state.
    - Bypass governance authorization.
    - Replace exact shadow execution for finalists.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def predict_next_state(self, state: Any, action: Any) -> ModelPrediction:
        """Predict the next structural state.

        Args:
            state: The current structural state (or its encoding).
            action: The action to apply.

        Returns:
            A ModelPrediction with predicted_next_state and uncertainty.
        """
        ...

    @abc.abstractmethod
    def rollout(
        self,
        initial_state: Any,
        actions: list[Any],
    ) -> list[ModelPrediction]:
        """Rollout a trajectory of actions from an initial state.

        Args:
            initial_state: The starting state.
            actions: Sequence of actions to apply.

        Returns:
            List of ModelPrediction, one per step, with predicted states
            and accumulated uncertainty.
        """
        ...

    @abc.abstractmethod
    def update(self, state: Any, action: Any, next_state: Any) -> None:
        """Update the model with a realized transition.

        Args:
            state: The state before the action.
            action: The action taken.
            next_state: The realized next state.
        """
        ...

    @abc.abstractmethod
    def trust_report(self) -> ModelTrustReport:
        """Produce a trust report for this model."""
        ...
