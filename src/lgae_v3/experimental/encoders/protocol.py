"""Canonical encoder protocols and representation contracts for v6.0-exp3.

Separates state encoding from action encoding from combined representation.
The same encoder can feed outcome prediction, risk estimation, world-model
rollouts, structural experience retrieval, and MPC.

Formal invariant for deterministic encoders:

    E(S, a, C) = constant

for identical state, action, and configuration.

Representation contract:

    @dataclass(frozen=True)
    class StateActionRepresentation:
        encoder_id: str
        encoder_version: str
        schema_hash: str
        vector: tuple[float, ...]
        dimension: int
        state_feature_hash: str
        action_feature_hash: str
        normalization_hash: str | None
        metadata: Mapping[str, Any]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable
import hashlib
import math
import numpy as np


# ---------------------------------------------------------------------------
# Representation dataclasses.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EncodedState:
    """Encoded structural state."""
    vector: tuple[float, ...]
    dimension: int
    encoder_id: str
    schema_hash: str
    missing_mask: tuple[bool, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "dimension": int(self.dimension),
            "encoder_id": self.encoder_id,
            "schema_hash": self.schema_hash,
            "vector": list(self.vector),
            "missing_mask": list(self.missing_mask) if self.missing_mask else [],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EncodedAction:
    """Encoded structural action."""
    vector: tuple[float, ...]
    dimension: int
    encoder_id: str
    schema_hash: str
    action_type: str
    missing_mask: tuple[bool, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "dimension": int(self.dimension),
            "encoder_id": self.encoder_id,
            "schema_hash": self.schema_hash,
            "action_type": self.action_type,
            "vector": list(self.vector),
            "missing_mask": list(self.missing_mask) if self.missing_mask else [],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class StateActionRepresentation:
    """Combined state-action representation.

    This is the canonical output of the encoder pipeline. It captures
    everything needed for downstream models (outcome, risk, cost, world
    model) with full provenance.

    Formal invariant:
        E(S, a, C) = constant
    for deterministic encoders with identical inputs and configuration.
    """
    encoder_id: str
    encoder_version: str
    schema_hash: str
    vector: tuple[float, ...]
    dimension: int
    state_feature_hash: str
    action_feature_hash: str
    normalization_hash: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "schema_hash": self.schema_hash,
            "dimension": int(self.dimension),
            "vector": list(self.vector),
            "state_feature_hash": self.state_feature_hash,
            "action_feature_hash": self.action_feature_hash,
            "normalization_hash": self.normalization_hash,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Encoder lifecycle.
# ---------------------------------------------------------------------------

class EncoderLifecycle(str):
    """Encoder fitting lifecycle."""
    UNFIT = "unfit"
    FITTED_TRAIN = "fitted_train"
    FROZEN = "frozen"


# ---------------------------------------------------------------------------
# Action encoding schema.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ActionEncodingSchema:
    """Canonical action encoding schema with deterministic mutation types."""
    mutation_types: tuple[str, ...] = (
        "ADD_EDGE",
        "REMOVE_EDGE",
        "UPDATE_WEIGHT",
        "REWEIGHT_AFFINITY",
        "SPAWN_FIBER",
        "PRUNE_FIBER",
        "CHANGE_GAUGE",
        "NO_OP",
    )
    version: str = "V6_EXP3_ACTION_SCHEMA_1"

    @property
    def schema_hash(self) -> str:
        content = f"{self.version}:{','.join(self.mutation_types)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def n_types(self) -> int:
        return len(self.mutation_types)

    def type_index(self, mutation_type: str) -> int:
        """Get the deterministic index for a mutation type."""
        try:
            return self.mutation_types.index(mutation_type)
        except ValueError:
            return -1  # Unknown type.

    def to_log(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mutation_types": list(self.mutation_types),
            "schema_hash": self.schema_hash,
            "n_types": int(self.n_types),
        }


# Default schema instance.
DEFAULT_ACTION_SCHEMA = ActionEncodingSchema()


# ---------------------------------------------------------------------------
# Protocols.
# ---------------------------------------------------------------------------

@runtime_checkable
class StructuralStateEncoder(Protocol):
    """Protocol for state encoders."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def schema_hash(self) -> str: ...

    @property
    def requires_fit(self) -> bool: ...

    @property
    def deterministic(self) -> bool: ...

    @property
    def lifecycle(self) -> str: ...

    def encode_state(
        self,
        state: Any,
        global_features: Sequence[float],
    ) -> EncodedState:
        ...


@runtime_checkable
class StructuralActionEncoder(Protocol):
    """Protocol for action encoders."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def schema_hash(self) -> str: ...

    def encode_action(
        self,
        action_type: str,
        action_target: dict[str, Any],
        local_features: Sequence[float],
    ) -> EncodedAction:
        ...


@runtime_checkable
class StateActionEncoder(Protocol):
    """Protocol for combined state-action encoders."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def schema_hash(self) -> str: ...

    @property
    def requires_fit(self) -> bool: ...

    @property
    def deterministic(self) -> bool: ...

    @property
    def lifecycle(self) -> str: ...

    def encode(
        self,
        state: Any,
        global_features: Sequence[float],
        action_type: str,
        action_target: dict[str, Any],
        local_features: Sequence[float],
    ) -> StateActionRepresentation:
        ...


# ---------------------------------------------------------------------------
# Utility functions.
# ---------------------------------------------------------------------------

def feature_hash(features: Sequence[float]) -> str:
    """Deterministic hash of a feature vector."""
    content = ",".join(f"{v:.10f}" for v in features)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def safe_log1p(x: float) -> float:
    """Bounded log(1+x) that handles negative and nonfinite values."""
    if not math.isfinite(x):
        return 0.0
    if x < -0.999:
        return -20.0  # bounded
    return math.log1p(max(x, -0.999))


def safe_normalize(
    values: Sequence[float],
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
    missing_mask: Sequence[bool] | None = None,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Normalize features with missing-value handling.

    Returns (normalized_values, missing_mask).
    Missing values are set to 0.0 with mask=True.
    """
    n = len(values)
    mask = tuple(bool(missing_mask[i]) if missing_mask else False for i in range(n))
    result = []
    for i, v in enumerate(values):
        if mask[i] or not math.isfinite(v):
            result.append(0.0)
            continue
        if mean is not None and std is not None:
            s = std[i] if std[i] > 1e-10 else 1.0
            result.append((v - mean[i]) / s)
        else:
            result.append(float(v))
    return tuple(result), mask


def ensure_finite(values: Sequence[float], default: float = 0.0) -> tuple[float, ...]:
    """Replace nonfinite values with default."""
    return tuple(v if math.isfinite(v) else default for v in values)
