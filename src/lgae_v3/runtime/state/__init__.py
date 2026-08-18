"""State isolation and ownership module (v5.11 Phases 1-3).

This module defines:
- AuthoritativeState: the single object that owns all mutable runtime state
- _AuthorityCapability: internal token required for state mutation
- StateBundle: complete candidate state for atomic swap
- Frozen views: immutable views of state for read-only access
- State hashing: deterministic hashing (no Python hash())
- State errors: violation types

The runtime invariant is:

    S_{t+1} = Commit(S_t, T_t, A_t)

No other path may alter S_t.
"""
from __future__ import annotations

from .authoritative_state import (
    AuthoritativeState, CalibrationState, ModelReference,
)
from .authority_token import _AuthorityCapability
from .state_bundle import StateBundle
from .state_hashing import (
    canonical_encode, state_hash, graph_hash, fiber_hash, gauge_hash,
)
from .state_errors import (
    StateOwnershipError, StaleStateError, StateValidationError,
    DeterminismError, CapabilityError,
)
from .frozen_views import (
    FrozenGraphView, FrozenFiberView, FrozenGaugeView,
    StaleSnapshotError,
)
from .immutable_views import EngineFacade

__all__ = [
    "AuthoritativeState",
    "CalibrationState",
    "ModelReference",
    "StateBundle",
    "EngineFacade",
    "canonical_encode",
    "state_hash",
    "graph_hash",
    "fiber_hash",
    "gauge_hash",
    "StateOwnershipError",
    "StaleStateError",
    "StateValidationError",
    "DeterminismError",
    "CapabilityError",
    "FrozenGraphView",
    "FrozenFiberView",
    "FrozenGaugeView",
    "StaleSnapshotError",
]
