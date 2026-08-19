"""Canonical state and action encoding for the lightweight world model.

The state vector z_t is derived from the StructuralStateSummary's 11
numeric fields, plus derived features (log-density, log-spectral-gap,
spectral-gap-per-node). This gives a compact, fixed-dimensional
representation that the exp4.2 study showed carries useful predictive
signal.

The action vector a_t encodes the mutation type (one-hot) plus the
action target features (u, v normalized by graph size, degree-based
features).

These encodings are FROZEN for exp5. No changes after training begins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import json
import math
import numpy as np


# ---------------------------------------------------------------------------
# Dimensions (frozen).
# ---------------------------------------------------------------------------

# 11 raw state fields + 3 derived features = 14.
STATE_DIM = 14

# Mutation types (frozen, must match runtime operators).
MUTATION_TYPES = (
    "ADD_EDGE",
    "REMOVE_EDGE",
    "REWIRE",
    "ADD_FIBER",
    "REMOVE_FIBER",
    "GAUGE_TRANSFORM",
)
N_MUTATION_TYPES = len(MUTATION_TYPES)

# Action: one-hot mutation type (6) + target features (6) = 12.
ACTION_DIM = N_MUTATION_TYPES + 6


# ---------------------------------------------------------------------------
# State encoding.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StateVector:
    """Encoded structural state vector."""
    vector: np.ndarray
    dim: int = STATE_DIM
    schema_hash: str = ""

    def __post_init__(self) -> None:
        if not self.schema_hash:
            object.__setattr__(self, "schema_hash", _state_schema_hash())

    def to_log(self) -> dict[str, Any]:
        return {
            "dim": int(self.dim),
            "schema_hash": self.schema_hash,
            "vector": [float(x) for x in self.vector],
        }


def _state_schema_hash() -> str:
    content = json.dumps({
        "dim": STATE_DIM,
        "fields": [
            "n_nodes", "n_edges", "density", "spectral_gap",
            "degree_mean", "degree_std", "n_components",
            "avg_clustering", "fiber_count", "fiber_width", "gauge_dim",
            "log_density", "log_spectral_gap", "spectral_gap_per_node",
        ],
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def encode_state(state: Any) -> StateVector:
    """Encode a StructuralStateSummary into a fixed-dimensional vector.

    Args:
        state: A StructuralStateSummary or compatible object with
            n_nodes, n_edges, density, spectral_gap, degree_mean,
            degree_std, n_components, avg_clustering, fiber_count,
            fiber_width, gauge_dim.

    Returns:
        StateVector of dimension STATE_DIM.
    """
    n_nodes = float(getattr(state, "n_nodes", 10))
    n_edges = float(getattr(state, "n_edges", 9))
    density = float(getattr(state, "density", 0.0))
    spectral_gap = float(getattr(state, "spectral_gap", 0.0))
    degree_mean = float(getattr(state, "degree_mean", 0.0))
    degree_std = float(getattr(state, "degree_std", 0.0))
    n_components = float(getattr(state, "n_components", 1))
    avg_clustering = float(getattr(state, "avg_clustering", 0.0))
    fiber_count = float(getattr(state, "fiber_count", 0))
    fiber_width = float(getattr(state, "fiber_width", 0))
    gauge_dim = float(getattr(state, "gauge_dim", 0))

    # Derived features.
    log_density = math.log1p(max(density, 0.0))
    log_spectral_gap = math.log1p(max(abs(spectral_gap), 1e-10))
    spectral_gap_per_node = spectral_gap / max(n_nodes, 1.0)

    vec = np.array([
        n_nodes, n_edges, density, spectral_gap,
        degree_mean, degree_std, n_components,
        avg_clustering, fiber_count, fiber_width, gauge_dim,
        log_density, log_spectral_gap, spectral_gap_per_node,
    ], dtype=np.float64)

    return StateVector(vector=vec)


def decode_state(sv: StateVector) -> dict[str, float]:
    """Decode a StateVector back to a dictionary of state fields.

    This is for inspection/verification only — the decoded values
    are approximate because derived features are not invertible.
    """
    v = sv.vector
    return {
        "n_nodes": float(v[0]),
        "n_edges": float(v[1]),
        "density": float(v[2]),
        "spectral_gap": float(v[3]),
        "degree_mean": float(v[4]),
        "degree_std": float(v[5]),
        "n_components": float(v[6]),
        "avg_clustering": float(v[7]),
        "fiber_count": float(v[8]),
        "fiber_width": float(v[9]),
        "gauge_dim": float(v[10]),
    }


# ---------------------------------------------------------------------------
# Action encoding.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ActionVector:
    """Encoded action vector."""
    vector: np.ndarray
    dim: int = ACTION_DIM
    action_type: str = ""
    schema_hash: str = ""

    def __post_init__(self) -> None:
        if not self.schema_hash:
            object.__setattr__(self, "schema_hash", _action_schema_hash())

    def to_log(self) -> dict[str, Any]:
        return {
            "dim": int(self.dim),
            "action_type": self.action_type,
            "schema_hash": self.schema_hash,
            "vector": [float(x) for x in self.vector],
        }


def _action_schema_hash() -> str:
    content = json.dumps({
        "dim": ACTION_DIM,
        "mutation_types": list(MUTATION_TYPES),
        "target_features": [
            "u_normalized", "v_normalized",
            "u_degree_proxy", "v_degree_proxy",
            "edge_exists", "same_component",
        ],
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def encode_action(
    action_type: str,
    action_target: dict[str, Any],
    *,
    n_nodes: int = 20,
    degree_mean: float = 2.0,
) -> ActionVector:
    """Encode an action into a fixed-dimensional vector.

    Args:
        action_type: The mutation type string (e.g., "ADD_EDGE").
        action_target: Dict with "u" and "v" keys (or empty).
        n_nodes: Number of nodes in the graph (for normalization).
        degree_mean: Average degree (for degree proxy).

    Returns:
        ActionVector of dimension ACTION_DIM.
    """
    # One-hot encode mutation type.
    one_hot = np.zeros(N_MUTATION_TYPES, dtype=np.float64)
    if action_type in MUTATION_TYPES:
        one_hot[MUTATION_TYPES.index(action_type)] = 1.0
    else:
        # Unknown mutation type — use uniform encoding.
        one_hot[:] = 1.0 / N_MUTATION_TYPES

    # Target features.
    u = float(action_target.get("u", 0)) if isinstance(action_target, dict) else 0.0
    v = float(action_target.get("v", 0)) if isinstance(action_target, dict) else 0.0
    u_norm = u / max(n_nodes, 1)
    v_norm = v / max(n_nodes, 1)
    u_deg_proxy = u_norm * degree_mean
    v_deg_proxy = v_norm * degree_mean
    edge_exists = 1.0 if abs(u - v) <= 1 else 0.0  # proxy
    same_component = 1.0  # default assumption

    target_feats = np.array([
        u_norm, v_norm, u_deg_proxy, v_deg_proxy, edge_exists, same_component,
    ], dtype=np.float64)

    vec = np.concatenate([one_hot, target_feats])
    return ActionVector(vector=vec, action_type=action_type)


def state_action_schema_hash() -> str:
    """Combined schema hash for state + action encoding."""
    content = json.dumps({
        "state_schema": _state_schema_hash(),
        "action_schema": _action_schema_hash(),
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
