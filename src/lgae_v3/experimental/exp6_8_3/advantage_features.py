"""Pairwise advantage features for exp6.8.3.

Constructs features for the advantage prediction problem:
  A(S) = Q_H(S, a_learned) - Q_H(S, a_baseline)

Features: [phi_L, phi_B, phi_L - phi_B, objective_features]

where phi_X encodes the action identity and state context.
"""
from __future__ import annotations

import numpy as np

from ..exp6_3.exact_mpc import ActionIdentity
from ..exp6_6.objective_spec import ObjectiveSpec, encode_objective, OBJECTIVE_ENCODING_DIM
from ..exp6_8_1.split_state import (
    SplitStructuralState, EXACT_STATE_DIM, CERTIFIED_STATE_DIM, LEARNED_STATE_DIM,
)


# Action type one-hot encoding.
ACTION_TYPES = ["add_edge", "remove_edge", "reweight_edge", "edge_swap"]
ACTION_TYPE_DIM = len(ACTION_TYPES)

# Per-action feature: [type_one_hot, u_normalized, v_normalized, factor, weight, new_target]
ACTION_FEATURE_DIM = ACTION_TYPE_DIM + 5

# Pairwise: [phi_L, phi_B, phi_L - phi_B]
PAIRWISE_FEATURE_DIM = 3 * ACTION_FEATURE_DIM

# Full feature: state + objective + pairwise
STATE_FEATURE_DIM = EXACT_STATE_DIM + CERTIFIED_STATE_DIM + LEARNED_STATE_DIM
FULL_FEATURE_DIM = STATE_FEATURE_DIM + OBJECTIVE_ENCODING_DIM + PAIRWISE_FEATURE_DIM


def encode_action(action: tuple, action_id: ActionIdentity) -> np.ndarray:
    """Encode a single action as a feature vector.

    [type_one_hot, u_norm, v_norm, factor, weight, new_target]
    """
    type_onehot = np.zeros(ACTION_TYPE_DIM, dtype=np.float32)
    mt = action[0] if len(action) > 0 else ""
    if mt in ACTION_TYPES:
        type_onehot[ACTION_TYPES.index(mt)] = 1.0

    u = float(action[1]) / 30.0 if len(action) > 1 else 0.0
    v = float(action[2]) / 30.0 if len(action) > 2 else 0.0

    params = action[3] if len(action) > 3 else {}
    factor = float(params.get("factor", 1.0)) / 5.0
    weight = float(params.get("weight", 1.0)) / 5.0
    new_target = float(params.get("new_target", 0.0)) / 30.0

    return np.concatenate([type_onehot, [u, v, factor, weight, new_target]]).astype(np.float32)


def extract_pairwise_features(
    baseline_action: tuple,
    learned_action: tuple,
    baseline_id: ActionIdentity,
    learned_id: ActionIdentity,
) -> np.ndarray:
    """Extract pairwise features: [phi_L, phi_B, phi_L - phi_B]."""
    phi_L = encode_action(learned_action, learned_id)
    phi_B = encode_action(baseline_action, baseline_id)
    return np.concatenate([phi_L, phi_B, phi_L - phi_B]).astype(np.float32)


def extract_state_features(state: SplitStructuralState) -> np.ndarray:
    """Extract state features: [exact, certified, learned]."""
    return state.to_full_array().astype(np.float32)


def extract_objective_features(spec: ObjectiveSpec) -> np.ndarray:
    """Extract objective encoding."""
    return encode_objective(spec).astype(np.float32)


def build_full_features(
    state: SplitStructuralState,
    objective: ObjectiveSpec,
    baseline_action: tuple,
    learned_action: tuple,
    baseline_id: ActionIdentity,
    learned_id: ActionIdentity,
) -> np.ndarray:
    """Build the full feature vector for advantage prediction."""
    state_feat = extract_state_features(state)
    obj_feat = extract_objective_features(objective)
    pairwise = extract_pairwise_features(baseline_action, learned_action, baseline_id, learned_id)
    return np.concatenate([state_feat, obj_feat, pairwise]).astype(np.float32)
