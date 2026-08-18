"""Encoder 0: MinimalControlEncoder — intentionally weak identity floor.

Uses only:
- graph family (one-hot)
- node count
- edge count
- action type (one-hot)

This gives a floor. If future encoders cannot materially beat this,
something is wrong with the benchmark or label signal.
"""
from __future__ import annotations

from typing import Any, Sequence
import hashlib

from .protocol import (
    EncodedState, EncodedAction, StateActionRepresentation,
    EncoderLifecycle, ActionEncodingSchema, DEFAULT_ACTION_SCHEMA,
    feature_hash, ensure_finite,
)
from ..graph_families import FROZEN_TRAIN_FAMILIES


# Known graph families for one-hot encoding.
KNOWN_FAMILIES = tuple(f.value for f in FROZEN_TRAIN_FAMILIES) + (
    "random_ws", "bipartite", "complete", "tree",
)
FAMILY_INDEX = {name: i for i, name in enumerate(KNOWN_FAMILIES)}


class MinimalControlEncoder:
    """Encoder 0: Minimal control baseline.

    Encodes only graph family, node count, edge count, and action type.
    This is the floor that all other encoders must beat.
    """

    name = "minimal-control"
    version = "v1"
    requires_fit = False
    deterministic = True

    def __init__(self) -> None:
        self._schema = ActionEncodingSchema()
        self._lifecycle = EncoderLifecycle.UNFIT
        # State: family_one_hot + log(n_nodes) + log(n_edges) = len(KNOWN_FAMILIES) + 2
        self._state_dim = len(KNOWN_FAMILIES) + 2
        # Action: action_type_one_hot = n_types
        self._action_dim = self._schema.n_types

    @property
    def dimension(self) -> int:
        return self._state_dim + self._action_dim

    @property
    def schema_hash(self) -> str:
        content = f"{self.name}:{self.version}:{self._state_dim}:{self._action_dim}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def encode_state(self, state: Any, global_features: Sequence[float]) -> EncodedState:
        family = getattr(state, "graph_family", "") or ""
        n_nodes = getattr(state, "n_nodes", 0) or 0
        n_edges = getattr(state, "n_edges", 0) or 0
        import math
        vec = [0.0] * len(KNOWN_FAMILIES)
        if family in FAMILY_INDEX:
            vec[FAMILY_INDEX[family]] = 1.0
        vec.append(math.log(max(n_nodes, 1)))
        vec.append(math.log(max(n_edges, 1)))
        vec = ensure_finite(vec)
        return EncodedState(
            vector=vec,
            dimension=len(vec),
            encoder_id=self.name,
            schema_hash=self.schema_hash,
        )

    def encode_action(
        self, action_type: str, action_target: dict[str, Any],
        local_features: Sequence[float],
    ) -> EncodedAction:
        vec = [0.0] * self._schema.n_types
        idx = self._schema.type_index(action_type)
        if idx >= 0:
            vec[idx] = 1.0
        vec = ensure_finite(vec)
        return EncodedAction(
            vector=vec,
            dimension=len(vec),
            encoder_id=self.name,
            schema_hash=self.schema_hash,
            action_type=action_type,
        )

    def encode(
        self, state: Any, global_features: Sequence[float],
        action_type: str, action_target: dict[str, Any],
        local_features: Sequence[float],
    ) -> StateActionRepresentation:
        es = self.encode_state(state, global_features)
        ea = self.encode_action(action_type, action_target, local_features)
        combined = es.vector + ea.vector
        return StateActionRepresentation(
            encoder_id=self.name,
            encoder_version=self.version,
            schema_hash=self.schema_hash,
            vector=combined,
            dimension=len(combined),
            state_feature_hash=feature_hash(es.vector),
            action_feature_hash=feature_hash(ea.vector),
            normalization_hash=None,
        )
