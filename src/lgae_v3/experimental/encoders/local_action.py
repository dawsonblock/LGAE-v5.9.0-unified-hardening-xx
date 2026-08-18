"""Encoder 2: LocalActionEncoder — 24+12=36-dim global+local representation.

Combines the 24-dim global state features with the 12-dim local action
features. This is the first serious baseline.

The consequences of adding/removing one edge are highly local, so local
action features may be more important than global graph embeddings for
many structural interventions.
"""
from __future__ import annotations

from typing import Any, Sequence
import hashlib

from .protocol import (
    EncodedState, EncodedAction, StateActionRepresentation,
    EncoderLifecycle, ActionEncodingSchema, DEFAULT_ACTION_SCHEMA,
    feature_hash, ensure_finite,
)
from .normalization import NormalizationStatistics


class LocalActionEncoder:
    """Encoder 2: Global (24) + Local action (12) = 36-dim + action one-hot."""

    name = "global-local"
    version = "v1"
    deterministic = True

    def __init__(self) -> None:
        self._global_norm = NormalizationStatistics()
        self._local_norm = NormalizationStatistics()
        self._schema = ActionEncodingSchema()
        self._global_dim = 24
        self._local_dim = 12
        self._action_dim = self._schema.n_types

    @property
    def dimension(self) -> int:
        return self._global_dim + self._local_dim + self._action_dim

    @property
    def schema_hash(self) -> str:
        content = f"{self.name}:{self.version}:{self._global_dim}:{self._local_dim}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def lifecycle(self) -> str:
        return self._global_norm.state

    def fit(
        self,
        global_features_list: Sequence[Sequence[float]],
        local_features_list: Sequence[Sequence[float]],
        *,
        split: str = "train",
        dataset_hash: str = "",
        feature_schema_hash: str = "",
    ) -> None:
        self._global_norm.fit(global_features_list, split=split,
                              dataset_hash=dataset_hash,
                              feature_schema_hash=feature_schema_hash)
        self._local_norm.fit(local_features_list, split=split,
                             dataset_hash=dataset_hash,
                             feature_schema_hash=feature_schema_hash)

    def freeze(self) -> None:
        self._global_norm.freeze()
        self._local_norm.freeze()

    @property
    def normalization_hash(self) -> str:
        return self._global_norm.normalization_hash + self._local_norm.normalization_hash

    def encode_state(self, state: Any, global_features: Sequence[float]) -> EncodedState:
        normed, mask = self._global_norm.transform(global_features)
        normed = ensure_finite(normed)
        return EncodedState(
            vector=normed, dimension=len(normed),
            encoder_id=self.name, schema_hash=self.schema_hash,
            missing_mask=mask,
        )

    def encode_action(
        self, action_type: str, action_target: dict[str, Any],
        local_features: Sequence[float],
    ) -> EncodedAction:
        normed, mask = self._local_norm.transform(local_features)
        # Append action type one-hot.
        type_vec = [0.0] * self._schema.n_types
        idx = self._schema.type_index(action_type)
        if idx >= 0:
            type_vec[idx] = 1.0
        vec = list(normed) + type_vec
        vec = ensure_finite(vec)
        full_mask = mask + tuple(False for _ in type_vec)
        return EncodedAction(
            vector=vec, dimension=len(vec),
            encoder_id=self.name, schema_hash=self.schema_hash,
            action_type=action_type, missing_mask=full_mask,
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
            encoder_id=self.name, encoder_version=self.version,
            schema_hash=self.schema_hash, vector=combined,
            dimension=len(combined),
            state_feature_hash=feature_hash(es.vector),
            action_feature_hash=feature_hash(ea.vector),
            normalization_hash=self.normalization_hash,
        )
