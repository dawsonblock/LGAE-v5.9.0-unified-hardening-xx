"""Encoder 1: GlobalStateEncoder — 24-dim handcrafted global features.

Uses the 24-dimensional global feature vector from exp2.
Normalization is fitted on train data only and frozen before
validation/held-out evaluation.
"""
from __future__ import annotations

from typing import Any, Sequence
import hashlib
import math

from .protocol import (
    EncodedState, EncodedAction, StateActionRepresentation,
    EncoderLifecycle, ActionEncodingSchema, DEFAULT_ACTION_SCHEMA,
    feature_hash, ensure_finite,
)
from .normalization import NormalizationStatistics


class GlobalStateEncoder:
    """Encoder 1: Handcrafted global state features (24-dim)."""

    name = "global"
    version = "v1"
    deterministic = True

    def __init__(self) -> None:
        self._norm = NormalizationStatistics()
        self._schema = ActionEncodingSchema()
        self._state_dim = 24
        self._action_dim = self._schema.n_types

    @property
    def dimension(self) -> int:
        return self._state_dim + self._action_dim

    @property
    def schema_hash(self) -> str:
        content = f"{self.name}:{self.version}:{self._state_dim}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def lifecycle(self) -> str:
        return self._norm.state

    def fit(
        self, features: Sequence[Sequence[float]], *,
        split: str = "train", dataset_hash: str = "",
        feature_schema_hash: str = "",
    ) -> None:
        self._norm.fit(features, split=split, dataset_hash=dataset_hash,
                       feature_schema_hash=feature_schema_hash)

    def freeze(self) -> None:
        self._norm.freeze()

    @property
    def normalization_hash(self) -> str:
        return self._norm.normalization_hash

    def encode_state(self, state: Any, global_features: Sequence[float]) -> EncodedState:
        normed, mask = self._norm.transform(global_features)
        normed = ensure_finite(normed)
        return EncodedState(
            vector=normed,
            dimension=len(normed),
            encoder_id=self.name,
            schema_hash=self.schema_hash,
            missing_mask=mask,
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
            vector=vec, dimension=len(vec),
            encoder_id=self.name, schema_hash=self.schema_hash,
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
            encoder_id=self.name, encoder_version=self.version,
            schema_hash=self.schema_hash, vector=combined,
            dimension=len(combined),
            state_feature_hash=feature_hash(es.vector),
            action_feature_hash=feature_hash(ea.vector),
            normalization_hash=self._norm.normalization_hash,
        )
