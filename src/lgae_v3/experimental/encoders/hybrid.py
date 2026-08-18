"""Encoder 8: HybridEncoder — concatenation of all representations.

    z = [z_global ‖ z_local ‖ z_geometric ‖ z_graph ‖ z_action]

This allows the later outcome model to decide what information matters.
But regularize and ablate it — do not assume more features are better.
"""
from __future__ import annotations

from typing import Any, Sequence
import hashlib

from .protocol import (
    EncodedState, EncodedAction, StateActionRepresentation,
    ActionEncodingSchema, DEFAULT_ACTION_SCHEMA,
    feature_hash, ensure_finite,
)
from .normalization import NormalizationStatistics


class HybridEncoder:
    """Encoder 8: Hybrid representation combining multiple encoders.

    Concatenates the outputs of multiple sub-encoders to create a rich
    combined representation. The sub-encoders must be fitted/frozen
    independently before this encoder is used.
    """

    name = "hybrid"
    version = "v1"
    deterministic = True  # If all sub-encoders are deterministic

    def __init__(self, sub_encoders: list[Any] | None = None) -> None:
        self._sub_encoders = sub_encoders or []
        self._schema = DEFAULT_ACTION_SCHEMA
        self._lifecycle = "unfit"

    @property
    def dimension(self) -> int:
        return sum(e.dimension for e in self._sub_encoders)

    @property
    def schema_hash(self) -> str:
        sub_hashes = ":".join(e.schema_hash for e in self._sub_encoders)
        content = f"{self.name}:{self.version}:{sub_hashes}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def requires_fit(self) -> bool:
        return any(e.requires_fit for e in self._sub_encoders)

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def add_encoder(self, encoder: Any) -> None:
        """Add a sub-encoder to the hybrid."""
        self._sub_encoders.append(encoder)

    def freeze(self) -> None:
        """Freeze all sub-encoders."""
        for e in self._sub_encoders:
            if hasattr(e, "freeze"):
                e.freeze()
        self._lifecycle = "frozen"

    def encode_state(self, state: Any, global_features: Sequence[float]) -> EncodedState:
        vectors = []
        masks = []
        for e in self._sub_encoders:
            es = e.encode_state(state, global_features)
            vectors.extend(es.vector)
            masks.extend(es.missing_mask)
        vec = ensure_finite(vectors)
        return EncodedState(
            vector=vec, dimension=len(vec),
            encoder_id=self.name, schema_hash=self.schema_hash,
            missing_mask=tuple(masks) if masks else (),
        )

    def encode_action(
        self, action_type: str, action_target: dict[str, Any],
        local_features: Sequence[float],
    ) -> EncodedAction:
        vectors = []
        masks = []
        for e in self._sub_encoders:
            ea = e.encode_action(action_type, action_target, local_features)
            vectors.extend(ea.vector)
            masks.extend(ea.missing_mask)
        vec = ensure_finite(vectors)
        return EncodedAction(
            vector=vec, dimension=len(vec),
            encoder_id=self.name, schema_hash=self.schema_hash,
            action_type=action_type, missing_mask=tuple(masks) if masks else (),
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
            normalization_hash=None,
            metadata={"n_sub_encoders": len(self._sub_encoders)},
        )
