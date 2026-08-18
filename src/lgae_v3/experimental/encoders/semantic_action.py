"""Encoder 3: SemanticActionEncoder — mutation-semantic encoding.

Does not treat ADD_EDGE, REMOVE_EDGE, UPDATE_WEIGHT as interchangeable
numeric actions. Creates a canonical action representation with explicit
fields:

- mutation type (one-hot, deterministic schema)
- source node degree
- target node degree
- current weight
- proposed weight
- weight delta
- edge existed before (bool)
- source zone (community proxy)
- target zone (community proxy)
- protected status (bool)

The schema itself has a hash so that action encoding is stable across
runs and model versions.
"""
from __future__ import annotations

from typing import Any, Sequence
import hashlib
import math

from .protocol import (
    EncodedState, EncodedAction, StateActionRepresentation,
    ActionEncodingSchema, DEFAULT_ACTION_SCHEMA,
    feature_hash, ensure_finite,
)
from .normalization import NormalizationStatistics


class SemanticActionEncoder:
    """Encoder 3: Mutation-semantic action encoding.

    Combines global state features with a semantically rich action
    representation that captures mutation type, node properties, weight
    changes, and edge existence.
    """

    name = "semantic-action"
    version = "v1"
    deterministic = True

    def __init__(self) -> None:
        self._global_norm = NormalizationStatistics()
        self._semantic_norm = NormalizationStatistics()
        self._schema = DEFAULT_ACTION_SCHEMA
        # Semantic action fields:
        # mutation_type_one_hot (n_types) +
        # source_degree, target_degree, current_weight, proposed_weight,
        # weight_delta, edge_existed, source_zone, target_zone, protected
        # = n_types + 9
        self._global_dim = 24
        self._semantic_action_dim = self._schema.n_types + 9

    @property
    def dimension(self) -> int:
        return self._global_dim + self._semantic_action_dim

    @property
    def schema_hash(self) -> str:
        content = f"{self.name}:{self.version}:{self._schema.schema_hash}"
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
        semantic_action_list: Sequence[Sequence[float]],
        *,
        split: str = "train",
        dataset_hash: str = "",
        feature_schema_hash: str = "",
    ) -> None:
        self._global_norm.fit(global_features_list, split=split,
                              dataset_hash=dataset_hash,
                              feature_schema_hash=feature_schema_hash)
        self._semantic_norm.fit(semantic_action_list, split=split,
                                dataset_hash=dataset_hash,
                                feature_schema_hash=feature_schema_hash)

    def freeze(self) -> None:
        self._global_norm.freeze()
        self._semantic_norm.freeze()

    @property
    def normalization_hash(self) -> str:
        return self._global_norm.normalization_hash + self._semantic_norm.normalization_hash

    def _build_semantic_action(
        self, action_type: str, action_target: dict[str, Any],
        local_features: Sequence[float],
    ) -> list[float]:
        """Build the semantic action vector from action target and local features."""
        # Mutation type one-hot.
        type_vec = [0.0] * self._schema.n_types
        idx = self._schema.type_index(action_type)
        if idx >= 0:
            type_vec[idx] = 1.0

        # Extract semantic fields from action_target and local_features.
        u = action_target.get("u", 0)
        v = action_target.get("v", 0)
        # local_features: [source_degree, target_degree, shortest_path_distance,
        #   local_curvature, same_community, bridge_likelihood,
        #   effective_resistance, predicted_connectivity_effect,
        #   source_degree_centrality, target_degree_centrality,
        #   common_neighbors, jaccard_coefficient]
        source_degree = float(local_features[0]) if len(local_features) > 0 else 0.0
        target_degree = float(local_features[1]) if len(local_features) > 1 else 0.0
        current_weight = float(action_target.get("current_weight", 1.0))
        proposed_weight = float(action_target.get("proposed_weight", 1.0))
        weight_delta = proposed_weight - current_weight
        edge_existed = 1.0 if action_target.get("edge_existed", False) else 0.0
        source_zone = float(local_features[4]) if len(local_features) > 4 else 0.0  # same_community
        target_zone = float(local_features[4]) if len(local_features) > 4 else 0.0
        protected = 1.0 if action_target.get("protected", False) else 0.0

        semantic = [
            source_degree, target_degree,
            current_weight, proposed_weight, weight_delta,
            edge_existed, source_zone, target_zone, protected,
        ]
        return type_vec + semantic

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
        semantic = self._build_semantic_action(action_type, action_target, local_features)
        normed, mask = self._semantic_norm.transform(semantic)
        normed = ensure_finite(normed)
        return EncodedAction(
            vector=normed, dimension=len(normed),
            encoder_id=self.name, schema_hash=self.schema_hash,
            action_type=action_type, missing_mask=mask,
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
