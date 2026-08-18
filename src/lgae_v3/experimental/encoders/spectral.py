"""Encoder 6: SpectralEncoder — deterministic spectral embedding.

Creates a graph-native representation without learning. Uses the first K
eigenvectors/eigenvalues of the normalized Laplacian to create a canonical
pooled spectral embedding.

This gives a baseline between handcrafted features and neural graph
encoders.
"""
from __future__ import annotations

from typing import Any, Sequence
import hashlib
import math
import numpy as np
import networkx as nx

from .protocol import (
    EncodedState, EncodedAction, StateActionRepresentation,
    ActionEncodingSchema, DEFAULT_ACTION_SCHEMA,
    feature_hash, ensure_finite,
)
from .normalization import NormalizationStatistics


class SpectralEncoder:
    """Encoder 6: Deterministic spectral embedding.

    Uses the first K eigenvalues of the normalized Laplacian and the
    corresponding eigenvector statistics (mean, std, max) as a pooled
    representation. This is deterministic and requires no training.
    """

    name = "spectral"
    version = "v1"
    deterministic = True

    def __init__(self, k_eigenvalues: int = 8) -> None:
        self.k = int(k_eigenvalues)
        self._global_norm = NormalizationStatistics()
        self._spectral_norm = NormalizationStatistics()
        self._schema = DEFAULT_ACTION_SCHEMA
        self._global_dim = 24
        # Spectral features: k eigenvalues + k eigenvector means + k eigenvector stds
        # = 3*k
        self._spectral_dim = 3 * self.k
        self._action_dim = self._schema.n_types

    @property
    def dimension(self) -> int:
        return self._global_dim + self._spectral_dim + self._action_dim

    @property
    def schema_hash(self) -> str:
        content = f"{self.name}:{self.version}:{self.k}"
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
        spectral_features_list: Sequence[Sequence[float]],
        *,
        split: str = "train",
        dataset_hash: str = "",
        feature_schema_hash: str = "",
    ) -> None:
        self._global_norm.fit(global_features_list, split=split,
                              dataset_hash=dataset_hash,
                              feature_schema_hash=feature_schema_hash)
        self._spectral_norm.fit(spectral_features_list, split=split,
                                dataset_hash=dataset_hash,
                                feature_schema_hash=feature_schema_hash)

    def freeze(self) -> None:
        self._global_norm.freeze()
        self._spectral_norm.freeze()

    @property
    def normalization_hash(self) -> str:
        return self._global_norm.normalization_hash + self._spectral_norm.normalization_hash

    def extract_spectral_features(self, graph: Any) -> list[float]:
        """Extract 3*k spectral features from a graph.

        Returns eigenvalues, eigenvector means, and eigenvector stds.
        """
        if hasattr(graph, "num_nodes"):
            G = self._graphbuffers_to_nx(graph)
        else:
            G = graph

        n = G.number_of_nodes()
        if n < 2 or G.number_of_edges() == 0:
            return [0.0] * (3 * self.k)

        try:
            # Normalized Laplacian.
            L = nx.normalized_laplacian_matrix(G).toarray().astype(float)
            eigvals, eigvecs = np.linalg.eigh(L)

            # Take first k eigenvalues (skip the first which is always 0 for connected).
            k_actual = min(self.k, len(eigvals) - 1)
            vals = [float(eigvals[i + 1]) for i in range(k_actual)] if k_actual > 0 else []
            vals.extend([0.0] * (self.k - len(vals)))

            # Eigenvector statistics (skip first eigenvector).
            means = []
            stds = []
            for i in range(min(self.k, len(eigvals) - 1)):
                vec = eigvecs[:, i + 1]
                means.append(float(np.mean(vec)))
                stds.append(float(np.std(vec)))
            means.extend([0.0] * (self.k - len(means)))
            stds.extend([0.0] * (self.k - len(stds)))

            return vals + means + stds
        except Exception:
            return [0.0] * (3 * self.k)

    @staticmethod
    def _graphbuffers_to_nx(graph: Any) -> nx.Graph:
        n = int(graph.num_nodes)
        valid = graph.valid.bool()
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(graph.src.shape[0]):
            if valid[i]:
                G.add_edge(int(graph.src[i].item()), int(graph.dst[i].item()))
        return G

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
        # Spectral features from local_features (first 3*k).
        spectral_feats = list(local_features[:self._spectral_dim]) if len(local_features) >= self._spectral_dim else [0.0] * self._spectral_dim
        normed, mask = self._spectral_norm.transform(spectral_feats)
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
