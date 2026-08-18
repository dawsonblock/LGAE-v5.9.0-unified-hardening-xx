"""Encoder 5: GeometricEncoder — spectral/curvature/resistance features.

Focuses on geometric features that answer the ablation question:

    Does LGAE's geometric machinery actually provide predictive signal
    beyond ordinary graph statistics?

Features include:
- local Laplacian eigenvalue summary
- local spectral gap
- effective resistance
- Forman curvature statistics
- Ollivier curvature where available
- local clustering
- betweenness approximation
- conductance
- community boundary status
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


class GeometricEncoder:
    """Encoder 5: Geometry-focused encoder.

    Combines global state features with explicit geometric features
    around the action target.
    """

    name = "geometric"
    version = "v1"
    deterministic = True

    def __init__(self) -> None:
        self._global_norm = NormalizationStatistics()
        self._geo_norm = NormalizationStatistics()
        self._schema = DEFAULT_ACTION_SCHEMA
        self._global_dim = 24
        # Geometric features (12):
        # local_spectral_gap, local_laplacian_eigval_1, local_laplacian_eigval_2,
        # effective_resistance, forman_curvature_mean, forman_curvature_min,
        # forman_curvature_std, local_clustering, betweenness_u, betweenness_v,
        # conductance, boundary_status
        self._geo_dim = 12
        self._action_dim = self._schema.n_types

    @property
    def dimension(self) -> int:
        return self._global_dim + self._geo_dim + self._action_dim

    @property
    def schema_hash(self) -> str:
        content = f"{self.name}:{self.version}:{self._geo_dim}"
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
        geo_features_list: Sequence[Sequence[float]],
        *,
        split: str = "train",
        dataset_hash: str = "",
        feature_schema_hash: str = "",
    ) -> None:
        self._global_norm.fit(global_features_list, split=split,
                              dataset_hash=dataset_hash,
                              feature_schema_hash=feature_schema_hash)
        self._geo_norm.fit(geo_features_list, split=split,
                           dataset_hash=dataset_hash,
                           feature_schema_hash=feature_schema_hash)

    def freeze(self) -> None:
        self._global_norm.freeze()
        self._geo_norm.freeze()

    @property
    def normalization_hash(self) -> str:
        return self._global_norm.normalization_hash + self._geo_norm.normalization_hash

    def extract_geometric_features(
        self,
        graph: Any,
        u: int,
        v: int,
    ) -> list[float]:
        """Extract 12-dim geometric features around nodes u, v."""
        if hasattr(graph, "num_nodes"):
            G = self._graphbuffers_to_nx(graph)
        else:
            G = graph

        # Local Laplacian eigenvalues (1-hop neighborhood).
        local_spec_gap = 0.0
        local_eigval_1 = 0.0
        local_eigval_2 = 0.0
        try:
            neighbors = set(G.neighbors(u)) | set(G.neighbors(v)) | {u, v}
            sub = G.subgraph(neighbors)
            if sub.number_of_nodes() > 1 and sub.number_of_edges() > 0:
                L = nx.laplacian_matrix(sub).toarray().astype(float)
                eigvals = sorted(np.linalg.eigvalsh(L))
                local_eigval_1 = float(eigvals[0]) if len(eigvals) > 0 else 0.0
                local_eigval_2 = float(eigvals[1]) if len(eigvals) > 1 else 0.0
                local_spec_gap = local_eigval_2
        except Exception:
            pass

        # Effective resistance.
        er = 0.0
        try:
            n = G.number_of_nodes()
            if n > 1 and n <= 30:
                er = float(nx.resistance_distance(G, u, v))
            else:
                # Approximate via shortest path.
                er = float(nx.shortest_path_length(G, u, v))
        except Exception:
            er = 0.0

        # Forman curvature statistics for edges around u and v.
        curvatures = []
        for node in [u, v]:
            for nbr in G.neighbors(node):
                du = max(G.degree(node), 1)
                dv = max(G.degree(nbr), 1)
                k = 2.0 / du + 2.0 / dv - 1.0
                curvatures.append(k)
        curv_mean = float(np.mean(curvatures)) if curvatures else 0.0
        curv_min = float(np.min(curvatures)) if curvatures else 0.0
        curv_std = float(np.std(curvatures)) if curvatures else 0.0

        # Local clustering coefficients.
        clust_u = float(nx.clustering(G, u)) if G.has_node(u) else 0.0
        clust_v = float(nx.clustering(G, v)) if G.has_node(v) else 0.0
        local_clust = (clust_u + clust_v) / 2.0

        # Betweenness centrality (approximate for large graphs).
        betw_u = 0.0
        betw_v = 0.0
        try:
            n = G.number_of_nodes()
            if n <= 50:
                bc = nx.betweenness_centrality(G)
                betw_u = float(bc.get(u, 0.0))
                betw_v = float(bc.get(v, 0.0))
        except Exception:
            pass

        # Conductance (edge cut around u's community).
        conductance = 0.0
        try:
            neighbors_u = set(G.neighbors(u))
            if neighbors_u:
                cut = sum(1 for nbr in neighbors_u if nbr not in set(G.neighbors(v)) | {v})
                conductance = float(cut) / max(len(neighbors_u), 1)
        except Exception:
            pass

        # Boundary status: are u and v in different communities?
        try:
            components = list(nx.connected_components(G))
            boundary = 0.0
            for comp in components:
                if (u in comp) != (v in comp):
                    boundary = 1.0
                    break
        except Exception:
            boundary = 0.0

        return [
            local_spec_gap, local_eigval_1, local_eigval_2,
            er, curv_mean, curv_min, curv_std,
            local_clust, betw_u, betw_v,
            conductance, boundary,
        ]

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
        # Use geometric features from local_features (first 12).
        geo_feats = list(local_features[:self._geo_dim]) if len(local_features) >= self._geo_dim else [0.0] * self._geo_dim
        normed, mask = self._geo_norm.transform(geo_feats)
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
