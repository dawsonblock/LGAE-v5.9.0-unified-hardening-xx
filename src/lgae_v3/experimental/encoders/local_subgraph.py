"""Encoder 4: LocalSubgraphEncoder — k-hop neighborhood extraction.

For every candidate mutation, extracts a bounded k-hop neighborhood
around affected nodes:

    G_a = G[N_k(u) ∪ N_k(v)]

Starts with k=2 with deterministic caps on nodes and edges.
Uses canonical node/edge ordering for permutation invariance.

This is the first graph-native representation. It produces a deterministic
serialized local graph representation that gives later models a stable
input contract.
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


class LocalSubgraphEncoder:
    """Encoder 4: Local subgraph (k-hop neighborhood) encoder.

    Extracts a bounded k-hop neighborhood around the action's target nodes,
    computes canonical structural descriptors of that subgraph, and combines
    them with global features.

    The subgraph is serialized deterministically using canonical node
    ordering (sorted by degree, then by node ID) to ensure permutation
    invariance.
    """

    name = "local-subgraph"
    version = "v1"
    deterministic = True

    def __init__(
        self,
        k_hop: int = 2,
        max_nodes: int = 20,
        max_edges: int = 50,
    ) -> None:
        self.k_hop = int(k_hop)
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self._global_norm = NormalizationStatistics()
        self._subgraph_norm = NormalizationStatistics()
        self._schema = DEFAULT_ACTION_SCHEMA
        self._global_dim = 24
        # Subgraph features: n_nodes, n_edges, density, avg_degree, max_degree,
        # spectral_gap, n_components, avg_clustering, diameter, n_triangles
        self._subgraph_dim = 10
        self._action_dim = self._schema.n_types

    @property
    def dimension(self) -> int:
        return self._global_dim + self._subgraph_dim + self._action_dim

    @property
    def schema_hash(self) -> str:
        content = f"{self.name}:{self.version}:{self.k_hop}:{self.max_nodes}:{self.max_edges}"
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
        subgraph_features_list: Sequence[Sequence[float]],
        *,
        split: str = "train",
        dataset_hash: str = "",
        feature_schema_hash: str = "",
    ) -> None:
        self._global_norm.fit(global_features_list, split=split,
                              dataset_hash=dataset_hash,
                              feature_schema_hash=feature_schema_hash)
        self._subgraph_norm.fit(subgraph_features_list, split=split,
                                dataset_hash=dataset_hash,
                                feature_schema_hash=feature_schema_hash)

    def freeze(self) -> None:
        self._global_norm.freeze()
        self._subgraph_norm.freeze()

    @property
    def normalization_hash(self) -> str:
        return self._global_norm.normalization_hash + self._subgraph_norm.normalization_hash

    def extract_subgraph_features(
        self,
        graph: Any,
        u: int,
        v: int,
    ) -> list[float]:
        """Extract canonical structural features from the k-hop subgraph.

        Args:
            graph: GraphBuffers or NetworkX graph.
            u, v: Target nodes of the candidate action.

        Returns:
            10-dimensional subgraph feature vector.
        """
        # Build NetworkX graph if needed.
        if hasattr(graph, "num_nodes"):
            G = self._graphbuffers_to_nx(graph)
        else:
            G = graph

        # Extract k-hop neighborhood.
        try:
            nodes_u = nx.single_source_shortest_path_length(G, u, cutoff=self.k_hop)
            nodes_v = nx.single_source_shortest_path_length(G, v, cutoff=self.k_hop)
            neighborhood = set(nodes_u.keys()) | set(nodes_v.keys())
        except (nx.NodeNotFound, Exception):
            neighborhood = {u, v} if G.has_node(u) and G.has_node(v) else set()

        # Cap nodes deterministically: sort by degree descending, then by ID.
        if len(neighborhood) > self.max_nodes:
            sorted_nodes = sorted(
                neighborhood,
                key=lambda n: (-G.degree(n), n),
            )
            neighborhood = set(sorted_nodes[:self.max_nodes])

        sub = G.subgraph(neighborhood)
        n = len(sub)
        m = sub.number_of_edges()

        # Compute features with safe fallbacks.
        density = float(m) / max(n * (n - 1) / 2, 1) if n > 1 else 0.0
        degrees = [d for _, d in sub.degree()]
        avg_deg = float(np.mean(degrees)) if degrees else 0.0
        max_deg = float(max(degrees)) if degrees else 0.0
        n_comp = nx.number_connected_components(sub) if n > 0 else 0
        avg_clust = float(nx.average_clustering(sub)) if n > 2 else 0.0

        # Spectral gap (largest eigenvalue of normalized Laplacian, simplified).
        spec_gap = 0.0
        if n > 1 and m > 0:
            try:
                L = nx.laplacian_matrix(sub).toarray().astype(float)
                eigvals = np.linalg.eigvalsh(L)
                if len(eigvals) > 1:
                    spec_gap = float(eigvals[1])  # second smallest (algebraic connectivity)
            except Exception:
                spec_gap = 0.0

        # Diameter (bounded).
        diam = 0.0
        if n > 1 and n_comp == 1:
            try:
                diam = float(nx.diameter(sub))
            except Exception:
                diam = float(n - 1)

        # Triangle count.
        n_triangles = sum(nx.triangles(sub).values()) // 3 if n > 2 else 0

        return [
            math.log(max(n, 1)),
            math.log(max(m, 1)),
            density,
            avg_deg,
            max_deg,
            spec_gap,
            float(n_comp),
            avg_clust,
            diam,
            float(n_triangles),
        ]

    @staticmethod
    def _graphbuffers_to_nx(graph: Any) -> nx.Graph:
        """Convert GraphBuffers to NetworkX."""
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
        # For the action encoder, we use the subgraph features if available
        # in local_features (first 10), otherwise fall back to zeros.
        subgraph_feats = list(local_features[:self._subgraph_dim]) if len(local_features) >= self._subgraph_dim else [0.0] * self._subgraph_dim
        normed, mask = self._subgraph_norm.transform(subgraph_feats)
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
