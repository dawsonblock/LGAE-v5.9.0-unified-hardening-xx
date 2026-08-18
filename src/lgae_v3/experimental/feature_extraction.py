"""Canonical structural feature extraction for v6.0-exp2.

Provides a stable feature schema so that exp3 (state encoder) can build on
top of the dataset without redesigning it.

Two types of features:

1. **Global structural features**: describe the entire graph state.
   - node count, edge count, density, degree moments, component count,
     spectral gap, effective resistance summary, curvature statistics,
     community structure, fiber/gauge summaries, OOD score, diagnosis
     indicators, recent mutation history.

2. **Local action features**: describe the neighborhood around a candidate
   action's target (u, v).
   - source degree, target degree, shortest-path distance, local curvature,
     community membership, bridge likelihood, effective resistance,
     predicted connectivity effect.

These features are deterministic, serializable, and do not depend on
PYTHONHASHSEED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import hashlib

import torch
import numpy as np
import networkx as nx

from ..types import GraphBuffers
from ..operators import spectral_gap_graphbuffers


# ---------------------------------------------------------------------------
# Global structural features
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GlobalStructuralFeatures:
    """Canonical global structural feature vector.

    24-dimensional vector capturing the full graph state summary.
    This is a stable schema — future encoders can rely on these fields
    being present in this order.
    """
    # Basic topology (6).
    log_n_nodes: float
    log_n_edges: float
    density: float
    degree_mean: float
    degree_std: float
    degree_max: float

    # Spectral (3).
    spectral_gap: float
    log_spectral_gap: float
    spectral_gap_normalized: float

    # Connectivity (3).
    n_components: float
    avg_clustering: float
    diameter_estimate: float

    # Curvature proxy (3).
    forman_curvature_mean: float
    forman_curvature_min: float
    forman_curvature_std: float

    # Effective resistance (2).
    effective_resistance_mean: float
    effective_resistance_max: float

    # Fiber/gauge (3).
    fiber_count: float
    fiber_width: float
    gauge_dim: float

    # Diagnosis (2).
    oversquashing_score: float
    bottleneck_score: float

    # History (2).
    recent_mutation_count: float
    recent_success_rate: float

    @property
    def vector(self) -> list[float]:
        """Return as a flat list in canonical order."""
        return [
            self.log_n_nodes, self.log_n_edges, self.density,
            self.degree_mean, self.degree_std, self.degree_max,
            self.spectral_gap, self.log_spectral_gap, self.spectral_gap_normalized,
            self.n_components, self.avg_clustering, self.diameter_estimate,
            self.forman_curvature_mean, self.forman_curvature_min, self.forman_curvature_std,
            self.effective_resistance_mean, self.effective_resistance_max,
            self.fiber_count, self.fiber_width, self.gauge_dim,
            self.oversquashing_score, self.bottleneck_score,
            self.recent_mutation_count, self.recent_success_rate,
        ]

    @property
    def dim(self) -> int:
        return 24

    def to_log(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "vector": self.vector,
            "field_names": [
                "log_n_nodes", "log_n_edges", "density",
                "degree_mean", "degree_std", "degree_max",
                "spectral_gap", "log_spectral_gap", "spectral_gap_normalized",
                "n_components", "avg_clustering", "diameter_estimate",
                "forman_curvature_mean", "forman_curvature_min", "forman_curvature_std",
                "effective_resistance_mean", "effective_resistance_max",
                "fiber_count", "fiber_width", "gauge_dim",
                "oversquashing_score", "bottleneck_score",
                "recent_mutation_count", "recent_success_rate",
            ],
        }


def extract_global_features(
    graph: GraphBuffers,
    *,
    fiber_count: int = 1,
    fiber_width: int = 2,
    gauge_dim: int = 0,
    oversquashing_score: float = 0.0,
    bottleneck_score: float = 0.0,
    recent_mutation_count: int = 0,
    recent_success_rate: float = 0.0,
) -> GlobalStructuralFeatures:
    """Extract canonical global structural features from a GraphBuffers.

    Args:
        graph: The graph to extract features from.
        fiber_count: Number of active fibers.
        fiber_width: Width of fiber latent space.
        gauge_dim: Gauge connection dimension.
        oversquashing_score: Structural diagnosis oversquashing score.
        bottleneck_score: Structural diagnosis bottleneck score.
        recent_mutation_count: Number of recent mutations (history).
        recent_success_rate: Fraction of recent mutations that succeeded.

    Returns:
        GlobalStructuralFeatures with 24-dimensional feature vector.
    """
    n = int(graph.num_nodes)
    valid = graph.valid.bool()
    n_edges = int(valid.sum().item())

    # Degree statistics.
    degrees = [0] * n
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n:
                degrees[s] += 1
            if d < n:
                degrees[d] += 1
    deg_mean = float(np.mean(degrees)) if degrees else 0.0
    deg_std = float(np.std(degrees)) if degrees else 0.0
    deg_max = float(max(degrees)) if degrees else 0.0

    # Density.
    max_edges = n * (n - 1) / 2
    density = float(n_edges) / max(max_edges, 1.0)

    # Spectral gap.
    try:
        lam, _ = spectral_gap_graphbuffers(graph)
        spec_gap = float(lam)
    except Exception:
        spec_gap = 0.0
    log_spec_gap = math.log(max(spec_gap, 1e-10))
    spec_gap_norm = spec_gap / max(n, 1)

    # Build NetworkX for complex metrics.
    edges = []
    for i in range(graph.src.shape[0]):
        if valid[i]:
            edges.append((int(graph.src[i].item()), int(graph.dst[i].item())))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    # Connectivity.
    n_comp = nx.number_connected_components(G) if n > 0 else 0
    avg_clust = float(nx.average_clustering(G)) if n > 2 else 0.0
    # Diameter estimate (use largest component, bounded for speed).
    diam_est = 0.0
    if n > 0 and n_comp > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        sub = G.subgraph(largest_cc)
        if len(sub) > 1:
            try:
                diam_est = float(nx.diameter(sub))
            except Exception:
                diam_est = float(len(largest_cc) - 1)

    # Forman curvature (simplified).
    curvatures = []
    for u, v in edges:
        du = max(degrees[u], 1) if u < n else 1
        dv = max(degrees[v], 1) if v < n else 1
        # Forman curvature: 4/|e| - deg(u) - deg(v) simplified.
        # Using the edge-based formula: 2/du + 2/dv - 1.
        k = 2.0 / du + 2.0 / dv - 1.0
        curvatures.append(k)
    curv_mean = float(np.mean(curvatures)) if curvatures else 0.0
    curv_min = float(np.min(curvatures)) if curvatures else 0.0
    curv_std = float(np.std(curvatures)) if curvatures else 0.0

    # Effective resistance (simplified: use resistance distance on small graphs).
    er_mean = 0.0
    er_max = 0.0
    if n > 1 and n <= 30 and n_comp == 1:
        try:
            # Sample a few pairs for speed.
            resistances = []
            nodes = list(G.nodes())
            n_samples = min(20, len(edges))
            rng = np.random.RandomState(42)  # deterministic
            for _ in range(n_samples):
                u, v = rng.choice(nodes, 2, replace=False)
                try:
                    r = nx.resistance_distance(G, int(u), int(v))
                    resistances.append(r)
                except Exception:
                    pass
            if resistances:
                er_mean = float(np.mean(resistances))
                er_max = float(np.max(resistances))
        except Exception:
            pass

    return GlobalStructuralFeatures(
        log_n_nodes=math.log(max(n, 1)),
        log_n_edges=math.log(max(n_edges, 1)),
        density=density,
        degree_mean=deg_mean,
        degree_std=deg_std,
        degree_max=deg_max,
        spectral_gap=spec_gap,
        log_spectral_gap=log_spec_gap,
        spectral_gap_normalized=spec_gap_norm,
        n_components=float(n_comp),
        avg_clustering=avg_clust,
        diameter_estimate=diam_est,
        forman_curvature_mean=curv_mean,
        forman_curvature_min=curv_min,
        forman_curvature_std=curv_std,
        effective_resistance_mean=er_mean,
        effective_resistance_max=er_max,
        fiber_count=float(fiber_count),
        fiber_width=float(fiber_width),
        gauge_dim=float(gauge_dim),
        oversquashing_score=float(oversquashing_score),
        bottleneck_score=float(bottleneck_score),
        recent_mutation_count=float(recent_mutation_count),
        recent_success_rate=float(recent_success_rate),
    )


# ---------------------------------------------------------------------------
# Local action features
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LocalActionFeatures:
    """Canonical local action feature vector.

    12-dimensional vector capturing the neighborhood around a candidate
    action's target (u, v).
    """
    source_degree: float
    target_degree: float
    shortest_path_distance: float
    local_curvature: float
    same_community: float
    bridge_likelihood: float
    effective_resistance: float
    predicted_connectivity_effect: float
    source_degree_centrality: float
    target_degree_centrality: float
    common_neighbors: float
    jaccard_coefficient: float

    @property
    def vector(self) -> list[float]:
        return [
            self.source_degree, self.target_degree,
            self.shortest_path_distance, self.local_curvature,
            self.same_community, self.bridge_likelihood,
            self.effective_resistance, self.predicted_connectivity_effect,
            self.source_degree_centrality, self.target_degree_centrality,
            self.common_neighbors, self.jaccard_coefficient,
        ]

    @property
    def dim(self) -> int:
        return 12

    def to_log(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "vector": self.vector,
            "field_names": [
                "source_degree", "target_degree",
                "shortest_path_distance", "local_curvature",
                "same_community", "bridge_likelihood",
                "effective_resistance", "predicted_connectivity_effect",
                "source_degree_centrality", "target_degree_centrality",
                "common_neighbors", "jaccard_coefficient",
            ],
        }


def extract_local_action_features(
    graph: GraphBuffers,
    u: int,
    v: int,
) -> LocalActionFeatures:
    """Extract canonical local action features for a candidate edge (u, v).

    Args:
        graph: The current graph state.
        u: Source node of the candidate action.
        v: Target node of the candidate action.

    Returns:
        LocalActionFeatures with 12-dimensional feature vector.
    """
    n = int(graph.num_nodes)
    valid = graph.valid.bool()
    edges = []
    for i in range(graph.src.shape[0]):
        if valid[i]:
            edges.append((int(graph.src[i].item()), int(graph.dst[i].item())))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    degrees = dict(G.degree())
    deg_u = float(degrees.get(u, 0))
    deg_v = float(degrees.get(v, 0))

    # Shortest path distance.
    try:
        dist = float(nx.shortest_path_length(G, u, v))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist = float(n)  # disconnected → high distance

    # Local curvature (Forman for edge (u,v) if it exists, else proxy).
    du = max(int(deg_u), 1)
    dv = max(int(deg_v), 1)
    local_curv = 2.0 / du + 2.0 / dv - 1.0

    # Community membership (using connected components as proxy).
    try:
        components = {frozenset(c) for c in nx.connected_components(G)}
        same_comm = 0.0
        for comp in components:
            if u in comp and v in comp:
                same_comm = 1.0
                break
    except Exception:
        same_comm = 0.0

    # Bridge likelihood: is (u,v) a bridge if added/removed?
    bridge_lik = 0.0
    if G.has_edge(u, v):
        try:
            bridge_lik = 1.0 if nx.has_bridges(G) and (u, v) in nx.bridges(G) else 0.0
        except Exception:
            bridge_lik = 0.0
    else:
        # If adding (u,v) would connect components, it's bridge-like.
        try:
            if not nx.has_path(G, u, v):
                bridge_lik = 1.0
        except Exception:
            bridge_lik = 0.0

    # Effective resistance.
    er = 0.0
    try:
        if n > 1 and n <= 30:
            er = float(nx.resistance_distance(G, u, v))
    except Exception:
        er = float(dist)  # fallback to distance

    # Predicted connectivity effect: how much would adding (u,v) change λ₂?
    # Proxy: inverse of distance (closer nodes → smaller effect).
    pred_conn = 1.0 / max(dist, 1.0)

    # Degree centrality.
    total = max(n - 1, 1)
    dc_u = deg_u / total
    dc_v = deg_v / total

    # Common neighbors.
    common = float(len(set(G.neighbors(u)) & set(G.neighbors(v)))) if G.has_node(u) and G.has_node(v) else 0.0

    # Jaccard coefficient.
    neighbors_u = set(G.neighbors(u)) if G.has_node(u) else set()
    neighbors_v = set(G.neighbors(v)) if G.has_node(v) else set()
    union = neighbors_u | neighbors_v
    jac = float(len(neighbors_u & neighbors_v)) / max(len(union), 1)

    return LocalActionFeatures(
        source_degree=deg_u,
        target_degree=deg_v,
        shortest_path_distance=dist,
        local_curvature=local_curv,
        same_community=same_comm,
        bridge_likelihood=bridge_lik,
        effective_resistance=er,
        predicted_connectivity_effect=pred_conn,
        source_degree_centrality=dc_u,
        target_degree_centrality=dc_v,
        common_neighbors=common,
        jaccard_coefficient=jac,
    )


# ---------------------------------------------------------------------------
# Combined feature vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StructuralFeatureVector:
    """Combined global + local action feature vector."""
    global_features: GlobalStructuralFeatures
    local_features: LocalActionFeatures | None

    @property
    def vector(self) -> list[float]:
        v = list(self.global_features.vector)
        if self.local_features is not None:
            v.extend(self.local_features.vector)
        return v

    @property
    def dim(self) -> int:
        d = self.global_features.dim
        if self.local_features is not None:
            d += self.local_features.dim
        return d

    def to_log(self) -> dict[str, Any]:
        return {
            "global": self.global_features.to_log(),
            "local": self.local_features.to_log() if self.local_features else None,
            "dim": self.dim,
            "vector": self.vector,
        }
