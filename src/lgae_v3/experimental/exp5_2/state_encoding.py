"""Normalized and topology-invariant state encoding for exp5.2.

Key principles:
1. Scale-invariant features: density, normalized degree, spectral gap per node
2. Delta-state target: predict Δz = z_{t+1} - z_t, not z_{t+1} directly
3. Topology-invariant descriptors: graphlet frequencies, degree entropy,
   modularity, transitivity — these transfer across graph families
4. No raw absolute counts (n_nodes, n_edges) that enable family identification

The encoding is FROZEN once training begins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import json
import math
import numpy as np


# ---------------------------------------------------------------------------
# Dimensions (frozen for exp5.2).
# ---------------------------------------------------------------------------

# Normalized state features (no raw counts).
# 1. density
# 2. normalized_degree_mean = 2|E| / |V|
# 3. normalized_degree_std
# 4. spectral_gap_normalized = spectral_gap / max(degree_mean, 1)
# 5. spectral_gap_per_node
# 6. n_components_normalized = n_components / |V|
# 7. avg_clustering
# 8. log_density
# 9. log_spectral_gap
# 10. degree_entropy
# 11. transitivity (if available, else 0)
# 12. modularity_proxy
# 13. diameter_proxy
# 14. avg_path_length_proxy
# 15. triangle_count_normalized
# 16. wedge_count_normalized
# 17. 4_cycle_count_normalized
# 18. 3_star_count_normalized
# 19. assortativity_proxy
# 20. fiber_count_normalized
NORM_STATE_DIM = 20

# Action encoding (same as exp5 but with normalized local features).
# 6 mutation types + 8 local features = 14.
NORM_ACTION_DIM = 14


# ---------------------------------------------------------------------------
# State encoding.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NormalizedStateVector:
    """Encoded normalized structural state vector."""
    vector: np.ndarray
    dim: int = NORM_STATE_DIM
    schema_hash: str = ""

    def __post_init__(self) -> None:
        if not self.schema_hash:
            object.__setattr__(self, "schema_hash", _norm_state_schema_hash())

    def to_log(self) -> dict[str, Any]:
        return {
            "dim": int(self.dim),
            "schema_hash": self.schema_hash,
            "vector": [float(x) for x in self.vector],
        }


def _norm_state_schema_hash() -> str:
    content = json.dumps({
        "dim": NORM_STATE_DIM,
        "fields": [
            "density", "norm_degree_mean", "norm_degree_std",
            "spectral_gap_normalized", "spectral_gap_per_node",
            "n_components_normalized", "avg_clustering",
            "log_density", "log_spectral_gap",
            "degree_entropy", "transitivity", "modularity_proxy",
            "diameter_proxy", "avg_path_length_proxy",
            "triangle_count_norm", "wedge_count_norm",
            "four_cycle_count_norm", "three_star_count_norm",
            "assortativity_proxy", "fiber_count_normalized",
        ],
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _compute_degree_entropy(degrees: list[int]) -> float:
    """Shannon entropy of the degree distribution."""
    if not degrees:
        return 0.0
    n = len(degrees)
    # Count degree frequencies.
    from collections import Counter
    counts = Counter(degrees)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _count_triangles(adj: dict[int, set[int]], n: int) -> int:
    """Count triangles using adjacency sets."""
    count = 0
    for u in range(min(n, len(adj))):
        for v in adj.get(u, set()):
            if v > u:
                for w in adj.get(u, set()):
                    if w > v and w in adj.get(v, set()):
                        count += 1
    return count


def _count_wedges(adj: dict[int, set[int]], n: int) -> int:
    """Count wedges (2-paths)."""
    count = 0
    for u in range(min(n, len(adj))):
        d = len(adj.get(u, set()))
        count += d * (d - 1) // 2
    return count


def _count_3_stars(adj: dict[int, set[int]], n: int) -> int:
    """Count 3-stars (K_{1,3})."""
    count = 0
    for u in range(min(n, len(adj))):
        d = len(adj.get(u, set()))
        if d >= 3:
            count += d * (d - 1) * (d - 2) // 6
    return count


def _count_4_cycles(adj: dict[int, set[int]], n: int) -> int:
    """Approximate 4-cycle count."""
    count = 0
    nodes = list(range(min(n, len(adj))))
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            common = len(adj.get(u, set()) & adj.get(v, set()))
            if common >= 2:
                count += common * (common - 1) // 2
    return count


def _assortativity_proxy(degrees: list[int], adj: dict[int, set[int]]) -> float:
    """Degree assortativity proxy (Pearson correlation of degrees at edge endpoints)."""
    src_degs = []
    dst_degs = []
    for u in adj:
        for v in adj[u]:
            if v > u:
                src_degs.append(degrees[u] if u < len(degrees) else 0)
                dst_degs.append(degrees[v] if v < len(degrees) else 0)
    if len(src_degs) < 2:
        return 0.0
    src_arr = np.array(src_degs, dtype=float)
    dst_arr = np.array(dst_degs, dtype=float)
    if np.std(src_arr) < 1e-10 or np.std(dst_arr) < 1e-10:
        return 0.0
    return float(np.corrcoef(src_arr, dst_arr)[0, 1])


def encode_normalized_state(
    state: Any,
    *,
    graph: Any = None,
) -> NormalizedStateVector:
    """Encode a StructuralStateSummary into a normalized, topology-invariant vector.

    Args:
        state: A StructuralStateSummary with n_nodes, n_edges, density, etc.
            The `extra` dict may contain graphlet features.
        graph: Optional GraphBuffers for computing graphlet frequencies.
            If not provided, graphlet counts come from the state's extra dict.

    Returns:
        NormalizedStateVector of dimension NORM_STATE_DIM.
    """
    n_nodes = float(getattr(state, "n_nodes", 10))
    n_edges = float(getattr(state, "n_edges", 9))
    density = float(getattr(state, "density", 0.0))
    spectral_gap = float(getattr(state, "spectral_gap", 0.0))
    degree_mean = float(getattr(state, "degree_mean", 0.0))
    degree_std = float(getattr(state, "degree_std", 0.0))
    n_components = float(getattr(state, "n_components", 1))
    avg_clustering = float(getattr(state, "avg_clustering", 0.0))
    fiber_count = float(getattr(state, "fiber_count", 0))

    # Normalized features (scale-invariant).
    norm_degree_mean = (2.0 * n_edges) / max(n_nodes, 1.0)
    norm_degree_std = degree_std / max(n_nodes - 1, 1.0)
    spectral_gap_normalized = spectral_gap / max(degree_mean, 1.0)
    spectral_gap_per_node = spectral_gap / max(n_nodes, 1.0)
    n_components_normalized = n_components / max(n_nodes, 1.0)
    log_density = math.log1p(max(density, 0.0))
    log_spectral_gap = math.log1p(max(abs(spectral_gap), 1e-10))
    fiber_count_normalized = fiber_count / max(n_nodes, 1.0)

    # Graphlet features from extra dict (populated by dataset generator).
    extra = getattr(state, "extra", {}) or {}

    degree_entropy = float(extra.get("degree_entropy", 0.0))
    triangle_count_norm = float(extra.get("triangle_count_norm", 0.0))
    wedge_count_norm = float(extra.get("wedge_count_norm", 0.0))
    four_cycle_count_norm = float(extra.get("four_cycle_count_norm", 0.0))
    three_star_count_norm = float(extra.get("three_star_count_norm", 0.0))
    assortativity_proxy = float(extra.get("assortativity", 0.0))
    transitivity = float(extra.get("transitivity", 0.0))

    # If graph is provided, compute directly (overrides extra).
    if graph is not None:
        n = int(graph.num_nodes)
        valid = graph.valid.bool()
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        degrees = [0] * n
        for i in range(graph.src.shape[0]):
            if valid[i]:
                s = int(graph.src[i].item())
                d = int(graph.dst[i].item())
                if s < n and d < n:
                    adj[s].add(d)
                    adj[d].add(s)
                    degrees[s] += 1
                    degrees[d] += 1

        degree_entropy = _compute_degree_entropy(degrees)
        n3 = max(n ** 3, 1)
        triangle_count = _count_triangles(adj, n)
        wedge_count = _count_wedges(adj, n)
        three_star_count = _count_3_stars(adj, n)
        four_cycle_count = _count_4_cycles(adj, n)
        triangle_count_norm = triangle_count / n3
        wedge_count_norm = wedge_count / n3
        four_cycle_count_norm = four_cycle_count / n3
        three_star_count_norm = three_star_count / n3
        assortativity_proxy = _assortativity_proxy(degrees, adj)
        transitivity = 3.0 * triangle_count / max(wedge_count, 1)

    # Modularity proxy.
    modularity_proxy = max(0.0, 1.0 - avg_clustering)

    # Diameter and path length proxies.
    if spectral_gap > 1e-10:
        diameter_proxy = 1.0 / spectral_gap
        avg_path_length_proxy = 1.0 / (2.0 * spectral_gap)
    else:
        diameter_proxy = float(n_nodes)
        avg_path_length_proxy = float(n_nodes) / 2.0
    diameter_proxy /= max(n_nodes, 1)
    avg_path_length_proxy /= max(n_nodes, 1)

    vec = np.array([
        density,
        norm_degree_mean,
        norm_degree_std,
        spectral_gap_normalized,
        spectral_gap_per_node,
        n_components_normalized,
        avg_clustering,
        log_density,
        log_spectral_gap,
        degree_entropy,
        transitivity,
        modularity_proxy,
        diameter_proxy,
        avg_path_length_proxy,
        triangle_count_norm,
        wedge_count_norm,
        four_cycle_count_norm,
        three_star_count_norm,
        assortativity_proxy,
        fiber_count_normalized,
    ], dtype=np.float64)

    return NormalizedStateVector(vector=vec)


# ---------------------------------------------------------------------------
# Action encoding (normalized local features).
# ---------------------------------------------------------------------------

MUTATION_TYPES = (
    "ADD_EDGE",
    "REMOVE_EDGE",
    "REWIRE",
    "ADD_FIBER",
    "REMOVE_FIBER",
    "GAUGE_TRANSFORM",
)
N_MUTATION_TYPES = len(MUTATION_TYPES)


@dataclass(frozen=True, slots=True)
class NormalizedActionVector:
    """Encoded action vector with normalized local features."""
    vector: np.ndarray
    dim: int = NORM_ACTION_DIM
    action_type: str = ""
    schema_hash: str = ""

    def __post_init__(self) -> None:
        if not self.schema_hash:
            object.__setattr__(self, "schema_hash", _norm_action_schema_hash())

    def to_log(self) -> dict[str, Any]:
        return {
            "dim": int(self.dim),
            "action_type": self.action_type,
            "schema_hash": self.schema_hash,
            "vector": [float(x) for x in self.vector],
        }


def _norm_action_schema_hash() -> str:
    content = json.dumps({
        "dim": NORM_ACTION_DIM,
        "mutation_types": list(MUTATION_TYPES),
        "target_features": [
            "u_normalized", "v_normalized",
            "u_degree_normalized", "v_degree_normalized",
            "common_neighbor_ratio", "jaccard_overlap",
            "local_clustering", "same_component",
        ],
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def encode_normalized_action(
    action_type: str,
    action_target: dict[str, Any],
    *,
    n_nodes: int = 20,
    degree_mean: float = 2.0,
    graph: Any = None,
) -> NormalizedActionVector:
    """Encode an action with normalized local features.

    Local features are topology-invariant:
    - Normalized node indices
    - Normalized degrees
    - Common-neighbor ratio
    - Jaccard neighborhood overlap
    - Local clustering coefficient
    """
    # One-hot encode mutation type.
    one_hot = np.zeros(N_MUTATION_TYPES, dtype=np.float64)
    # Normalize action type string.
    at = action_type.upper().replace("ADD_EDGE", "ADD_EDGE").replace("REMOVE_EDGE", "REMOVE_EDGE")
    if at in MUTATION_TYPES:
        one_hot[MUTATION_TYPES.index(at)] = 1.0
    else:
        one_hot[:] = 1.0 / N_MUTATION_TYPES

    u = float(action_target.get("u", 0)) if isinstance(action_target, dict) else 0.0
    v = float(action_target.get("v", 0)) if isinstance(action_target, dict) else 0.0
    u_norm = u / max(n_nodes, 1)
    v_norm = v / max(n_nodes, 1)

    # Normalized degree proxies.
    u_deg_norm = u_norm * degree_mean / max(n_nodes - 1, 1)
    v_deg_norm = v_norm * degree_mean / max(n_nodes - 1, 1)

    # Topology-invariant local features.
    common_neighbor_ratio = 0.0
    jaccard_overlap = 0.0
    local_clustering = 0.0
    same_component = 1.0

    if graph is not None:
        n = int(graph.num_nodes)
        valid = graph.valid.bool()
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(graph.src.shape[0]):
            if valid[i]:
                s = int(graph.src[i].item())
                d = int(graph.dst[i].item())
                if s < n and d < n:
                    adj[s].add(d)
                    adj[d].add(s)

        u_int, v_int = int(u), int(v)
        if u_int < n and v_int < n:
            nu = adj.get(u_int, set())
            nv = adj.get(v_int, set())
            common = len(nu & nv)
            union = len(nu | nv)
            common_neighbor_ratio = common / max(max(len(nu), len(nv)), 1)
            jaccard_overlap = common / max(union, 1)
            # Local clustering of u.
            du = len(nu)
            if du >= 2:
                links = sum(1 for a in nu for b in nu if a < b and b in adj.get(a, set()))
                local_clustering = 2.0 * links / (du * (du - 1))

    target_feats = np.array([
        u_norm, v_norm, u_deg_norm, v_deg_norm,
        common_neighbor_ratio, jaccard_overlap,
        local_clustering, same_component,
    ], dtype=np.float64)

    vec = np.concatenate([one_hot, target_feats])
    return NormalizedActionVector(vector=vec, action_type=action_type)


def norm_state_action_schema_hash() -> str:
    """Combined schema hash for normalized state + action encoding."""
    content = json.dumps({
        "state_schema": _norm_state_schema_hash(),
        "action_schema": _norm_action_schema_hash(),
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
