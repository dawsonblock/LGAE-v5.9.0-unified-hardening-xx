"""Rich feature extractors for exp6.8.4.

Four feature levels:
  F1_current: exp6.8.3 features (state + action encoding)
  F2_action_effects: + exact action effects (delta edge count, delta degrees, etc.)
  F3_local_topology: + local topology around endpoints (clustering, common neighbors, bridge)
  F4_full: + global structure (spectral sensitivity, component sizes, path stats)

Principle: calculate known mechanics; learn only the residual uncertainty.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from ...types import GraphBuffers
from ..exp6_3.exact_mpc import ActionIdentity, apply_action, apply_action_with_status
from ..exp6_6.objective_spec import ObjectiveSpec, encode_objective, OBJECTIVE_ENCODING_DIM
from ..exp6_8_1.split_state import (
    SplitStructuralState, EXACT_STATE_DIM, CERTIFIED_STATE_DIM, LEARNED_STATE_DIM,
)
from ..exp6_8_1.deterministic_oracles import compute_spectral_gap_deterministic

# Reuse action encoding from exp6.8.3.
from ..exp6_8_3.advantage_features import (
    encode_action, ACTION_TYPE_DIM, ACTION_FEATURE_DIM, PAIRWISE_FEATURE_DIM,
)


# F2: Exact action effect features.
ACTION_EFFECT_DIM = 12  # delta_edge_count, delta_deg_u, delta_deg_v, etc.

# F3: Local topology features (per endpoint pair).
LOCAL_TOPOLOGY_DIM = 10

# F4: Global structure features.
GLOBAL_STRUCTURE_DIM = 8

# Feature level dimensions (cumulative).
# F1: state + objective + pairwise (3 * ACTION_FEATURE_DIM)
F1_DIM = EXACT_STATE_DIM + CERTIFIED_STATE_DIM + LEARNED_STATE_DIM + OBJECTIVE_ENCODING_DIM + PAIRWISE_FEATURE_DIM
# F2: F1 + 3 * action effects (learned, baseline, diff)
F2_DIM = F1_DIM + 3 * ACTION_EFFECT_DIM
# F3: F2 + 3 * local topology (learned, baseline, diff)
F3_DIM = F2_DIM + 3 * LOCAL_TOPOLOGY_DIM
# F4: F3 + global structure
F4_DIM = F3_DIM + GLOBAL_STRUCTURE_DIM


def extract_action_effects(
    graph: GraphBuffers,
    action: tuple,
    n_nodes: int,
) -> np.ndarray:
    """Extract exact action effect features.

    These are cheaply computable consequences of the action:
      - delta edge count
      - delta degree of endpoint u
      - delta degree of endpoint v
      - same component flag (before)
      - component size of u (before)
      - component size of v (before)
      - shortest path distance u-v (before, -1 if disconnected)
      - common neighbors count
      - clustering coefficient around u
      - clustering coefficient around v
      - bridge status (is edge a bridge?)
      - exact immediate delta U (utility change)
    """
    action_type = action[0] if len(action) > 0 else ""
    u = action[1] if len(action) > 1 else 0
    v = action[2] if len(action) > 2 else 0

    # Current graph state.
    adj = _build_adjacency(graph, n_nodes)
    deg = adj.sum(axis=1)

    # Delta edge count.
    if action_type == "add_edge":
        delta_edges = 1.0
        delta_deg_u = 1.0
        delta_deg_v = 1.0
    elif action_type == "remove_edge":
        delta_edges = -1.0
        delta_deg_u = -1.0
        delta_deg_v = -1.0
    elif action_type == "reweight_edge":
        delta_edges = 0.0
        delta_deg_u = 0.0
        delta_deg_v = 0.0
    elif action_type == "edge_swap":
        delta_edges = 0.0
        delta_deg_u = 0.0
        delta_deg_v = 0.0
    else:
        delta_edges = 0.0
        delta_deg_u = 0.0
        delta_deg_v = 0.0

    # Same component check.
    comp_u = _find_component(adj, u, n_nodes)
    comp_v = _find_component(adj, v, n_nodes)
    same_component = 1.0 if comp_u == comp_v else 0.0

    # Component sizes.
    comp_size_u = float(_component_size(adj, u, n_nodes))
    comp_size_v = float(_component_size(adj, v, n_nodes))

    # Shortest path distance (BFS, limited depth).
    dist_uv = _shortest_path(adj, u, v, n_nodes, max_depth=10)

    # Common neighbors.
    common_neighbors = float(np.sum(adj[u] & adj[v])) if u < n_nodes and v < n_nodes else 0.0

    # Clustering coefficients.
    clustering_u = _clustering_coefficient(adj, u, n_nodes)
    clustering_v = _clustering_coefficient(adj, v, n_nodes)

    # Bridge status: is (u,v) a bridge?
    is_bridge = _is_bridge(adj, u, v, n_nodes)

    # Exact immediate delta U is computed externally (needs utility_fn).
    # Placeholder: 0.0, will be filled by caller if available.
    delta_u_immediate = 0.0

    return np.array([
        delta_edges / 5.0,
        delta_deg_u / 10.0,
        delta_deg_v / 10.0,
        same_component,
        comp_size_u / 30.0,
        comp_size_v / 30.0,
        dist_uv / 10.0,
        common_neighbors / 10.0,
        clustering_u,
        clustering_v,
        is_bridge,
        delta_u_immediate,
    ], dtype=np.float32)


def extract_local_topology(
    graph: GraphBuffers,
    action: tuple,
    n_nodes: int,
) -> np.ndarray:
    """Extract local topology features around action endpoints.

      - degree of u (normalized)
      - degree of v (normalized)
      - degree difference
      - max degree in neighborhood
      - min degree in neighborhood
      - neighborhood size (2-hop)
      - edge density in neighborhood
      - articulation point status (u)
      - articulation point status (v)
      - triangle count involving (u,v)
    """
    action_type = action[0] if len(action) > 0 else ""
    u = action[1] if len(action) > 1 else 0
    v = action[2] if len(action) > 2 else 0

    adj = _build_adjacency(graph, n_nodes)
    deg = adj.sum(axis=1)

    deg_u = float(deg[u]) / 30.0 if u < n_nodes else 0.0
    deg_v = float(deg[v]) / 30.0 if v < n_nodes else 0.0
    deg_diff = abs(deg_u - deg_v)

    # Neighborhood degrees.
    neighbors_u = set(np.where(adj[u])[0]) if u < n_nodes else set()
    neighbors_v = set(np.where(adj[v])[0]) if v < n_nodes else set()

    all_neighbors = neighbors_u | neighbors_v
    if all_neighbors:
        neigh_degs = [float(deg[n]) for n in all_neighbors if n < n_nodes]
        max_neigh_deg = max(neigh_degs) / 30.0 if neigh_degs else 0.0
        min_neigh_deg = min(neigh_degs) / 30.0 if neigh_degs else 0.0
    else:
        max_neigh_deg = 0.0
        min_neigh_deg = 0.0

    # 2-hop neighborhood size.
    two_hop = set()
    for n in neighbors_u:
        two_hop.update(np.where(adj[n])[0])
    two_hop = two_hop - {u}
    neigh_2hop_size = float(len(two_hop)) / 50.0

    # Edge density in neighborhood.
    if len(all_neighbors) > 1:
        edges_in_neigh = 0
        neigh_list = list(all_neighbors)
        for i in range(len(neigh_list)):
            for j in range(i+1, len(neigh_list)):
                if adj[neigh_list[i], neigh_list[j]]:
                    edges_in_neigh += 1
        max_possible = len(neigh_list) * (len(neigh_list) - 1) / 2
        density = edges_in_neigh / max_possible if max_possible > 0 else 0.0
    else:
        density = 0.0

    # Articulation points.
    is_artic_u = _is_articulation_point(adj, u, n_nodes)
    is_artic_v = _is_articulation_point(adj, v, n_nodes)

    # Triangle count.
    triangles = float(len(neighbors_u & neighbors_v))

    return np.array([
        deg_u, deg_v, deg_diff,
        max_neigh_deg, min_neigh_deg,
        neigh_2hop_size, density,
        float(is_artic_u), float(is_artic_v),
        triangles / 10.0,
    ], dtype=np.float32)


def extract_global_structure(
    graph: GraphBuffers,
    n_nodes: int,
) -> np.ndarray:
    """Extract global structure features.

      - number of components (normalized)
      - largest component size (normalized)
      - edge density
      - average degree
      - degree variance
      - spectral gap (deterministic)
      - diameter estimate (max shortest path, capped)
      - modularity proxy
    """
    adj = _build_adjacency(graph, n_nodes)
    deg = adj.sum(axis=1)

    # Components.
    visited = np.zeros(n_nodes, dtype=bool)
    n_components = 0
    largest_comp = 0
    for start in range(n_nodes):
        if visited[start]:
            continue
        comp = _bfs_component(adj, start, n_nodes)
        visited[list(comp)] = True
        n_components += 1
        largest_comp = max(largest_comp, len(comp))

    # Edge density.
    n_edges = int(adj.sum() // 2)
    max_edges = n_nodes * (n_nodes - 1) / 2
    edge_density = n_edges / max_edges if max_edges > 0 else 0.0

    # Degree stats.
    avg_deg = float(np.mean(deg)) / 30.0
    deg_var = float(np.var(deg)) / 100.0

    # Spectral gap (deterministic).
    try:
        spectral_gap = compute_spectral_gap_deterministic(graph, n_nodes)
    except Exception:
        spectral_gap = 0.0
    spectral_gap_norm = spectral_gap / 10.0

    # Diameter estimate (sampled BFS).
    diameter = _estimate_diameter(adj, n_nodes, max_samples=10)

    # Modularity proxy: fraction of edges within communities.
    # Simple proxy: 1 - edge_density (sparse graphs have higher modularity potential).
    modularity_proxy = 1.0 - edge_density

    return np.array([
        n_components / 10.0,
        largest_comp / 30.0,
        edge_density,
        avg_deg,
        deg_var,
        spectral_gap_norm,
        diameter / 10.0,
        modularity_proxy,
    ], dtype=np.float32)


def extract_features_level(
    graph: GraphBuffers,
    z: torch.Tensor,
    state: SplitStructuralState,
    objective: ObjectiveSpec,
    baseline_action: tuple,
    learned_action: tuple,
    baseline_id: ActionIdentity,
    learned_id: ActionIdentity,
    level: str = "F4_full",
    utility_fn=None,
) -> np.ndarray:
    """Extract features at the specified level.

    Levels:
      F1_current: state + objective + pairwise action
      F2_action_effects: F1 + exact action effects
      F3_local_topology: F2 + local topology
      F4_full: F3 + global structure
    """
    from ..exp6_8_3.advantage_features import (
        extract_state_features, extract_objective_features,
        extract_pairwise_features,
    )

    n_nodes = graph.num_nodes

    # F1: Base features.
    state_feat = extract_state_features(state)
    obj_feat = extract_objective_features(objective)
    pairwise = extract_pairwise_features(baseline_action, learned_action, baseline_id, learned_id)

    features = np.concatenate([state_feat, obj_feat, pairwise])

    if level == "F1_current":
        return features

    # F2: + Action effects.
    baseline_effects = extract_action_effects(graph, baseline_action, n_nodes)
    learned_effects = extract_action_effects(graph, learned_action, n_nodes)
    effects_diff = learned_effects - baseline_effects
    features = np.concatenate([features, learned_effects, baseline_effects, effects_diff])

    if level == "F2_action_effects":
        return features

    # F3: + Local topology.
    baseline_local = extract_local_topology(graph, baseline_action, n_nodes)
    learned_local = extract_local_topology(graph, learned_action, n_nodes)
    local_diff = learned_local - baseline_local
    features = np.concatenate([features, learned_local, baseline_local, local_diff])

    if level == "F3_local_topology":
        return features

    # F4: + Global structure.
    global_feat = extract_global_structure(graph, n_nodes)
    features = np.concatenate([features, global_feat])

    return features


def get_feature_dim(level: str) -> int:
    """Get feature dimension for a level."""
    if level == "F1_current":
        return F1_DIM
    elif level == "F2_action_effects":
        return F1_DIM + 3 * ACTION_EFFECT_DIM
    elif level == "F3_local_topology":
        return F1_DIM + 3 * ACTION_EFFECT_DIM + 3 * LOCAL_TOPOLOGY_DIM
    elif level == "F4_full":
        return F1_DIM + 3 * ACTION_EFFECT_DIM + 3 * LOCAL_TOPOLOGY_DIM + GLOBAL_STRUCTURE_DIM
    else:
        return F4_DIM


# === Helper functions ===

def _build_adjacency(graph: GraphBuffers, n_nodes: int) -> np.ndarray:
    """Build dense adjacency matrix."""
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    src = graph.src.numpy() if hasattr(graph.src, 'numpy') else np.array(graph.src)
    dst = graph.dst.numpy() if hasattr(graph.dst, 'numpy') else np.array(graph.dst)
    for s, d in zip(src, dst):
        if s < n_nodes and d < n_nodes:
            adj[s, d] = True
            adj[d, s] = True
    return adj


def _find_component(adj: np.ndarray, node: int, n_nodes: int) -> int:
    """Find the component label for a node (returns first node in component)."""
    visited = np.zeros(n_nodes, dtype=bool)
    queue = [node]
    visited[node] = True
    first = node
    while queue:
        curr = queue.pop(0)
        first = min(first, curr)
        for neighbor in np.where(adj[curr])[0]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return first


def _component_size(adj: np.ndarray, node: int, n_nodes: int) -> int:
    """Get the size of the component containing node."""
    visited = np.zeros(n_nodes, dtype=bool)
    queue = [node]
    visited[node] = True
    size = 1
    while queue:
        curr = queue.pop(0)
        for neighbor in np.where(adj[curr])[0]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                size += 1
    return size


def _bfs_component(adj: np.ndarray, start: int, n_nodes: int) -> set:
    """BFS to find all nodes in the same component as start."""
    visited = {start}
    queue = [start]
    while queue:
        curr = queue.pop(0)
        for neighbor in np.where(adj[curr])[0]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def _shortest_path(adj: np.ndarray, u: int, v: int, n_nodes: int, max_depth: int = 10) -> float:
    """BFS shortest path, capped at max_depth. Returns -1 if disconnected."""
    if u == v:
        return 0.0
    if u >= n_nodes or v >= n_nodes:
        return -1.0
    visited = {u}
    queue = [(u, 0)]
    while queue:
        curr, depth = queue.pop(0)
        if depth >= max_depth:
            return float(max_depth)
        for neighbor in np.where(adj[curr])[0]:
            if neighbor == v:
                return float(depth + 1)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return -1.0


def _clustering_coefficient(adj: np.ndarray, node: int, n_nodes: int) -> float:
    """Local clustering coefficient."""
    if node >= n_nodes:
        return 0.0
    neighbors = np.where(adj[node])[0]
    k = len(neighbors)
    if k < 2:
        return 0.0
    links = 0
    for i in range(len(neighbors)):
        for j in range(i+1, len(neighbors)):
            if adj[neighbors[i], neighbors[j]]:
                links += 1
    return 2.0 * links / (k * (k - 1))


def _is_bridge(adj: np.ndarray, u: int, v: int, n_nodes: int) -> float:
    """Check if edge (u,v) is a bridge (removing it disconnects the component)."""
    if u >= n_nodes or v >= n_nodes or not adj[u, v]:
        return 0.0
    # Temporarily remove edge.
    adj[u, v] = False
    adj[v, u] = False
    # Check connectivity.
    comp_size_before = _component_size(adj, u, n_nodes) + 1  # +1 for v
    comp_u = _component_size(adj, u, n_nodes)
    # Restore.
    adj[u, v] = True
    adj[v, u] = True
    # If u's component shrank, it was a bridge.
    return 1.0 if comp_u < _component_size(adj, u, n_nodes) - 1 else 0.0


def _is_articulation_point(adj: np.ndarray, node: int, n_nodes: int) -> float:
    """Check if node is an articulation point (removing it disconnects the graph)."""
    if node >= n_nodes:
        return 0.0
    neighbors = list(np.where(adj[node])[0])
    if len(neighbors) <= 1:
        return 0.0
    # Temporarily remove node.
    old_edges = adj[node].copy()
    adj[node, :] = False
    adj[:, node] = False
    # Check if first two neighbors are still connected.
    comp_first = _bfs_component(adj, neighbors[0], n_nodes)
    is_artic = 0.0
    if neighbors[1] not in comp_first:
        is_artic = 1.0
    # Restore.
    adj[node, :] = old_edges
    adj[:, node] = old_edges
    return is_artic


def _estimate_diameter(adj: np.ndarray, n_nodes: int, max_samples: int = 10) -> float:
    """Estimate graph diameter via sampled BFS."""
    if n_nodes == 0:
        return 0.0
    max_dist = 0
    sample_nodes = np.linspace(0, n_nodes - 1, min(max_samples, n_nodes), dtype=int)
    for start in sample_nodes:
        visited = {start}
        queue = [(start, 0)]
        while queue:
            curr, depth = queue.pop(0)
            max_dist = max(max_dist, depth)
            for neighbor in np.where(adj[curr])[0]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
    return float(max_dist)
