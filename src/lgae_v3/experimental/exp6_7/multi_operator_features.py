"""Multi-operator observable feature extractor for exp6.7.1.

Correct semantics for ADD, REMOVE, REWEIGHT, EDGE_SWAP:
  - Action type one-hot correctly distinguishes all 4 operators
  - REMOVE_EDGE component prediction: only bridges change components
  - Redundancy proxy: correct for REWEIGHT (no degree change) and SWAP
  - Degree after: correct per-operator semantics
"""
from __future__ import annotations

import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_4.structural_features import compute_component_info
from ..exp6_5.observable_features import (
    _compute_degree_stats, _compute_spectral_gap, OBSERVABLE_FEATURE_DIM,
)


# Mutation type one-hot encoding.
MUTATION_TYPES_ORDERED = ["add_edge", "remove_edge", "reweight_edge", "edge_swap"]
N_MUTATION_TYPES = len(MUTATION_TYPES_ORDERED)


def _is_bridge(graph: GraphBuffers, u: int, v: int, n: int) -> bool:
    """Check if edge (u,v) is a bridge (removing it disconnects a component)."""
    comp_before = compute_component_info(graph, n)
    # Temporarily check by seeing if u and v are in the same component
    # and there's no alternative path.
    # Simple BFS without the edge.
    adj: list[list[int]] = [[] for _ in range(n)]
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n and d < n:
                if (s == u and d == v) or (s == v and d == u):
                    continue  # Skip this edge.
                adj[s].append(d)
                adj[d].append(s)

    from collections import deque
    dist = [-1] * n
    dist[u] = 0
    queue = deque([u])
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if dist[neighbor] < 0:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist[v] < 0  # v is unreachable from u without this edge.


def extract_multi_operator_features(
    graph: GraphBuffers,
    z: torch.Tensor,
    action: tuple[str, int, int, dict],
    *,
    threshold: int = 1,
    horizon: int = 2,
) -> np.ndarray:
    """Extract observable features for a multi-operator action.

    Features are objective-independent and mechanism-label-free.
    Correct operator semantics for all 4 mutation types.
    """
    n = int(graph.num_nodes)
    mt, u, v, params = action
    params = params or {}

    # --- State features (same as before) ---
    comp_info = compute_component_info(graph, n)
    n_comp = comp_info.n_components
    _degrees_arr, mean_deg, std_deg, max_deg, min_deg = _compute_degree_stats(graph, n)
    spec_gap = _compute_spectral_gap(graph, n)

    # Component sizes.
    comp_sizes = np.zeros(6)
    for i in range(min(n_comp, 6)):
        comp_sizes[i] = comp_info.component_sizes[i] if i < len(comp_info.component_sizes) else 0

    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1

    # Latent distance.
    d_sq = float(torch.norm(z[u] - z[v]).item() ** 2) if u < n and v < n else 0.0
    weight = params.get("weight", 1.0)
    factor = params.get("factor", 1.0)
    new_target = params.get("new_target", v)

    # --- Action type one-hot (4 types, correctly distinguished) ---
    action_type = np.zeros(N_MUTATION_TYPES)
    if mt in MUTATION_TYPES_ORDERED:
        action_type[MUTATION_TYPES_ORDERED.index(mt)] = 1.0
    else:
        # Unknown type — all zeros (will be caught by apply_action_with_status).
        pass

    # --- Operator-correct structural predictions ---

    # Component change prediction.
    if mt == "add_edge":
        # Adding a non-existing edge between different components merges them.
        same_comp = comp_info.component_ids[u] == comp_info.component_ids[v] if u < n and v < n else False
        merges_components = -1 if not same_comp else 0
    elif mt == "remove_edge":
        # Only bridges change component count.
        is_bridge = _is_bridge(graph, u, v, n) if u < n and v < n else False
        merges_components = 1 if is_bridge else 0  # Removing a bridge increases components.
    elif mt == "reweight_edge":
        # Reweighting never changes components.
        merges_components = 0
    elif mt == "edge_swap":
        # Swap removes (u,v) and adds (u, new_target).
        # Component change depends on whether (u,v) is a bridge and
        # whether new_target is in a different component.
        is_bridge = _is_bridge(graph, u, v, n) if u < n and v < n else False
        if is_bridge:
            # Removing bridge splits, adding new edge may reconnect.
            if new_target < n:
                same_comp_new = comp_info.component_ids[u] == comp_info.component_ids[new_target]
                merges_components = 0 if not same_comp_new else 1  # split if new target same comp
            else:
                merges_components = 1
        else:
            # Not a bridge — removing doesn't split. Adding may merge.
            if new_target < n:
                same_comp_new = comp_info.component_ids[u] == comp_info.component_ids[new_target]
                merges_components = -1 if not same_comp_new else 0
            else:
                merges_components = 0
    else:
        merges_components = 0

    n_comp_after = n_comp + merges_components
    components_remaining_after = max(0, n_comp_after - threshold)
    reaches_threshold = 1.0 if n_comp_after <= threshold else 0.0

    # Degree after action (operator-correct).
    if mt == "add_edge" and u < n and v < n:
        deg_u_after = degrees[u] + 1
        deg_v_after = degrees[v] + 1
        max_deg_after = max(max_deg, deg_u_after, deg_v_after)
        min_deg_after = min(min_deg, deg_u_after, deg_v_after)
        mean_deg_after = mean_deg + 2.0 / n
    elif mt == "remove_edge" and u < n and v < n:
        deg_u_after = max(0, degrees[u] - 1)
        deg_v_after = max(0, degrees[v] - 1)
        max_deg_after = max(deg_u_after, deg_v_after, max(0, max_deg - 1) if max_deg == degrees[u] or max_deg == degrees[v] else max_deg)
        min_deg_after = min(deg_u_after, deg_v_after)
        mean_deg_after = max(0.0, mean_deg - 2.0 / n)
    elif mt == "reweight_edge" and u < n and v < n:
        # Reweight doesn't change degree.
        deg_u_after = degrees[u]
        deg_v_after = degrees[v]
        max_deg_after = max_deg
        min_deg_after = min_deg
        mean_deg_after = mean_deg
    elif mt == "edge_swap" and u < n:
        # Remove (u,v), add (u, new_target).
        deg_u_after = degrees[u]  # net zero
        deg_v_after = max(0, degrees[v] - 1)
        if new_target < n:
            deg_new = degrees[new_target] + 1
            max_deg_after = max(max_deg, deg_new)
            min_deg_after = min(min_deg, deg_v_after, deg_new)
        else:
            max_deg_after = max_deg
            min_deg_after = min(min_deg, deg_v_after)
        mean_deg_after = mean_deg  # edge count unchanged
    else:
        deg_u_after = degrees[u] if u < n else 0.0
        deg_v_after = degrees[v] if v < n else 0.0
        max_deg_after = max_deg
        min_deg_after = min_deg
        mean_deg_after = mean_deg

    hub_load_after = max_deg_after / max(n - 1, 1)
    redundancy_after = min_deg_after / max(n - 1, 1)

    # --- Build feature vector ---
    state_features = np.array([
        n / 30.0,
        n_comp / 6.0,
        mean_deg / 10.0,
        std_deg / 5.0,
        max_deg / 20.0,
        min_deg / 5.0,
        spec_gap,
        hub_load_after,
        redundancy_after,
    ])

    comp_size_features = comp_sizes / n

    action_features = np.array([
        d_sq / 10.0,
        weight,
        factor,  # reweight factor
        float(new_target) / n if new_target < n else 1.0,  # swap target
        merges_components,
        n_comp_after / 10.0,
        components_remaining_after / 10.0,
        reaches_threshold,
        max_deg_after / 20.0,
        min_deg_after / 5.0,
        mean_deg_after / 10.0,
        deg_u_after / 20.0 if u < n else 0.0,
        deg_v_after / 20.0 if v < n else 0.0,
    ])

    all_features = np.concatenate([
        state_features,  # 9
        comp_size_features,  # 6
        action_type,  # 4
        action_features,  # 14
    ])
    # Total: 33. Pad to OBSERVABLE_FEATURE_DIM (64).
    if len(all_features) < OBSERVABLE_FEATURE_DIM:
        all_features = np.pad(all_features, (0, OBSERVABLE_FEATURE_DIM - len(all_features)))
    else:
        all_features = all_features[:OBSERVABLE_FEATURE_DIM]

    return all_features
