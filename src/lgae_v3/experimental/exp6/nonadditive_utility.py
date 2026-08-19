"""Non-additive utility functions for exp6.3.

The default utility U = -sum(w * d²) is additive/separable: each edge
contributes independently. This means greedy = optimal for any horizon.

For multi-step planning to matter, the utility must be NON-ADDITIVE:
edges must interact. These utilities create genuine delayed-value structure.

Non-additive utilities:
1. Connectivity-aware: rewards having few connected components
2. Spectral: rewards low spectral radius (well-connected but not too dense)
3. Diameter-aware: rewards small graph diameter
4. Connectivity + utility: combines additive term with connectivity bonus
"""
from __future__ import annotations

import numpy as np
import torch
from ...types import GraphBuffers


def compute_connectivity_utility(graph: GraphBuffers, z: torch.Tensor) -> float:
    """Non-additive utility: rewards connectivity (few components).

    U = -sum(w * d²) + λ * (n - n_components)

    The connectivity bonus is NON-ADDITIVE: adding a bridge edge between
    two components gives a large bonus, while adding an edge within a
    component gives no bonus. This creates delayed-value structure:
    bridging components now (negative additive term) can enable
    better future structure (large connectivity bonus).
    """
    from .candidate_generator import compute_exact_utility

    n = int(graph.num_nodes)
    # Compute additive part.
    u_add = compute_exact_utility(graph, z)

    # Compute number of connected components (non-additive).
    n_components = _count_components(graph, n)

    # Connectivity bonus: reward having fewer components.
    lambda_conn = 5.0
    u_total = u_add + lambda_conn * (n - n_components)

    return u_total


def compute_spectral_utility(graph: GraphBuffers, z: torch.Tensor) -> float:
    """Non-additive utility: rewards low spectral radius.

    U = -sum(w * d²) - λ * spectral_radius

    The spectral radius depends on the GLOBAL structure, not just
    individual edges. Adding/removing one edge changes the spectral
    radius in a non-additive way.
    """
    from .candidate_generator import compute_exact_utility

    u_add = compute_exact_utility(graph, z)

    # Compute spectral radius (largest eigenvalue of adjacency).
    n = int(graph.num_nodes)
    adj = np.zeros((n, n))
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n and d < n:
                w = float(graph.weight[i].item())
                adj[s, d] = w
                adj[d, s] = w

    try:
        eigenvalues = np.linalg.eigvalsh(adj)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
    except Exception:
        spectral_radius = 0.0

    lambda_spec = 2.0
    u_total = u_add - lambda_spec * spectral_radius

    return u_total


def compute_diameter_utility(graph: GraphBuffers, z: torch.Tensor) -> float:
    """Non-additive utility: rewards small diameter.

    U = -sum(w * d²) - λ * diameter

    The diameter depends on the global path structure. Adding a
    shortcut edge can dramatically reduce diameter, creating
    delayed-value opportunities.
    """
    from .candidate_generator import compute_exact_utility

    u_add = compute_exact_utility(graph, z)

    # Compute diameter (longest shortest path).
    n = int(graph.num_nodes)
    adj = _build_adjacency_dict(graph, n)
    diameter = _compute_diameter(adj, n)

    lambda_diam = 1.0
    u_total = u_add - lambda_diam * diameter

    return u_total


def compute_connectivity_plus_utility(graph: GraphBuffers, z: torch.Tensor) -> float:
    """Combined: additive utility + connectivity bonus.

    This is the key non-additive utility for delayed-value tasks.
    Adding a bridge between components has:
    - Negative additive term: -w * d² (bad for one-step)
    - Large connectivity bonus: +λ (good globally)

    So greedy (which only sees the additive term) picks the WRONG action,
    while multi-step MPC (which sees the total) picks the bridge.
    """
    return compute_connectivity_utility(graph, z)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_adjacency_dict(graph: GraphBuffers, n: int) -> dict[int, set[int]]:
    """Build adjacency dict from graph buffers."""
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n and d < n:
                adj[s].add(d)
                adj[d].add(s)
    return adj


def _count_components(graph: GraphBuffers, n: int) -> int:
    """Count connected components using BFS."""
    adj = _build_adjacency_dict(graph, n)
    visited = set()
    n_components = 0

    for start in range(n):
        if start in visited:
            continue
        n_components += 1
        # BFS
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return n_components


def _compute_diameter(adj: dict[int, set[int]], n: int) -> int:
    """Compute graph diameter using BFS from each node."""
    if n <= 1:
        return 0

    max_dist = 0
    for start in range(n):
        # BFS from start
        dist = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        if len(dist) < n:
            return n  # disconnected, large diameter
        max_dist = max(max_dist, max(dist.values()))

    return max_dist


# ---------------------------------------------------------------------------
# Non-additive exact MPC
# ---------------------------------------------------------------------------

def compute_nonadditive_delta(
    graph: GraphBuffers,
    z: torch.Tensor,
    action: tuple[str, int, int, dict],
    utility_fn,
) -> float:
    """Compute delta utility for a non-additive utility function.

    Unlike the additive case, this requires applying the action and
    recomputing the full utility (O(E) per evaluation, not O(1)).
    """
    from .mpc_planner import apply_action_to_graph

    u_before = utility_fn(graph, z)
    new_graph = apply_action_to_graph(graph, action)
    u_after = utility_fn(new_graph, z)
    return float(u_after - u_before)


def exact_mpc_nonadditive(
    graph: GraphBuffers,
    z: torch.Tensor,
    available_actions: list[tuple[str, int, int, dict]],
    utility_fn,
    *,
    horizon: int = 2,
    gamma: float = 0.9,
) -> dict:
    """Exact MPC with a non-additive utility function.

    For non-additive utilities, each step requires full utility recomputation.
    """
    from .mpc_planner import apply_action_to_graph

    if horizon == 0 or len(available_actions) == 0:
        return {"best_first_action": ("", 0, 0), "best_total_utility": float("-inf"),
                "n_evaluations": 0, "all_first_action_utilities": {}}

    from itertools import product

    n_evaluations = len(available_actions) ** horizon
    best_utility = float("-inf")
    best_sequence = []
    first_action_utilities: dict[str, float] = {}

    for seq in product(available_actions, repeat=horizon):
        current_graph = graph
        total_u = 0.0

        for t, action in enumerate(seq):
            delta_u = compute_nonadditive_delta(current_graph, z, action, utility_fn)
            total_u += (gamma ** t) * delta_u
            current_graph = apply_action_to_graph(current_graph, action)

        first_key = f"{seq[0][0]}_{seq[0][1]}_{seq[0][2]}"
        if first_key not in first_action_utilities or total_u > first_action_utilities[first_key]:
            first_action_utilities[first_key] = total_u

        if total_u > best_utility:
            best_utility = total_u
            best_sequence = list(seq)

    best_first = ("", 0, 0)
    if best_sequence:
        a = best_sequence[0]
        best_first = (a[0], a[1], a[2])

    return {
        "best_first_action": best_first,
        "best_total_utility": best_utility,
        "n_evaluations": n_evaluations,
        "all_first_action_utilities": first_action_utilities,
        "best_sequence": best_sequence,
    }
