"""Extended structural effects for exp6.7.

7 supervised heads:
  1. delta_n_components
  2. delta_redundancy (min degree change)
  3. delta_hub_load (max degree change)
  4. delta_spectral_gap
  5. delta_path_length (avg shortest path change)
  6. delta_efficiency (global efficiency change)
  7. delta_curvature (approximate curvature change)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_5.observable_features import _compute_degree_stats, _compute_spectral_gap
from ..exp6_4.structural_features import compute_component_info
from ..exp6_3.exact_mpc import apply_action


EXTENDED_EFFECT_DIM = 7


@dataclass
class ExtendedEffect:
    """7 structural effect predictions."""
    delta_n_components: float = 0.0
    delta_redundancy: float = 0.0
    delta_hub_load: float = 0.0
    delta_spectral_gap: float = 0.0
    delta_path_length: float = 0.0
    delta_efficiency: float = 0.0
    delta_curvature: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.delta_n_components,
            self.delta_redundancy,
            self.delta_hub_load,
            self.delta_spectral_gap,
            self.delta_path_length,
            self.delta_efficiency,
            self.delta_curvature,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ExtendedEffect":
        return cls(
            delta_n_components=float(arr[0]),
            delta_redundancy=float(arr[1]),
            delta_hub_load=float(arr[2]),
            delta_spectral_gap=float(arr[3]),
            delta_path_length=float(arr[4]) if len(arr) > 4 else 0.0,
            delta_efficiency=float(arr[5]) if len(arr) > 5 else 0.0,
            delta_curvature=float(arr[6]) if len(arr) > 6 else 0.0,
        )


def _compute_avg_path_length(graph: GraphBuffers, n: int) -> float:
    """Compute average shortest path length using BFS."""
    if n <= 1:
        return 0.0
    # Build adjacency list.
    adj: list[list[int]] = [[] for _ in range(n)]
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n and d < n:
                adj[s].append(d)
                adj[d].append(s)

    total_dist = 0
    n_pairs = 0
    for start in range(min(n, 20)):  # Sample for efficiency.
        from collections import deque
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] < 0:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        for end in range(n):
            if end != start and dist[end] > 0:
                total_dist += dist[end]
                n_pairs += 1

    return total_dist / max(n_pairs, 1)


def _compute_global_efficiency(graph: GraphBuffers, n: int) -> float:
    """Compute global efficiency (sum of 1/distance)."""
    if n <= 1:
        return 0.0
    adj: list[list[int]] = [[] for _ in range(n)]
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n and d < n:
                adj[s].append(d)
                adj[d].append(s)

    total_eff = 0.0
    n_pairs = 0
    for start in range(min(n, 20)):
        from collections import deque
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] < 0:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        for end in range(n):
            if end != start and dist[end] > 0:
                total_eff += 1.0 / dist[end]
                n_pairs += 1

    return total_eff / max(n_pairs, 1)


def _compute_curvature_proxy(graph: GraphBuffers, n: int) -> float:
    """Compute a curvature proxy: negative degree variance.

    Higher curvature = more balanced = lower variance.
    """
    if n <= 1:
        return 0.0
    _, _, _, _, _ = _compute_degree_stats(graph, n)
    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1
    return -float(np.var(degrees))


def compute_extended_effect_labels(
    graph: GraphBuffers, z: torch.Tensor,
    action: tuple[str, int, int, dict],
) -> ExtendedEffect:
    """Compute 7 structural effect labels for training.

    These are objective-independent: they measure how the action
    changes structural properties regardless of the objective.
    """
    n = int(graph.num_nodes)

    # Before state.
    comp_before = compute_component_info(graph, n)
    _, _, _, max_deg_before, min_deg_before = _compute_degree_stats(graph, n)
    spec_before = _compute_spectral_gap(graph, n)
    path_before = _compute_avg_path_length(graph, n)
    eff_before = _compute_global_efficiency(graph, n)
    curv_before = _compute_curvature_proxy(graph, n)

    # After state.
    next_graph = apply_action(graph, action)
    comp_after = compute_component_info(next_graph, n)
    _, _, _, max_deg_after, min_deg_after = _compute_degree_stats(next_graph, n)
    spec_after = _compute_spectral_gap(next_graph, n)
    path_after = _compute_avg_path_length(next_graph, n)
    eff_after = _compute_global_efficiency(next_graph, n)
    curv_after = _compute_curvature_proxy(next_graph, n)

    return ExtendedEffect(
        delta_n_components=float(comp_after.n_components - comp_before.n_components),
        delta_redundancy=float(min_deg_after - min_deg_before),
        delta_hub_load=float(max_deg_after - max_deg_before),
        delta_spectral_gap=float(spec_after - spec_before),
        delta_path_length=float(path_after - path_before),
        delta_efficiency=float(eff_after - eff_before),
        delta_curvature=float(curv_after - curv_before),
    )
