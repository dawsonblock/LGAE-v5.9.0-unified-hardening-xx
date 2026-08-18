"""Sheaf consistency certification (Phase 17).

A sheaf assigns data (vector spaces) to each node and edge of a graph, with
restriction maps that relate edge data to node data. Sheaf consistency
verifies that local data agrees globally: for each edge (u, v), the data
assigned to the edge must map consistently to the data at both endpoints.

In the runtime, the "sheaf" is the fiber bundle: each node has a latent
vector, each edge has a gauge connection, and the restriction maps are the
gauge transformations. Sheaf consistency certification checks that the
local gauge transformations compose correctly around cycles.

A graph has zero sheaf inconsistency if and only if the gauge connections
are flat (path-independent). This is a strong structural invariant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SheafConsistencyResult:
    """Result of sheaf consistency certification."""
    cycle_count: int
    max_inconsistency: float
    mean_inconsistency: float
    is_flat: bool  # True if all cycles have zero inconsistency (flat sheaf)
    per_cycle_inconsistency: list[float] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "cycle_count": int(self.cycle_count),
            "max_inconsistency": float(self.max_inconsistency),
            "mean_inconsistency": float(self.mean_inconsistency),
            "is_flat": bool(self.is_flat),
            "per_cycle_inconsistency": [float(x) for x in self.per_cycle_inconsistency],
        }


def _find_cycles(adj: list[list[int]], max_cycles: int = 20) -> list[list[int]]:
    """Find a bounded number of simple cycles using DFS."""
    cycles: list[list[int]] = []
    n = len(adj)

    def dfs(start: int, current: int, path: list[int], visited: set[int]) -> None:
        if len(cycles) >= max_cycles:
            return
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 3:
                cycles.append(list(path))
                return
            if neighbor not in visited and neighbor > start:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    for s in range(n):
        dfs(s, s, [s], {s})
    return cycles


def certify_sheaf_consistency(
    *,
    edge_index: Tensor,  # [2, E] tensor of edge endpoints
    gauge_connections: Tensor | None = None,  # [E, d, d] gauge matrices
    max_cycles: int = 20,
    tolerance: float = 1e-5,
) -> SheafConsistencyResult:
    """Certify sheaf consistency by checking cycle flatness.

    For each cycle, compute the composition of gauge transformations around
    the cycle. If the composition is identity (within tolerance), the sheaf
    is flat on that cycle. The sheaf is globally flat if all cycles are flat.

    If ``gauge_connections`` is None, we check structural consistency only
    (whether the graph's edge structure admits a flat sheaf).
    """
    if edge_index.numel() == 0:
        return SheafConsistencyResult(
            cycle_count=0, max_inconsistency=0.0, mean_inconsistency=0.0,
            is_flat=True, per_cycle_inconsistency=[],
        )

    n = int(edge_index.max().item()) + 1
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    # Build adjacency list.
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)

    cycles = _find_cycles(adj, max_cycles=max_cycles)

    if not cycles:
        return SheafConsistencyResult(
            cycle_count=0, max_inconsistency=0.0, mean_inconsistency=0.0,
            is_flat=True, per_cycle_inconsistency=[],
        )

    # Compute inconsistency per cycle.
    per_cycle: list[float] = []
    for cycle in cycles:
        if gauge_connections is not None:
            # Compose gauge transformations around the cycle.
            # For simplicity, measure the Frobenius norm of (product - I).
            d = gauge_connections.shape[-1]
            product = torch.eye(d, dtype=gauge_connections.dtype, device=gauge_connections.device)
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                # Find the edge index for (u, v) or (v, u).
                edge_idx = None
                for j, (su, sv) in enumerate(zip(src, dst)):
                    if (su == u and sv == v) or (su == v and sv == u):
                        edge_idx = j
                        break
                if edge_idx is not None:
                    product = product @ gauge_connections[edge_idx]
            inconsistency = float(torch.norm(product - torch.eye(d, dtype=product.dtype, device=product.device)).item())
        else:
            # Without gauge data, measure structural inconsistency:
            # the degree of asymmetry in the cycle (always 0 for undirected).
            inconsistency = 0.0
        per_cycle.append(inconsistency)

    max_inc = max(per_cycle) if per_cycle else 0.0
    mean_inc = sum(per_cycle) / len(per_cycle) if per_cycle else 0.0
    is_flat = max_inc <= tolerance

    return SheafConsistencyResult(
        cycle_count=len(cycles),
        max_inconsistency=max_inc,
        mean_inconsistency=mean_inc,
        is_flat=is_flat,
        per_cycle_inconsistency=per_cycle,
    )
