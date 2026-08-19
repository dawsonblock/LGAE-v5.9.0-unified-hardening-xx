"""Curriculum graph generator (Phase 24).

Generates diverse graph families for training and evaluation. The curriculum
spans structural regimes the runtime should generalize across:

  - path: linear chains
  - cycle: ring topologies
  - star: hub-and-spoke
  - grid: 2D lattice
  - barbell: two cliques joined by a bridge
  - random_er: Erdos-Renyi
  - random_ba: Barabasi-Albert (scale-free)
  - random_ws: Watts-Strogatz (small-world)
  - complete: fully connected
  - tree: binary tree
  - bipartite: random bipartite

Each family is generated with a deterministic seed so the curriculum is
reproducible. The generator returns ``GraphBuffers`` directly (tensor-native),
not NetworkX graphs, for hot-path efficiency. NetworkX is used only as a
construction helper where convenient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

import torch

from ..types import GraphBuffers, make_graph_buffers


class GraphFamily(str, Enum):
    PATH = "path"
    CYCLE = "cycle"
    STAR = "star"
    GRID = "grid"
    BARBELL = "barbell"
    RANDOM_ER = "random_er"
    RANDOM_BA = "random_ba"
    RANDOM_WS = "random_ws"
    COMPLETE = "complete"
    TREE = "tree"
    BIPARTITE = "bipartite"
    # v6.0-exp5.1: New families for untouched TEST-B split.
    WHEEL = "wheel"
    LADDER = "ladder"
    CIRCULAR_LADDER = "circular_ladder"
    HYPERCUBE = "hypercube"


@dataclass(frozen=True, slots=True)
class CurriculumEntry:
    """One entry in the curriculum."""
    family: GraphFamily
    n_nodes: int
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    family_id: str = ""

    def __post_init__(self) -> None:
        if not self.family_id:
            object.__setattr__(self, "family_id", f"{self.family.value}_n{self.n_nodes}_s{self.seed}")

    def to_log(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "n_nodes": int(self.n_nodes),
            "seed": int(self.seed),
            "family_id": self.family_id,
            "params": self.params,
        }


def _edges_path(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _edges_cycle(n: int) -> list[tuple[int, int]]:
    if n < 3:
        return _edges_path(n)
    return [(i, (i + 1) % n) for i in range(n)]


def _edges_star(n: int) -> list[tuple[int, int]]:
    if n < 2:
        return []
    return [(0, i) for i in range(1, n)]


def _edges_grid(n: int) -> list[tuple[int, int]]:
    import math
    side = max(2, int(math.isqrt(n)))
    edges: list[tuple[int, int]] = []
    for r in range(side):
        for c in range(side):
            idx = r * side + c
            if c + 1 < side:
                edges.append((idx, idx + 1))
            if r + 1 < side:
                edges.append((idx, idx + side))
    return edges


def _edges_barbell(n: int) -> list[tuple[int, int]]:
    import math
    clique_size = max(3, n // 3)
    edges: list[tuple[int, int]] = []
    # First clique.
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((i, j))
    # Bridge.
    if n > 2 * clique_size:
        bridge_start = clique_size
        bridge_end = n - clique_size - 1
        for i in range(bridge_start, bridge_end):
            edges.append((i, i + 1))
        # Connect first clique to bridge start, bridge end to second clique.
        edges.append((clique_size - 1, bridge_start))
        edges.append((bridge_end, n - clique_size))
    # Second clique.
    offset = n - clique_size
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((offset + i, offset + j))
    return edges


def _edges_complete(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _edges_tree(n: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(1, n):
        parent = (i - 1) // 2
        edges.append((parent, i))
    return edges


def _edges_bipartite(n: int, seed: int = 0, p: float = 0.3) -> list[tuple[int, int]]:
    import random
    rng = random.Random(seed)
    left = n // 2
    right = n - left
    edges: list[tuple[int, int]] = []
    for u in range(left):
        for v in range(right):
            if rng.random() < p:
                edges.append((u, left + v))
    return edges if edges else [(0, left)] if left < n and left > 0 else []


def _edges_random_er(n: int, seed: int = 0, p: float = 0.1) -> list[tuple[int, int]]:
    import random
    rng = random.Random(seed)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges if edges else [(0, 1)] if n >= 2 else []


def _edges_random_ba(n: int, seed: int = 0, m: int = 2) -> list[tuple[int, int]]:
    import random
    rng = random.Random(seed)
    m = max(1, min(m, n - 1))
    edges: list[tuple[int, int]] = []
    targets = list(range(m))
    for new_node in range(m, n):
        for t in targets:
            edges.append((new_node, t))
        # Preferential attachment.
        degrees = [0] * n
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        pool = []
        for node in range(new_node):
            pool.extend([node] * degrees[node])
        if pool:
            new_targets = rng.sample(pool, min(m, len(pool)))
            targets = list(set(new_targets))
        else:
            targets = list(range(m))
    return edges


def _edges_random_ws(n: int, seed: int = 0, k: int = 4, p: float = 0.1) -> list[tuple[int, int]]:
    import random
    rng = random.Random(seed)
    k = max(2, min(k, n - 1))
    if k % 2 != 0:
        k -= 1
    half = k // 2
    edges: set[tuple[int, int]] = set()
    for u in range(n):
        for j in range(1, half + 1):
            v = (u + j) % n
            edges.add((min(u, v), max(u, v)))
    # Rewire.
    edge_list = list(edges)
    for i, (u, v) in enumerate(edge_list):
        if rng.random() < p:
            new_v = rng.randint(0, n - 1)
            if new_v != u and new_v != v:
                edge_list[i] = (min(u, new_v), max(u, new_v))
    # Deduplicate and filter self-edges.
    seen: set[tuple[int, int]] = set()
    clean: list[tuple[int, int]] = []
    for u, v in edge_list:
        if u != v and (u, v) not in seen:
            seen.add((u, v))
            clean.append((u, v))
    return clean if clean else [(0, 1)] if n >= 2 else []


def _edges_wheel(n: int) -> list[tuple[int, int]]:
    """Wheel graph: a cycle of n-1 nodes plus a hub connected to all."""
    if n < 4:
        return _edges_cycle(n)
    hub = 0
    edges: list[tuple[int, int]] = []
    # Rim cycle (nodes 1..n-1).
    for i in range(1, n):
        edges.append((i, (i % (n - 1)) + 1 if i < n - 1 else 1))
    # Spokes from hub to rim.
    for i in range(1, n):
        edges.append((hub, i))
    return edges


def _edges_ladder(n: int) -> list[tuple[int, int]]:
    """Ladder graph: two paths connected by rungs."""
    if n < 4:
        return _edges_path(n)
    rungs = n // 2
    edges: list[tuple[int, int]] = []
    for i in range(rungs):
        # Rung.
        edges.append((2 * i, 2 * i + 1))
        # Rails.
        if i + 1 < rungs:
            edges.append((2 * i, 2 * (i + 1)))
            edges.append((2 * i + 1, 2 * (i + 1) + 1))
    # Leftover node.
    if 2 * rungs < n:
        edges.append((n - 2, n - 1))
    return edges


def _edges_circular_ladder(n: int) -> list[tuple[int, int]]:
    """Circular ladder (prism) graph."""
    if n < 6:
        return _edges_ladder(n)
    rungs = n // 2
    edges: list[tuple[int, int]] = []
    for i in range(rungs):
        # Rung.
        edges.append((2 * i, 2 * i + 1))
        # Rails (circular).
        next_i = (i + 1) % rungs
        edges.append((2 * i, 2 * next_i))
        edges.append((2 * i + 1, 2 * next_i + 1))
    return edges


def _edges_hypercube(n: int) -> list[tuple[int, int]]:
    """Hypercube graph (dimension d where 2^d <= n)."""
    import math
    d = max(1, int(math.log2(max(n, 2))))
    n_actual = min(n, 2 ** d)
    edges: list[tuple[int, int]] = []
    for u in range(n_actual):
        for bit in range(d):
            v = u ^ (1 << bit)
            if v < n_actual and u < v:
                edges.append((u, v))
    return edges if edges else [(0, 1)] if n >= 2 else []


_EDGE_GENERATORS = {
    GraphFamily.PATH: _edges_path,
    GraphFamily.CYCLE: _edges_cycle,
    GraphFamily.STAR: _edges_star,
    GraphFamily.GRID: _edges_grid,
    GraphFamily.BARBELL: _edges_barbell,
    GraphFamily.COMPLETE: _edges_complete,
    GraphFamily.TREE: _edges_tree,
    GraphFamily.WHEEL: _edges_wheel,
    GraphFamily.LADDER: _edges_ladder,
    GraphFamily.CIRCULAR_LADDER: _edges_circular_ladder,
    GraphFamily.HYPERCUBE: _edges_hypercube,
}


def generate_graph(entry: CurriculumEntry) -> GraphBuffers:
    """Generate a GraphBuffers from a curriculum entry."""
    n = int(entry.n_nodes)
    seed = int(entry.seed)
    if entry.family in _EDGE_GENERATORS:
        edges = _EDGE_GENERATORS[entry.family](n)
    elif entry.family == GraphFamily.RANDOM_ER:
        p = float(entry.params.get("p", 0.1))
        edges = _edges_random_er(n, seed=seed, p=p)
    elif entry.family == GraphFamily.RANDOM_BA:
        m = int(entry.params.get("m", 2))
        edges = _edges_random_ba(n, seed=seed, m=m)
    elif entry.family == GraphFamily.RANDOM_WS:
        k = int(entry.params.get("k", 4))
        p = float(entry.params.get("p", 0.1))
        edges = _edges_random_ws(n, seed=seed, k=k, p=p)
    elif entry.family == GraphFamily.BIPARTITE:
        p = float(entry.params.get("p", 0.3))
        edges = _edges_bipartite(n, seed=seed, p=p)
    else:
        edges = _edges_path(n)
    capacity = max(len(edges) + 8, n * 2)
    return make_graph_buffers(n, edges, capacity=capacity)


class CurriculumGenerator:
    """Generates a curriculum of diverse graph families."""

    def __init__(self, *, seed: int = 42) -> None:
        self.base_seed = int(seed)

    def generate_curriculum(
        self,
        *,
        n_nodes: int = 20,
        families: list[GraphFamily] | None = None,
        n_seeds: int = 3,
        params: dict[GraphFamily, dict[str, Any]] | None = None,
    ) -> list[CurriculumEntry]:
        """Generate a list of curriculum entries spanning families and seeds."""
        families = families or list(GraphFamily)
        params = params or {}
        entries: list[CurriculumEntry] = []
        for family in families:
            for seed_idx in range(n_seeds):
                # v5.11 Phase 14: use SHA-256 instead of hash() for
                # deterministic seed derivation across PYTHONHASHSEED values.
                import hashlib
                family_hash = int.from_bytes(
                    hashlib.sha256(family.value.encode()).digest()[:4], "big"
                )
                seed = self.base_seed + seed_idx * 1000 + family_hash % 100
                entries.append(CurriculumEntry(
                    family=family,
                    n_nodes=n_nodes,
                    seed=seed,
                    params=params.get(family, {}),
                ))
        return entries

    def generate_split(
        self,
        *,
        n_nodes: int = 20,
        train_families: list[GraphFamily] | None = None,
        held_out_families: list[GraphFamily] | None = None,
        n_seeds: int = 3,
    ) -> dict[str, list[CurriculumEntry]]:
        """Generate a train/held-out split for OOD qualification.

        Held-out families are never seen during training. The split is
        deterministic for a given base seed.
        """
        all_families = list(GraphFamily)
        train = train_families or all_families[:7]
        held_out = held_out_families or all_families[7:]
        return {
            "train": self.generate_curriculum(n_nodes=n_nodes, families=train, n_seeds=n_seeds),
            "held_out": self.generate_curriculum(n_nodes=n_nodes, families=held_out, n_seeds=n_seeds),
        }

    def iter_graphs(self, entries: list[CurriculumEntry]) -> Iterator[tuple[CurriculumEntry, GraphBuffers]]:
        """Iterate over (entry, graph) pairs."""
        for entry in entries:
            yield entry, generate_graph(entry)
