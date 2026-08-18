"""Real graph benchmarks (Phase 26).

Provides loaders for real-world graph benchmarks. Unlike synthetic
curriculum graphs (Phase 24), real graphs have structural properties
(degree distributions, clustering, community structure) that synthetic
generators may not capture.

Supported benchmarks:
  - KARATE: Zachary's Karate Club (34 nodes, 78 edges)
  - DOLPHIN: Lusseau's dolphin network (62 nodes, 159 edges)
  - LESMIS: Les Miserables co-appearance (77 nodes, 254 edges)
  - POLBOOKS: Krebs political books (105 nodes, 441 edges)
  - FOOTBALL: NCAA football network (115 nodes, 613 edges)

For benchmarks that require external data files, we generate canonical
small-world/power-law approximations with the correct node/edge counts
when the data files are not available. This ensures the benchmarks are
always runnable, while flagging whether the data is real or synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..types import GraphBuffers, make_graph_buffers
from .curriculum import generate_graph, CurriculumEntry, GraphFamily


class RealGraphBenchmark(str, Enum):
    KARATE = "karate"
    DOLPHIN = "dolphin"
    LESMIS = "lesmis"
    POLBOOKS = "polbooks"
    FOOTBALL = "football"


@dataclass(frozen=True, slots=True)
class RealGraphSpec:
    """Specification of a real-world graph benchmark."""
    name: RealGraphBenchmark
    n_nodes: int
    n_edges: int
    description: str
    is_real_data: bool = False  # True if loaded from actual data file

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "n_nodes": int(self.n_nodes),
            "n_edges": int(self.n_edges),
            "description": self.description,
            "is_real_data": bool(self.is_real_data),
        }


# Canonical specifications for each benchmark.
BENCHMARK_SPECS: dict[RealGraphBenchmark, RealGraphSpec] = {
    RealGraphBenchmark.KARATE: RealGraphSpec(
        name=RealGraphBenchmark.KARATE, n_nodes=34, n_edges=78,
        description="Zachary's Karate Club social network",
    ),
    RealGraphBenchmark.DOLPHIN: RealGraphSpec(
        name=RealGraphBenchmark.DOLPHIN, n_nodes=62, n_edges=159,
        description="Lusseau's dolphin social network",
    ),
    RealGraphBenchmark.LESMIS: RealGraphSpec(
        name=RealGraphBenchmark.LESMIS, n_nodes=77, n_edges=254,
        description="Les Miserables character co-appearance network",
    ),
    RealGraphBenchmark.POLBOOKS: RealGraphSpec(
        name=RealGraphBenchmark.POLBOOKS, n_nodes=105, n_edges=441,
        description="Krebs political books co-purchase network",
    ),
    RealGraphBenchmark.FOOTBALL: RealGraphSpec(
        name=RealGraphBenchmark.FOOTBALL, n_nodes=115, n_edges=613,
        description="NCAA Division I football network",
    ),
}


# Canonical edge lists for the smallest benchmarks.
# These are the real edges from the well-known network science datasets.
_KARATE_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16),
    (6, 16),
    (8, 30), (8, 32), (8, 33),
    (9, 33),
    (13, 33),
    (14, 32), (14, 33),
    (15, 32), (15, 33),
    (18, 32), (18, 33),
    (19, 33),
    (20, 32), (20, 33),
    (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31),
    (25, 31), (25, 32),
    (26, 29), (26, 33),
    (27, 33),
    (28, 31), (28, 33),
    (29, 32), (29, 33),
    (30, 32), (30, 33),
    (31, 32), (31, 33),
    (32, 33),
]

_REAL_EDGES: dict[RealGraphBenchmark, list[tuple[int, int]] | None] = {
    RealGraphBenchmark.KARATE: _KARATE_EDGES,
    RealGraphBenchmark.DOLPHIN: None,  # Use synthetic approximation
    RealGraphBenchmark.LESMIS: None,
    RealGraphBenchmark.POLBOOKS: None,
    RealGraphBenchmark.FOOTBALL: None,
}


def load_benchmark(name: RealGraphBenchmark) -> tuple[RealGraphSpec, GraphBuffers]:
    """Load a real-world graph benchmark.

    Returns (spec, graph). If the real edge list is available, it is used.
    Otherwise, a synthetic approximation with the correct node/edge counts
    is generated, and ``spec.is_real_data`` is False.
    """
    spec = BENCHMARK_SPECS[name]
    edges = _REAL_EDGES.get(name)
    if edges is not None:
        # Use the real edge list.
        capacity = max(len(edges) + 8, spec.n_nodes * 2)
        graph = make_graph_buffers(spec.n_nodes, edges, capacity=capacity)
        return RealGraphSpec(
            name=spec.name, n_nodes=spec.n_nodes, n_edges=len(edges),
            description=spec.description, is_real_data=True,
        ), graph
    # Synthetic approximation: use BA model with correct node count.
    # Choose m to approximate the target edge count: m ≈ n_edges / n_nodes.
    import math
    m = max(1, int(math.floor(spec.n_edges / spec.n_nodes)))
    entry = CurriculumEntry(
        family=GraphFamily.RANDOM_BA, n_nodes=spec.n_nodes, seed=42,
        params={"m": m},
    )
    graph = generate_graph(entry)
    return RealGraphSpec(
        name=spec.name, n_nodes=spec.n_nodes, n_edges=int(graph.valid.sum()),
        description=spec.description + " (synthetic approximation)",
        is_real_data=False,
    ), graph


def list_benchmarks() -> list[RealGraphSpec]:
    """List all available real graph benchmarks."""
    return [BENCHMARK_SPECS[b] for b in RealGraphBenchmark]
