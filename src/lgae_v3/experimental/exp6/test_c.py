"""TEST-C: Untouched graph generators for exp6.1 final evaluation.

These generators are DIFFERENT from the training and TEST-B families.
They must not be inspected until the protocol is frozen.

TEST-C families:
- SBM (stochastic block model) — community structure
- Random geometric — spatial proximity
- Random regular — uniform degree
- Power-law cluster — scale-free with clustering
- Lollipop — path + complete graph
- Lobster — caterpillar tree with leaves
- Higher-dimensional grids — 3D, 4D lattices

Each family is generated with continuous parameter variation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random
import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


@dataclass(frozen=True, slots=True)
class TestCFamilyConfig:
    """Configuration for a TEST-C graph family."""
    name: str
    generator: str
    n_nodes: int
    params: dict[str, Any]
    seed: int = 42


def generate_sbm(n: int, n_communities: int, mixing: float, seed: int) -> list[tuple[int, int]]:
    """Generate stochastic block model."""
    rng = np.random.RandomState(seed)
    sizes = [n // n_communities] * n_communities
    sizes[-1] += n - sum(sizes)
    p_in = 1.0 - mixing
    p_out = mixing
    edges = []
    for ci in range(n_communities):
        for cj in range(ci, n_communities):
            p = p_in if ci == cj else p_out
            for i in range(sizes[ci]):
                for j in range(i + 1, sizes[cj]) if ci != cj else range(i + 1, sizes[ci]):
                    if rng.random() < p:
                        u = sum(sizes[:ci]) + i
                        v = sum(sizes[:cj]) + j
                        edges.append((u, v))
    return edges


def generate_random_geometric(n: int, radius: float, seed: int) -> list[tuple[int, int]]:
    """Generate random geometric graph."""
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radius:
                edges.append((i, j))
    return edges


def generate_random_regular(n: int, degree: int, seed: int) -> list[tuple[int, int]]:
    """Generate random regular graph."""
    if not HAS_NX:
        # Fallback: cycle graph.
        return [(i, (i + 1) % n) for i in range(n)]
    try:
        G = nx.random_regular_graph(degree, n, seed=seed)
        return list(G.edges())
    except Exception:
        G = nx.cycle_graph(n)
        return list(G.edges())


def generate_powerlaw_cluster(n: int, m: int, p: float, seed: int) -> list[tuple[int, int]]:
    """Generate power-law cluster graph."""
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
    return list(G.edges())


def generate_lollipop(n: int, path_length: int, seed: int) -> list[tuple[int, int]]:
    """Generate lollipop graph (path + complete graph)."""
    if not HAS_NX:
        # Fallback: path.
        return [(i, i + 1) for i in range(n - 1)]
    clique_size = n - path_length
    if clique_size < 2:
        clique_size = n // 2
        path_length = n - clique_size
    G = nx.lollipop_graph(clique_size, path_length)
    return list(G.edges())


def generate_lobster(n: int, leaf_prob: float, seed: int) -> list[tuple[int, int]]:
    """Generate lobster graph (caterpillar with leaves)."""
    rng = random.Random(seed)
    # Start with a path (caterpillar spine).
    spine_length = max(2, n // 3)
    edges = [(i, i + 1) for i in range(spine_length - 1)]
    # Add leaves to spine nodes.
    next_node = spine_length
    for i in range(spine_length):
        while next_node < n and rng.random() < leaf_prob:
            edges.append((i, next_node))
            next_node += 1
    # Add leaves to leaves (second level).
    leaf_nodes = [v for u, v in edges if v >= spine_length]
    for leaf in leaf_nodes:
        while next_node < n and rng.random() < leaf_prob * 0.5:
            edges.append((leaf, next_node))
            next_node += 1
    return edges


def generate_highdim_grid(n: int, dim: int, seed: int) -> list[tuple[int, int]]:
    """Generate higher-dimensional grid graph."""
    if not HAS_NX:
        return [(i, i + 1) for i in range(n - 1)]
    # Find grid dimensions that give approximately n nodes.
    side = int(round(n ** (1.0 / dim)))
    dims = tuple([side] * dim)
    actual_n = 1
    for d in dims:
        actual_n *= d
    try:
        if dim == 3:
            G = nx.grid_graph(dims)
        elif dim == 4:
            G = nx.grid_graph(dims)
        else:
            G = nx.grid_graph(dims)
        G = nx.convert_node_labels_to_integers(G)
        # Trim to n nodes if needed.
        nodes = list(G.nodes())[:n]
        G = G.subgraph(nodes)
        return list(G.edges())
    except Exception:
        return [(i, i + 1) for i in range(n - 1)]


def generate_test_c_configs(
    *,
    n_per_family: int = 5,
    seed: int = 7777,
) -> list[TestCFamilyConfig]:
    """Generate TEST-C configurations from untouched generators.

    These generators are NEVER used in training or TEST-B.
    """
    families = [
        "sbm", "geometric", "regular", "powerlaw_cluster",
        "lollipop", "lobster", "highdim_grid",
    ]

    configs: list[TestCFamilyConfig] = []
    rng = np.random.RandomState(seed)
    cfg_idx = 0

    n_nodes_options = [15, 20, 25, 30, 40]

    for fam in families:
        for _ in range(n_per_family):
            n = int(rng.choice(n_nodes_options))
            params: dict[str, Any] = {}

            if fam == "sbm":
                params["n_communities"] = int(rng.choice([2, 3, 4, 5]))
                params["mixing"] = float(rng.uniform(0.05, 0.3))
            elif fam == "geometric":
                params["radius"] = float(rng.uniform(0.3, 0.6))
            elif fam == "regular":
                params["degree"] = int(rng.choice([2, 3, 4]))
            elif fam == "powerlaw_cluster":
                params["m"] = int(rng.choice([1, 2, 3]))
                params["p"] = float(rng.uniform(0.1, 0.5))
            elif fam == "lollipop":
                params["path_length"] = int(rng.choice([n // 3, n // 2, 2 * n // 3]))
            elif fam == "lobster":
                params["leaf_prob"] = float(rng.uniform(0.3, 0.7))
            elif fam == "highdim_grid":
                params["dim"] = int(rng.choice([3, 4]))

            configs.append(TestCFamilyConfig(
                name=f"test_c_{fam}_{cfg_idx}",
                generator=fam,
                n_nodes=n,
                params=params,
                seed=int(rng.randint(0, 100000)),
            ))
            cfg_idx += 1

    return configs


def generate_test_c_graph(config: TestCFamilyConfig) -> list[tuple[int, int]]:
    """Generate a graph from a TEST-C configuration.

    Returns a list of edges.
    """
    gen = config.generator
    n = config.n_nodes
    p = config.params
    seed = config.seed

    if gen == "sbm":
        edges = generate_sbm(n, p["n_communities"], p["mixing"], seed)
    elif gen == "geometric":
        edges = generate_random_geometric(n, p["radius"], seed)
    elif gen == "regular":
        edges = generate_random_regular(n, p["degree"], seed)
    elif gen == "powerlaw_cluster":
        edges = generate_powerlaw_cluster(n, p["m"], p["p"], seed)
    elif gen == "lollipop":
        edges = generate_lollipop(n, p["path_length"], seed)
    elif gen == "lobster":
        edges = generate_lobster(n, p["leaf_prob"], seed)
    elif gen == "highdim_grid":
        edges = generate_highdim_grid(n, p["dim"], seed)
    else:
        edges = [(i, (i + 1) % n) for i in range(n)]

    return edges
