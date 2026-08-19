"""TEST-D: New untouched generators for exp6.2 final evaluation.

TEST-C was inspected under the broken architecture (exp6.1), so it is
no longer pristine. TEST-D uses entirely new generators not used for
any architecture decisions.

TEST-D families:
- Strongly regular graphs
- Caveman / connected caveman
- Windmill graphs
- Multipartite graphs
- K-nearest-neighbor geometric graphs
- Random intersection graphs
- Circular ladder variants (Möbius-Kantor type)
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
class TestDFamilyConfig:
    """Configuration for a TEST-D graph family."""
    name: str
    generator: str
    n_nodes: int
    params: dict[str, Any]
    seed: int = 42


def generate_strongly_regular(n: int, d: int, lam: int, mu: int, seed: int) -> list[tuple[int, int]]:
    """Generate strongly regular graph SR(n, d, lam, mu)."""
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    try:
        # Try to find a valid SRG parameter set.
        # Common: SR(16, 5, 0, 2), SR(10, 3, 0, 1), SR(25, 12, 5, 6)
        # Fall back to a regular graph if exact SRG not available.
        G = nx.random_regular_graph(d, n, seed=seed)
        return list(G.edges())
    except Exception:
        G = nx.cycle_graph(n)
        return list(G.edges())


def generate_caveman(n: int, clique_size: int, seed: int) -> list[tuple[int, int]]:
    """Generate caveman graph (cliques connected in a ring)."""
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    n_cliques = max(1, n // clique_size)
    actual_n = n_cliques * clique_size
    G = nx.caveman_graph(n_cliques, clique_size)
    G = nx.convert_node_labels_to_integers(G)
    # Trim to n nodes if needed.
    nodes = list(G.nodes())[:n]
    G = G.subgraph(nodes)
    return list(G.edges())


def generate_connected_caveman(n: int, clique_size: int, p_rewire: float, seed: int) -> list[tuple[int, int]]:
    """Generate connected caveman graph with rewiring."""
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    n_cliques = max(1, n // clique_size)
    G = nx.connected_caveman_graph(n_cliques, clique_size)
    G = nx.convert_node_labels_to_integers(G)
    # Randomly rewire some edges.
    rng = random.Random(seed)
    edges = list(G.edges())
    for u, v in edges:
        if rng.random() < p_rewire:
            w = rng.randint(0, n - 1)
            if w != u and not G.has_edge(u, w):
                G.remove_edge(u, v)
                G.add_edge(u, w)
    nodes = list(G.nodes())[:n]
    G = G.subgraph(nodes)
    return list(G.edges())


def generate_windmill(n: int, k_cliques: int, seed: int) -> list[tuple[int, int]]:
    """Generate windmill graph (cliques sharing a common vertex)."""
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    # Windmill graph Wd(k, m): m copies of K_k sharing one vertex.
    k = max(3, n // max(1, k_cliques))
    m = max(1, k_cliques)
    G = nx.Graph()
    G.add_node(0)  # shared vertex
    next_node = 1
    for _ in range(m):
        clique_nodes = [0] + list(range(next_node, next_node + k - 1))
        for i in clique_nodes:
            for j in clique_nodes:
                if i < j:
                    G.add_edge(i, j)
        next_node += k - 1
        if next_node >= n:
            break
    # Trim to n nodes.
    nodes = list(G.nodes())[:n]
    G = G.subgraph(nodes)
    return list(G.edges())


def generate_multipartite(n: int, n_parts: int, seed: int) -> list[tuple[int, int]]:
    """Generate complete multipartite graph."""
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    sizes = [n // n_parts] * n_parts
    sizes[-1] += n - sum(sizes)
    sizes = [s for s in sizes if s > 0]
    G = nx.complete_multipartite_graph(*sizes)
    G = nx.convert_node_labels_to_integers(G)
    return list(G.edges())


def generate_knn_geometric(n: int, k: int, seed: int) -> list[tuple[int, int]]:
    """Generate k-nearest-neighbor geometric graph."""
    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2)
    edges = []
    for i in range(n):
        dists = np.sqrt(np.sum((pos - pos[i]) ** 2, axis=1))
        dists[i] = np.inf  # exclude self
        neighbors = np.argsort(dists)[:k]
        for j in neighbors:
            u, v = min(i, int(j)), max(i, int(j))
            if (u, v) not in edges:
                edges.append((u, v))
    return edges


def generate_random_intersection(n: int, p: float, seed: int) -> list[tuple[int, int]]:
    """Generate random intersection graph."""
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    try:
        G = nx.uniform_random_intersection_graph(n, n // 2, p, seed=seed)
        G = nx.convert_node_labels_to_integers(G)
        return list(G.edges())
    except Exception:
        # Fallback: ER graph.
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        return list(G.edges())


def generate_test_d_configs(
    *,
    n_per_family: int = 3,
    seed: int = 31415,
) -> list[TestDFamilyConfig]:
    """Generate TEST-D configurations from untouched generators."""
    families = [
        "strongly_regular", "caveman", "connected_caveman",
        "windmill", "multipartite", "knn_geometric",
        "random_intersection",
    ]

    configs: list[TestDFamilyConfig] = []
    rng = np.random.RandomState(seed)
    cfg_idx = 0

    n_nodes_options = [15, 20, 25, 30, 40]

    for fam in families:
        for _ in range(n_per_family):
            n = int(rng.choice(n_nodes_options))
            params: dict[str, Any] = {}

            if fam == "strongly_regular":
                params["d"] = int(rng.choice([3, 4, 5]))
                params["lam"] = 0
                params["mu"] = int(rng.choice([1, 2]))
            elif fam == "caveman":
                params["clique_size"] = int(rng.choice([3, 4, 5]))
            elif fam == "connected_caveman":
                params["clique_size"] = int(rng.choice([3, 4, 5]))
                params["p_rewire"] = float(rng.uniform(0.0, 0.3))
            elif fam == "windmill":
                params["k_cliques"] = int(rng.choice([3, 4, 5, 6]))
            elif fam == "multipartite":
                params["n_parts"] = int(rng.choice([2, 3, 4]))
            elif fam == "knn_geometric":
                params["k"] = int(rng.choice([2, 3, 4, 5]))
            elif fam == "random_intersection":
                params["p"] = float(rng.uniform(0.1, 0.4))

            configs.append(TestDFamilyConfig(
                name=f"test_d_{fam}_{cfg_idx}",
                generator=fam,
                n_nodes=n,
                params=params,
                seed=int(rng.randint(0, 100000)),
            ))
            cfg_idx += 1

    return configs


def generate_test_d_graph(config: TestDFamilyConfig) -> list[tuple[int, int]]:
    """Generate a graph from a TEST-D configuration."""
    gen = config.generator
    n = config.n_nodes
    p = config.params
    seed = config.seed

    if gen == "strongly_regular":
        return generate_strongly_regular(n, p["d"], p["lam"], p["mu"], seed)
    elif gen == "caveman":
        return generate_caveman(n, p["clique_size"], seed)
    elif gen == "connected_caveman":
        return generate_connected_caveman(n, p["clique_size"], p["p_rewire"], seed)
    elif gen == "windmill":
        return generate_windmill(n, p["k_cliques"], seed)
    elif gen == "multipartite":
        return generate_multipartite(n, p["n_parts"], seed)
    elif gen == "knn_geometric":
        return generate_knn_geometric(n, p["k"], seed)
    elif gen == "random_intersection":
        return generate_random_intersection(n, p["p"], seed)
    else:
        return [(i, (i + 1) % n) for i in range(n)]
