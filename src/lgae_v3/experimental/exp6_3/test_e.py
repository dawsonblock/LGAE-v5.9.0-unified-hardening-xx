"""TEST-E: Fresh untouched generators for exp6.3 final evaluation.

TEST-D has been inspected during exp6.2 architecture decisions.
TEST-E uses entirely new generators not used for any prior decisions.

TEST-E families:
- Generalized Petersen graphs
- Higher-order SBM variants
- Random DAG-derived undirected graphs
- Connected caveman (different from TEST-D)
- Random circulant graphs
- Full rary tree
- Anti-regular graphs
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
class TestEConfig:
    name: str
    generator: str
    n_nodes: int
    params: dict[str, Any]
    seed: int = 42


def generate_generalized_petersen(n: int, k: int, seed: int) -> list[tuple[int, int]]:
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    try:
        G = nx.generators.generalized_petersen_graph(n, k)
        return list(G.edges())
    except Exception:
        G = nx.cycle_graph(n)
        return list(G.edges())


def generate_higher_order_sbm(n: int, n_comm: int, p_in: float, p_out: float, seed: int) -> list[tuple[int, int]]:
    rng = np.random.RandomState(seed)
    sizes = [n // n_comm] * n_comm
    sizes[-1] += n - sum(sizes)
    edges = []
    for ci in range(n_comm):
        for cj in range(ci, n_comm):
            p = p_in if ci == cj else p_out
            for i in range(sizes[ci]):
                for j in range(i + 1, sizes[cj]) if ci != cj else range(i + 1, sizes[ci]):
                    if rng.random() < p:
                        u = sum(sizes[:ci]) + i
                        v = sum(sizes[:cj]) + j
                        edges.append((u, v))
    return edges


def generate_dag_derived(n: int, p: float, seed: int) -> list[tuple[int, int]]:
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    G = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    G = G.to_undirected()
    return list(G.edges())


def generate_connected_caveman_e(n: int, clique_size: int, seed: int) -> list[tuple[int, int]]:
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    n_cliques = max(1, n // clique_size)
    G = nx.connected_caveman_graph(n_cliques, clique_size)
    G = nx.convert_node_labels_to_integers(G)
    nodes = list(G.nodes())[:n]
    G = G.subgraph(nodes)
    return list(G.edges())


def generate_circulant(n: int, steps: list[int], seed: int) -> list[tuple[int, int]]:
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    G = nx.generators.circulant_graph(n, steps)
    return list(G.edges())


def generate_rary_tree(n: int, r: int, seed: int) -> list[tuple[int, int]]:
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    try:
        G = nx.full_rary_tree(r, n)
        return list(G.edges())
    except Exception:
        G = nx.balanced_tree(r, max(1, n // r - 1))
        return list(G.edges())[:n-1]


def generate_anti_regular(n: int, seed: int) -> list[tuple[int, int]]:
    """Anti-regular graph: no two vertices have the same degree."""
    if not HAS_NX:
        return [(i, (i + 1) % n) for i in range(n)]
    # Construct anti-regular graph manually.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # Connect vertex i to all j where i + j >= n - 1 and i != j.
    for i in range(n):
        for j in range(i + 1, n):
            if i + j >= n - 1:
                G.add_edge(i, j)
    return list(G.edges())


def generate_test_e_configs(*, n_per_family: int = 3, seed: int = 27183) -> list[TestEConfig]:
    families = [
        "generalized_petersen", "higher_order_sbm", "dag_derived",
        "connected_caveman_e", "circulant", "rary_tree", "anti_regular",
    ]
    configs: list[TestEConfig] = []
    rng = np.random.RandomState(seed)
    idx = 0
    n_options = [15, 20, 25, 30]

    for fam in families:
        for _ in range(n_per_family):
            n = int(rng.choice(n_options))
            params: dict[str, Any] = {}
            if fam == "generalized_petersen":
                params["k"] = int(rng.choice([2, 3, 4]))
                n = max(2 * params["k"] + 1, n)
            elif fam == "higher_order_sbm":
                params["n_comm"] = int(rng.choice([3, 4, 5, 6]))
                params["p_in"] = float(rng.uniform(0.5, 0.9))
                params["p_out"] = float(rng.uniform(0.01, 0.1))
            elif fam == "dag_derived":
                params["p"] = float(rng.uniform(0.1, 0.3))
            elif fam == "connected_caveman_e":
                params["clique_size"] = int(rng.choice([3, 4, 5]))
            elif fam == "circulant":
                params["steps"] = [int(rng.choice([1, 2, 3]))]
            elif fam == "rary_tree":
                params["r"] = int(rng.choice([2, 3, 4]))
            elif fam == "anti_regular":
                pass

            configs.append(TestEConfig(
                name=f"test_e_{fam}_{idx}",
                generator=fam, n_nodes=n, params=params,
                seed=int(rng.randint(0, 100000)),
            ))
            idx += 1
    return configs


def generate_test_e_graph(config: TestEConfig) -> list[tuple[int, int]]:
    g = config.generator
    n, p, s = config.n_nodes, config.params, config.seed
    if g == "generalized_petersen":
        return generate_generalized_petersen(n, p["k"], s)
    elif g == "higher_order_sbm":
        return generate_higher_order_sbm(n, p["n_comm"], p["p_in"], p["p_out"], s)
    elif g == "dag_derived":
        return generate_dag_derived(n, p["p"], s)
    elif g == "connected_caveman_e":
        return generate_connected_caveman_e(n, p["clique_size"], s)
    elif g == "circulant":
        return generate_circulant(n, p["steps"], s)
    elif g == "rary_tree":
        return generate_rary_tree(n, p["r"], s)
    elif g == "anti_regular":
        return generate_anti_regular(n, s)
    else:
        return [(i, (i + 1) % n) for i in range(n)]
