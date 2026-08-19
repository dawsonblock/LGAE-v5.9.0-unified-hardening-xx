"""Parametric graph family generators for exp5.3.

Instead of treating each named topology as one category, generate
families parametrically across continuous structural parameters.

This creates a continuous structural distribution rather than
memorizing a few topology labels.

Families:
- ER across p (connection probability)
- BA across m (preferential attachment parameter)
- WS across k and β (rewiring probability)
- Trees across branching factor
- Grids across dimensions/aspect ratios
- Bipartite across partition ratios
- SBM across community counts/mixing
- Random geometric graphs across radius
- Regular graphs across degree
- Power-law cluster graphs

Graph size is varied independently (n=10, 15, 20, 25, 30).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random
import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


@dataclass(frozen=True, slots=True)
class ParametricFamilyConfig:
    """Configuration for a parametric graph family."""
    name: str
    generator: str  # "er", "ba", "ws", "tree", "grid", "bipartite", "sbm", "geometric", "regular", "pl_cluster"
    n_nodes: int
    params: dict[str, Any]
    seed: int = 42


def generate_parametric_graph(config: ParametricFamilyConfig) -> Any:
    """Generate a graph from a parametric family configuration.

    Returns a networkx Graph (or a simple adjacency representation if
    networkx is not available).
    """
    if not HAS_NX:
        return _generate_simple_graph(config)

    rng = np.random.RandomState(config.seed)
    n = config.n_nodes
    gen = config.generator
    p = config.params

    if gen == "er":
        prob = p.get("p", 0.2)
        G = nx.erdos_renyi_graph(n, prob, seed=config.seed)
    elif gen == "ba":
        m = p.get("m", 2)
        G = nx.barabasi_albert_graph(n, m, seed=config.seed)
    elif gen == "ws":
        k = p.get("k", 4)
        beta = p.get("beta", 0.3)
        G = nx.watts_strogatz_graph(n, k, beta, seed=config.seed)
    elif gen == "tree":
        branching = p.get("branching", 2)
        G = nx.full_rary_tree(branching, n)
    elif gen == "grid":
        dims = p.get("dims", (4, 5))
        if dims[0] * dims[1] <= n:
            G = nx.grid_2d_graph(dims[0], dims[1])
            G = nx.convert_node_labels_to_integers(G)
        else:
            G = nx.path_graph(n)
    elif gen == "bipartite":
        ratio = p.get("ratio", 0.5)
        n1 = max(1, int(n * ratio))
        n2 = n - n1
        G = nx.complete_bipartite_graph(n1, n2)
    elif gen == "sbm":
        n_comm = p.get("n_communities", 2)
        mixing = p.get("mixing", 0.1)
        sizes = [n // n_comm] * n_comm
        sizes[-1] += n - sum(sizes)
        p_in = 1.0 - mixing
        p_out = mixing
        probs = [[p_in if i == j else p_out for j in range(n_comm)]
                 for i in range(n_comm)]
        G = nx.stochastic_block_model(sizes, probs, seed=config.seed)
    elif gen == "geometric":
        radius = p.get("radius", 0.4)
        pos = rng.rand(n, 2)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radius:
                    G.add_edge(i, j)
    elif gen == "regular":
        degree = p.get("degree", 3)
        try:
            G = nx.random_regular_graph(degree, n, seed=config.seed)
        except Exception:
            G = nx.cycle_graph(n)
    elif gen == "pl_cluster":
        m = p.get("m", 2)
        p_edge = p.get("p", 0.3)
        G = nx.powerlaw_cluster_graph(n, m, p_edge, seed=config.seed)
    else:
        G = nx.path_graph(n)

    # Ensure connected (for spectral gap computation).
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            u = next(iter(components[0]))
            v = next(iter(components[i]))
            G.add_edge(u, v)

    return G


def _generate_simple_graph(config: ParametricFamilyConfig) -> dict:
    """Generate a simple adjacency representation without networkx."""
    n = config.n_nodes
    adj = {i: set() for i in range(n)}
    rng = random.Random(config.seed)
    gen = config.generator
    p = config.params

    if gen == "er":
        prob = p.get("p", 0.2)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < prob:
                    adj[i].add(j)
                    adj[j].add(i)
    elif gen == "path":
        for i in range(n - 1):
            adj[i].add(i + 1)
            adj[i + 1].add(i)
    elif gen == "cycle":
        for i in range(n):
            adj[i].add((i + 1) % n)
            adj[(i + 1) % n].add(i)
    else:
        # Default: path graph.
        for i in range(n - 1):
            adj[i].add(i + 1)
            adj[i + 1].add(i)

    return {"adj": adj, "n": n}


def generate_parametric_dataset(
    *,
    n_per_family: int = 5,
    n_steps: int = 5,
    seed: int = 42,
    n_nodes_options: list[int] | None = None,
    families: list[str] | None = None,
) -> list[ParametricFamilyConfig]:
    """Generate a dataset of parametric graph configurations.

    Creates a continuous structural distribution by varying parameters
    within each family.
    """
    if n_nodes_options is None:
        n_nodes_options = [10, 15, 20, 25, 30]

    if families is None:
        families = [
            "er", "ba", "ws", "tree", "grid",
            "bipartite", "sbm", "geometric", "regular", "pl_cluster",
        ]

    configs: list[ParametricFamilyConfig] = []
    rng = np.random.RandomState(seed)
    cfg_idx = 0

    for fam in families:
        for _ in range(n_per_family):
            n = int(rng.choice(n_nodes_options))
            params: dict[str, Any] = {}

            if fam == "er":
                params["p"] = float(rng.uniform(0.1, 0.5))
            elif fam == "ba":
                params["m"] = int(rng.choice([1, 2, 3, 4]))
            elif fam == "ws":
                params["k"] = int(rng.choice([2, 4, 6]))
                params["beta"] = float(rng.uniform(0.0, 0.5))
            elif fam == "tree":
                params["branching"] = int(rng.choice([2, 3, 4]))
            elif fam == "grid":
                dims = [(3, 4), (4, 5), (5, 5), (4, 6), (3, 7)]
                params["dims"] = dims[int(rng.randint(0, len(dims)))]
            elif fam == "bipartite":
                params["ratio"] = float(rng.uniform(0.3, 0.7))
            elif fam == "sbm":
                params["n_communities"] = int(rng.choice([2, 3, 4]))
                params["mixing"] = float(rng.uniform(0.05, 0.3))
            elif fam == "geometric":
                params["radius"] = float(rng.uniform(0.3, 0.6))
            elif fam == "regular":
                params["degree"] = int(rng.choice([2, 3, 4]))
            elif fam == "pl_cluster":
                params["m"] = int(rng.choice([1, 2, 3]))
                params["p"] = float(rng.uniform(0.1, 0.5))

            configs.append(ParametricFamilyConfig(
                name=f"{fam}_{cfg_idx}",
                generator=fam,
                n_nodes=n,
                params=params,
                seed=int(rng.randint(0, 100000)),
            ))
            cfg_idx += 1

    return configs


# ---------------------------------------------------------------------------
# TEST-C: held-out generators.
# ---------------------------------------------------------------------------

def generate_test_c_configs(
    *,
    n_per_family: int = 5,
    seed: int = 999,
) -> list[ParametricFamilyConfig]:
    """Generate TEST-C configurations from different generators.

    TEST-C uses different parameter ranges and some entirely different
    generators than training, to test generator extrapolation.
    """
    # Different parameter ranges + some new generators.
    test_families = [
        "er",  # same generator, different p range
        "ba",  # same generator, different m range
        "ws",  # same generator, extreme beta
        "geometric",  # same generator, different radius
        "sbm",  # same generator, more communities
    ]

    configs: list[ParametricFamilyConfig] = []
    rng = np.random.RandomState(seed)
    cfg_idx = 0

    n_nodes_options = [12, 18, 22, 28, 35]  # different sizes

    for fam in test_families:
        for _ in range(n_per_family):
            n = int(rng.choice(n_nodes_options))
            params: dict[str, Any] = {}

            if fam == "er":
                params["p"] = float(rng.uniform(0.05, 0.15))  # sparser
            elif fam == "ba":
                params["m"] = int(rng.choice([5, 6, 7]))  # higher m
            elif fam == "ws":
                params["k"] = int(rng.choice([8, 10]))  # higher k
                params["beta"] = float(rng.uniform(0.7, 1.0))  # high rewiring
            elif fam == "geometric":
                params["radius"] = float(rng.uniform(0.6, 0.9))  # denser
            elif fam == "sbm":
                params["n_communities"] = int(rng.choice([5, 6, 7]))  # more communities
                params["mixing"] = float(rng.uniform(0.3, 0.5))  # more mixing

            configs.append(ParametricFamilyConfig(
                name=f"test_c_{fam}_{cfg_idx}",
                generator=fam,
                n_nodes=n,
                params=params,
                seed=int(rng.randint(0, 100000)),
            ))
            cfg_idx += 1

    return configs
