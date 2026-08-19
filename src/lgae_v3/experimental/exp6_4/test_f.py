"""TEST-F: Unseen delayed-value mechanisms for exp6.4.

Unlike TEST-E (unseen graph generators), TEST-F uses unseen
delayed-value MECHANISMS — different non-additive utility functions
that create delayed value through different structural properties.

TEST-F mechanisms:
1. connectivity_threshold: bonus when n_components <= threshold
2. redundancy_threshold: bonus when min degree >= threshold
3. spectral_gap_threshold: bonus when spectral gap >= threshold
4. community_mixing_threshold: bonus when modularity <= threshold
5. hub_load_threshold: bonus when max degree <= threshold
6. minimum_path_threshold: bonus when diameter <= threshold
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import torch

from ...types import GraphBuffers
from .structural_features import compute_component_info
from ..exp6_3.split_utility import compute_additive_utility


@dataclass(frozen=True, slots=True)
class TestFConfig:
    name: str
    mechanism: str
    n_nodes: int
    n_components: int
    component_sizes: list[int]
    lambda_bonus: float
    threshold: int
    latent_seed: int = 42
    cluster_spacing: float = 1.0


# ---------------------------------------------------------------------------
# Unseen delayed-value mechanisms
# ---------------------------------------------------------------------------

def utility_redundancy_threshold(graph: GraphBuffers, z: torch.Tensor,
                                  lambda_bonus: float = 25.0,
                                  threshold: int = 2) -> float:
    """Bonus when enough nodes have degree >= threshold.

    U = U_additive + lambda * max(0, n_nodes_with_degree_ge_threshold - target_count)

    Delayed value: adding edges to low-degree nodes has negative immediate
    additive utility (cross-cluster) but increases the count of nodes
    meeting the degree threshold. The bonus triggers when enough nodes
    reach the threshold, creating a delayed-value structure.
    """
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1
    n_above = int(np.sum(degrees >= threshold))
    # Target: at least 60% of nodes with degree >= threshold.
    target = int(n * 0.6)
    bonus = lambda_bonus * max(0, n_above - target)
    return u_add + bonus


def utility_hub_load_threshold(graph: GraphBuffers, z: torch.Tensor,
                                lambda_bonus: float = 25.0,
                                threshold: int = 5) -> float:
    """Bonus when degree variance is low (balanced graph).

    U = U_additive + lambda * max(0, target_variance - degree_variance)

    Delayed value: adding edges to low-degree nodes has negative immediate
    additive utility (cross-cluster) but reduces degree variance, moving
    toward a balanced graph. The bonus triggers when variance drops below
    the threshold, creating delayed value from balance-promoting actions.
    """
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1
    mean_deg = float(np.mean(degrees))
    var_deg = float(np.var(degrees))
    # Target: variance below threshold/n (normalized).
    target_var = threshold / max(n, 1)
    bonus = lambda_bonus * max(0, target_var - var_deg) * n
    return u_add + bonus


def utility_spectral_gap_threshold(graph: GraphBuffers, z: torch.Tensor,
                                    lambda_bonus: float = 20.0,
                                    threshold: float = 0.5) -> float:
    """Bonus when spectral gap >= threshold.

    U = U_additive + lambda * max(0, spectral_gap - threshold) * 10

    Delayed value: adding strategic edges increases spectral gap.
    """
    u_add = compute_additive_utility(graph, z)
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
        sorted_eg = np.sort(eigenvalues)
        spectral_gap = float(sorted_eg[-1] - sorted_eg[-2]) if n > 1 else 0.0
    except Exception:
        spectral_gap = 0.0
    bonus = lambda_bonus * max(0, spectral_gap - threshold) * 10
    return u_add + bonus


def utility_connectivity_threshold_f(graph: GraphBuffers, z: torch.Tensor,
                                      lambda_bonus: float = 30.0,
                                      threshold: int = 1) -> float:
    """Same threshold connectivity as exp6.3, but used as TEST-F baseline."""
    from ..exp6_3.split_utility import compute_bonus
    u_add = compute_additive_utility(graph, z)
    u_bonus = compute_bonus(graph, z, lambda_bonus, threshold)
    return u_add + u_bonus


MECHANISMS: dict[str, Callable] = {
    "connectivity_threshold": utility_connectivity_threshold_f,
    "redundancy_threshold": utility_redundancy_threshold,
    "hub_load_threshold": utility_hub_load_threshold,
    "spectral_gap_threshold": utility_spectral_gap_threshold,
}


def make_test_f_utility(mechanism: str, lambda_bonus: float = 25.0,
                        threshold: int = 2) -> Callable:
    """Create a utility function for a TEST-F mechanism."""
    fn = MECHANISMS[mechanism]
    return lambda g, z: fn(g, z, lambda_bonus=lambda_bonus, threshold=threshold)


def generate_test_f_configs(*, n_per_mechanism: int = 3, seed: int = 99999) -> list[TestFConfig]:
    """Generate TEST-F configurations."""
    rng = np.random.RandomState(seed)
    configs: list[TestFConfig] = []
    mechanisms = list(MECHANISMS.keys())

    n_options = [15, 20, 25, 30, 35]  # Larger than training range
    n_comp_options = [3, 4, 5, 6]
    lambda_options = [30.0, 35.0, 40.0, 45.0, 50.0]  # Higher lambda for larger graphs
    threshold_options = [1]  # Keep threshold=1 for clear delayed value

    for mech in mechanisms:
        for _ in range(n_per_mechanism):
            n = int(rng.choice(n_options))
            n_comp = int(rng.choice(n_comp_options))
            n_comp = min(n_comp, n // 3)
            n_comp = max(n_comp, 3)  # At least 3 components

            # Component sizes.
            min_size = max(2, n // (n_comp * 3))
            sizes = []
            remaining = n
            for j in range(n_comp):
                if j == n_comp - 1:
                    sizes.append(remaining)
                else:
                    s = int(rng.randint(min_size, max(min_size + 1, remaining // (n_comp - j) * 2)))
                    s = min(s, remaining - min_size * (n_comp - j - 1))
                    sizes.append(s)
                    remaining -= s

            configs.append(TestFConfig(
                name=f"test_f_{mech}_{len(configs)}",
                mechanism=mech,
                n_nodes=n,
                n_components=n_comp,
                component_sizes=sizes,
                lambda_bonus=float(rng.choice(lambda_options)),
                threshold=int(rng.choice(threshold_options)),
                latent_seed=int(rng.randint(0, 100000)),
                cluster_spacing=float(rng.uniform(0.5, 1.5)),
            ))

    return configs


def generate_test_f_graph(config: TestFConfig) -> tuple[GraphBuffers, torch.Tensor, list[tuple[int, int]]]:
    """Generate a TEST-F graph with disconnected components."""
    from .procedural_tasks import make_procedural_graph
    from .procedural_tasks import ProceduralTaskConfig

    proc_config = ProceduralTaskConfig(
        n_nodes=config.n_nodes,
        n_components=config.n_components,
        component_sizes=list(config.component_sizes),
        latent_dim=4,
        latent_seed=config.latent_seed,
        cluster_spacing=config.cluster_spacing,
        lambda_conn=config.lambda_bonus,
        threshold=config.threshold,
    )
    return make_procedural_graph(proc_config)
