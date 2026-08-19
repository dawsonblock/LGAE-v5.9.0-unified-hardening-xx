"""Scaling benchmark for exp6.5.

Tests how the model-assisted planner scales with:
- candidate set size: 25, 50, 100, 250, 500, 1000
- graph size: 20, 50, 100, 250, 500 nodes

Measures actual wall-clock time, not just node expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time
import numpy as np
import torch

from ...types import GraphBuffers, make_graph_buffers
from ..exp6_3.exact_mpc import exact_mpc, greedy_one_step, apply_action
from ..exp6_3.split_utility import make_total_utility_fn
from ..exp6_4.procedural_tasks import (
    ProceduralTaskConfig, make_procedural_graph, generate_candidates,
)
from .adaptive_beam import adaptive_beam_search


@dataclass
class ScalingConfig:
    n_nodes: int = 20
    n_candidates: int = 25
    n_components: int = 3
    lambda_conn: float = 30.0
    threshold: int = 1
    seed: int = 42


@dataclass
class ScalingResult:
    config: ScalingConfig
    exact_mpc_time: float = 0.0
    exact_mpc_nodes: int = 0
    model_assisted_time: float = 0.0
    model_assisted_nodes: int = 0
    greedy_time: float = 0.0
    speedup: float = 0.0
    search_savings: float = 0.0
    first_action_agreement: bool = False
    regret: float = 0.0


def _generate_large_candidate_set(
    graph: GraphBuffers, z: torch.Tensor, n: int, n_candidates: int,
    seed: int,
) -> list[tuple[str, int, int, dict]]:
    """Generate a large candidate set for scaling tests."""
    import random
    rng = random.Random(seed)

    from ..exp6_4.structural_features import compute_component_info
    comp_info = compute_component_info(graph, n)
    comp_ids = comp_info.component_ids

    # Find all possible cross-component and within-component pairs.
    cross_pairs: list[tuple[int, int]] = []
    within_pairs: list[tuple[int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            if comp_ids[u] != comp_ids[v]:
                cross_pairs.append((u, v))
            else:
                within_pairs.append((u, v))

    candidates: list[tuple[str, int, int, dict]] = []

    # Mix of cross and within.
    n_cross = min(n_candidates // 2, len(cross_pairs))
    n_within = min(n_candidates - n_cross, len(within_pairs))

    if n_cross > 0:
        sampled = rng.sample(cross_pairs, min(n_cross, len(cross_pairs)))
        for u, v in sampled:
            candidates.append(("add_edge", u, v, {"weight": 1.0}))

    if n_within > 0:
        sampled = rng.sample(within_pairs, min(n_within, len(within_pairs)))
        for u, v in sampled:
            candidates.append(("add_edge", u, v, {"weight": 1.0}))

    # Pad with random cross-component if needed.
    while len(candidates) < n_candidates and cross_pairs:
        u, v = rng.choice(cross_pairs)
        candidates.append(("add_edge", u, v, {"weight": 1.0}))

    return candidates[:n_candidates]


def run_scaling_benchmark(
    model,
    *,
    configs: list[ScalingConfig] | None = None,
    gamma: float = 0.9,
) -> list[ScalingResult]:
    """Run scaling benchmark with given model.

    Measures wall-clock time for exact MPC vs model-assisted search.
    """
    if configs is None:
        configs = [
            ScalingConfig(n_nodes=20, n_candidates=25, seed=100),
            ScalingConfig(n_nodes=20, n_candidates=50, seed=101),
            ScalingConfig(n_nodes=50, n_candidates=100, seed=102),
            ScalingConfig(n_nodes=50, n_candidates=250, seed=103),
            ScalingConfig(n_nodes=100, n_candidates=250, seed=104),
            ScalingConfig(n_nodes=100, n_candidates=500, seed=105),
        ]

    results: list[ScalingResult] = []

    for config in configs:
        result = ScalingResult(config=config)

        # Generate graph.
        sizes = []
        n = config.n_nodes
        n_comp = config.n_components
        min_size = max(2, n // (n_comp * 3))
        remaining = n
        for j in range(n_comp):
            if j == n_comp - 1:
                sizes.append(remaining)
            else:
                s = max(min_size, remaining // (n_comp - j))
                sizes.append(s)
                remaining -= s

        proc_config = ProceduralTaskConfig(
            n_nodes=n,
            n_components=n_comp,
            component_sizes=sizes,
            latent_dim=4,
            latent_seed=config.seed,
            cluster_spacing=1.0,
            lambda_conn=config.lambda_conn,
            threshold=config.threshold,
            n_candidates=config.n_candidates,
            n_within_candidates=config.n_candidates // 2,
            seed=config.seed,
        )

        graph, z, _ = make_procedural_graph(proc_config)
        candidates = _generate_large_candidate_set(
            graph, z, n, config.n_candidates, config.seed,
        )

        if len(candidates) < 4:
            continue

        utility_fn = make_total_utility_fn(config.lambda_conn, config.threshold)

        # Exact MPC H=2.
        t0 = time.time()
        exact = exact_mpc(graph, z, candidates, utility_fn, horizon=2, gamma=gamma)
        result.exact_mpc_time = time.time() - t0
        result.exact_mpc_nodes = exact.nodes_expanded

        # Greedy.
        t0 = time.time()
        greedy = greedy_one_step(graph, z, candidates, utility_fn)
        result.greedy_time = time.time() - t0

        # Model-assisted adaptive beam search.
        t0 = time.time()
        bs = adaptive_beam_search(
            graph, z, candidates, model,
            horizon=2, gamma=gamma,
            min_beam_width=2, max_beam_width=min(10, len(candidates) // 2),
            threshold=config.threshold,
        )
        result.model_assisted_time = time.time() - t0
        result.model_assisted_nodes = bs.nodes_expanded

        # Metrics.
        result.speedup = result.exact_mpc_time / max(result.model_assisted_time, 1e-6)
        result.search_savings = 1.0 - result.model_assisted_nodes / max(result.exact_mpc_nodes, 1)
        result.first_action_agreement = bs.first_action == exact.first_action

        # Regret.
        exact_key = f"{exact.first_action[0]}_{exact.first_action[1]}_{exact.first_action[2]}"
        model_key = f"{bs.first_action[0]}_{bs.first_action[1]}_{bs.first_action[2]}"
        exact_val = exact.all_first_action_values.get(exact_key, exact.total_value)
        model_val = exact.all_first_action_values.get(model_key, bs.total_value)
        result.regret = float(exact_val - model_val)

        results.append(result)

    return results
