"""Procedural delayed-value task generation for exp6.4.

Instead of 5 hand-designed tasks, generate thousands of random
delayed-value scenarios with varied:
- graph size (10-40 nodes)
- component count (2-6)
- component size imbalance
- latent spacing
- bonus threshold
- lambda
- candidate count
- mutation types
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random
import numpy as np
import torch

from ...types import GraphBuffers, make_graph_buffers
from ..exp6_3.split_utility import compute_total_utility, make_total_utility_fn
from .structural_features import compute_component_info


@dataclass
class ProceduralTaskConfig:
    """Configuration for a procedurally generated delayed-value task."""
    n_nodes: int = 15
    n_components: int = 3
    component_sizes: list[int] = field(default_factory=list)
    latent_dim: int = 4
    latent_seed: int = 42
    cluster_spacing: float = 1.0
    lambda_conn: float = 30.0
    threshold: int = 1
    n_candidates: int = 8
    n_within_candidates: int = 4
    seed: int = 42


def generate_procedural_tasks(
    *,
    n_tasks: int = 200,
    seed: int = 42,
    n_nodes_range: tuple[int, int] = (10, 30),
    n_components_range: tuple[int, int] = (3, 5),
    lambda_range: tuple[float, float] = (25.0, 45.0),
    threshold_range: tuple[int, int] = (1, 1),
    spacing_range: tuple[float, float] = (0.8, 1.2),
) -> list[ProceduralTaskConfig]:
    """Generate random procedural task configurations.

    Constrained to create delayed-value scenarios:
    - n_components >= 3 (so at least 2 bridges needed for threshold=1)
    - lambda large enough relative to cluster spacing
    - threshold=1 (so bonus requires full connectivity)
    """
    rng = random.Random(seed)
    configs: list[ProceduralTaskConfig] = []

    for i in range(n_tasks):
        n = rng.randint(*n_nodes_range)
        n_comp = rng.randint(*n_components_range)
        n_comp = min(n_comp, n // 3)  # ensure components have at least 3 nodes
        n_comp = max(n_comp, 3)  # at least 3 components for delayed value

        # Generate component sizes that sum to n.
        min_size = max(2, n // (n_comp * 3))
        sizes = []
        remaining = n
        for j in range(n_comp):
            if j == n_comp - 1:
                sizes.append(remaining)
            else:
                s = rng.randint(min_size, max(min_size, remaining // (n_comp - j) * 2))
                s = min(s, remaining - min_size * (n_comp - j - 1))
                sizes.append(s)
                remaining -= s

        spacing = rng.uniform(*spacing_range)
        # Ensure lambda > 2 * cross_d² (≈ 2 * 8 * spacing² for dim=4)
        min_lambda = 2 * 8 * spacing * spacing + 5
        lambda_conn = max(rng.uniform(*lambda_range), min_lambda)

        configs.append(ProceduralTaskConfig(
            n_nodes=n,
            n_components=n_comp,
            component_sizes=sizes,
            latent_dim=4,
            latent_seed=rng.randint(0, 100000),
            cluster_spacing=spacing,
            lambda_conn=lambda_conn,
            threshold=1,  # always threshold=1 for clear delayed value
            n_candidates=rng.randint(6, 12),
            n_within_candidates=rng.randint(2, 6),
            seed=rng.randint(0, 100000),
        ))

    return configs


def make_procedural_graph(config: ProceduralTaskConfig) -> tuple[GraphBuffers, torch.Tensor, list[tuple[int, int]]]:
    """Create a graph with disconnected components and clustered latent states."""
    rng = torch.Generator().manual_seed(config.latent_seed)
    n = config.n_nodes
    sizes = config.component_sizes
    n_comp = len(sizes)

    # Generate edges within each component (path graph).
    edges: list[tuple[int, int]] = []
    offset = 0
    for size in sizes:
        for j in range(size - 1):
            edges.append((offset + j, offset + j + 1))
        offset += size

    # Generate clustered latent states.
    z = torch.randn(n, config.latent_dim, generator=rng) * 0.3
    centers = torch.randn(n_comp, config.latent_dim, generator=rng) * config.cluster_spacing
    offset = 0
    for ci, size in enumerate(sizes):
        for j in range(size):
            z[offset + j] += centers[ci]
        offset += size

    graph = make_graph_buffers(num_nodes=n, edges=edges, capacity=max(len(edges) * 3, n * 3))
    return graph, z, edges


def generate_candidates(
    config: ProceduralTaskConfig,
    graph: GraphBuffers,
    z: torch.Tensor,
) -> list[tuple[str, int, int, dict]]:
    """Generate candidate actions: cross-component bridges and within-component edges."""
    import random as pyrandom
    rng = pyrandom.Random(config.seed)
    n = config.n_nodes

    # Compute component assignments.
    comp_info = compute_component_info(graph, n)
    comp_ids = comp_info.component_ids

    # Find cross-component pairs (bridges).
    cross_pairs: list[tuple[int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            if comp_ids[u] != comp_ids[v]:
                cross_pairs.append((u, v))

    # Find within-component pairs.
    within_pairs: list[tuple[int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            if comp_ids[u] == comp_ids[v]:
                # Check if edge already exists.
                exists = False
                valid = graph.valid.bool()
                for i in range(graph.src.shape[0]):
                    if valid[i]:
                        s, d = int(graph.src[i].item()), int(graph.dst[i].item())
                        if (s == u and d == v) or (s == v and d == u):
                            exists = True
                            break
                if not exists:
                    within_pairs.append((u, v))

    # Sample candidates.
    candidates: list[tuple[str, int, int, dict]] = []

    # Cross-component bridges.
    n_cross = min(config.n_candidates - config.n_within_candidates, len(cross_pairs))
    if n_cross > 0:
        sampled = rng.sample(cross_pairs, min(n_cross, len(cross_pairs)))
        for u, v in sampled:
            candidates.append(("add_edge", u, v, {"weight": 1.0}))

    # Within-component edges.
    n_within = min(config.n_within_candidates, len(within_pairs))
    if n_within > 0:
        sampled = rng.sample(within_pairs, min(n_within, len(within_pairs)))
        for u, v in sampled:
            candidates.append(("add_edge", u, v, {"weight": 1.0}))

    # Pad if needed.
    while len(candidates) < config.n_candidates and cross_pairs:
        u, v = rng.choice(cross_pairs)
        candidates.append(("add_edge", u, v, {"weight": 1.0}))

    return candidates


def generate_procedural_training_data(
    *,
    n_tasks: int = 500,
    seed: int = 42,
    horizons: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Generate large-scale training data from procedural tasks.

    Returns dict with:
    - X: feature matrix (n_samples x n_features)
    - y_bonus: exact bonus of S'
    - y_threshold: binary threshold reached label
    - y_delta_comp: delta n_components
    - y_q_h2: exact Q_H2 (if horizons includes 2)
    """
    from .structural_features import extract_structural_features, compute_component_info
    from .causal_targets import compute_causal_targets
    from ..exp6_3.split_utility import compute_bonus
    from ..exp6_3.exact_mpc import apply_action, exact_mpc
    import random as pyrandom

    if horizons is None:
        horizons = [2]

    configs = generate_procedural_tasks(n_tasks=n_tasks, seed=seed)

    all_X: list[np.ndarray] = []
    all_y_bonus: list[float] = []
    all_y_threshold: list[int] = []
    all_y_delta_comp: list[int] = []

    for config in configs:
        graph, z, edges = make_procedural_graph(config)
        candidates = generate_candidates(config, graph, z)

        if not candidates:
            continue

        utility_fn = make_total_utility_fn(config.lambda_conn, config.threshold)

        # Generate samples at multiple rollout depths so the model
        # sees states near the threshold (not just initial states).
        rng = pyrandom.Random(config.seed)

        # Depth 0: initial state + each action.
        # Target: exact future residual = Q_H(S,a) - delta_U_add(S,a)
        # This captures the FUTURE bonus over the full horizon, not just 1-step.
        from ...runtime.analytical_utility import AnalyticalUtilityOracle
        oracle = AnalyticalUtilityOracle()

        for action in candidates:
            x = extract_structural_features(
                graph, z, action,
                threshold=config.threshold, horizon=2,
            )

            # Exact additive delta.
            mt, u, v, params = action
            try:
                delta_add = oracle.delta_for_mutation(graph, z, mt, u, v, params)
            except Exception:
                delta_add = 0.0

            # Exact Q_H2 = best future total utility.
            next_graph = apply_action(graph, action)
            exact_h1 = exact_mpc(next_graph, z, candidates, utility_fn, horizon=1, gamma=0.9)
            q_h2 = delta_add + 0.9 * exact_h1.total_value

            # Future residual = Q_H2 - delta_add = gamma * V(S')
            future_residual = q_h2 - delta_add

            # Also compute 1-step bonus for reference.
            target = compute_causal_targets(
                graph, z, action,
                lambda_conn=config.lambda_conn,
                threshold=config.threshold,
                horizon=2,
                available_actions=candidates,
                utility_fn=utility_fn,
            )

            all_X.append(x)
            all_y_bonus.append(future_residual)  # Train on future residual!
            all_y_threshold.append(int(target.threshold_reached))
            all_y_delta_comp.append(target.delta_n_components)

        # Depth 1-2: apply random bridge actions, then sample from S'.
        current = graph
        for step in range(2):
            # Apply a random cross-component bridge.
            cross_actions = [a for a in candidates
                             if a[0] == "add_edge" and a[1] < config.n_nodes and a[2] < config.n_nodes]
            if not cross_actions:
                break
            comp_info = compute_component_info(current, config.n_nodes)
            bridge_actions = []
            for a in cross_actions:
                _, u, v, _ = a
                if u < config.n_nodes and v < config.n_nodes:
                    if comp_info.component_ids[u] != comp_info.component_ids[v]:
                        bridge_actions.append(a)
            if not bridge_actions:
                break
            action = rng.choice(bridge_actions)
            current = apply_action(current, action)

            # Generate candidates from this intermediate state.
            updated_candidates = generate_candidates(config, current, z)
            if not updated_candidates:
                continue

            for action2 in updated_candidates:
                x = extract_structural_features(
                    current, z, action2,
                    threshold=config.threshold, horizon=2,
                )

                # Exact future residual from this intermediate state.
                mt2, u2, v2, params2 = action2
                try:
                    delta_add2 = oracle.delta_for_mutation(current, z, mt2, u2, v2, params2)
                except Exception:
                    delta_add2 = 0.0

                next_graph2 = apply_action(current, action2)
                exact_h1_2 = exact_mpc(next_graph2, z, updated_candidates, utility_fn, horizon=1, gamma=0.9)
                q_h2_2 = delta_add2 + 0.9 * exact_h1_2.total_value
                future_residual2 = q_h2_2 - delta_add2

                target2 = compute_causal_targets(
                    current, z, action2,
                    lambda_conn=config.lambda_conn,
                    threshold=config.threshold,
                    horizon=2,
                    available_actions=updated_candidates,
                    utility_fn=utility_fn,
                )

                all_X.append(x)
                all_y_bonus.append(future_residual2)  # Train on future residual!
                all_y_threshold.append(int(target2.threshold_reached))
                all_y_delta_comp.append(target2.delta_n_components)

    X = np.array(all_X) if all_X else np.array([]).reshape(0, 0)
    y_bonus = np.array(all_y_bonus)
    y_threshold = np.array(all_y_threshold)
    y_delta_comp = np.array(all_y_delta_comp)

    return {
        "X": X,
        "y_bonus": y_bonus,
        "y_threshold": y_threshold,
        "y_delta_comp": y_delta_comp,
    }
