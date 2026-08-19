"""Multi-mechanism training data generator for exp6.5.

Generates training data from ALL 4 delayed-value mechanisms so the
model sees a distribution of mechanisms rather than only connectivity.

The mechanism label is NOT included in features. The model must
infer the mechanism from observables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import random
import numpy as np
import torch

from ...types import GraphBuffers, make_graph_buffers
from ..exp6_3.split_utility import compute_additive_utility, make_total_utility_fn
from ..exp6_3.exact_mpc import apply_action, exact_mpc
from ..exp6_4.structural_features import compute_component_info
from ..exp6_4.test_f import (
    MECHANISMS, make_test_f_utility,
    utility_connectivity_threshold_f,
    utility_redundancy_threshold,
    utility_hub_load_threshold,
    utility_spectral_gap_threshold,
)
from ..exp6_4.procedural_tasks import (
    ProceduralTaskConfig, make_procedural_graph, generate_candidates,
)
from .observable_features import extract_observable_features


MECHANISM_NAMES = list(MECHANISMS.keys())


@dataclass
class MechanismTaskConfig:
    """A task config with an associated mechanism."""
    mechanism: str
    n_nodes: int
    n_components: int
    component_sizes: list[int]
    lambda_bonus: float
    threshold: int
    latent_seed: int
    cluster_spacing: float
    n_candidates: int
    n_within_candidates: int
    seed: int


def _generate_component_sizes(n: int, n_comp: int, rng: random.Random) -> list[int]:
    """Generate component sizes that sum to n."""
    min_size = max(2, n // (n_comp * 3))
    sizes = []
    remaining = n
    for j in range(n_comp):
        if j == n_comp - 1:
            sizes.append(remaining)
        else:
            s = rng.randint(min_size, max(min_size + 1, remaining // (n_comp - j) * 2))
            s = min(s, remaining - min_size * (n_comp - j - 1))
            sizes.append(s)
            remaining -= s
    return sizes


def generate_mechanism_task_configs(
    *,
    mechanism: str,
    n_tasks: int,
    seed: int,
    n_nodes_range: tuple[int, int] = (12, 25),
    n_components_range: tuple[int, int] = (3, 5),
    lambda_range: tuple[float, float] = (25.0, 45.0),
    threshold_range: tuple[int, int] = (1, 2),
    spacing_range: tuple[float, float] = (0.8, 1.2),
) -> list[MechanismTaskConfig]:
    """Generate task configs for a specific mechanism."""
    rng = random.Random(seed)
    configs: list[MechanismTaskConfig] = []

    for _ in range(n_tasks):
        n = rng.randint(*n_nodes_range)
        n_comp = rng.randint(*n_components_range)
        n_comp = min(n_comp, n // 3)
        n_comp = max(n_comp, 3)

        sizes = _generate_component_sizes(n, n_comp, rng)
        spacing = rng.uniform(*spacing_range)
        min_lambda = 2 * 8 * spacing * spacing + 5
        lambda_bonus = max(rng.uniform(*lambda_range), min_lambda)

        configs.append(MechanismTaskConfig(
            mechanism=mechanism,
            n_nodes=n,
            n_components=n_comp,
            component_sizes=sizes,
            lambda_bonus=lambda_bonus,
            threshold=rng.randint(*threshold_range),
            latent_seed=rng.randint(0, 100000),
            cluster_spacing=spacing,
            n_candidates=rng.randint(6, 12),
            n_within_candidates=rng.randint(2, 6),
            seed=rng.randint(0, 100000),
        ))

    return configs


def _make_graph_from_config(config: MechanismTaskConfig) -> tuple[GraphBuffers, torch.Tensor]:
    """Create graph and latent states from a mechanism task config."""
    proc_config = ProceduralTaskConfig(
        n_nodes=config.n_nodes,
        n_components=config.n_components,
        component_sizes=list(config.component_sizes),
        latent_dim=4,
        latent_seed=config.latent_seed,
        cluster_spacing=config.cluster_spacing,
        lambda_conn=config.lambda_bonus,
        threshold=config.threshold,
        n_candidates=config.n_candidates,
        n_within_candidates=config.n_within_candidates,
        seed=config.seed,
    )
    graph, z, _ = make_procedural_graph(proc_config)
    return graph, z


def generate_multi_mechanism_training_data(
    *,
    n_tasks_per_mechanism: int = 150,
    seed: int = 42,
    mechanisms: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Generate training data from ALL mechanisms.

    Returns dict with:
    - X: observable features (n_samples x OBSERVABLE_FEATURE_DIM)
    - y_residual: future residual = Q_H2 - delta_U_additive
    - y_bonus: 1-step exact bonus
    - y_threshold: binary threshold reached
    - mechanism: mechanism label (for split only, NOT for model)
    """
    from ...runtime.analytical_utility import AnalyticalUtilityOracle
    oracle = AnalyticalUtilityOracle()

    if mechanisms is None:
        mechanisms = MECHANISM_NAMES

    all_X: list[np.ndarray] = []
    all_y_residual: list[float] = []
    all_y_bonus: list[float] = []
    all_y_threshold: list[int] = []
    all_mechanism: list[str] = []

    for mech_idx, mechanism in enumerate(mechanisms):
        configs = generate_mechanism_task_configs(
            mechanism=mechanism,
            n_tasks=n_tasks_per_mechanism,
            seed=seed + mech_idx * 1000,
        )

        utility_fn = make_test_f_utility(mechanism, lambda_bonus=30.0, threshold=1)

        for config in configs:
            graph, z = _make_graph_from_config(config)
            proc_config = ProceduralTaskConfig(
                n_nodes=config.n_nodes,
                n_components=config.n_components,
                component_sizes=list(config.component_sizes),
                latent_dim=4,
                latent_seed=config.latent_seed,
                cluster_spacing=config.cluster_spacing,
                lambda_conn=config.lambda_bonus,
                threshold=config.threshold,
                n_candidates=config.n_candidates,
                n_within_candidates=config.n_within_candidates,
                seed=config.seed,
            )
            candidates = generate_candidates(proc_config, graph, z)

            if not candidates:
                continue

            # Use the mechanism-specific utility.
            utility_fn = make_test_f_utility(mechanism, config.lambda_bonus, config.threshold)

            # Generate samples at multiple rollout depths.
            rng = random.Random(config.seed)

            # Depth 0: initial state + each action.
            for action in candidates:
                x = extract_observable_features(
                    graph, z, action,
                    threshold=config.threshold, horizon=2,
                )

                # Exact future residual.
                mt, u, v, params = action
                try:
                    delta_add = oracle.delta_for_mutation(graph, z, mt, u, v, params)
                except Exception:
                    delta_add = 0.0

                next_graph = apply_action(graph, action)
                exact_h1 = exact_mpc(next_graph, z, candidates, utility_fn, horizon=1, gamma=0.9)
                q_h2 = delta_add + 0.9 * exact_h1.total_value
                future_residual = q_h2 - delta_add

                # 1-step bonus.
                from ..exp6_4.causal_targets import compute_causal_targets
                target = compute_causal_targets(
                    graph, z, action,
                    lambda_conn=config.lambda_bonus,
                    threshold=config.threshold,
                )

                all_X.append(x)
                all_y_residual.append(future_residual)
                all_y_bonus.append(target.exact_bonus)
                all_y_threshold.append(int(target.threshold_reached))
                all_mechanism.append(mechanism)

            # Depth 1-2: apply random bridge, then sample.
            current = graph
            for step in range(2):
                comp_info = compute_component_info(current, config.n_nodes)
                bridge_actions = []
                for a in candidates:
                    _, u, v, _ = a
                    if u < config.n_nodes and v < config.n_nodes:
                        if comp_info.component_ids[u] != comp_info.component_ids[v]:
                            bridge_actions.append(a)
                if not bridge_actions:
                    break
                action = rng.choice(bridge_actions)
                current = apply_action(current, action)

                updated_candidates = generate_candidates(proc_config, current, z)
                if not updated_candidates:
                    continue

                for action2 in updated_candidates:
                    x = extract_observable_features(
                        current, z, action2,
                        threshold=config.threshold, horizon=2,
                    )

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
                        lambda_conn=config.lambda_bonus,
                        threshold=config.threshold,
                    )

                    all_X.append(x)
                    all_y_residual.append(future_residual2)
                    all_y_bonus.append(target2.exact_bonus)
                    all_y_threshold.append(int(target2.threshold_reached))
                    all_mechanism.append(mechanism)

    X = np.array(all_X) if all_X else np.array([]).reshape(0, 64)
    y_residual = np.array(all_y_residual)
    y_bonus = np.array(all_y_bonus)
    y_threshold = np.array(all_y_threshold)
    mechanism_labels = np.array(all_mechanism)

    return {
        "X": X,
        "y_residual": y_residual,
        "y_bonus": y_bonus,
        "y_threshold": y_threshold,
        "mechanism": mechanism_labels,
    }


def generate_mechanism_eval_tasks(
    *,
    mechanism: str,
    n_tasks: int = 30,
    seed: int = 999,
    n_nodes_range: tuple[int, int] = (15, 30),
) -> list[MechanismTaskConfig]:
    """Generate evaluation tasks for a specific mechanism."""
    return generate_mechanism_task_configs(
        mechanism=mechanism,
        n_tasks=n_tasks,
        seed=seed,
        n_nodes_range=n_nodes_range,
    )
