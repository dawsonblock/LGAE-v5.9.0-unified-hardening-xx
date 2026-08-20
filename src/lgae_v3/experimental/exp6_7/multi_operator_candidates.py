"""Multi-operator candidate generation for exp6.7.

Generates candidates from 4 mutation types:
  - ADD_EDGE: add a new edge
  - REMOVE_EDGE: remove an existing edge
  - REWEIGHT_EDGE: change an edge's weight
  - EDGE_SWAP: remove one edge and add another

This enables objectives that naturally require different operators
(e.g., hub_load needs REMOVE_EDGE, spectral gap benefits from SWAP).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_4.structural_features import compute_component_info
from ..exp6_5.multi_mechanism_data import (
    MechanismTaskConfig, _make_graph_from_config,
    generate_mechanism_task_configs,
)
from ..exp6_4.procedural_tasks import ProceduralTaskConfig, generate_candidates
from ..exp6_5.observable_features import extract_observable_features, OBSERVABLE_FEATURE_DIM
from ..exp6_3.exact_mpc import apply_action, exact_mpc
from ..exp6_3.split_utility import compute_additive_utility
from ..exp6_4.test_f import make_test_f_utility


MUTATION_TYPES = ["add_edge", "remove_edge", "reweight_edge", "edge_swap"]


def _get_existing_edges(graph: GraphBuffers, n: int) -> list[tuple[int, int]]:
    """Get list of existing edges."""
    edges = []
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n and d < n:
                edges.append((s, d))
    return edges


def _get_non_edges(graph: GraphBuffers, n: int) -> list[tuple[int, int]]:
    """Get list of non-existing node pairs."""
    existing = set()
    for s, d in _get_existing_edges(graph, n):
        existing.add((min(s, d), max(s, d)))
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in existing:
                non_edges.append((u, v))
    return non_edges


def generate_multi_operator_candidates(
    graph: GraphBuffers,
    z: torch.Tensor,
    config: MechanismTaskConfig,
    *,
    n_add: int = 4,
    n_remove: int = 3,
    n_reweight: int = 2,
    n_swap: int = 2,
    rng: random.Random | None = None,
) -> list[tuple[str, int, int, dict]]:
    """Generate candidates from all 4 mutation types."""
    if rng is None:
        rng = random.Random(config.seed)
    n = int(graph.num_nodes)
    candidates: list[tuple[str, int, int, dict]] = []

    # ADD_EDGE: cross-component and within-component.
    comp_info = compute_component_info(graph, n)
    cross_pairs = []
    within_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if comp_info.component_ids[u] != comp_info.component_ids[v]:
                cross_pairs.append((u, v))
            else:
                within_pairs.append((u, v))
    rng.shuffle(cross_pairs)
    rng.shuffle(within_pairs)
    for u, v in cross_pairs[:n_add // 2]:
        candidates.append(("add_edge", u, v, {"weight": 1.0}))
    for u, v in within_pairs[:n_add - n_add // 2]:
        candidates.append(("add_edge", u, v, {"weight": 1.0}))

    # REMOVE_EDGE: remove existing edges.
    existing = _get_existing_edges(graph, n)
    rng.shuffle(existing)
    for u, v in existing[:n_remove]:
        candidates.append(("remove_edge", u, v, {}))

    # REWEIGHT_EDGE: reweight existing edges up and down.
    for u, v in existing[:n_reweight]:
        candidates.append(("reweight_edge", u, v, {"factor": 2.0}))
        candidates.append(("reweight_edge", u, v, {"factor": 0.5}))

    # EDGE_SWAP: remove (u,v), add (u,w).
    non_edges = _get_non_edges(graph, n)
    rng.shuffle(non_edges)
    swap_count = 0
    for u, v in existing:
        if swap_count >= n_swap:
            break
        # Find a new target for u that's not v and not already connected.
        for w, x in non_edges:
            if w == u and x != v:
                candidates.append(("edge_swap", u, v, {"new_target": x, "weight": 1.0}))
                swap_count += 1
                break
            elif x == u and w != v:
                candidates.append(("edge_swap", u, v, {"new_target": w, "weight": 1.0}))
                swap_count += 1
                break

    return candidates


def generate_multi_operator_training_data(
    *,
    n_tasks_per_mechanism: int = 200,
    seed: int = 42,
    mechanisms: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Generate training data with multi-operator candidates.

    Returns dict with X, y_residual, y_effects, mechanism labels.
    """
    from ...runtime.analytical_utility import AnalyticalUtilityOracle
    from ..exp6_3.exact_mpc import apply_action as apply_act
    from ..exp6_6.objective_spec import get_objective_spec
    from ..exp6_6.causal_effect_model import compute_effect_labels
    from .extended_effects import compute_extended_effect_labels

    oracle = AnalyticalUtilityOracle()

    from ..exp6_5.multi_mechanism_data import MECHANISM_NAMES
    if mechanisms is None:
        mechanisms = MECHANISM_NAMES

    all_X: list[np.ndarray] = []
    all_y_residual: list[float] = []
    all_y_effects: list[np.ndarray] = []
    all_mechanism: list[str] = []

    for mech_idx, mechanism in enumerate(mechanisms):
        configs = generate_mechanism_task_configs(
            mechanism=mechanism,
            n_tasks=n_tasks_per_mechanism,
            seed=seed + mech_idx * 1000,
        )
        obj_spec = get_objective_spec(mechanism)
        utility_fn = make_test_f_utility(mechanism, obj_spec.magnitude, int(obj_spec.threshold))

        for config in configs:
            graph, z = _make_graph_from_config(config)
            candidates = generate_multi_operator_candidates(
                graph, z, config, rng=random.Random(config.seed),
            )
            if len(candidates) < 4:
                continue

            for action in candidates:
                x = extract_observable_features(graph, z, action, threshold=config.threshold, horizon=2)

                # Exact future residual.
                mt, u, v, params = action
                # Normalize reweight_edge to reweight_up/down.
                if mt == "reweight_edge":
                    factor = params.get("factor", 2.0)
                    mt_oracle = "reweight_up" if factor > 1 else "reweight_down"
                else:
                    mt_oracle = mt

                try:
                    delta_add = oracle.delta_for_mutation(graph, z, mt_oracle, u, v, params)
                except Exception:
                    delta_add = 0.0

                next_graph = apply_act(graph, action)
                exact_h1 = exact_mpc(next_graph, z, candidates, utility_fn, horizon=1, gamma=0.9)
                q_h2 = delta_add + 0.9 * exact_h1.total_value
                future_residual = q_h2 - delta_add

                # Extended effect labels (7 heads).
                effects = compute_extended_effect_labels(graph, z, action)

                all_X.append(x)
                all_y_residual.append(future_residual)
                all_y_effects.append(effects.to_array())
                all_mechanism.append(mechanism)

    X = np.array(all_X) if all_X else np.array([]).reshape(0, OBSERVABLE_FEATURE_DIM)
    y_residual = np.array(all_y_residual)
    y_effects = np.array(all_y_effects)
    mechanism_labels = np.array(all_mechanism)

    return {
        "X": X,
        "y_residual": y_residual,
        "y_effects": y_effects,
        "mechanism": mechanism_labels,
    }
