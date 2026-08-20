"""Graph-storing advantage records for exp6.8.5.

Unlike exp6.8.3/6.8.4, these records store the graph adjacency matrix
so that F4 (full structural) features can be computed at evaluation time.

This is the critical difference: F4 features require the graph, which
previous experiments didn't store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_3.exact_mpc import (
    exact_mpc, greedy_one_step, apply_action, apply_action_with_status,
    ActionIdentity,
)
from ..exp6_5.multi_mechanism_data import (
    generate_mechanism_task_configs, MechanismTaskConfig, _make_graph_from_config,
)
from ..exp6_6.objective_spec import get_objective_spec, ObjectiveSpec, encode_objective, OBJECTIVE_ENCODING_DIM
from ..exp6_7.multi_operator_candidates import generate_multi_operator_candidates
from ..exp6_8_1.split_state import SplitStructuralState
from ..exp6_8_3.advantage_dataset import compute_exact_q_h2, AdvantageRecord
from ..exp6_8_4.rich_features import extract_features_level, get_feature_dim


@dataclass
class GraphAdvantageRecord:
    """Advantage record with stored graph adjacency for F4 features."""
    state_id: int
    adjacency: np.ndarray  # (n, n) boolean adjacency matrix
    n_nodes: int
    z: torch.Tensor  # node features
    state_features: np.ndarray  # split state array
    objective_features: np.ndarray
    baseline_action: tuple
    learned_action: tuple
    baseline_action_id: ActionIdentity
    learned_action_id: ActionIdentity
    baseline_q: float
    learned_q: float
    advantage: float
    mechanism: str
    split: str
    config_seed: int  # for candidate regeneration if needed
    threshold: int
    lambda_bonus: float

    @property
    def is_beneficial(self) -> bool:
        return self.advantage > 0


def _graph_to_adjacency(graph: GraphBuffers, n_nodes: int) -> np.ndarray:
    """Convert GraphBuffers to dense adjacency matrix."""
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    src = graph.src.numpy() if hasattr(graph.src, 'numpy') else np.array(graph.src)
    dst = graph.dst.numpy() if hasattr(graph.dst, 'numpy') else np.array(graph.dst)
    for s, d in zip(src, dst):
        if s < n_nodes and d < n_nodes:
            adj[s, d] = True
            adj[d, s] = True
    return adj


def _adjacency_to_graph_buffers(adj: np.ndarray, n_nodes: int, capacity: int = 50) -> GraphBuffers:
    """Convert adjacency matrix back to GraphBuffers."""
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adj[i, j]:
                edges.append((i, j))
    if not edges:
        edges = [(0, 1)]
    return make_graph_buffers_safe(num_nodes=n_nodes, edges=edges, capacity=capacity)


def make_graph_buffers_safe(num_nodes: int, edges: list, capacity: int) -> GraphBuffers:
    """Safe graph construction."""
    from ...types import make_graph_buffers
    return make_graph_buffers(num_nodes=num_nodes, edges=edges, capacity=capacity)


def generate_graph_advantage_records(
    mechanism: str,
    n_tasks: int,
    seed: int,
    split: str,
) -> list[GraphAdvantageRecord]:
    """Generate advantage records with stored graph adjacency."""
    from ..exp6_4.test_f import make_test_f_utility

    obj_spec = get_objective_spec(mechanism)
    configs = generate_mechanism_task_configs(
        mechanism=mechanism, n_tasks=n_tasks, seed=seed,
    )

    records = []
    state_id = 0

    for config in configs:
        graph, z = _make_graph_from_config(config)
        candidates = generate_multi_operator_candidates(
            graph, z, config, rng=__import__("random").Random(config.seed),
        )
        if len(candidates) < 4:
            continue

        utility_fn = make_test_f_utility(
            mechanism, config.lambda_bonus, int(obj_spec.threshold),
        )

        # Baseline: greedy.
        greedy = greedy_one_step(graph, z, candidates, utility_fn)
        baseline_action = None
        if greedy.first_action[0]:
            for action in candidates:
                if (action[0] == greedy.first_action[0]
                        and action[1] == greedy.first_action[1]
                        and action[2] == greedy.first_action[2]):
                    baseline_action = action
                    break
        if baseline_action is None:
            continue

        # Learned: exact MPC.
        exact = exact_mpc(
            graph, z, candidates, utility_fn, horizon=2, gamma=0.9,
            regenerate_candidates=True,
            candidate_generator=lambda g, z2, **kw: generate_multi_operator_candidates(
                g, z2, config, rng=__import__("random").Random(config.seed + 100),
            ),
        )
        learned_action = None
        if exact.first_action_identity:
            for action in candidates:
                aid = ActionIdentity.from_action(action)
                if aid == exact.first_action_identity:
                    learned_action = action
                    break
        if learned_action is None:
            continue

        baseline_id = ActionIdentity.from_action(baseline_action)
        learned_id = ActionIdentity.from_action(learned_action)
        if baseline_id == learned_id:
            continue

        baseline_q = compute_exact_q_h2(graph, z, baseline_action, config, utility_fn)
        learned_q = compute_exact_q_h2(graph, z, learned_action, config, utility_fn)
        advantage = learned_q - baseline_q

        # Store graph adjacency.
        n_nodes = graph.num_nodes
        adj = _graph_to_adjacency(graph, n_nodes)
        state = SplitStructuralState.from_graph(graph)
        obj_feat = encode_objective(obj_spec)

        records.append(GraphAdvantageRecord(
            state_id=state_id,
            adjacency=adj,
            n_nodes=n_nodes,
            z=z,
            state_features=state.to_full_array(),
            objective_features=obj_feat,
            baseline_action=baseline_action,
            learned_action=learned_action,
            baseline_action_id=baseline_id,
            learned_action_id=learned_id,
            baseline_q=baseline_q,
            learned_q=learned_q,
            advantage=advantage,
            mechanism=mechanism,
            split=split,
            config_seed=config.seed,
            threshold=int(obj_spec.threshold),
            lambda_bonus=config.lambda_bonus,
        ))
        state_id += 1

    return records


def build_features_for_records(
    records: list[GraphAdvantageRecord],
    feature_level: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature matrix using stored graphs for F4 features.

    Returns (X, advantages, baseline_qs).
    """
    X = []
    y = []
    baseline_qs = []

    for rec in records:
        # Reconstruct graph from adjacency.
        graph = _adjacency_to_graph_buffers(rec.adjacency, rec.n_nodes, capacity=50)
        state = SplitStructuralState.from_graph(graph)
        obj_spec = get_objective_spec(rec.mechanism)

        x = extract_features_level(
            graph=graph,
            z=rec.z,
            state=state,
            objective=obj_spec,
            baseline_action=rec.baseline_action,
            learned_action=rec.learned_action,
            baseline_id=rec.baseline_action_id,
            learned_id=rec.learned_action_id,
            level=feature_level,
        )
        X.append(x)
        y.append(rec.advantage)
        baseline_qs.append(rec.baseline_q)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(baseline_qs, dtype=np.float32),
    )
