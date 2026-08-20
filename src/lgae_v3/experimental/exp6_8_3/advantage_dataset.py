"""Advantage dataset generation for exp6.8.3.

For every (state, action) pair, compute the exact advantage:
  A* = Q_H(S, a_learned) - Q_H(S, a_baseline)

where Q_H is the exact H-step oracle value.

The advantage is signed: positive means the learned override was
beneficial, negative means it was harmful.

AdvantageRecord schema includes complete ActionIdentity.
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
    MECHANISM_NAMES,
    generate_mechanism_task_configs, MechanismTaskConfig, _make_graph_from_config,
)
from ..exp6_6.objective_spec import get_objective_spec, ObjectiveSpec
from ..exp6_7.multi_operator_candidates import generate_multi_operator_candidates
from ..exp6_8_1.split_state import SplitStructuralState


@dataclass
class AdvantageRecord:
    """A single advantage training example."""
    state_id: int
    state_features: np.ndarray  # exact + certified + learned state features
    objective_features: np.ndarray  # objective encoding
    baseline_action: tuple  # (type, u, v, params)
    learned_action: tuple
    baseline_action_id: ActionIdentity
    learned_action_id: ActionIdentity
    baseline_q: float  # exact H-step Q for baseline action
    learned_q: float  # exact H-step Q for learned action
    advantage: float  # learned_q - baseline_q
    mechanism: str
    split: str  # "train", "calibration", or "test"

    @property
    def is_beneficial(self) -> bool:
        """True if the learned override was beneficial."""
        return self.advantage > 0


def compute_exact_q_h2(
    graph: GraphBuffers,
    z: torch.Tensor,
    first_action: tuple,
    config: MechanismTaskConfig,
    utility_fn,
    gamma: float = 0.9,
) -> float:
    """Compute exact Q_H for a specific first action at H=2.

    Q_H(S, a) = U(S') + gamma * max_{a'} U(S'')

    where S' = T_exact(S, a), S'' = T_exact(S', a'),
    and U measures the utility of a graph state.
    """
    # First step: apply action exactly.
    status = apply_action_with_status(graph, first_action)
    if status.status != "VALID":
        return -1e9  # invalid action has very low Q

    new_graph = apply_action(graph, first_action)

    # Utility of S' (after first action).
    first_utility = utility_fn(new_graph, z) if callable(utility_fn) else 0.0

    # Generate candidates at S'.
    future_candidates = generate_multi_operator_candidates(
        new_graph, z, config, rng=__import__("random").Random(config.seed + 1),
    )

    # Filter valid at S'.
    valid_future = []
    for action in future_candidates:
        st = apply_action_with_status(new_graph, action)
        if st.status == "VALID":
            valid_future.append(action)

    if not valid_future:
        return float(first_utility)

    # Best second-step utility.
    best_second = -1e9
    for second_action in valid_future:
        g2 = apply_action(new_graph, second_action)
        u2 = utility_fn(g2, z) if callable(utility_fn) else 0.0
        best_second = max(best_second, u2)

    return float(first_utility + gamma * best_second)


def generate_advantage_dataset(
    mechanisms: list[str],
    n_tasks_per_mechanism: int = 200,
    seed: int = 42,
    split: str = "train",
) -> list[AdvantageRecord]:
    """Generate advantage dataset for all mechanisms.

    For each task:
    1. Generate graph and candidates.
    2. Compute baseline action (greedy).
    3. Compute learned action (recursive MPC with ensemble mean).
    4. Compute exact Q_H for both.
    5. Record advantage = Q_learned - Q_baseline.
    """
    from ..exp6_4.test_f import make_test_f_utility
    from ..exp6_8.recursive_planner import recursive_causal_mpc
    from ..exp6_8_1.learned_state_model import LearnedStateModel
    from ..exp6_8_1.split_state import SplitStructuralState
    from .advantage_features import extract_state_features, extract_objective_features

    records = []
    state_id_counter = 0

    for mech_idx, mechanism in enumerate(mechanisms):
        obj_spec = get_objective_spec(mechanism)
        configs = generate_mechanism_task_configs(
            mechanism=mechanism,
            n_tasks=n_tasks_per_mechanism,
            seed=seed + mech_idx * 1000,
        )

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

            # Learned: recursive MPC (using a simple model for dataset gen).
            # We use the recursive planner with use_predicted=True.
            # For dataset generation, we don't need a fitted model —
            # we use the exact MPC's first action as "learned" to generate
            # diverse advantage labels. The actual learned model will be
            # trained on these labels.
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

            # Skip if baseline and learned are the same.
            baseline_id = ActionIdentity.from_action(baseline_action)
            learned_id = ActionIdentity.from_action(learned_action)
            if baseline_id == learned_id:
                continue  # no advantage to predict

            # Compute exact Q_H for both.
            baseline_q = compute_exact_q_h2(
                graph, z, baseline_action, config, utility_fn,
            )
            learned_q = compute_exact_q_h2(
                graph, z, learned_action, config, utility_fn,
            )

            advantage = learned_q - baseline_q

            # Extract features.
            state = SplitStructuralState.from_graph(graph)
            state_feat = extract_state_features(state)
            obj_feat = extract_objective_features(obj_spec)

            records.append(AdvantageRecord(
                state_id=state_id_counter,
                state_features=state_feat,
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
            ))
            state_id_counter += 1

    return records


def records_to_arrays(records: list[AdvantageRecord]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert records to (X, y, mechanisms) arrays.

    X = [state_features, objective_features, pairwise_action_features]
    y = advantage (signed)
    """
    from .advantage_features import extract_pairwise_features

    X = []
    y = []
    mechanisms = []

    for rec in records:
        # Pairwise features need the graph, but we don't store it.
        # Instead, we use state + objective + action identity features.
        # The pairwise features are built from action identities.
        pairwise = extract_pairwise_features(
            rec.baseline_action, rec.learned_action,
            rec.baseline_action_id, rec.learned_action_id,
        )
        x = np.concatenate([rec.state_features, rec.objective_features, pairwise])
        X.append(x)
        y.append(rec.advantage)
        mechanisms.append(rec.mechanism)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        mechanisms,
    )
