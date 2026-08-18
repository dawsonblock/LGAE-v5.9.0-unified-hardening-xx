"""Structural-policy qualification utilities for LGAE v5.2.

This module is intentionally separate from the governor qualification suite.
It answers a different question: can the learned proposal model recover known
structural interventions from held-out states after being trained on structural
outcomes?  The engine/governor remains the execution authority.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import math
import random

import torch

from .tasks import BenchmarkTask, StructuralAction, ALL_TASKS, canonical_action, ACTION_TO_INDEX
from .metrics import BenchmarkResult, run_benchmark
from ..executive import StructuralExecutive, ACTION_LIST

POLICY_QUALIFICATION_TASKS = tuple(t for t in ALL_TASKS if t.name != "G_info_gain")


_ACTION_COST = {
    StructuralAction.NO_OP: 0.0,
    StructuralAction.ADD_EDGE: 1.0,
    StructuralAction.PRUNE_EDGE: 0.5,
    StructuralAction.REWEIGHT_AFFINITY: 0.25,
    StructuralAction.REWEIGHT_LENGTH: 0.25,
    StructuralAction.COUPLED_REWEIGHT: 0.35,
    StructuralAction.SPAWN_FIBER: 1.0,
    StructuralAction.PRUNE_FIBER: 0.5,
    StructuralAction.CHANGE_GAUGE: 0.25,
}


@dataclass
class PolicyQualificationResult:
    train_samples: int
    heldout_seeds: list[int]
    diagnosis_accuracy: float
    mean_regret: float
    per_seed_accuracy: dict[int, float]
    per_seed_regret: dict[int, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _risk_from_delta(delta: float, best_delta: float) -> float:
    """Bounded empirical downside label for the risk head."""
    scale = max(abs(best_delta), 1e-6)
    if delta >= 0:
        return 0.0
    return float(min(1.0, abs(delta) / scale))


def add_oracle_experience(
    executive: StructuralExecutive,
    task: BenchmarkTask,
    seed: int,
) -> int:
    """Add full-information structural outcomes for one synthetic state."""
    state = task.initial_state(seed=seed)
    obs = executive.observe(state.graph, state.z)
    outcomes = task.evaluate_all(state)
    best_outcome = max(outcomes, key=lambda o: float(o.delta_utility))
    best = float(best_outcome.delta_utility)
    # The benchmark defines semantically correct actions. Prefer that label when
    # available so near-tied utility alternatives do not erase diagnosis intent.
    # When multiple actions are correct, pick the one with highest ΔU (same
    # logic as OracleController).  Using list(set)[0] is nondeterministic under
    # PYTHONHASHSEED variation.
    correct = task.correct_actions()
    if correct:
        if len(correct) == 1:
            policy_target = canonical_action(correct)
        else:
            # Pick highest-ΔU among correct actions, ties broken canonically
            best_correct_delta = float("-inf")
            policy_target = canonical_action(correct)
            for action in sorted(correct, key=lambda a: ACTION_TO_INDEX[a]):
                outcome = task.evaluate(state, action)
                if outcome.delta_utility > best_correct_delta:
                    best_correct_delta = outcome.delta_utility
                    policy_target = action
    else:
        policy_target = best_outcome.action
    executive.record_policy_label(obs, policy_target)
    for outcome in outcomes:
        executive.record_outcome(
            obs,
            outcome.action,
            float(outcome.delta_utility),
            cost_target=float(_ACTION_COST.get(outcome.action, 1.0)),
            risk_target=_risk_from_delta(float(outcome.delta_utility), best),
            ig_target=0.0,
            uncertainty_target=0.0,
            sample_weight=1.0,
        )
    return len(outcomes)


def train_on_structural_oracles(
    executive: StructuralExecutive,
    *,
    tasks: Iterable[BenchmarkTask] = POLICY_QUALIFICATION_TASKS,
    train_seeds: Iterable[int] = range(16),
    gradient_steps: int = 500,
    batch_size: int = 64,
    seed: int = 0,
) -> int:
    """Train the proposal-value heads on synthetic full-information outcomes.

    This is a qualification/training scaffold, not a claim that real systems have
    oracle counterfactual labels.  In live operation the same heads are updated
    from committed outcomes, governance risk labels and long-horizon credit.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    count = 0
    task_list = list(tasks)
    for s in train_seeds:
        for task in task_list:
            count += add_oracle_experience(executive, task, int(s))
    for _ in range(int(gradient_steps)):
        executive.train_step(batch_size=min(batch_size, len(executive._experience)))
    return count


def evaluate_policy(
    executive: StructuralExecutive,
    *,
    tasks: Iterable[BenchmarkTask] = POLICY_QUALIFICATION_TASKS,
    heldout_seeds: Iterable[int] = (101, 102, 103, 104, 105),
) -> PolicyQualificationResult:
    task_list = list(tasks)
    per_acc: dict[int, float] = {}
    per_regret: dict[int, float] = {}
    all_acc: list[float] = []
    all_regret: list[float] = []
    for seed in heldout_seeds:
        proposals: dict[str, StructuralAction] = {}
        for task in task_list:
            state = task.initial_state(seed=int(seed))
            obs = executive.observe(state.graph, state.z)
            proposals[task.name] = executive.best_proposal(obs).action
        result = run_benchmark(proposals=proposals, seed=int(seed), tasks=task_list)
        per_acc[int(seed)] = float(result.diagnosis_accuracy)
        per_regret[int(seed)] = float(result.mean_regret)
        all_acc.append(float(result.diagnosis_accuracy))
        all_regret.append(float(result.mean_regret))
    return PolicyQualificationResult(
        train_samples=0,
        heldout_seeds=[int(s) for s in heldout_seeds],
        diagnosis_accuracy=float(sum(all_acc) / max(len(all_acc), 1)),
        mean_regret=float(sum(all_regret) / max(len(all_regret), 1)),
        per_seed_accuracy=per_acc,
        per_seed_regret=per_regret,
    )


def qualify_structural_policy(
    *,
    train_seeds: Iterable[int] = range(16),
    heldout_seeds: Iterable[int] = (101, 102, 103, 104, 105),
    gradient_steps: int = 500,
    seed: int = 0,
) -> tuple[StructuralExecutive, PolicyQualificationResult]:
    # Seed before constructing the executive so network/target-scorer weight
    # initialization is deterministic. Previously the seed was only applied
    # inside train_on_structural_oracles, *after* the networks were already
    # initialized, which made the release-gate qualification nondeterministic
    # (observed runs ranged 86.7%..100% diagnosis accuracy on the same code).
    random.seed(seed)
    torch.manual_seed(seed)
    executive = StructuralExecutive(hidden_dim=96, lr=2e-3)
    samples = train_on_structural_oracles(
        executive,
        train_seeds=train_seeds,
        gradient_steps=gradient_steps,
        seed=seed,
    )
    result = evaluate_policy(executive, heldout_seeds=heldout_seeds)
    result.train_samples = int(samples)
    return executive, result
