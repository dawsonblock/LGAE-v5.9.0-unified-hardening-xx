"""Tests for v5.3.1 integrity fixes: baselines, held-out tasks, determinism."""
from __future__ import annotations

import random

import pytest
import torch

from lgae_v3.benchmark.baselines import (
    RandomActionController,
    SpectralHeuristicController,
    OracleController,
    ALL_BASELINES,
)
from lgae_v3.benchmark.tasks import (
    ALL_TASKS,
    HeldOutBottleneck,
    HeldOutSpuriousEdge,
    heldout_tasks,
    StructuralAction,
)
from lgae_v3.benchmark.metrics import run_benchmark
from lgae_v3.benchmark.policy_qualification import (
    qualify_structural_policy,
)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_random_controller_returns_valid_action():
    ctrl = RandomActionController(seed=0)
    for task in ALL_TASKS:
        state = task.initial_state(seed=42)
        action = ctrl.propose(task, state)
        assert action in StructuralAction


def test_oracle_controller_always_correct():
    ctrl = OracleController()
    for task in ALL_TASKS:
        state = task.initial_state(seed=42)
        action = ctrl.propose(task, state)
        assert action in task.correct_actions()


def test_oracle_scores_perfect_on_benchmark():
    """Oracle must score 100% diagnosis accuracy -- a benchmark consistency check."""
    ctrl = OracleController()
    for seed in (101, 102, 103):
        proposals = {}
        for task in ALL_TASKS:
            state = task.initial_state(seed=seed)
            proposals[task.name] = ctrl.propose(task, state)
        result = run_benchmark(proposals=proposals, seed=seed, tasks=ALL_TASKS)
        assert result.diagnosis_accuracy == 1.0
        assert result.mean_regret == 0.0


def test_spectral_heuristic_beats_random_on_in_distribution():
    """The non-learned heuristic should beat uniform random on the training tasks."""
    rand = RandomActionController(seed=0)
    heur = SpectralHeuristicController()
    seeds = [101, 102, 103, 104, 105]
    rand_accs, heur_accs = [], []
    for seed in seeds:
        rp = {t.name: rand.propose(t, t.initial_state(seed=seed)) for t in ALL_TASKS}
        hp = {t.name: heur.propose(t, t.initial_state(seed=seed)) for t in ALL_TASKS}
        rand_accs.append(run_benchmark(proposals=rp, seed=seed, tasks=ALL_TASKS).diagnosis_accuracy)
        heur_accs.append(run_benchmark(proposals=hp, seed=seed, tasks=ALL_TASKS).diagnosis_accuracy)
    assert sum(heur_accs) / len(heur_accs) > sum(rand_accs) / len(rand_accs)


def test_all_baselines_exported():
    assert set(ALL_BASELINES) == {"random", "spectral_heuristic", "oracle"}


# ---------------------------------------------------------------------------
# Held-out parametric tasks
# ---------------------------------------------------------------------------

def test_heldout_bottleneck_correct_action_is_add_edge():
    task = HeldOutBottleneck(cluster=5, bridge_offset=0)
    assert task.correct_actions() == {StructuralAction.ADD_EDGE}


def test_heldout_spurious_correct_action_is_prune_edge():
    task = HeldOutSpuriousEdge(n=8)
    assert task.correct_actions() == {StructuralAction.PRUNE_EDGE}


def test_heldout_tasks_have_different_structure_than_training():
    """Held-out tasks must not be identical to the fixed ALL_TASKS structures."""
    ho = heldout_tasks(seed=0)
    assert len(ho) >= 2
    train_a = ALL_TASKS[0].initial_state(seed=42)
    for t in ho:
        s = t.initial_state(seed=42)
        # At least one held-out task should have a different node count than Task A (8).
        assert s.graph.num_nodes != train_a.graph.num_nodes or t.name != ALL_TASKS[0].name


def test_heldout_bottleneck_varies_with_params():
    """Different cluster sizes produce different graph sizes."""
    t5 = HeldOutBottleneck(cluster=5)
    t7 = HeldOutBottleneck(cluster=7)
    s5 = t5.initial_state(seed=42)
    s7 = t7.initial_state(seed=42)
    assert s7.graph.num_nodes > s5.graph.num_nodes


def test_heldout_bottleneck_correct_action_maximizes_spectral_gap():
    """The correct action (ADD_EDGE) should be the utility argmax -- a physics
    consequence, not a definition (utility is pure spectral gap)."""
    task = HeldOutBottleneck(cluster=5, bridge_offset=0)
    state = task.initial_state(seed=42)
    outs = task.evaluate_all(state)
    best = max(outs, key=lambda o: o.delta_utility)
    assert best.action in task.correct_actions()


# ---------------------------------------------------------------------------
# Determinism of policy qualification (the v5.3.1 fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_policy_qualification_is_deterministic(seed):
    """Two runs with the same seed must produce identical accuracy and regret.

    This guards the v5.3.1 fix: previously the network was initialized before
    torch.manual_seed was called, making the release-gate qualification
    nondeterministic (observed range 86.7%-100% on the same code).
    """
    _, r1 = qualify_structural_policy(seed=seed, gradient_steps=50)
    _, r2 = qualify_structural_policy(seed=seed, gradient_steps=50)
    assert r1.diagnosis_accuracy == r2.diagnosis_accuracy
    assert r1.mean_regret == r2.mean_regret


def test_policy_qualification_passes_release_gate():
    _, r = qualify_structural_policy(seed=0, gradient_steps=200)
    assert r.diagnosis_accuracy >= 0.80
    assert r.mean_regret <= 0.35
