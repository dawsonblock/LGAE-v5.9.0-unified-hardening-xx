"""Non-learned baseline controllers for benchmark comparison.

These exist so the learned ``StructuralExecutive`` is not evaluated in a
vacuum.  Every baseline implements the same protocol::

    propose(task, state, rng) -> StructuralAction

and is scored by the same ``run_benchmark`` machinery as the learned
executive.  This makes diagnosis-accuracy / regret numbers directly
comparable across controllers.

Baselines:

- :class:`RandomActionController` -- uniform over the 9 actions.  Lower bar.
- :class:`SpectralHeuristicController` -- a non-learned "reasonable engineer"
  controller that uses the same cheap observables the executive sees
  (spectral gap, edge-latent mismatch, fiber utilization, latent norm) to
  pick an action via fixed thresholds.  It is *not* tuned per task.
- :class:`OracleController` -- always returns the task's labeled correct
  action.  Upper bound (and a consistency check on the benchmark itself).

Why this matters: without these, the learned executive's headline numbers
have no reference point.  Beating random is the minimum credible bar;
matching or losing to the spectral heuristic would indicate the learned
policy adds no value over cheap non-learned rules on these synthetic tasks.
"""
from __future__ import annotations

import random
from typing import Callable

import torch

from .tasks import BenchmarkTask, StructuralAction, TaskState, canonical_action, ACTION_TO_INDEX
from ..operators import spectral_gap_graphbuffers


Controller = Callable[[BenchmarkTask, TaskState, random.Random], StructuralAction]


class RandomActionController:
    """Uniform-random action selection.  Lower-bound baseline."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def propose(self, task: BenchmarkTask, state: TaskState) -> StructuralAction:
        return self._rng.choice(list(StructuralAction))


class OracleController:
    """Always returns the task's labeled correct action.  Upper bound.

    Also a consistency check: if the oracle does not score 100% diagnosis
    accuracy, the benchmark's own regret computation is inconsistent.

    v5.3.2: When multiple actions are correct, picks the one with the
    highest ΔU (zero regret).  Previously picked an arbitrary element
    from the set, which could have nonzero regret if the set contained
    both high- and low-ΔU actions.
    """

    def propose(self, task: BenchmarkTask, state: TaskState) -> StructuralAction:
        correct = task.correct_actions()
        if not correct:
            return StructuralAction.NO_OP
        if len(correct) == 1:
            return canonical_action(correct)
        # Multiple correct actions: pick the one with highest ΔU.
        # Ties broken by canonical action ordering (deterministic).
        best_action = None
        best_delta = float("-inf")
        for action in sorted(correct, key=lambda a: ACTION_TO_INDEX[a]):
            outcome = task.evaluate(state, action)
            if outcome.delta_utility > best_delta:
                best_delta = outcome.delta_utility
                best_action = action
        return best_action or canonical_action(correct)


class SpectralHeuristicController:
    """Non-learned controller using cheap spectral / geometry signals.

    Decision rules (applied in order, first match wins):

    - very low spectral gap (λ₂ < 0.15) and graph is sparse  -> ADD_EDGE
    - max edge-latent mismatch is large (top edge far in latent space)
      and there are enough edges to spare                      -> PRUNE_EDGE
    - latent width below configured max and a node has very high
      latent norm relative to the mean                        -> SPAWN_FIBER
    - otherwise                                               -> NO_OP

    The thresholds are intentionally *not* tuned to the benchmark tasks;
    they are round-number rules of thumb a practitioner might write before
    seeing the data.  This keeps the baseline honest rather than a
    reverse-engineered optimum.
    """

    def propose(self, task: BenchmarkTask, state: TaskState) -> StructuralAction:
        graph = state.graph
        z = state.z
        cfg = state.config
        try:
            lam, _ = spectral_gap_graphbuffers(graph)
            lam = float(lam)
        except Exception:
            lam = 0.0
        valid = graph.valid.bool()
        n_edges = int(valid.sum().item())
        n_nodes = int(graph.num_nodes)

        # Edge-latent mismatch.
        max_mismatch = 0.0
        if z.ndim == 2 and n_edges:
            ids = torch.where(valid)[0]
            src = graph.src[ids]
            dst = graph.dst[ids]
            mm = torch.linalg.vector_norm(z[src] - z[dst], dim=-1)
            max_mismatch = float(mm.max().item()) if mm.numel() else 0.0

        # Latent norm statistics.
        mean_norm = 0.0
        max_norm = 0.0
        if z.ndim == 2 and z.numel():
            zn = torch.linalg.vector_norm(z.detach(), dim=-1).to(torch.float32)
            mean_norm = float(zn.mean().item())
            max_norm = float(zn.max().item())

        d_max = float(cfg.fiber.d_max)
        width = float(z.shape[1]) if z.ndim == 2 else 0.0

        # Rules of thumb (no per-task tuning).
        if lam < 0.15 and n_edges < n_nodes * 2:
            return StructuralAction.ADD_EDGE
        if max_mismatch > 3.0 * max(mean_norm, 1e-6) and n_edges > 3:
            return StructuralAction.PRUNE_EDGE
        if width < d_max and max_norm > 2.5 * max(mean_norm, 1e-6):
            return StructuralAction.SPAWN_FIBER
        return StructuralAction.NO_OP


ALL_BASELINES: dict[str, Controller] = {
    "random": RandomActionController().propose,
    "spectral_heuristic": SpectralHeuristicController().propose,
    "oracle": OracleController().propose,
}
