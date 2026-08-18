#!/usr/bin/env python3
"""Compare the learned structural executive against non-learned baselines.

This is the comparison the original v5.2/v5.3 qualification lacked.  It runs
four controllers on the *same* benchmark tasks and reports diagnosis accuracy
and mean regret for each:

    - random            : uniform random action (lower bound)
    - spectral_heuristic: non-learned threshold rules on cheap observables
    - learned           : the trained StructuralExecutive
    - oracle            : always the task's labeled correct action (upper bound)

It reports on two splits:

    - in_distribution   : the original 6 ALL_TASKS (seeds 101-105)
    - held_out_structure: parametric variants with different graph sizes /
                          cluster splits / spurious-edge positions (the
                          "held-out seeds 101-105" in the original report
                          reused identical graph structures, so they were
                          not actually held out)

Usage::

    python scripts/compare_baselines.py
    python scripts/compare_baselines.py --gradient-steps 500 --seed 0

Exit code 0 if the learned executive beats random on both splits, else 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import random

import torch

from lgae_v3.benchmark.tasks import (
    ALL_TASKS, StructuralAction, BenchmarkTask, heldout_tasks,
)
from lgae_v3.benchmark.metrics import run_benchmark
from lgae_v3.benchmark.baselines import (
    RandomActionController, SpectralHeuristicController, OracleController,
)
from lgae_v3.benchmark.policy_qualification import (
    qualify_structural_policy,
)


def _baseline_proposals(
    controller, tasks: list[BenchmarkTask], seeds: list[int],
) -> dict[str, dict[str, StructuralAction]]:
    """Return {seed: {task_name: action}} for a baseline controller."""
    out: dict[str, dict[str, StructuralAction]] = {}
    for seed in seeds:
        per_task: dict[str, StructuralAction] = {}
        for task in tasks:
            state = task.initial_state(seed=seed)
            per_task[task.name] = controller(task, state)
        out[str(seed)] = per_task
    return out


def _learned_proposals(executive, tasks, seeds):
    out: dict[str, dict[str, StructuralAction]] = {}
    for seed in seeds:
        per_task: dict[str, StructuralAction] = {}
        for task in tasks:
            state = task.initial_state(seed=seed)
            obs = executive.observe(state.graph, state.z)
            per_task[task.name] = executive.best_proposal(obs).action
        out[str(seed)] = per_task
    return out


def _score(proposals_by_seed, tasks, seeds):
    accs, regrets = [], []
    per_seed = {}
    for seed in seeds:
        props = proposals_by_seed[str(seed)]
        res = run_benchmark(proposals=props, seed=seed, tasks=tasks)
        accs.append(res.diagnosis_accuracy)
        regrets.append(res.mean_regret)
        per_seed[int(seed)] = {
            "accuracy": res.diagnosis_accuracy, "regret": res.mean_regret,
        }
    return {
        "diagnosis_accuracy": sum(accs) / max(len(accs), 1),
        "mean_regret": sum(regrets) / max(len(regrets), 1),
        "per_seed": per_seed,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gradient-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heldout-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    seeds = args.heldout_seeds

    # Train the learned executive (deterministic after the v5.3.1 seed fix).
    executive, _ = qualify_structural_policy(
        gradient_steps=args.gradient_steps, seed=args.seed,
    )

    # Held-out *structurally distinct* tasks (not in ALL_TASKS).
    heldout = heldout_tasks(seed=0)

    controllers: dict[str, Callable] = {
        "random": RandomActionController(seed=args.seed).propose,
        "spectral_heuristic": SpectralHeuristicController().propose,
        "oracle": OracleController().propose,
    }

    results: dict[str, dict] = {}

    for split_name, tasks in [("in_distribution", ALL_TASKS), ("held_out_structure", heldout)]:
        results[split_name] = {}
        # Baselines.
        for cname, ctrl in controllers.items():
            props = _baseline_proposals(ctrl, tasks, seeds)
            results[split_name][cname] = _score(props, tasks, seeds)
        # Learned.
        lprops = _learned_proposals(executive, tasks, seeds)
        results[split_name]["learned"] = _score(lprops, tasks, seeds)

    # Summary table.
    print(f"{'controller':22s} {'split':22s} {'accuracy':>10s} {'regret':>10s}")
    print("-" * 66)
    for split_name in results:
        for cname in ["random", "spectral_heuristic", "learned", "oracle"]:
            r = results[split_name][cname]
            print(f"{cname:22s} {split_name:22s} {r['diagnosis_accuracy']:10.4f} {r['mean_regret']:10.4f}")

    payload = {
        "schema": "LGAE_BASELINE_COMPARISON",
        "seed": args.seed,
        "gradient_steps": args.gradient_steps,
        "heldout_seeds": seeds,
        "results": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
    else:
        print("\n" + json.dumps(payload, indent=2))

    # Pass only if learned beats random on both splits.
    learned_beats_random = (
        results["in_distribution"]["learned"]["diagnosis_accuracy"]
        > results["in_distribution"]["random"]["diagnosis_accuracy"]
        and
        results["held_out_structure"]["learned"]["diagnosis_accuracy"]
        > results["held_out_structure"]["random"]["diagnosis_accuracy"]
    )
    return 0 if learned_beats_random else 1


if __name__ == "__main__":
    raise SystemExit(main())
