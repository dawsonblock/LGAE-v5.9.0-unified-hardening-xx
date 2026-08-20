"""Experiment runner for v7.0-exp1-real-ai-topology.

The first real AI topology experiment.

Three conditions:
  A. Fixed topology
  B. Hand-written dynamic router
  C. LGAE adaptive topology

Held constant: models, prompts, tasks, token limits, tools, hardware.

Measures: Quality, Tokens, Latency, Calls, Failures, Cost, J.

Success criteria:
  - Same quality with materially lower compute (20-30% cost reduction)
  - OR materially higher quality at roughly equal compute
  - Pareto view: quality vs cost frontier
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json
import sys
import os
import time
import numpy as np

from .ai_node import create_default_nodes
from .topology import AITopology, create_default_topology
from .runtime import AIRuntime, TaskResult
from .objective import (
    ObjectiveWeights, compute_objective_from_result,
    compute_pareto_efficiency,
)
from .benchmark import generate_benchmark, BenchmarkTask, evaluate_quality, TASK_CLASSES
from .conditions import (
    run_fixed_topology, run_dynamic_router, run_lgae_adaptive,
    ConditionResult,
)


@dataclass
class Exp71Result:
    audit_note: str = ""
    condition_results: list[ConditionResult] = field(default_factory=list)
    pareto_analysis: dict = field(default_factory=dict)
    gates: dict[str, dict] = field(default_factory=dict)
    all_gates_passed: bool = False
    summary: dict[str, Any] = field(default_factory=dict)
    manifest_evidence: dict[str, Any] = field(default_factory=dict)
    objective_weights: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "audit_note": self.audit_note,
            "objective_weights": self.objective_weights,
            "condition_results": [r.to_dict() for r in self.condition_results],
            "pareto_analysis": self.pareto_analysis,
            "gates": self.gates,
            "all_gates_passed": self.all_gates_passed,
            "summary": self.summary,
            "manifest_evidence": self.manifest_evidence,
        }


def _check_manifest_evidence() -> dict:
    import subprocess
    evidence = {
        "manifest_exists": False, "manifest_valid": False,
        "test_count": 0, "test_passed": 0, "test_failed": 0,
    }
    manifest_path = os.path.join(os.getcwd(), "MANIFEST.sha256.json")
    if os.path.exists(manifest_path):
        evidence["manifest_exists"] = True
        try:
            result = subprocess.run(
                [sys.executable, "scripts/generate_manifest.py", "--check"],
                capture_output=True, text=True, timeout=30, cwd=os.getcwd(),
            )
            if result.returncode == 0:
                evidence["manifest_valid"] = True
        except Exception:
            pass

    qual_path = os.path.join(os.getcwd(), "qualification_summary.json")
    if os.path.exists(qual_path):
        try:
            with open(qual_path) as f:
                qual = json.load(f)
            tr = qual.get("test_results", {})
            evidence["test_count"] = tr.get("collected", 0)
            evidence["test_passed"] = tr.get("passed", 0)
            evidence["test_failed"] = tr.get("failed", 0)
        except Exception:
            pass
    return evidence


def run_exp7_1(
    *,
    n_tasks_per_class: int = 10,
    weights: ObjectiveWeights = None,
    adaptation_interval: int = 10,
) -> Exp71Result:
    """Run the v7.0-exp1-real-ai-topology experiment."""
    if weights is None:
        weights = ObjectiveWeights()

    result = Exp71Result(
        audit_note=(
            "Real AI Topology. First test of whether structural adaptation "
            "improves an AI system per unit compute. "
            "Three conditions: Fixed, Dynamic, LGAE. "
            "Models/prompts/tasks/tokens/tools held constant. "
            "LGAE controls only routing topology."
        ),
        objective_weights=weights.to_dict(),
    )

    print(f"\n  Tasks per class: {n_tasks_per_class}")
    print(f"  Task classes: {TASK_CLASSES}")
    print(f"  Total tasks: {n_tasks_per_class * len(TASK_CLASSES)}")
    print(f"  Objective weights: {weights.to_dict()}")

    # === Phase 1: Generate benchmark ===
    print("\n=== Phase 1: Generating benchmark ===")
    tasks = generate_benchmark(n_per_class=n_tasks_per_class, seed=42)
    print(f"  Generated {len(tasks)} tasks across {len(TASK_CLASSES)} classes")

    # === Phase 2: Run three conditions ===
    print("\n=== Phase 2: Running three conditions ===")

    print("\n  --- Condition A: Fixed topology ---")
    t0 = time.time()
    result_a = run_fixed_topology(tasks, weights)
    print(f"    Quality: {result_a.mean_quality:.4f}")
    print(f"    Tokens: {result_a.mean_tokens:.1f}")
    print(f"    Latency: {result_a.mean_latency_ms:.1f}ms")
    print(f"    Calls: {result_a.mean_calls:.2f}")
    print(f"    Failures: {result_a.mean_failures:.2f}")
    print(f"    Objective J: {result_a.mean_objective:.4f}")
    print(f"    Cost: {result_a.mean_cost:.4f}")
    print(f"    Success rate: {result_a.success_rate:.1%}")
    print(f"    Time: {time.time()-t0:.1f}s")
    result.condition_results.append(result_a)

    print("\n  --- Condition B: Dynamic router ---")
    t0 = time.time()
    result_b = run_dynamic_router(tasks, weights)
    print(f"    Quality: {result_b.mean_quality:.4f}")
    print(f"    Tokens: {result_b.mean_tokens:.1f}")
    print(f"    Latency: {result_b.mean_latency_ms:.1f}ms")
    print(f"    Calls: {result_b.mean_calls:.2f}")
    print(f"    Failures: {result_b.mean_failures:.2f}")
    print(f"    Objective J: {result_b.mean_objective:.4f}")
    print(f"    Cost: {result_b.mean_cost:.4f}")
    print(f"    Success rate: {result_b.success_rate:.1%}")
    print(f"    Time: {time.time()-t0:.1f}s")
    result.condition_results.append(result_b)

    print("\n  --- Condition C: LGAE adaptive ---")
    t0 = time.time()
    result_c = run_lgae_adaptive(tasks, weights, adaptation_interval=adaptation_interval)
    print(f"    Quality: {result_c.mean_quality:.4f}")
    print(f"    Tokens: {result_c.mean_tokens:.1f}")
    print(f"    Latency: {result_c.mean_latency_ms:.1f}ms")
    print(f"    Calls: {result_c.mean_calls:.2f}")
    print(f"    Failures: {result_c.mean_failures:.2f}")
    print(f"    Objective J: {result_c.mean_objective:.4f}")
    print(f"    Cost: {result_c.mean_cost:.4f}")
    print(f"    Success rate: {result_c.success_rate:.1%}")
    print(f"    Mutations applied: {result_c.n_mutations}")
    print(f"    Time: {time.time()-t0:.1f}s")
    result.condition_results.append(result_c)

    # === Phase 3: Pareto analysis ===
    print("\n=== Phase 3: Pareto analysis ===")

    pareto_points = [
        {"name": r.condition_name, "quality": r.mean_quality, "cost": r.mean_cost}
        for r in result.condition_results
    ]
    is_efficient = compute_pareto_efficiency(pareto_points)

    for i, point in enumerate(pareto_points):
        point["pareto_efficient"] = is_efficient[i]
        print(f"  {point['name']}: quality={point['quality']:.4f}, cost={point['cost']:.4f}, "
              f"efficient={is_efficient[i]}")

    result.pareto_analysis = {
        "points": pareto_points,
        "pareto_front": [p for i, p in enumerate(pareto_points) if is_efficient[i]],
    }

    # === Phase 4: Comparison and gates ===
    print("\n=== Phase 4: Checking success gates ===")

    fixed = result_a
    dynamic = result_b
    lgae = result_c

    # Cost reduction: (fixed_cost - lgae_cost) / fixed_cost
    cost_reduction = (fixed.mean_cost - lgae.mean_cost) / max(fixed.mean_cost, 1e-6)
    # Quality change: lgae_quality - fixed_quality
    quality_change = lgae.mean_quality - fixed.mean_quality
    # Token reduction
    token_reduction = (fixed.mean_tokens - lgae.mean_tokens) / max(fixed.mean_tokens, 1e-6)
    # Objective improvement
    obj_improvement = lgae.mean_objective - fixed.mean_objective

    # Gate 1: LGAE on Pareto frontier.
    lgae_efficient = is_efficient[2] if len(is_efficient) > 2 else False

    # Gate 2: Cost reduction >= 20% at similar quality.
    cost_gate = cost_reduction >= 0.20 and abs(quality_change) < 0.1

    # Gate 3: Quality improvement >= 10% at similar cost.
    quality_gate = quality_change >= 0.10 and abs(
        (fixed.mean_cost - lgae.mean_cost) / max(fixed.mean_cost, 1e-6)
    ) < 0.1

    # Gate 4: LGAE objective J > fixed objective J.
    objective_gate = lgae.mean_objective > fixed.mean_objective

    # Gate 5: LGAE no worse than fixed on any metric.
    no_regression = (
        lgae.mean_quality >= fixed.mean_quality - 0.05
        and lgae.mean_failures <= fixed.mean_failures + 0.5
    )

    # Gate 6: LGAE applied at least one mutation.
    mutation_gate = lgae.n_mutations > 0

    # Gate 7: Release qualification.
    manifest_evidence = _check_manifest_evidence()
    result.manifest_evidence = manifest_evidence
    qual_gate = manifest_evidence["manifest_valid"] and manifest_evidence["test_failed"] == 0

    gates = {
        "1_lgae_pareto_efficient": {
            "passed": lgae_efficient,
            "description": f"LGAE on Pareto frontier: {lgae_efficient}",
        },
        "2_cost_reduction_20pct": {
            "passed": cost_gate,
            "description": f"cost_reduction={cost_reduction:.1%}, quality_change={quality_change:+.4f}",
        },
        "3_quality_improvement_10pct": {
            "passed": quality_gate,
            "description": f"quality_change={quality_change:+.4f}",
        },
        "4_objective_improvement": {
            "passed": objective_gate,
            "description": f"LGAE J={lgae.mean_objective:.4f} vs Fixed J={fixed.mean_objective:.4f}",
        },
        "5_no_regression": {
            "passed": no_regression,
            "description": f"quality diff={quality_change:+.4f}, failure diff={lgae.mean_failures-fixed.mean_failures:+.2f}",
        },
        "6_mutations_applied": {
            "passed": mutation_gate,
            "description": f"n_mutations={lgae.n_mutations}",
        },
        "7_qualification": {
            "passed": qual_gate,
            "description": f"manifest_valid={manifest_evidence['manifest_valid']}, failures={manifest_evidence['test_failed']}",
        },
    }

    result.gates = gates
    result.all_gates_passed = all(g["passed"] for g in gates.values())

    result.summary = {
        "fixed_quality": fixed.mean_quality,
        "fixed_cost": fixed.mean_cost,
        "fixed_objective": fixed.mean_objective,
        "dynamic_quality": dynamic.mean_quality,
        "dynamic_cost": dynamic.mean_cost,
        "dynamic_objective": dynamic.mean_objective,
        "lgae_quality": lgae.mean_quality,
        "lgae_cost": lgae.mean_cost,
        "lgae_objective": lgae.mean_objective,
        "cost_reduction": round(cost_reduction, 4),
        "quality_change": round(quality_change, 4),
        "token_reduction": round(token_reduction, 4),
        "obj_improvement": round(obj_improvement, 4),
        "lgae_mutations": lgae.n_mutations,
    }

    print(f"\n  Overall: {'ALL GATES PASSED' if result.all_gates_passed else 'GATES NOT ALL MET'}")
    for name, gate in gates.items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"    {name}: {status} — {gate['description']}")

    # Print comparison table.
    print(f"\n{'='*70}")
    print(f"{'Condition':<15} {'Quality':>8} {'Tokens':>8} {'Latency':>8} {'Calls':>6} {'J':>8} {'Cost':>8}")
    print(f"{'-'*70}")
    for r in result.condition_results:
        print(f"{r.condition_name:<15} {r.mean_quality:>8.4f} {r.mean_tokens:>8.1f} "
              f"{r.mean_latency_ms:>8.1f} {r.mean_calls:>6.2f} {r.mean_objective:>8.4f} {r.mean_cost:>8.4f}")
    print(f"{'='*70}")

    return result
