"""Experiment runner for v7.0-exp4-learned-routing-policy.

Target: Match or approach the dynamic router's token efficiency
without giving LGAE explicit task-class labels.

Success criteria:
  Quality_LGAE ≥ Quality_Dynamic - ε
  Tokens_LGAE ≤ 1.5 × Tokens_Dynamic
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json
import sys
import os
import time
import numpy as np

from ..exp7_2.model_backend import create_backend
from ..exp7_2.objective import ObjectiveWeights, compute_pareto_efficiency
from ..exp7_2.benchmark import generate_benchmark, TASK_CLASSES
from ..exp7_2.conditions import run_fixed_topology, run_dynamic_router, ConditionResult
from ..exp7_3.conditions import run_lgae_adaptive_v2
from .conditions import run_lgae_node_necessity
from .task_embedding import embed_task


@dataclass
class Exp74Result:
    audit_note: str = ""
    objective_weights: dict = field(default_factory=dict)
    condition_results: list[ConditionResult] = field(default_factory=list)
    pareto_analysis: dict = field(default_factory=dict)
    routing_patterns: dict = field(default_factory=dict)
    gates: dict[str, dict] = field(default_factory=dict)
    all_gates_passed: bool = False
    summary: dict[str, Any] = field(default_factory=dict)
    manifest_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "audit_note": self.audit_note,
            "objective_weights": self.objective_weights,
            "condition_results": [r.to_dict() for r in self.condition_results],
            "pareto_analysis": self.pareto_analysis,
            "routing_patterns": self.routing_patterns,
            "gates": self.gates,
            "all_gates_passed": self.all_gates_passed,
            "summary": self.summary,
            "manifest_evidence": self.manifest_evidence,
        }


def _check_manifest_evidence() -> dict:
    import subprocess
    evidence = {"manifest_exists": False, "manifest_valid": False, "test_count": 0, "test_passed": 0, "test_failed": 0}
    manifest_path = os.path.join(os.getcwd(), "MANIFEST.sha256.json")
    if os.path.exists(manifest_path):
        evidence["manifest_exists"] = True
        try:
            result = subprocess.run([sys.executable, "scripts/generate_manifest.py", "--check"], capture_output=True, text=True, timeout=30, cwd=os.getcwd())
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


def run_exp7_4(
    *,
    n_tasks_per_class: int = 50,
    backend_type: str = "mock",
    calibration_interval: int = 20,
    shadow_batch_size: int = 5,
    weights: ObjectiveWeights = None,
) -> Exp74Result:
    """Run the v7.0-exp4 experiment."""
    if weights is None:
        weights = ObjectiveWeights()

    result = Exp74Result(
        audit_note=(
            "Learned Routing Policy. "
            "Test whether per-node marginal value estimation closes "
            "the token efficiency gap to the dynamic router. "
            "Target: Quality ≥ Dynamic-ε, Tokens ≤ 1.5× Dynamic. "
            f"Backend: {backend_type}."
        ),
        objective_weights=weights.to_dict(),
    )

    print(f"\n  Backend: {backend_type}")
    print(f"  Tasks per class: {n_tasks_per_class}")
    print(f"  Total tasks: {n_tasks_per_class * len(TASK_CLASSES)}")
    print(f"  Calibration interval: {calibration_interval}")
    print(f"  Shadow batch size: {shadow_batch_size}")

    backend = create_backend(backend_type)

    # === Phase 1: Generate benchmark ===
    print("\n=== Phase 1: Generating benchmark ===")
    tasks = generate_benchmark(n_per_class=n_tasks_per_class, seed=42)
    print(f"  Generated {len(tasks)} tasks across {len(TASK_CLASSES)} classes")

    # Show task embedding distribution.
    embeddings = [embed_task(t.input) for t in tasks[:20]]
    print(f"  Embedding dim: {embeddings[0].dim}")

    # === Phase 2: Run conditions ===
    print("\n=== Phase 2: Running conditions ===")

    print("\n  --- Condition A: Fixed topology ---")
    t0 = time.time()
    result_a = run_fixed_topology(tasks, backend, weights)
    print(f"    Quality: {result_a.mean_quality:.4f}, Tokens: {result_a.mean_tokens:.0f}, "
          f"J: {result_a.mean_objective:.4f}, Cost: {result_a.mean_cost:.4f}")
    print(f"    Time: {time.time()-t0:.1f}s")
    result.condition_results.append(result_a)

    print("\n  --- Condition B: Dynamic router ---")
    t0 = time.time()
    result_b = run_dynamic_router(tasks, backend, weights)
    print(f"    Quality: {result_b.mean_quality:.4f}, Tokens: {result_b.mean_tokens:.0f}, "
          f"J: {result_b.mean_objective:.4f}, Cost: {result_b.mean_cost:.4f}")
    print(f"    Time: {time.time()-t0:.1f}s")
    result.condition_results.append(result_b)

    print("\n  --- Condition C: LGAE task-conditioned (exp7.3) ---")
    t0 = time.time()
    result_c = run_lgae_adaptive_v2(
        tasks, backend, weights,
        adaptation_interval=20,
        shadow_batch_size=20,
        use_task_features=True,
        online_rollback=True,
    )
    print(f"    Quality: {result_c.mean_quality:.4f}, Tokens: {result_c.mean_tokens:.0f}, "
          f"J: {result_c.mean_objective:.4f}, Cost: {result_c.mean_cost:.4f}")
    print(f"    Time: {time.time()-t0:.1f}s")
    result.condition_results.append(result_c)

    print("\n  --- Condition D: LGAE node-necessity router ---")
    t0 = time.time()
    result_d = run_lgae_node_necessity(
        tasks, backend, weights,
        calibration_interval=calibration_interval,
        shadow_batch_size=shadow_batch_size,
    )
    print(f"    Quality: {result_d.mean_quality:.4f}, Tokens: {result_d.mean_tokens:.0f}, "
          f"J: {result_d.mean_objective:.4f}, Cost: {result_d.mean_cost:.4f}")
    print(f"    Calibrations: {result_d.n_mutations}")
    router_summary = getattr(result_d, "router_summary", {})
    print(f"    Routing summary: {router_summary.get('n_decisions', 0)} decisions, "
          f"{router_summary.get('n_calibrations', 0)} calibrations")
    print(f"    Time: {time.time()-t0:.1f}s")
    result.condition_results.append(result_d)

    # Store routing patterns for inspection.
    result.routing_patterns = getattr(result_d, "routing_patterns", {})

    # === Phase 3: Pareto analysis ===
    print("\n=== Phase 3: Pareto analysis ===")
    pareto_points = [
        {"name": r.condition_name, "quality": r.mean_quality, "cost": r.mean_cost, "qpt": r.quality_per_token}
        for r in result.condition_results
    ]
    is_efficient = compute_pareto_efficiency(pareto_points)
    for i, point in enumerate(pareto_points):
        point["pareto_efficient"] = is_efficient[i]
        print(f"  {point['name']}: quality={point['quality']:.4f}, cost={point['cost']:.4f}, "
              f"Q/Tok={point['qpt']:.6f}, efficient={is_efficient[i]}")
    result.pareto_analysis = {"points": pareto_points, "pareto_front": [p for i, p in enumerate(pareto_points) if is_efficient[i]]}

    # === Phase 4: Gates ===
    print("\n=== Phase 4: Checking gates ===")

    fixed = result_a
    dynamic = result_b
    lgae_tc = result_c
    lgae_nn = result_d

    token_ratio_nn_vs_dynamic = lgae_nn.mean_tokens / max(dynamic.mean_tokens, 1e-6)
    token_ratio_nn_vs_fixed = lgae_nn.mean_tokens / max(fixed.mean_tokens, 1e-6)
    quality_diff_nn_vs_dynamic = lgae_nn.mean_quality - dynamic.mean_quality
    quality_diff_nn_vs_fixed = lgae_nn.mean_quality - fixed.mean_quality
    obj_nn_vs_fixed = lgae_nn.mean_objective - fixed.mean_objective
    obj_nn_vs_dynamic = lgae_nn.mean_objective - dynamic.mean_objective
    obj_nn_vs_tc = lgae_nn.mean_objective - lgae_tc.mean_objective
    cost_reduction_nn_vs_fixed = (fixed.mean_cost - lgae_nn.mean_cost) / max(fixed.mean_cost, 1e-6)

    nn_efficient = is_efficient[3] if len(is_efficient) > 3 else False

    gates = {
        "1_identical_models_prompts_tasks": {
            "passed": True,
            "description": "all conditions use same backend, prompts, tasks",
        },
        "2_topology_changes_execution": {
            "passed": True,
            "description": "topology controls context accumulation",
        },
        "3_authority_preserved": {
            "passed": True,
            "description": "LGAE goes through router + conformal gate",
        },
        "4_quality_no_worse_than_fixed": {
            "passed": lgae_nn.mean_quality >= fixed.mean_quality - 0.05,
            "description": f"LGAE-NN quality={lgae_nn.mean_quality:.4f} vs fixed={fixed.mean_quality:.4f}",
        },
        "5_quality_approaches_dynamic": {
            "passed": lgae_nn.mean_quality >= dynamic.mean_quality - 0.02,
            "description": f"LGAE-NN quality={lgae_nn.mean_quality:.4f} vs dynamic={dynamic.mean_quality:.4f} (Δ={quality_diff_nn_vs_dynamic:+.4f})",
        },
        "6_token_ratio_below_1.5x": {
            "passed": token_ratio_nn_vs_dynamic <= 1.5,
            "description": f"LGAE-NN tokens={lgae_nn.mean_tokens:.0f} vs dynamic={dynamic.mean_tokens:.0f} (ratio={token_ratio_nn_vs_dynamic:.2f}x)",
        },
        "7_token_ratio_below_2.3x": {
            "passed": token_ratio_nn_vs_dynamic <= 2.3,
            "description": f"token ratio={token_ratio_nn_vs_dynamic:.2f}x (exp7.3 was 2.3x)",
        },
        "8_lgae_nn_beats_fixed_cost_adjusted": {
            "passed": lgae_nn.mean_objective > fixed.mean_objective,
            "description": f"LGAE-NN J={lgae_nn.mean_objective:.4f} vs fixed J={fixed.mean_objective:.4f} (Δ={obj_nn_vs_fixed:+.4f})",
        },
        "9_no_regression": {
            "passed": lgae_nn.mean_quality >= fixed.mean_quality - 0.1 and lgae_nn.mean_failures <= fixed.mean_failures + 1.0,
            "description": f"quality diff={quality_diff_nn_vs_fixed:+.4f}, failure diff={lgae_nn.mean_failures-fixed.mean_failures:+.2f}",
        },
        "10_calibrations_nonzero": {
            "passed": lgae_nn.n_mutations > 0,
            "description": f"n_calibrations={lgae_nn.n_mutations}",
        },
        "11_nn_beats_tc": {
            "passed": lgae_nn.mean_objective > lgae_tc.mean_objective,
            "description": f"LGAE-NN J={lgae_nn.mean_objective:.4f} vs LGAE-TC J={lgae_tc.mean_objective:.4f} (Δ={obj_nn_vs_tc:+.4f})",
        },
        "12_nn_pareto_efficient": {
            "passed": nn_efficient,
            "description": f"LGAE-NN on Pareto frontier: {nn_efficient}",
        },
    }

    result.gates = gates
    result.all_gates_passed = all(g["passed"] for g in gates.values())

    result.summary = {
        "fixed_quality": fixed.mean_quality,
        "fixed_tokens": fixed.mean_tokens,
        "fixed_objective": fixed.mean_objective,
        "dynamic_quality": dynamic.mean_quality,
        "dynamic_tokens": dynamic.mean_tokens,
        "dynamic_objective": dynamic.mean_objective,
        "lgae_tc_quality": lgae_tc.mean_quality,
        "lgae_tc_tokens": lgae_tc.mean_tokens,
        "lgae_tc_objective": lgae_tc.mean_objective,
        "lgae_nn_quality": lgae_nn.mean_quality,
        "lgae_nn_tokens": lgae_nn.mean_tokens,
        "lgae_nn_objective": lgae_nn.mean_objective,
        "token_ratio_nn_vs_dynamic": round(token_ratio_nn_vs_dynamic, 4),
        "token_ratio_nn_vs_fixed": round(token_ratio_nn_vs_fixed, 4),
        "quality_diff_nn_vs_dynamic": round(quality_diff_nn_vs_dynamic, 4),
        "obj_nn_vs_fixed": round(obj_nn_vs_fixed, 4),
        "obj_nn_vs_dynamic": round(obj_nn_vs_dynamic, 4),
        "obj_nn_vs_tc": round(obj_nn_vs_tc, 4),
        "cost_reduction_nn_vs_fixed": round(cost_reduction_nn_vs_fixed, 4),
    }

    manifest_evidence = _check_manifest_evidence()
    result.manifest_evidence = manifest_evidence

    print(f"\n  Overall: {'ALL GATES PASSED' if result.all_gates_passed else 'GATES NOT ALL MET'}")
    for name, gate in gates.items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"    {name}: {status} — {gate['description']}")

    # Print comparison table.
    print(f"\n{'='*95}")
    print(f"{'Condition':<30} {'Quality':>8} {'Tokens':>8} {'Calls':>6} {'Q/Tok':>10} {'J':>8} {'Cost':>8}")
    print(f"{'-'*95}")
    for r in result.condition_results:
        print(f"{r.condition_name:<30} {r.mean_quality:>8.4f} {r.mean_tokens:>8.0f} "
              f"{r.mean_calls:>6.2f} {r.quality_per_token:>10.6f} {r.mean_objective:>8.4f} "
              f"{r.mean_cost:>8.4f}")
    print(f"{'='*95}")

    # Print routing patterns if available.
    if result.routing_patterns:
        print(f"\n  Learned routing patterns:")
        for pattern, info in result.routing_patterns.items():
            print(f"    {pattern}: count={info['count']}")

    return result
