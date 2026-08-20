"""Three experimental conditions for exp7.

A. Fixed topology — no adaptation
B. Hand-written dynamic router — rule-based adaptation
C. LGAE adaptive topology — structural planning with conformal arbitration

Everything else is held constant: models, prompts, tasks, tokens, tools, hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time

from .ai_node import AINode, create_default_nodes
from .topology import AITopology, TopologyEdge, create_default_topology
from .runtime import AIRuntime, TaskResult
from .objective import ObjectiveWeights, compute_objective_from_result
from .benchmark import BenchmarkTask, evaluate_quality
from .topology_controller import TopologyController


@dataclass
class ConditionResult:
    """Results for one experimental condition."""
    condition_name: str
    task_results: list[TaskResult] = field(default_factory=list)
    # Aggregated metrics.
    mean_quality: float = 0.0
    mean_tokens: float = 0.0
    mean_latency_ms: float = 0.0
    mean_calls: float = 0.0
    mean_failures: float = 0.0
    mean_objective: float = 0.0
    mean_cost: float = 0.0
    success_rate: float = 0.0
    # Per-class metrics.
    per_class: dict = field(default_factory=dict)
    # Topology info.
    final_topology_summary: dict = field(default_factory=dict)
    n_mutations: int = 0

    def to_dict(self) -> dict:
        return {
            "condition_name": self.condition_name,
            "mean_quality": round(self.mean_quality, 4),
            "mean_tokens": round(self.mean_tokens, 1),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "mean_calls": round(self.mean_calls, 2),
            "mean_failures": round(self.mean_failures, 2),
            "mean_objective": round(self.mean_objective, 4),
            "mean_cost": round(self.mean_cost, 4),
            "success_rate": round(self.success_rate, 4),
            "per_class": self.per_class,
            "final_topology_summary": self.final_topology_summary,
            "n_mutations": self.n_mutations,
        }


def run_fixed_topology(
    tasks: list[BenchmarkTask],
    weights: ObjectiveWeights,
) -> ConditionResult:
    """Condition A: Fixed topology — no adaptation."""
    nodes = create_default_nodes()
    topology = create_default_topology(nodes)
    runtime = AIRuntime(topology)

    results = []
    for task in tasks:
        result = runtime.execute_task(
            task_id=task.task_id,
            task_input=task.input,
            task_class=task.task_class,
        )
        result.quality_score = evaluate_quality(result, task)
        result.objective_value = compute_objective_from_result(result, weights)
        results.append(result)

    return _aggregate_results("A_fixed", results, topology)


def run_dynamic_router(
    tasks: list[BenchmarkTask],
    weights: ObjectiveWeights,
) -> ConditionResult:
    """Condition B: Hand-written dynamic router.

    Uses simple rules to adapt the topology:
    - If task requires verification, increase verifier edge weight
    - If task requires memory, increase memory edge weight
    - If task is simple, bypass critic (skip review)
    - If task is hard, add planner→verifier direct edge
    """
    nodes = create_default_nodes()
    topology = create_default_topology(nodes)

    results = []
    for task in tasks:
        # Apply rule-based adaptation for this task.
        task_topology = topology.clone()

        if task.requires_verification:
            task_topology.reweight_edge("critic", "verifier", 2.0)
        if task.requires_memory:
            task_topology.reweight_edge("memory", "planner", 2.0)
            task_topology.reweight_edge("memory", "worker", 1.5)
        if task.difficulty < 0.3:
            # Simple task: bypass critic.
            task_topology.reweight_edge("worker", "critic", 0.0)
            # Add direct worker→verifier edge.
            task_topology.add_edge("worker", "verifier", 1.0)
        if task.difficulty > 0.7:
            # Hard task: add planner→verifier for early validation.
            task_topology.add_edge("planner", "verifier", 0.3)

        runtime = AIRuntime(task_topology)
        result = runtime.execute_task(
            task_id=task.task_id,
            task_input=task.input,
            task_class=task.task_class,
        )
        result.quality_score = evaluate_quality(result, task)
        result.objective_value = compute_objective_from_result(result, weights)
        results.append(result)

    return _aggregate_results("B_dynamic", results, topology)


def run_lgae_adaptive(
    tasks: list[BenchmarkTask],
    weights: ObjectiveWeights,
    *,
    adaptation_interval: int = 10,
) -> ConditionResult:
    """Condition C: LGAE adaptive topology.

    Uses the TopologyController to propose and evaluate mutations.
    Adapts every N tasks based on observed performance.
    """
    nodes = create_default_nodes()
    topology = create_default_topology(nodes)
    controller = TopologyController(topology, weights)
    runtime = AIRuntime(topology)

    results = []
    eval_task_batch = []

    for i, task in enumerate(tasks):
        result = runtime.execute_task(
            task_id=task.task_id,
            task_input=task.input,
            task_class=task.task_class,
        )
        result.quality_score = evaluate_quality(result, task)
        result.objective_value = compute_objective_from_result(result, weights)
        results.append(result)

        eval_task_batch.append({"task_id": task.task_id, "input": task.input})

        # Adapt every adaptation_interval tasks.
        if (i + 1) % adaptation_interval == 0 and len(eval_task_batch) >= 3:
            # Propose and evaluate mutations.
            mutation_records = controller.propose_and_evaluate(eval_task_batch[-5:])

            # Check rollback.
            current_obj = float(np.mean([r.objective_value for r in results[-adaptation_interval:]]))
            if controller.check_rollback(current_obj):
                controller.rollback()
                # Rebuild runtime with rolled-back topology.
                runtime = AIRuntime(controller.topology)

            # Rebuild runtime with potentially mutated topology.
            runtime = AIRuntime(controller.topology)
            eval_task_batch.clear()

    n_mutations = sum(1 for r in controller.mutation_history if r.applied)

    cond_result = _aggregate_results("C_lgae", results, controller.topology)
    cond_result.n_mutations = n_mutations
    return cond_result


def _aggregate_results(
    condition_name: str,
    results: list[TaskResult],
    topology: AITopology,
) -> ConditionResult:
    """Aggregate task results into a condition result."""
    if not results:
        return ConditionResult(condition_name=condition_name)

    qualities = [r.quality_score for r in results]
    tokens = [r.total_tokens for r in results]
    latencies = [r.total_latency_ms for r in results]
    calls = [r.total_llm_calls for r in results]
    failures = [r.total_failures for r in results]
    objectives = [r.objective_value for r in results]
    costs = [r.cost_proxy for r in results]
    successes = [1.0 if r.success else 0.0 for r in results]

    # Per-class breakdown.
    per_class = {}
    for r in results:
        cls = r.task_class
        if cls not in per_class:
            per_class[cls] = {
                "quality": [], "tokens": [], "latency": [],
                "calls": [], "failures": [], "objective": [],
                "success": [], "cost": [],
            }
        per_class[cls]["quality"].append(r.quality_score)
        per_class[cls]["tokens"].append(r.total_tokens)
        per_class[cls]["latency"].append(r.total_latency_ms)
        per_class[cls]["calls"].append(r.total_llm_calls)
        per_class[cls]["failures"].append(r.total_failures)
        per_class[cls]["objective"].append(r.objective_value)
        per_class[cls]["success"].append(1.0 if r.success else 0.0)
        per_class[cls]["cost"].append(r.cost_proxy)

    # Aggregate per-class.
    for cls in per_class:
        for key in per_class[cls]:
            vals = per_class[cls][key]
            per_class[cls][key] = round(float(np.mean(vals)), 4) if vals else 0.0

    return ConditionResult(
        condition_name=condition_name,
        task_results=results,
        mean_quality=float(np.mean(qualities)),
        mean_tokens=float(np.mean(tokens)),
        mean_latency_ms=float(np.mean(latencies)),
        mean_calls=float(np.mean(calls)),
        mean_failures=float(np.mean(failures)),
        mean_objective=float(np.mean(objectives)),
        mean_cost=float(np.mean(costs)),
        success_rate=float(np.mean(successes)),
        per_class=per_class,
        final_topology_summary=topology.summary(),
    )
