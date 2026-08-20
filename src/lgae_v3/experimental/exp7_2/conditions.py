"""Three experimental conditions for exp7.2.

A. Fixed topology — no adaptation
B. Hand-written dynamic router — rule-based, uses task metadata
C. LGAE adaptive topology — structural planning, NO task metadata

The key difference from exp7.1: the dynamic router uses task flags
(benefits_from_research, etc.) while LGAE only sees telemetry.
This is a fair test: can LGAE learn what the human router knows?
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .ai_node import create_default_nodes
from .topology_runtime import AITopology, TopologyEdge, AIRuntime, StructuralTransitionRecord, create_default_topology
from .model_backend import ModelBackend, MockModelBackend
from .objective import ObjectiveWeights, compute_objective_from_record, compute_quality_per_token, compute_quality_per_cost
from .benchmark import BenchmarkTask, generate_benchmark
from .quality_evaluators import evaluate_quality
from .topology_controller import TopologyController


@dataclass
class ConditionResult:
    condition_name: str
    records: list[StructuralTransitionRecord] = field(default_factory=list)
    mean_quality: float = 0.0
    mean_tokens: float = 0.0
    mean_latency_ms: float = 0.0
    mean_calls: float = 0.0
    mean_failures: float = 0.0
    mean_objective: float = 0.0
    mean_cost: float = 0.0
    success_rate: float = 0.0
    quality_per_token: float = 0.0
    quality_per_cost: float = 0.0
    per_class: dict = field(default_factory=dict)
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
            "quality_per_token": round(self.quality_per_token, 6),
            "quality_per_cost": round(self.quality_per_cost, 4),
            "per_class": self.per_class,
            "final_topology_summary": self.final_topology_summary,
            "n_mutations": self.n_mutations,
        }


def run_fixed_topology(
    tasks: list[BenchmarkTask],
    backend: ModelBackend,
    weights: ObjectiveWeights,
) -> ConditionResult:
    """Condition A: Fixed topology — no adaptation."""
    nodes = create_default_nodes()
    topology = create_default_topology(nodes)
    runtime = AIRuntime(topology, backend)

    records = []
    for task in tasks:
        record = runtime.execute_task(task.task_id, task.input, task.task_class)
        record.final_quality = evaluate_quality(
            task.task_class, record.output, task.expected_output,
            record.verification_outcome, record.output,
        )
        record.objective_value = compute_objective_from_record(record, weights)
        records.append(record)

    return _aggregate("A_fixed", records, topology)


def run_dynamic_router(
    tasks: list[BenchmarkTask],
    backend: ModelBackend,
    weights: ObjectiveWeights,
) -> ConditionResult:
    """Condition B: Hand-written dynamic router.

    Uses task metadata (benefits_from_*) to adapt topology per task.
    This is the competent human baseline — LGAE must beat this.
    """
    nodes = create_default_nodes()
    base_topology = create_default_topology(nodes)

    records = []
    for task in tasks:
        topo = base_topology.clone()

        # Rule-based adaptation using task metadata.
        if task.benefits_from_research:
            topo.reweight_edge("planner", "researcher", 2.0)
            topo.reweight_edge("researcher", "worker", 2.0)
        else:
            topo.reweight_edge("planner", "researcher", 0.0)
            topo.reweight_edge("researcher", "worker", 0.0)

        if task.benefits_from_critic:
            topo.reweight_edge("worker", "critic", 2.0)
            topo.reweight_edge("critic", "worker", 1.0)  # retry loop
        else:
            # Bypass critic, go directly to verifier.
            topo.reweight_edge("worker", "critic", 0.0)
            topo.add_edge("worker", "verifier", 1.5)

        if task.benefits_from_memory:
            topo.reweight_edge("memory", "planner", 2.0)
            topo.reweight_edge("memory", "worker", 1.5)
        else:
            topo.reweight_edge("memory", "planner", 0.0)
            topo.reweight_edge("memory", "worker", 0.0)

        if task.benefits_from_planning and not task.benefits_from_research:
            topo.reweight_edge("planner", "worker", 2.0)

        if not task.benefits_from_verification:
            # Lighter verification path.
            topo.reweight_edge("critic", "verifier", 0.3)

        runtime = AIRuntime(topo, backend)
        record = runtime.execute_task(task.task_id, task.input, task.task_class)
        record.final_quality = evaluate_quality(
            task.task_class, record.output, task.expected_output,
            record.verification_outcome, record.output,
        )
        record.objective_value = compute_objective_from_record(record, weights)
        records.append(record)

    return _aggregate("B_dynamic", records, base_topology)


def run_lgae_adaptive(
    tasks: list[BenchmarkTask],
    backend: ModelBackend,
    weights: ObjectiveWeights,
    *,
    adaptation_interval: int = 20,
    shadow_batch_size: int = 5,
) -> ConditionResult:
    """Condition C: LGAE adaptive topology.

    LGAE does NOT see task metadata. It only sees telemetry and
    topology structure. It learns which mutations improve the
    objective via shadow execution.
    """
    nodes = create_default_nodes()
    topology = create_default_topology(nodes)
    controller = TopologyController(
        topology, backend, weights,
        shadow_batch_size=shadow_batch_size,
        max_mutations_per_cycle=3,
        rollback_threshold=0.1,
    )
    runtime = AIRuntime(topology, backend)

    records = []
    shadow_batch = []

    for i, task in enumerate(tasks):
        # LGAE does NOT use task metadata — only task input.
        record = runtime.execute_task(task.task_id, task.input, task.task_class)
        record.final_quality = evaluate_quality(
            task.task_class, record.output, task.expected_output,
            record.verification_outcome, record.output,
        )
        record.objective_value = compute_objective_from_record(record, weights)
        records.append(record)

        # Collect shadow tasks (LGAE sees these as generic inputs).
        shadow_batch.append({"task_id": task.task_id, "input": task.input, "task_class": task.task_class})

        # Adapt every adaptation_interval tasks.
        if (i + 1) % adaptation_interval == 0 and len(shadow_batch) >= shadow_batch_size:
            controller.adapt(shadow_batch[-shadow_batch_size:])

            # Check rollback.
            recent_obj = float(np.mean([r.objective_value for r in records[-adaptation_interval:]]))
            if controller.check_rollback(recent_obj):
                controller.rollback()

            # Rebuild runtime with current topology.
            runtime = AIRuntime(controller.topology, backend)
            shadow_batch.clear()

    n_mutations = sum(1 for r in controller.mutation_history if r.applied)
    result = _aggregate("C_lgae", records, controller.topology)
    result.n_mutations = n_mutations
    return result


def _aggregate(
    condition_name: str,
    records: list[StructuralTransitionRecord],
    topology: AITopology,
) -> ConditionResult:
    """Aggregate records into a condition result."""
    if not records:
        return ConditionResult(condition_name=condition_name)

    qualities = [r.final_quality for r in records]
    tokens = [r.total_tokens for r in records]
    latencies = [r.total_latency_ms for r in records]
    calls = [r.total_llm_calls for r in records]
    failures = [r.total_failures for r in records]
    objectives = [r.objective_value for r in records]
    costs = [r.total_cost for r in records]
    successes = [1.0 if r.success else 0.0 for r in records]

    qpt = [compute_quality_per_token(q, t) for q, t in zip(qualities, tokens)]
    qpc = [compute_quality_per_cost(q, c) for q, c in zip(qualities, costs)]

    # Per-class breakdown.
    per_class = {}
    for r in records:
        cls = r.task_class
        if cls not in per_class:
            per_class[cls] = {
                "quality": [], "tokens": [], "latency": [],
                "calls": [], "failures": [], "objective": [],
                "success": [], "cost": [], "qpt": [],
            }
        per_class[cls]["quality"].append(r.final_quality)
        per_class[cls]["tokens"].append(r.total_tokens)
        per_class[cls]["latency"].append(r.total_latency_ms)
        per_class[cls]["calls"].append(r.total_llm_calls)
        per_class[cls]["failures"].append(r.total_failures)
        per_class[cls]["objective"].append(r.objective_value)
        per_class[cls]["success"].append(1.0 if r.success else 0.0)
        per_class[cls]["cost"].append(r.total_cost)
        per_class[cls]["qpt"].append(compute_quality_per_token(r.final_quality, r.total_tokens))

    for cls in per_class:
        for key in per_class[cls]:
            vals = per_class[cls][key]
            per_class[cls][key] = round(float(np.mean(vals)), 4) if vals else 0.0

    return ConditionResult(
        condition_name=condition_name,
        records=records,
        mean_quality=float(np.mean(qualities)),
        mean_tokens=float(np.mean(tokens)),
        mean_latency_ms=float(np.mean(latencies)),
        mean_calls=float(np.mean(calls)),
        mean_failures=float(np.mean(failures)),
        mean_objective=float(np.mean(objectives)),
        mean_cost=float(np.mean(costs)),
        success_rate=float(np.mean(successes)),
        quality_per_token=float(np.mean(qpt)),
        quality_per_cost=float(np.mean(qpc)),
        per_class=per_class,
        final_topology_summary=topology.summary(),
    )
