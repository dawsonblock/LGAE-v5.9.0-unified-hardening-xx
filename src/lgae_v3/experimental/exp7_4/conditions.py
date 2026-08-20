"""Conditions for exp7.4.

Five conditions:
  A. Fixed topology
  B. Rule-based dynamic router
  C. LGAE task-conditioned (exp7.3 style)
  D. LGAE learned node-necessity router (exp7.4)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..exp7_2.ai_node import create_default_nodes
from ..exp7_2.topology_runtime import AITopology, AIRuntime, StructuralTransitionRecord, create_default_topology
from ..exp7_2.model_backend import ModelBackend, MockModelBackend
from ..exp7_2.objective import ObjectiveWeights, compute_objective_from_record, compute_quality_per_token
from ..exp7_2.benchmark import BenchmarkTask, generate_benchmark
from ..exp7_2.quality_evaluators import evaluate_quality
from ..exp7_2.conditions import run_fixed_topology, run_dynamic_router, ConditionResult
from ..exp7_3.conditions import run_lgae_adaptive_v2
from .node_necessity_router import NodeNecessityRouter, RoutingDecision


def run_lgae_node_necessity(
    tasks: list[BenchmarkTask],
    backend: ModelBackend,
    weights: ObjectiveWeights,
    *,
    calibration_interval: int = 20,
    shadow_batch_size: int = 5,
    k_neighbors: int = 5,
    min_samples: int = 3,
) -> ConditionResult:
    """Condition D: LGAE learned node-necessity router.

    Uses per-node marginal value estimation to decide which nodes
    to include for each task. Learns from shadow executions.
    """
    router = NodeNecessityRouter(
        backend, weights,
        k_neighbors=k_neighbors,
        min_samples=min_samples,
        calibration_interval=calibration_interval,
        shadow_batch_size=shadow_batch_size,
    )

    records = []
    calibration_batch = []

    for i, task in enumerate(tasks):
        # Route the task using learned marginal values.
        topo, decision = router.route_task(task.task_id, task.input)

        runtime = AIRuntime(topo, backend)
        record = runtime.execute_task(task.task_id, task.input, task.task_class)
        record.final_quality = evaluate_quality(
            task.task_class, record.output, task.expected_output,
            record.verification_outcome, record.output,
        )
        record.objective_value = compute_objective_from_record(record, weights)
        records.append(record)

        calibration_batch.append({"task_id": task.task_id, "input": task.input, "task_class": task.task_class})

        # Calibrate periodically.
        if (i + 1) % calibration_interval == 0 and len(calibration_batch) >= shadow_batch_size:
            router.calibrate(calibration_batch[-shadow_batch_size:])
            calibration_batch.clear()

    return _aggregate("D_lgae_node_necessity", records, topo, router)


def _aggregate(
    condition_name: str,
    records: list[StructuralTransitionRecord],
    topology: AITopology,
    router: Optional[NodeNecessityRouter] = None,
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

    per_class = {}
    for r in records:
        cls = r.task_class
        if cls not in per_class:
            per_class[cls] = {"quality": [], "tokens": [], "calls": [], "objective": [], "cost": [], "qpt": []}
        per_class[cls]["quality"].append(r.final_quality)
        per_class[cls]["tokens"].append(r.total_tokens)
        per_class[cls]["calls"].append(r.total_llm_calls)
        per_class[cls]["objective"].append(r.objective_value)
        per_class[cls]["cost"].append(r.total_cost)
        per_class[cls]["qpt"].append(compute_quality_per_token(r.final_quality, r.total_tokens))

    for cls in per_class:
        for key in per_class[cls]:
            vals = per_class[cls][key]
            per_class[cls][key] = round(float(np.mean(vals)), 4) if vals else 0.0

    result = ConditionResult(
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
        per_class=per_class,
        final_topology_summary=topology.summary(),
    )

    if router:
        result.n_mutations = router.n_calibrations  # type: ignore
        result.router_summary = router.get_routing_summary()  # type: ignore
        result.routing_patterns = router.inspect_routing_patterns()  # type: ignore

    return result
