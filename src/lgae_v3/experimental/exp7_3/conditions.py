"""Three conditions for exp7.3.

A. Fixed topology — no adaptation
B. Hand-written dynamic router — rule-based, uses task metadata
C. LGAE adaptive topology V2 — feature-aware, conformal-gated, incremental

Condition C uses the improved controller with:
  - Task feature extraction (no labels)
  - Larger shadow batches (20 tasks)
  - Incremental adaptation
  - Conformal advantage gate
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
from .topology_controller_v2 import TopologyControllerV2
from .task_features import extract_features


def run_lgae_adaptive_v2(
    tasks: list[BenchmarkTask],
    backend: ModelBackend,
    weights: ObjectiveWeights,
    *,
    adaptation_interval: int = 20,
    shadow_batch_size: int = 20,
    conformal_alpha: float = 0.2,
    use_task_features: bool = True,
    online_rollback: bool = True,
) -> ConditionResult:
    """Condition C/D: LGAE adaptive topology V2.

    With use_task_features=True: Condition D (task-conditioned)
    With use_task_features=False: Condition C (telemetry-only)

    Improvements over exp7.2:
    - Task feature extraction (no labels, just text analysis) [optional]
    - Larger shadow batches (20 tasks default)
    - Incremental adaptation (cumulative mutation effects)
    - Conformal advantage gate (data-driven threshold)
    - Online rollback based on rolling J degradation
    """
    nodes = create_default_nodes()
    topology = create_default_topology(nodes)
    controller = TopologyControllerV2(
        topology, backend, weights,
        shadow_batch_size=shadow_batch_size,
        max_mutations_per_cycle=3,
        rollback_threshold=0.1,
        conformal_alpha=conformal_alpha,
        use_task_features=use_task_features,
        online_rollback_window=10,
        online_rollback_epsilon=0.05,
    )
    runtime = AIRuntime(topology, backend)

    records = []
    shadow_batch = []
    n_online_rollbacks = 0

    for i, task in enumerate(tasks):
        # LGAE sees only the task input, NOT the task metadata.
        record = runtime.execute_task(task.task_id, task.input, task.task_class)
        record.final_quality = evaluate_quality(
            task.task_class, record.output, task.expected_output,
            record.verification_outcome, record.output,
        )
        record.objective_value = compute_objective_from_record(record, weights)
        records.append(record)

        # Online rollback check.
        if online_rollback:
            rolled_back = controller.observe_objective(record.objective_value)
            if rolled_back:
                n_online_rollbacks += 1
                runtime = AIRuntime(controller.topology, backend)

        shadow_batch.append({"task_id": task.task_id, "input": task.input, "task_class": task.task_class})

        if (i + 1) % adaptation_interval == 0 and len(shadow_batch) >= shadow_batch_size:
            controller.adapt(shadow_batch[-shadow_batch_size:])

            recent_obj = float(np.mean([r.objective_value for r in records[-adaptation_interval:]]))
            if controller.check_rollback(recent_obj):
                controller.rollback()

            runtime = AIRuntime(controller.topology, backend)
            shadow_batch.clear()

    n_mutations = sum(1 for r in controller.mutation_history if r.applied)
    condition_name = "D_lgae_task_conditioned" if use_task_features else "C_lgae_telemetry_only"
    result = _aggregate(condition_name, records, controller.topology)
    result.n_mutations = n_mutations
    # Store extra info in the result for analysis.
    result.n_online_rollbacks = n_online_rollbacks  # type: ignore
    result.shadow_advantages = controller.shadow_advantages  # type: ignore
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
        per_class=per_class,
        final_topology_summary=topology.summary(),
    )
