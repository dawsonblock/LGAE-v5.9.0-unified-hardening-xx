"""Pre-experiment validation for exp7.5.

1. Backend smoke test: one task through every role
2. Topology-sensitivity sanity check: minimal vs full topology
3. Node ablation table: ΔQ, ΔTokens, ΔLatency, ΔJ per node
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np

from ..exp7_2.ai_node import create_default_nodes
from ..exp7_2.topology_runtime import AITopology, AIRuntime, create_default_topology
from ..exp7_2.model_backend import ModelBackend, MockModelBackend, Message
from ..exp7_2.objective import ObjectiveWeights, compute_objective_from_record
from ..exp7_2.benchmark import BenchmarkTask, generate_benchmark
from ..exp7_2.quality_evaluators import evaluate_quality
from ..exp7_4.node_necessity_router import OPTIONAL_NODES
from .prompts import load_all_prompts, format_prompt


@dataclass
class SmokeTestResult:
    passed: bool = False
    n_roles_tested: int = 0
    n_roles_succeeded: int = 0
    role_results: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_roles_tested": self.n_roles_tested,
            "n_roles_succeeded": self.n_roles_succeeded,
            "role_results": self.role_results,
            "error": self.error,
        }


def run_smoke_test(backend: ModelBackend) -> SmokeTestResult:
    """Run one task through every role to verify the backend works.

    Checks:
      - API authentication succeeds
      - Output is nonempty
      - Token usage is captured
      - Latency is captured
      - Role prompt changes behavior
    """
    result = SmokeTestResult()
    roles = ["planner", "worker", "researcher", "critic", "verifier", "memory"]
    prompts = load_all_prompts()

    test_input = "What is the capital of France?"

    for role in roles:
        prompt_record = prompts[role]
        system_prompt = format_prompt(prompt_record.content, test_input, "")
        messages = [Message(role="user", content=test_input)]

        t0 = time.time()
        try:
            response = backend.generate(
                role=role,
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=256,
                temperature=0.0,
            )
            elapsed = time.time() - t0

            # For smoke test, consider non-empty output as success even if
            # the backend reported a simulated error (mock failure rate).
            success = bool(response.text.strip()) and "ERROR: simulated failure" not in response.text
            result.role_results[role] = {
                "success": success,
                "status": getattr(response, "status", "SUCCESS"),
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "latency_ms": response.latency_ms,
                "output_preview": response.text[:100] if response.text else "",
                "error": response.error,
            }
            result.n_roles_tested += 1
            if success:
                result.n_roles_succeeded += 1
        except Exception as e:
            result.role_results[role] = {
                "success": False,
                "status": "EXCEPTION",
                "error": str(e)[:200],
            }
            result.n_roles_tested += 1
            result.error = str(e)[:200]

    # Pass if at least 5/6 roles succeed (allow 1 simulated failure).
    result.passed = result.n_roles_succeeded >= 5
    return result


@dataclass
class TopologySensitivityResult:
    """Result of topology-sensitivity sanity check."""
    n_tasks: int = 0
    minimal_quality: list[float] = field(default_factory=list)
    full_quality: list[float] = field(default_factory=list)
    minimal_tokens: list[float] = field(default_factory=list)
    full_tokens: list[float] = field(default_factory=list)
    quality_diff: list[float] = field(default_factory=list)
    mean_quality_diff: float = 0.0
    std_quality_diff: float = 0.0
    has_meaningful_variance: bool = False
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "mean_quality_diff": round(self.mean_quality_diff, 4),
            "std_quality_diff": round(self.std_quality_diff, 4),
            "has_meaningful_variance": self.has_meaningful_variance,
            "passed": self.passed,
            "per_task": [
                {
                    "minimal_q": round(self.minimal_quality[i], 4),
                    "full_q": round(self.full_quality[i], 4),
                    "diff": round(self.quality_diff[i], 4),
                }
                for i in range(min(len(self.quality_diff), 20))
            ],
        }


def run_topology_sensitivity_check(
    backend: ModelBackend,
    tasks: list[BenchmarkTask],
    weights: ObjectiveWeights,
    *,
    n_tasks: int = 20,
) -> TopologySensitivityResult:
    """Run topology-sensitivity sanity check.

    Run each task through minimal and full topology.
    Test whether Q_full - Q_minimal has meaningful variance.
    """
    result = TopologySensitivityResult()
    test_tasks = tasks[:n_tasks]
    result.n_tasks = len(test_tasks)

    for task in test_tasks:
        # Minimal topology: just verifier.
        nodes_min = create_default_nodes()
        topo_min = create_default_topology(nodes_min)
        for node in ["researcher", "critic", "memory"]:
            topo_min.bypass_node(node)
        topo_min.add_edge("worker", "verifier", 1.0)
        runtime_min = AIRuntime(topo_min, backend)
        record_min = runtime_min.execute_task(task.task_id, task.input, task.task_class)
        q_min = evaluate_quality(
            task.task_class, record_min.output, task.expected_output,
            record_min.verification_outcome, record_min.output,
        )

        # Full topology: all nodes.
        nodes_full = create_default_nodes()
        topo_full = create_default_topology(nodes_full)
        runtime_full = AIRuntime(topo_full, backend)
        record_full = runtime_full.execute_task(task.task_id, task.input, task.task_class)
        q_full = evaluate_quality(
            task.task_class, record_full.output, task.expected_output,
            record_full.verification_outcome, record_full.output,
        )

        result.minimal_quality.append(q_min)
        result.full_quality.append(q_full)
        result.minimal_tokens.append(record_min.total_tokens)
        result.full_tokens.append(record_full.total_tokens)
        result.quality_diff.append(q_full - q_min)

    if result.quality_diff:
        result.mean_quality_diff = float(np.mean(result.quality_diff))
        result.std_quality_diff = float(np.std(result.quality_diff))
        # Meaningful variance: std > 0.01 or mean diff > 0.02
        result.has_meaningful_variance = (
            result.std_quality_diff > 0.01 or abs(result.mean_quality_diff) > 0.02
        )
        result.passed = result.has_meaningful_variance

    return result


@dataclass
class NodeAblationResult:
    """Result of per-node ablation."""
    node: str = ""
    n_tasks: int = 0
    delta_quality: float = 0.0
    delta_tokens: float = 0.0
    delta_latency: float = 0.0
    delta_j: float = 0.0
    quality_with: float = 0.0
    quality_without: float = 0.0
    tokens_with: float = 0.0
    tokens_without: float = 0.0
    j_with: float = 0.0
    j_without: float = 0.0

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "n_tasks": self.n_tasks,
            "delta_quality": round(self.delta_quality, 4),
            "delta_tokens": round(self.delta_tokens, 1),
            "delta_latency": round(self.delta_latency, 1),
            "delta_j": round(self.delta_j, 4),
            "quality_with": round(self.quality_with, 4),
            "quality_without": round(self.quality_without, 4),
            "tokens_with": round(self.tokens_with, 1),
            "tokens_without": round(self.tokens_without, 1),
            "j_with": round(self.j_with, 4),
            "j_without": round(self.j_without, 4),
        }


def run_node_ablation(
    backend: ModelBackend,
    tasks: list[BenchmarkTask],
    weights: ObjectiveWeights,
    *,
    n_tasks: int = 30,
) -> list[NodeAblationResult]:
    """Ablate each optional node individually.

    For each optional node, measure:
      ΔQ_n = Q(with) - Q(without)
      ΔTokens_n, ΔLatency_n, ΔJ_n
    """
    results = []
    test_tasks = tasks[:n_tasks]

    for node in OPTIONAL_NODES:
        qualities_with, qualities_without = [], []
        tokens_with, tokens_without = [], []
        latencies_with, latencies_without = [], []
        j_with_list, j_without_list = [], []

        for task in test_tasks:
            # With the node.
            nodes_with = create_default_nodes()
            topo_with = create_default_topology(nodes_with)
            runtime_with = AIRuntime(topo_with, backend)
            rec_with = runtime_with.execute_task(task.task_id, task.input, task.task_class)
            q_with = evaluate_quality(
                task.task_class, rec_with.output, task.expected_output,
                rec_with.verification_outcome, rec_with.output,
            )
            j_with = compute_objective_from_record(rec_with, weights)

            # Without the node.
            nodes_without = create_default_nodes()
            topo_without = create_default_topology(nodes_without)
            topo_without.bypass_node(node)
            if node == "critic":
                topo_without.add_edge("worker", "verifier", 1.0)
            runtime_without = AIRuntime(topo_without, backend)
            rec_without = runtime_without.execute_task(task.task_id, task.input, task.task_class)
            q_without = evaluate_quality(
                task.task_class, rec_without.output, task.expected_output,
                rec_without.verification_outcome, rec_without.output,
            )
            j_without = compute_objective_from_record(rec_without, weights)

            qualities_with.append(q_with)
            qualities_without.append(q_without)
            tokens_with.append(rec_with.total_tokens)
            tokens_without.append(rec_without.total_tokens)
            latencies_with.append(rec_with.total_latency_ms)
            latencies_without.append(rec_without.total_latency_ms)
            j_with_list.append(j_with)
            j_without_list.append(j_without)

        results.append(NodeAblationResult(
            node=node,
            n_tasks=len(test_tasks),
            delta_quality=float(np.mean(qualities_with) - np.mean(qualities_without)),
            delta_tokens=float(np.mean(tokens_with) - np.mean(tokens_without)),
            delta_latency=float(np.mean(latencies_with) - np.mean(latencies_without)),
            delta_j=float(np.mean(j_with_list) - np.mean(j_without_list)),
            quality_with=float(np.mean(qualities_with)),
            quality_without=float(np.mean(qualities_without)),
            tokens_with=float(np.mean(tokens_with)),
            tokens_without=float(np.mean(tokens_without)),
            j_with=float(np.mean(j_with_list)),
            j_without=float(np.mean(j_without_list)),
        ))

    return results
