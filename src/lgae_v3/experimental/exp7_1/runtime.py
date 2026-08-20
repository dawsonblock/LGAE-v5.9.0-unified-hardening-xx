"""AI Runtime: executes tasks through the topology.

The runtime is the execution engine. It takes a task, routes it
through the topology's active edges, and collects telemetry.

If LGAE removes an edge, the runtime stops using that route.
The graph is not merely observational — it controls execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np

from .ai_node import AINode, NodeRole, NodeTelemetry
from .topology import AITopology, TopologyEdge, EdgeTelemetry


@dataclass
class TaskResult:
    """Result of executing a task through the topology."""
    task_id: str
    task_class: str
    output: str = ""
    success: bool = False
    quality_score: float = 0.0
    # Aggregated metrics.
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_llm_calls: int = 0
    total_failures: int = 0
    total_tool_calls: int = 0
    # Per-node telemetry.
    node_telemetry: list[NodeTelemetry] = field(default_factory=list)
    # Per-edge telemetry.
    edge_traversals: list[dict] = field(default_factory=list)
    # Execution trace.
    execution_trace: list[str] = field(default_factory=list)
    # Final objective value.
    objective_value: float = 0.0

    @property
    def cost_proxy(self) -> float:
        """Cost proxy: tokens * 0.001 + latency_s * 0.01 + calls * 0.05."""
        return (
            self.total_tokens * 0.001
            + self.total_latency_ms / 1000 * 0.01
            + self.total_llm_calls * 0.05
        )


class AIRuntime:
    """Executes tasks through the AI topology.

    The runtime follows the active edges in the topology. If an edge
    is removed or bypassed, the runtime does not use that route.

    Execution model:
      1. Start at the planner (or memory if planner is bypassed)
      2. Follow active edges to process the task
      3. Collect telemetry at each node and edge
      4. Return the final result with aggregated metrics
    """

    def __init__(
        self,
        topology: AITopology,
        max_steps: int = 15,
        max_retries: int = 2,
    ) -> None:
        self.topology = topology
        self.max_steps = max_steps
        self.max_retries = max_retries

    def execute_task(
        self,
        task_id: str,
        task_input: str,
        task_class: str = "general",
    ) -> TaskResult:
        """Execute a single task through the topology."""
        result = TaskResult(task_id=task_id, task_class=task_class)
        context = ""
        current_node_id = "planner"
        visited = []
        retry_count = 0

        # If planner is bypassed, start at memory or worker.
        if "planner" not in self.topology.nodes:
            current_node_id = "memory" if "memory" in self.topology.nodes else "worker"

        for step in range(self.max_steps):
            if current_node_id not in self.topology.nodes:
                break

            node = self.topology.nodes[current_node_id]
            trace_entry = f"step={step} node={current_node_id}"
            result.execution_trace.append(trace_entry)

            # Invoke the node.
            output, telemetry = node.invoke(task_input, context=context)
            result.node_telemetry.append(telemetry)
            result.total_tokens += telemetry.total_tokens
            result.total_latency_ms += telemetry.latency_ms
            result.total_llm_calls += 1
            result.total_tool_calls += telemetry.tool_calls
            if not telemetry.success:
                result.total_failures += 1

            # Handle verification outcome.
            if telemetry.verification_outcome == "pass":
                result.success = True
                result.output = output
                result.quality_score = telemetry.confidence
                break
            elif telemetry.verification_outcome == "fail":
                # Check if we can retry via feedback loop.
                active_edges = self.topology.get_active_edges_from(current_node_id)
                feedback = [e for e in active_edges if e.destination == "planner"]
                if feedback and retry_count < self.max_retries:
                    current_node_id = "planner"
                    retry_count += 1
                    context = f"Previous attempt failed verification. Retry."
                    continue
                else:
                    result.success = False
                    result.output = output
                    break

            # Update context with node output.
            context = f"{context}\n[{current_node_id}]: {output}"

            # Follow active edges to next node.
            active_edges = self.topology.get_active_edges_from(current_node_id)
            if not active_edges:
                # No outgoing edges — end execution.
                result.output = output
                result.success = telemetry.success
                result.quality_score = telemetry.confidence
                break

            # Select the highest-weight edge.
            # If multiple edges, could use weighted sampling, but for
            # determinism, take the highest weight.
            next_edge = active_edges[0]
            self.topology.record_edge_usage(
                next_edge.source, next_edge.destination,
                tokens=telemetry.total_tokens,
                latency_ms=telemetry.latency_ms,
                success=telemetry.success,
            )
            result.edge_traversals.append({
                "edge_id": next_edge.edge_id,
                "source": next_edge.source,
                "destination": next_edge.destination,
                "weight": next_edge.weight,
            })

            current_node_id = next_edge.destination
            task_input = output  # pass output to next node

        # If we exhausted steps without verification, use last output.
        if not result.output:
            result.output = context[-200:] if context else "no output"
            result.success = False
            result.quality_score = 0.0

        return result

    def execute_batch(
        self,
        tasks: list[dict],
    ) -> list[TaskResult]:
        """Execute a batch of tasks."""
        results = []
        for task in tasks:
            result = self.execute_task(
                task_id=task["task_id"],
                task_input=task["input"],
                task_class=task.get("task_class", "general"),
            )
            results.append(result)
        return results
