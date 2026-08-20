"""Topology and runtime for exp7.2.

The runtime accumulates context from visited nodes. This is the
critical change: routing through Researcher before Worker gives
Worker different context than routing directly from Planner.

If LGAE removes the Planner→Researcher edge, the Worker never
sees research findings, and its output quality changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import random
import time

from .ai_node import AINode, NodeRole, NodeTelemetry, create_default_nodes
from .model_backend import ModelBackend, MockModelBackend


@dataclass
class TopologyEdge:
    source: str
    destination: str
    weight: float = 1.0
    active: bool = True
    edge_id: str = ""

    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = f"{self.source}->{self.destination}"

    @property
    def is_active(self) -> bool:
        return self.active and self.weight > 0.0


@dataclass
class EdgeTelemetry:
    edge_id: str
    source: str
    destination: str
    messages_sent: int = 0
    tokens_transferred: int = 0
    latency_contribution_ms: float = 0.0
    successful_downstream: int = 0
    failure_propagation: int = 0


class AITopology:
    """Routing topology for AI execution."""

    def __init__(self, nodes: dict[str, AINode], edges: list[TopologyEdge]) -> None:
        self.nodes = nodes
        self.edges: dict[str, TopologyEdge] = {}
        for edge in edges:
            self.edges[edge.edge_id] = edge
        self._edge_telemetry: dict[str, EdgeTelemetry] = {}
        for eid in self.edges:
            self._edge_telemetry[eid] = EdgeTelemetry(
                edge_id=eid,
                source=self.edges[eid].source,
                destination=self.edges[eid].destination,
            )

    def get_active_edges_from(self, node_id: str) -> list[TopologyEdge]:
        active = [e for e in self.edges.values() if e.source == node_id and e.is_active]
        return sorted(active, key=lambda e: -e.weight)

    def has_edge(self, source: str, destination: str) -> bool:
        edge = self.edges.get(f"{source}->{destination}")
        return edge is not None and edge.is_active

    def get_edge(self, source: str, destination: str) -> Optional[TopologyEdge]:
        return self.edges.get(f"{source}->{destination}")

    def add_edge(self, source: str, destination: str, weight: float = 1.0) -> TopologyEdge:
        eid = f"{source}->{destination}"
        if eid in self.edges:
            self.edges[eid].active = True
            self.edges[eid].weight = weight
            return self.edges[eid]
        edge = TopologyEdge(source=source, destination=destination, weight=weight)
        self.edges[eid] = edge
        self._edge_telemetry[eid] = EdgeTelemetry(edge_id=eid, source=source, destination=destination)
        return edge

    def remove_edge(self, source: str, destination: str) -> bool:
        eid = f"{source}->{destination}"
        if eid in self.edges:
            self.edges[eid].active = False
            self.edges[eid].weight = 0.0
            return True
        return False

    def reweight_edge(self, source: str, destination: str, weight: float) -> bool:
        eid = f"{source}->{destination}"
        if eid in self.edges:
            self.edges[eid].weight = weight
            self.edges[eid].active = weight > 0.0
            return True
        return False

    def bypass_node(self, node_id: str) -> int:
        count = 0
        for edge in self.edges.values():
            if edge.source == node_id or edge.destination == node_id:
                if edge.is_active:
                    edge.active = False
                    edge.weight = 0.0
                    count += 1
        return count

    def record_edge_usage(self, source: str, destination: str, tokens: int, latency_ms: float, success: bool) -> None:
        eid = f"{source}->{destination}"
        if eid in self._edge_telemetry:
            tel = self._edge_telemetry[eid]
            tel.messages_sent += 1
            tel.tokens_transferred += tokens
            tel.latency_contribution_ms += latency_ms
            if success:
                tel.successful_downstream += 1
            else:
                tel.failure_propagation += 1

    def get_edge_telemetry(self) -> dict[str, EdgeTelemetry]:
        return dict(self._edge_telemetry)

    def to_adjacency_matrix(self) -> np.ndarray:
        node_ids = sorted(self.nodes.keys())
        n = len(node_ids)
        idx = {nid: i for i, nid in enumerate(node_ids)}
        adj = np.zeros((n, n), dtype=np.float32)
        for edge in self.edges.values():
            if edge.is_active and edge.source in idx and edge.destination in idx:
                adj[idx[edge.source], idx[edge.destination]] = edge.weight
        return adj

    def get_node_ids(self) -> list[str]:
        return sorted(self.nodes.keys())

    def clone(self) -> "AITopology":
        edges = [TopologyEdge(source=e.source, destination=e.destination, weight=e.weight, active=e.active, edge_id=e.edge_id) for e in self.edges.values()]
        return AITopology(dict(self.nodes), edges)

    def summary(self) -> dict:
        active_edges = sum(1 for e in self.edges.values() if e.is_active)
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_active_edges": active_edges,
            "edges": [{"edge_id": e.edge_id, "source": e.source, "destination": e.destination, "weight": e.weight, "active": e.is_active} for e in self.edges.values()],
        }


def create_default_topology(nodes: dict[str, AINode]) -> AITopology:
    """Create the default 6-node topology.

    Planner → Worker → Critic → Verifier
    Planner → Researcher → Worker (optional research path)
    Memory ↔ Planner
    Memory ↔ Worker
    Critic → Worker (retry feedback)
    Verifier → Planner (replan feedback)
    """
    edges = [
        # Main path
        TopologyEdge("planner", "worker", weight=1.0),
        TopologyEdge("worker", "critic", weight=1.0),
        TopologyEdge("critic", "verifier", weight=1.0),
        # Research path (optional, expensive)
        TopologyEdge("planner", "researcher", weight=0.5),
        TopologyEdge("researcher", "worker", weight=0.8),
        # Memory
        TopologyEdge("memory", "planner", weight=1.0),
        TopologyEdge("planner", "memory", weight=0.5),
        TopologyEdge("memory", "worker", weight=0.5),
        TopologyEdge("worker", "memory", weight=0.3),
        # Feedback loops
        TopologyEdge("critic", "worker", weight=0.3),
        TopologyEdge("verifier", "planner", weight=0.2),
    ]
    return AITopology(nodes, edges)


@dataclass
class StructuralTransitionRecord:
    """Full record of a single task execution through the topology.

    This is the causal data that LGAE learns from.
    """
    task_id: str
    task_class: str
    # Topology state
    topology_before: dict = field(default_factory=dict)
    topology_after: dict = field(default_factory=dict)
    # Execution
    nodes_executed: list[str] = field(default_factory=list)
    routing_decisions: list[dict] = field(default_factory=list)
    messages_exchanged: list[dict] = field(default_factory=list)
    # Per-node telemetry
    node_telemetry: list[dict] = field(default_factory=list)
    # Aggregated
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_llm_calls: int = 0
    total_failures: int = 0
    total_tool_calls: int = 0
    # Outcomes
    critic_outcome: Optional[str] = None
    verification_outcome: Optional[str] = None
    final_quality: float = 0.0
    total_cost: float = 0.0
    objective_value: float = 0.0
    success: bool = False
    output: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_class": self.task_class,
            "topology_before": self.topology_before,
            "topology_after": self.topology_after,
            "nodes_executed": self.nodes_executed,
            "routing_decisions": self.routing_decisions,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "total_llm_calls": self.total_llm_calls,
            "total_failures": self.total_failures,
            "critic_outcome": self.critic_outcome,
            "verification_outcome": self.verification_outcome,
            "final_quality": self.final_quality,
            "total_cost": self.total_cost,
            "objective_value": self.objective_value,
            "success": self.success,
        }


class AIRuntime:
    """Executes tasks through the topology.

    Critical change from exp7.1: the runtime accumulates context
    from each visited node and passes it to the next node. This
    means topology genuinely changes what information each node
    sees, which changes the model's output.
    """

    def __init__(
        self,
        topology: AITopology,
        backend: ModelBackend,
        max_steps: int = 15,
        max_retries: int = 2,
        routing_seed: int = 42,
    ) -> None:
        self.topology = topology
        self.backend = backend
        self.max_steps = max_steps
        self.max_retries = max_retries
        self._rng = random.Random(routing_seed)

    def execute_task(
        self,
        task_id: str,
        task_input: str,
        task_class: str = "general",
    ) -> StructuralTransitionRecord:
        """Execute a task through the topology."""
        topo_before = self.topology.summary()
        record = StructuralTransitionRecord(
            task_id=task_id,
            task_class=task_class,
            topology_before=topo_before,
        )

        # Start at memory (to load context) if available, then planner.
        # If planner is bypassed, start at the first available node.
        current_node_id = self._find_start_node()
        accumulated_context = ""
        task_text = task_input
        retry_count = 0

        for step in range(self.max_steps):
            if current_node_id is None or current_node_id not in self.topology.nodes:
                break

            node = self.topology.nodes[current_node_id]
            record.nodes_executed.append(current_node_id)

            # Invoke the node with accumulated context.
            output, telemetry = node.invoke(
                task_input=task_text,
                accumulated_context=accumulated_context,
                backend=self.backend,
            )

            # Record telemetry.
            record.node_telemetry.append({
                "node_id": current_node_id,
                "role": telemetry.role.value,
                "tokens_in": telemetry.tokens_in,
                "tokens_out": telemetry.tokens_out,
                "latency_ms": telemetry.latency_ms,
                "success": telemetry.success,
                "confidence": telemetry.confidence,
                "verification_outcome": telemetry.verification_outcome,
            })
            record.total_tokens += telemetry.total_tokens
            record.total_latency_ms += telemetry.latency_ms
            record.total_llm_calls += 1
            record.total_tool_calls += telemetry.tool_calls
            if not telemetry.success:
                record.total_failures += 1

            # Track critic/verifier outcomes.
            if telemetry.role == NodeRole.CRITIC:
                record.critic_outcome = "good" if telemetry.success else "bad"
            if telemetry.role == NodeRole.VERIFIER:
                record.verification_outcome = telemetry.verification_outcome

            # Handle verification pass/fail.
            if telemetry.verification_outcome == "pass":
                record.success = True
                record.output = output
                # Extract quality from the execution.
                record.final_quality = self._extract_quality(accumulated_context + "\n" + output)
                break
            elif telemetry.verification_outcome == "fail":
                active_edges = self.topology.get_active_edges_from(current_node_id)
                feedback = [e for e in active_edges if e.destination == "planner"]
                if feedback and retry_count < self.max_retries:
                    # Record edge traversal.
                    self.topology.record_edge_usage(
                        feedback[0].source, feedback[0].destination,
                        tokens=telemetry.total_tokens, latency_ms=telemetry.latency_ms,
                        success=False,
                    )
                    record.routing_decisions.append({
                        "edge": feedback[0].edge_id, "reason": "verification_failed_retry",
                    })
                    current_node_id = "planner"
                    retry_count += 1
                    accumulated_context = f"Previous attempt failed verification.\n{accumulated_context}"
                    continue
                else:
                    record.success = False
                    record.output = output
                    record.final_quality = self._extract_quality(accumulated_context + "\n" + output)
                    break

            # Accumulate context — THIS is what makes topology matter.
            accumulated_context = f"{accumulated_context}\n[{current_node_id}]: {output}"

            # Follow active edges to next node using weighted random selection.
            # This makes topology matter: higher-weight edges are more likely
            # to be chosen, but lower-weight edges still get used.
            active_edges = self.topology.get_active_edges_from(current_node_id)
            if not active_edges:
                record.output = output
                record.success = telemetry.success
                record.final_quality = self._extract_quality(accumulated_context)
                break

            # Weighted random selection.
            weights = [e.weight for e in active_edges]
            total_weight = sum(weights)
            if total_weight <= 0:
                record.output = output
                record.success = telemetry.success
                record.final_quality = self._extract_quality(accumulated_context)
                break
            r = self._rng.random() * total_weight
            cumulative = 0.0
            next_edge = active_edges[-1]  # fallback
            for edge, w in zip(active_edges, weights):
                cumulative += w
                if r <= cumulative:
                    next_edge = edge
                    break

            self.topology.record_edge_usage(
                next_edge.source, next_edge.destination,
                tokens=telemetry.total_tokens, latency_ms=telemetry.latency_ms,
                success=telemetry.success,
            )
            record.routing_decisions.append({
                "edge": next_edge.edge_id,
                "weight": next_edge.weight,
                "reason": "highest_weight",
            })

            current_node_id = next_edge.destination
            # Pass the accumulated context as the new task input.
            task_text = output

        # If we exhausted steps.
        if not record.output:
            record.output = accumulated_context[-200:] if accumulated_context else "no output"
            record.final_quality = self._extract_quality(accumulated_context)

        record.topology_after = self.topology.summary()
        record.total_cost = (
            record.total_tokens * 0.001
            + record.total_latency_ms / 1000 * 0.01
            + record.total_llm_calls * 0.05
        )

        return record

    def execute_batch(self, tasks: list[dict]) -> list[StructuralTransitionRecord]:
        """Execute a batch of tasks."""
        return [
            self.execute_task(
                task_id=t["task_id"],
                task_input=t["input"],
                task_class=t.get("task_class", "general"),
            )
            for t in tasks
        ]

    def _find_start_node(self) -> Optional[str]:
        """Find the starting node for execution."""
        # Prefer memory → planner path if both exist.
        if "memory" in self.topology.nodes and "planner" in self.topology.nodes:
            if self.topology.has_edge("memory", "planner"):
                return "memory"
        if "planner" in self.topology.nodes:
            return "planner"
        # Fall back to any node with outgoing edges.
        for nid in self.topology.get_node_ids():
            if self.topology.get_active_edges_from(nid):
                return nid
        return None

    def _extract_quality(self, text: str) -> float:
        """Extract quality score from execution output."""
        for line in text.split("\n"):
            if "WORKER_QUALITY_SCORE:" in line:
                try:
                    return float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
        # Default: use confidence from last node if available.
        if self.topology.nodes:
            return 0.5
        return 0.0
