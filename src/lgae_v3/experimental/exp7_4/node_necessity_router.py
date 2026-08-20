"""Node-necessity router for exp7.4.

Uses learned per-node marginal value predictions to decide which
nodes to include for each task. This is the learned routing policy.

For each task:
  1. Embed the task
  2. Predict ΔJ_n for each optional node
  3. Include nodes with positive predicted marginal value
  4. Exclude nodes with negative predicted marginal value
  5. Build a task-specific topology

This replaces blind graph mutation with informed node selection.

The router still goes through the authority path:
  LGAE proposes topology → shadow eval → conformal gate → commit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..exp7_2.ai_node import create_default_nodes
from ..exp7_2.topology_runtime import AITopology, TopologyEdge, AIRuntime, StructuralTransitionRecord, create_default_topology
from ..exp7_2.model_backend import ModelBackend
from ..exp7_2.objective import ObjectiveWeights, compute_objective_from_record
from ..exp7_2.benchmark import BenchmarkTask
from ..exp7_2.quality_evaluators import evaluate_quality
from .task_embedding import embed_task
from .marginal_value import MarginalValueEstimator, OPTIONAL_NODES


@dataclass
class RoutingDecision:
    """A single routing decision for a task."""
    task_id: str
    included_nodes: list[str] = field(default_factory=list)
    excluded_nodes: list[str] = field(default_factory=list)
    marginal_values: dict = field(default_factory=dict)
    topology_summary: dict = field(default_factory=dict)


class NodeNecessityRouter:
    """Learned node-necessity router.

    Uses marginal value predictions to decide which nodes to
    include for each task. Learns from execution history via
    shadow evaluations.
    """

    def __init__(
        self,
        backend: ModelBackend,
        objective_weights: ObjectiveWeights,
        *,
        k_neighbors: int = 5,
        min_samples: int = 3,
        calibration_interval: int = 20,
        shadow_batch_size: int = 5,
    ) -> None:
        self.backend = backend
        self.objective_weights = objective_weights
        self.calibration_interval = calibration_interval
        self.shadow_batch_size = shadow_batch_size

        self.estimator = MarginalValueEstimator(
            k=k_neighbors,
            min_samples=min_samples,
        )

        self.routing_history: list[RoutingDecision] = []
        self.n_calibrations = 0

    def route_task(
        self,
        task_id: str,
        task_input: str,
    ) -> tuple[AITopology, RoutingDecision]:
        """Route a task by building a task-specific topology.

        Returns (topology, routing_decision).
        """
        nodes = create_default_nodes()
        topo = create_default_topology(nodes)

        # Predict marginal value for each optional node.
        predictions = self.estimator.predict_all_nodes(task_input)

        included = []
        excluded = []

        for node in OPTIONAL_NODES:
            pred = predictions[node]
            # Include node if predicted marginal value is positive,
            # OR if we have no data (default to include — conservative).
            if pred["include"] or pred["confidence"] < 0.01:
                included.append(node)
            else:
                excluded.append(node)
                # Disable edges to/from this node.
                topo.bypass_node(node)

        # If we excluded critic, add direct worker→verifier edge.
        if "critic" in excluded:
            topo.add_edge("worker", "verifier", 1.0)

        # If we excluded researcher, ensure planner→worker is strong.
        if "researcher" in excluded:
            topo.reweight_edge("planner", "worker", 1.5)

        # If we excluded memory, ensure planner starts directly.
        if "memory" in excluded:
            topo.reweight_edge("memory", "planner", 0.0)

        decision = RoutingDecision(
            task_id=task_id,
            included_nodes=included,
            excluded_nodes=excluded,
            marginal_values=predictions,
            topology_summary=topo.summary(),
        )

        self.routing_history.append(decision)

        return topo, decision

    def calibrate(
        self,
        calibration_tasks: list[dict],
    ) -> int:
        """Calibrate the marginal value estimator.

        For each calibration task and each optional node, run the
        task with and without the node, and record the marginal value.

        Returns the number of samples added.
        """
        n_samples = 0

        for task in calibration_tasks[:self.shadow_batch_size]:
            task_input = task["input"]
            task_id = task["task_id"]
            task_class = task.get("task_class", "general")

            for node in OPTIONAL_NODES:
                # Run with the node.
                nodes_with = create_default_nodes()
                topo_with = create_default_topology(nodes_with)
                runtime_with = AIRuntime(topo_with, self.backend)
                record_with = runtime_with.execute_task(task_id, task_input, task_class)
                record_with.final_quality = evaluate_quality(
                    task_class, record_with.output, "", record_with.verification_outcome, record_with.output,
                )
                j_with = compute_objective_from_record(record_with, self.objective_weights)

                # Run without the node.
                nodes_without = create_default_nodes()
                topo_without = create_default_topology(nodes_without)
                topo_without.bypass_node(node)
                # Add fallback edges.
                if node == "critic":
                    topo_without.add_edge("worker", "verifier", 1.0)
                runtime_without = AIRuntime(topo_without, self.backend)
                record_without = runtime_without.execute_task(task_id, task_input, task_class)
                record_without.final_quality = evaluate_quality(
                    task_class, record_without.output, "", record_without.verification_outcome, record_without.output,
                )
                j_without = compute_objective_from_record(record_without, self.objective_weights)

                self.estimator.add_sample(task_input, node, j_with, j_without)
                n_samples += 1

        self.n_calibrations += 1
        return n_samples

    def get_routing_summary(self) -> dict:
        """Get a summary of routing decisions."""
        if not self.routing_history:
            return {"n_decisions": 0}

        # Count how often each node is included/excluded.
        include_counts = {node: 0 for node in OPTIONAL_NODES}
        exclude_counts = {node: 0 for node in OPTIONAL_NODES}

        for decision in self.routing_history:
            for node in decision.included_nodes:
                include_counts[node] = include_counts.get(node, 0) + 1
            for node in decision.excluded_nodes:
                exclude_counts[node] = exclude_counts.get(node, 0) + 1

        return {
            "n_decisions": len(self.routing_history),
            "n_calibrations": self.n_calibrations,
            "include_counts": include_counts,
            "exclude_counts": exclude_counts,
            "estimator_summary": self.estimator.get_summary(),
        }

    def inspect_routing_patterns(self) -> dict[str, dict]:
        """Inspect learned routing patterns.

        Groups routing decisions by the dominant task category
        (inferred from embedding, not labels) and shows which
        nodes are typically included/excluded.
        """
        if not self.routing_history:
            return {}

        # Group by embedding similarity (not labels).
        # For simplicity, group by the category with highest score.
        patterns = defaultdict(lambda: {"included": defaultdict(int), "excluded": defaultdict(int), "count": 0})

        for decision in self.routing_history:
            # Find the original task input from the estimator's samples.
            # For inspection, we use the marginal values to infer pattern.
            # Group by which nodes are included.
            included_key = tuple(sorted(decision.included_nodes))
            patterns[str(included_key)]["count"] += 1
            for node in decision.included_nodes:
                patterns[str(included_key)]["included"][node] += 1
            for node in decision.excluded_nodes:
                patterns[str(included_key)]["excluded"][node] += 1

        return dict(patterns)


from collections import defaultdict
