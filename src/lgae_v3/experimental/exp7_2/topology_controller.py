"""Topology controller with real shadow evaluation for exp7.2.

LGAE proposes topology mutations and evaluates them via actual
shadow executions on a small calibration batch. Only mutations
with positive advantage are applied.

Authority pattern preserved:
  LGAE proposes → shadow eval → advantage check → governor → commit
  Rollback to KNOWN_GOOD_TOPOLOGY on degradation.

LGAE does NOT receive task labels — only telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .ai_node import create_default_nodes
from .topology_runtime import AITopology, TopologyEdge, AIRuntime, StructuralTransitionRecord, create_default_topology
from .model_backend import ModelBackend
from .objective import ObjectiveWeights, compute_objective_from_record
from .topology_actions import TopologyAction, TopologyActionType, generate_candidate_actions


@dataclass
class MutationRecord:
    action: TopologyAction
    shadow_objective: float = 0.0
    baseline_objective: float = 0.0
    advantage: float = 0.0
    applied: bool = False
    reason: str = ""


class TopologyController:
    """LGAE's structural planner for AI topology.

    Uses real shadow executions to evaluate candidate mutations.
    Does NOT receive task labels — only sees telemetry and
    structural features of the topology.
    """

    def __init__(
        self,
        topology: AITopology,
        backend: ModelBackend,
        objective_weights: ObjectiveWeights,
        *,
        shadow_batch_size: int = 5,
        max_mutations_per_cycle: int = 3,
        rollback_threshold: float = 0.1,
        advantage_threshold: float = 0.01,
    ) -> None:
        self.topology = topology
        self.backend = backend
        self.objective_weights = objective_weights
        self.shadow_batch_size = shadow_batch_size
        self.max_mutations_per_cycle = max_mutations_per_cycle
        self.rollback_threshold = rollback_threshold
        self.advantage_threshold = advantage_threshold

        self.known_good_topology = topology.clone()
        self.mutation_history: list[MutationRecord] = []
        self.best_objective = float("-inf")

    def adapt(
        self,
        shadow_tasks: list[dict],
    ) -> list[MutationRecord]:
        """Propose and evaluate topology mutations via shadow execution.

        1. Generate candidate actions from current topology
        2. For each candidate, run shadow executions on calibration batch
        3. Compute advantage vs baseline
        4. Apply only if advantage > threshold
        """
        candidates = generate_candidate_actions(self.topology)
        records = []

        # Baseline: run shadow tasks on current topology.
        baseline_runtime = AIRuntime(self.topology.clone(), self.backend)
        baseline_results = baseline_runtime.execute_batch(shadow_tasks[:self.shadow_batch_size])
        baseline_objectives = [compute_objective_from_record(r, self.objective_weights) for r in baseline_results]
        baseline_mean = float(np.mean(baseline_objectives)) if baseline_objectives else 0.0

        applied_this_cycle = 0

        for action in candidates[:15]:  # limit evaluations
            # Shadow evaluation: apply action to clone, run tasks.
            shadow_topology = self.topology.clone()
            action.apply(shadow_topology)
            shadow_runtime = AIRuntime(shadow_topology, self.backend)
            shadow_results = shadow_runtime.execute_batch(shadow_tasks[:self.shadow_batch_size])
            shadow_objectives = [compute_objective_from_record(r, self.objective_weights) for r in shadow_results]
            shadow_mean = float(np.mean(shadow_objectives)) if shadow_objectives else 0.0

            advantage = shadow_mean - baseline_mean

            record = MutationRecord(
                action=action,
                shadow_objective=shadow_mean,
                baseline_objective=baseline_mean,
                advantage=advantage,
                applied=False,
                reason="",
            )

            if advantage > self.advantage_threshold:
                record.applied = True
                record.reason = f"positive advantage ({advantage:.4f})"
                action.apply(self.topology)
                applied_this_cycle += 1
            else:
                record.reason = f"advantage below threshold ({advantage:.4f})"

            records.append(record)
            self.mutation_history.append(record)

            if applied_this_cycle >= self.max_mutations_per_cycle:
                break

        return records

    def check_rollback(self, current_objective: float) -> bool:
        """Check if we should rollback."""
        if self.best_objective == float("-inf"):
            self.best_objective = current_objective
            return False

        if current_objective > self.best_objective:
            self.best_objective = current_objective
            self.known_good_topology = self.topology.clone()
            return False

        degradation = (self.best_objective - current_objective) / max(abs(self.best_objective), 1e-6)
        return degradation > self.rollback_threshold

    def rollback(self) -> None:
        self.topology = self.known_good_topology.clone()

    def get_summary(self) -> dict:
        total = len(self.mutation_history)
        applied = sum(1 for r in self.mutation_history if r.applied)
        return {
            "total_proposed": total,
            "total_applied": applied,
            "best_objective": self.best_objective,
            "current_topology": self.topology.summary(),
        }
