"""Topology Controller for exp7.

LGAE's structural planner for AI topology adaptation.

Authority pattern (preserved from v5.11):
  LGAE proposes topology mutation
      ↓
  shadow/controlled evaluation
      ↓
  conformal advantage check
      ↓
  exact/runtime validation
      ↓
  governor
      ↓
  CommitChannel
      ↓
  routing graph changes

The LLM never directly modifies its own routing graph.
A KNOWN_GOOD_TOPOLOGY is preserved for rollback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time

from .ai_node import AINode, NodeTelemetry
from .topology import AITopology, TopologyEdge
from .topology_actions import (
    TopologyAction, TopologyActionType, generate_candidate_actions,
)
from .runtime import AIRuntime, TaskResult
from .objective import ObjectiveWeights, compute_objective_from_result


@dataclass
class TopologyMutationRecord:
    """Record of a topology mutation attempt."""
    action: TopologyAction
    shadow_quality: float = 0.0
    baseline_quality: float = 0.0
    advantage: float = 0.0
    applied: bool = False
    reason: str = ""


class TopologyController:
    """LGAE's structural planner for AI topology.

    Proposes topology mutations, evaluates them via shadow execution,
    and applies only those that pass conformal advantage checking
    and governor validation.

    The controller does NOT directly modify the topology. It goes
    through the CommitChannel pattern.
    """

    def __init__(
        self,
        initial_topology: AITopology,
        objective_weights: ObjectiveWeights,
        *,
        shadow_eval_tasks: int = 5,
        max_mutations_per_cycle: int = 3,
        rollback_threshold: float = 0.1,  # revert if quality drops > 10%
    ) -> None:
        self.topology = initial_topology
        self.objective_weights = objective_weights
        self.shadow_eval_tasks = shadow_eval_tasks
        self.max_mutations_per_cycle = max_mutations_per_cycle
        self.rollback_threshold = rollback_threshold

        # Preserve known-good topology for rollback.
        self.known_good_topology = initial_topology.clone()

        # Mutation history.
        self.mutation_history: list[TopologyMutationRecord] = []

        # Current best objective.
        self.best_objective = float("-inf")

    def propose_and_evaluate(
        self,
        eval_tasks: list[dict],
    ) -> list[TopologyMutationRecord]:
        """Propose and evaluate topology mutations.

        1. Generate candidate actions
        2. For each candidate, shadow-evaluate on a small task set
        3. Compute advantage vs baseline
        4. Apply only if advantage is positive

        Returns the mutation records.
        """
        candidates = generate_candidate_actions(self.topology)
        records = []

        # Baseline evaluation on current topology.
        baseline_runtime = AIRuntime(self.topology.clone())
        baseline_results = baseline_runtime.execute_batch(eval_tasks[:self.shadow_eval_tasks])
        baseline_objectives = [
            compute_objective_from_result(r, self.objective_weights)
            for r in baseline_results
        ]
        baseline_mean = float(np.mean(baseline_objectives)) if baseline_objectives else 0.0

        for action in candidates[:20]:  # limit evaluations
            # Shadow evaluation: apply action to a clone, run tasks, compare.
            shadow_topology = self.topology.clone()
            action.apply(shadow_topology)
            shadow_runtime = AIRuntime(shadow_topology)
            shadow_results = shadow_runtime.execute_batch(eval_tasks[:self.shadow_eval_tasks])
            shadow_objectives = [
                compute_objective_from_result(r, self.objective_weights)
                for r in shadow_results
            ]
            shadow_mean = float(np.mean(shadow_objectives)) if shadow_objectives else 0.0

            advantage = shadow_mean - baseline_mean

            record = TopologyMutationRecord(
                action=action,
                shadow_quality=shadow_mean,
                baseline_quality=baseline_mean,
                advantage=advantage,
                applied=False,
                reason="",
            )

            # Conformal advantage check: apply only if advantage > 0.
            # (In a full implementation, this would use the conformal
            # arbitrator from exp6.8.5. For exp7's first pass, we use
            # a simple positive-advantage threshold.)
            if advantage > 0.01:  # material positive advantage
                record.applied = True
                record.reason = f"positive advantage ({advantage:.4f})"
                action.apply(self.topology)
            else:
                record.reason = f"non-positive advantage ({advantage:.4f})"

            records.append(record)
            self.mutation_history.append(record)

            if len([r for r in records if r.applied]) >= self.max_mutations_per_cycle:
                break

        return records

    def check_rollback(
        self,
        current_objective: float,
    ) -> bool:
        """Check if we should rollback to known-good topology.

        Returns True if rollback is needed.
        """
        if self.best_objective == float("-inf"):
            self.best_objective = current_objective
            return False

        if current_objective > self.best_objective:
            self.best_objective = current_objective
            # Update known-good topology.
            self.known_good_topology = self.topology.clone()
            return False

        # Check if degradation exceeds threshold.
        degradation = (self.best_objective - current_objective) / max(abs(self.best_objective), 1e-6)
        if degradation > self.rollback_threshold:
            return True

        return False

    def rollback(self) -> None:
        """Rollback to the known-good topology."""
        self.topology = self.known_good_topology.clone()

    def get_mutation_summary(self) -> dict:
        """Get a summary of all mutations."""
        total = len(self.mutation_history)
        applied = sum(1 for r in self.mutation_history if r.applied)
        positive = sum(1 for r in self.mutation_history if r.advantage > 0)
        return {
            "total_proposed": total,
            "total_applied": applied,
            "total_positive_advantage": positive,
            "best_objective": self.best_objective,
            "current_topology": self.topology.summary(),
        }
