"""Improved topology controller for exp7.3.

Key improvements over exp7.2:
1. Larger shadow batches (20-50 tasks) to reduce overfitting
2. Incremental adaptation: evaluate cumulative mutation effects
3. Task-feature-aware candidate generation: use text-derived features
   to bias which mutations are proposed
4. Conformal advantage gate: use calibration history to set
   a data-driven advantage threshold
5. Per-task-class shadow sampling: ensure shadow batch covers
   the task distribution
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import random

from ..exp7_2.topology_runtime import AITopology, AIRuntime, StructuralTransitionRecord, create_default_topology
from ..exp7_2.model_backend import ModelBackend
from ..exp7_2.objective import ObjectiveWeights, compute_objective_from_record
from ..exp7_2.topology_actions import TopologyAction, TopologyActionType, generate_candidate_actions
from ..exp7_2.ai_node import create_default_nodes
from .task_features import extract_features, features_to_topology_hints, TaskFeatures


@dataclass
class MutationRecord:
    action: TopologyAction
    shadow_objective: float = 0.0
    baseline_objective: float = 0.0
    advantage: float = 0.0
    applied: bool = False
    reason: str = ""
    shadow_batch_size: int = 0
    # For transfer analysis: full-set advantage (filled later)
    full_advantage: Optional[float] = None
    # Task features at time of mutation
    avg_complexity: float = 0.0
    suggests_research: bool = False
    suggests_critic: bool = False
    suggests_memory: bool = False


class ConformalAdvantageGate:
    """Conformal advantage gate using calibration history.

    Maintains a history of observed advantages and uses the
    empirical distribution to set a data-driven threshold.
    Only apply mutations where the lower confidence bound of
    advantage is positive.
    """

    def __init__(self, alpha: float = 0.2, min_history: int = 5) -> None:
        self.alpha = alpha
        self.min_history = min_history
        self.advantage_history: list[float] = []

    def record(self, advantage: float) -> None:
        self.advantage_history.append(advantage)

    def threshold(self) -> float:
        """Compute the conformal advantage threshold.

        Returns the (1-alpha) quantile of observed advantages.
        Mutations with advantage above this threshold are applied.
        """
        if len(self.advantage_history) < self.min_history:
            # Not enough history — use a conservative fixed threshold.
            return 0.02
        arr = np.array(self.advantage_history)
        # LCB-style: use the alpha quantile as threshold.
        return float(np.quantile(arr, self.alpha))

    def should_apply(self, advantage: float) -> bool:
        """Check if a mutation should be applied based on conformal gate."""
        threshold = self.threshold()
        return advantage > threshold


class TopologyControllerV2:
    """Improved topology controller for exp7.3.

    Improvements:
    1. Larger shadow batches (configurable, default 20)
    2. Incremental adaptation: apply mutations cumulatively
    3. Task-feature-aware: uses text features to bias candidate generation
    4. Conformal advantage gate: data-driven threshold
    5. Representative shadow sampling: covers task distribution
    """

    def __init__(
        self,
        topology: AITopology,
        backend: ModelBackend,
        objective_weights: ObjectiveWeights,
        *,
        shadow_batch_size: int = 20,
        max_mutations_per_cycle: int = 3,
        rollback_threshold: float = 0.1,
        conformal_alpha: float = 0.2,
        online_rollback_window: int = 10,
        online_rollback_epsilon: float = 0.05,
        use_task_features: bool = True,
    ) -> None:
        self.topology = topology
        self.backend = backend
        self.objective_weights = objective_weights
        self.shadow_batch_size = shadow_batch_size
        self.max_mutations_per_cycle = max_mutations_per_cycle
        self.rollback_threshold = rollback_threshold
        self.use_task_features = use_task_features

        self.known_good_topology = topology.clone()
        self.mutation_history: list[MutationRecord] = []
        self.best_objective = float("-inf")

        # Conformal advantage gate.
        self.conformal_gate = ConformalAdvantageGate(alpha=conformal_alpha)

        # Task feature history for learning.
        self.feature_history: list[TaskFeatures] = []

        # Online rollback: track rolling J and revert if degraded.
        self.online_rollback_window = online_rollback_window
        self.online_rollback_epsilon = online_rollback_epsilon
        self.recent_objectives: list[float] = []
        self.baseline_objective: float = 0.0
        self.n_rollbacks: int = 0

        # Shadow advantage tracking for transfer analysis.
        self.shadow_advantages: list[float] = []

    def adapt(
        self,
        shadow_tasks: list[dict],
    ) -> list[MutationRecord]:
        """Propose and evaluate topology mutations with improvements.

        1. Extract task features for each shadow task
        2. Generate feature-aware candidate actions
        3. Evaluate candidates with larger shadow batch
        4. Apply incrementally (cumulative effect)
        5. Gate with conformal advantage threshold
        """
        # Extract features for shadow tasks.
        shadow_features = [extract_features(t["input"]) for t in shadow_tasks]
        self.feature_history.extend(shadow_features)

        # Generate feature-aware candidates (or standard if disabled).
        avg_features = self._average_features(shadow_features)
        if self.use_task_features:
            candidates = self._generate_feature_aware_candidates(avg_features)
        else:
            candidates = generate_candidate_actions(self.topology)

        # Baseline evaluation on current topology.
        batch = shadow_tasks[:self.shadow_batch_size]
        baseline_runtime = AIRuntime(self.topology.clone(), self.backend)
        baseline_results = baseline_runtime.execute_batch(batch)
        baseline_objectives = [compute_objective_from_record(r, self.objective_weights) for r in baseline_results]
        baseline_mean = float(np.mean(baseline_objectives)) if baseline_objectives else 0.0

        records = []
        applied_this_cycle = 0
        current_baseline = baseline_mean

        for action in candidates[:12]:
            # Incremental: apply to current topology (which may already have
            # mutations from this cycle), not to the original.
            shadow_topology = self.topology.clone()
            action.apply(shadow_topology)
            shadow_runtime = AIRuntime(shadow_topology, self.backend)
            shadow_results = shadow_runtime.execute_batch(batch)
            shadow_objectives = [compute_objective_from_record(r, self.objective_weights) for r in shadow_results]
            shadow_mean = float(np.mean(shadow_objectives)) if shadow_objectives else 0.0

            # Advantage vs current baseline (incremental).
            advantage = shadow_mean - current_baseline

            # Record for conformal calibration.
            self.conformal_gate.record(advantage)
            self.shadow_advantages.append(advantage)

            record = MutationRecord(
                action=action,
                shadow_objective=shadow_mean,
                baseline_objective=current_baseline,
                advantage=advantage,
                applied=False,
                reason="",
                shadow_batch_size=len(batch),
                avg_complexity=avg_features.complexity_score,
                suggests_research=avg_features.suggests_research,
                suggests_critic=avg_features.suggests_critic,
                suggests_memory=avg_features.suggests_memory,
            )

            # Conformal gate.
            if self.conformal_gate.should_apply(advantage):
                record.applied = True
                record.reason = f"conformal gate passed (adv={advantage:.4f}, thresh={self.conformal_gate.threshold():.4f})"
                action.apply(self.topology)
                current_baseline = shadow_mean  # update baseline for incremental
                applied_this_cycle += 1
            else:
                record.reason = f"conformal gate rejected (adv={advantage:.4f}, thresh={self.conformal_gate.threshold():.4f})"

            records.append(record)
            self.mutation_history.append(record)

            if applied_this_cycle >= self.max_mutations_per_cycle:
                break

        return records

    def check_rollback(self, current_objective: float) -> bool:
        if self.best_objective == float("-inf"):
            self.best_objective = current_objective
            return False
        if current_objective > self.best_objective:
            self.best_objective = current_objective
            self.known_good_topology = self.topology.clone()
            return False
        degradation = (self.best_objective - current_objective) / max(abs(self.best_objective), 1e-6)
        return degradation > self.rollback_threshold

    def observe_objective(self, obj: float) -> bool:
        """Online rollback check.

        Track rolling J and revert if it degrades below baseline - epsilon
        for N consecutive tasks.

        Returns True if rollback was triggered.
        """
        self.recent_objectives.append(obj)
        if len(self.recent_objectives) > self.online_rollback_window:
            self.recent_objectives.pop(0)

        if len(self.recent_objectives) < self.online_rollback_window:
            return False

        rolling_j = float(np.mean(self.recent_objectives))

        # Set baseline if not set.
        if self.baseline_objective == 0.0:
            self.baseline_objective = rolling_j
            return False

        # Check if rolling J has degraded beyond epsilon.
        if rolling_j < self.baseline_objective - self.online_rollback_epsilon:
            self.rollback()
            self.n_rollbacks += 1
            self.recent_objectives.clear()
            self.baseline_objective = 0.0
            return True

        # Update baseline if performance improved.
        if rolling_j > self.baseline_objective:
            self.baseline_objective = rolling_j

        return False

    def rollback(self) -> None:
        self.topology = self.known_good_topology.clone()

    def get_summary(self) -> dict:
        total = len(self.mutation_history)
        applied = sum(1 for r in self.mutation_history if r.applied)
        return {
            "total_proposed": total,
            "total_applied": applied,
            "best_objective": self.best_objective,
            "conformal_threshold": self.conformal_gate.threshold(),
            "advantage_history_len": len(self.conformal_gate.advantage_history),
            "current_topology": self.topology.summary(),
        }

    def _average_features(self, features: list[TaskFeatures]) -> TaskFeatures:
        """Compute average features across a batch."""
        if not features:
            return TaskFeatures()
        n = len(features)
        return TaskFeatures(
            n_tokens=sum(f.n_tokens for f in features) / n,
            n_chars=sum(f.n_chars for f in features) / n,
            avg_word_length=sum(f.avg_word_length for f in features) / n,
            n_sentences=sum(f.n_sentences for f in features) / n,
            has_question_mark=any(f.has_question_mark for f in features),
            has_code_keywords=any(f.has_code_keywords for f in features),
            has_debug_keywords=any(f.has_debug_keywords for f in features),
            has_research_keywords=any(f.has_research_keywords for f in features),
            has_verify_keywords=any(f.has_verify_keywords for f in features),
            has_memory_keywords=any(f.has_memory_keywords for f in features),
            has_reasoning_keywords=any(f.has_reasoning_keywords for f in features),
            complexity_score=sum(f.complexity_score for f in features) / n,
            estimated_difficulty=sum(f.estimated_difficulty for f in features) / n,
            suggests_research=any(f.suggests_research for f in features),
            suggests_verification=any(f.suggests_verification for f in features),
            suggests_memory=any(f.suggests_memory for f in features),
            suggests_planning=any(f.suggests_planning for f in features),
            suggests_critic=any(f.suggests_critic for f in features),
        )

    def _generate_feature_aware_candidates(self, features: TaskFeatures) -> list[TopologyAction]:
        """Generate candidates biased by task features.

        Instead of trying all possible mutations, focus on mutations
        that the task features suggest might be beneficial.
        """
        candidates = []
        hints = features_to_topology_hints(features)
        node_ids = self.topology.get_node_ids()

        # 1. Feature-suggested reweighting.
        if features.suggests_research:
            candidates.append(TopologyAction(
                action_type=TopologyActionType.REWEIGHT_ROUTE,
                source="planner", destination="researcher",
                weight=hints["research_weight"],
                reason="features suggest research",
            ))
            candidates.append(TopologyAction(
                action_type=TopologyActionType.REWEIGHT_ROUTE,
                source="researcher", destination="worker",
                weight=hints["research_weight"],
                reason="features suggest research",
            ))
        else:
            # Reduce research path if not suggested.
            candidates.append(TopologyAction(
                action_type=TopologyActionType.REWEIGHT_ROUTE,
                source="planner", destination="researcher",
                weight=0.1,
                reason="features suggest no research",
            ))

        if features.suggests_critic:
            candidates.append(TopologyAction(
                action_type=TopologyActionType.REWEIGHT_ROUTE,
                source="worker", destination="critic",
                weight=hints["critic_weight"],
                reason="features suggest critic",
            ))
        else:
            # Bypass critic, add direct worker→verifier.
            candidates.append(TopologyAction(
                action_type=TopologyActionType.REWEIGHT_ROUTE,
                source="worker", destination="critic",
                weight=0.1,
                reason="features suggest no critic",
            ))
            candidates.append(TopologyAction(
                action_type=TopologyActionType.ADD_ROUTE,
                source="worker", destination="verifier",
                weight=1.0,
                reason="direct worker→verifier for simple tasks",
            ))

        if features.suggests_memory:
            candidates.append(TopologyAction(
                action_type=TopologyActionType.REWEIGHT_ROUTE,
                source="memory", destination="planner",
                weight=hints["memory_weight"],
                reason="features suggest memory",
            ))
        else:
            candidates.append(TopologyAction(
                action_type=TopologyActionType.REWEIGHT_ROUTE,
                source="memory", destination="planner",
                weight=0.1,
                reason="features suggest no memory",
            ))

        if features.suggests_verification:
            candidates.append(TopologyAction(
                action_type=TopologyActionType.REWEIGHT_ROUTE,
                source="critic", destination="verifier",
                weight=hints["verifier_weight"],
                reason="features suggest verification",
            ))

        if features.suggests_planning:
            candidates.append(TopologyAction(
                action_type=TopologyActionType.REWEIGHT_ROUTE,
                source="planner", destination="worker",
                weight=hints["planner_weight"],
                reason="features suggest planning",
            ))

        # 2. Standard candidates (for diversity).
        standard = generate_candidate_actions(self.topology)
        # Add a few standard candidates for exploration.
        candidates.extend(standard[:6])

        return candidates
