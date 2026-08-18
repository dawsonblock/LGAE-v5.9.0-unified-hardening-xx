"""v5.1.1 closed structural-learning loop.

Authority model
---------------
The learned executive is a proposal mechanism.  ``LGAEEngine`` is the only
component allowed to commit graph, fiber, or gauge changes.  ``QUARANTINE`` is
never treated as execution.  Long-horizon outcomes are fed back into both the
executive and the bootstrap uncertainty ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import math

import torch
from torch import Tensor

from .executive import (
    StructuralExecutive, ActionProposal, StructuralObservation,
    StructuralAction, ACTION_TO_IDX,
)
from .counterfactual import StructuralCounterfactualEngine, CounterfactualResult
from .uncertainty import EnsembleUncertainty, ConformalCalibrator, uncertainty_gated_decision
from .credit import MutationCreditTracker
from .consolidation import StabilityPlasticityController
from .action_bridge import certify_action_through_governor, action_to_mutation
from .types import GraphBuffers, MutationDecision, MutationResult
from .config import LGAEConfig, config_governance_hash
from .timescales import MultiTimescaleController, Timescale
from .version import VERSION


@dataclass
class StructuralLoopResult:
    step: int
    observation: StructuralObservation
    counterfactual: CounterfactualResult
    chosen_action: StructuralAction
    uncertainty_decision: str
    governance_decision: str
    executed: bool
    utility_before: float
    utility_after: float
    delta_utility: float
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuralLearningLoop:
    """Closed structural-learning loop with single-authority commit semantics.

    ``engine`` is the preferred integration.  The older ``governor`` argument
    remains supported as a read-only certification path for backwards
    compatibility, but without an engine a structural action is never reported
    as executed because there is no authoritative commit owner.
    """

    def __init__(
        self,
        config: LGAEConfig | None = None,
        executive: StructuralExecutive | None = None,
        governor: Any | None = None,
        engine: Any | None = None,
        ensemble_size: int = 5,
        beta: float = 1.0,
        gamma: float = 0.99,
        credit_horizons: list[int] | None = None,
        max_budget: float = float("inf"),
        tau_efficiency: float = 0.01,
        probation_length: int = 100,
        max_candidates: int = 5,
        no_op_penalty: float = 0.0,
        timescale_controller: MultiTimescaleController | None = None,
        enforce_timescales: bool = False,
    ):
        self.config = config or (engine.cfg if engine is not None else LGAEConfig())
        self.engine = engine
        self.governor = engine.governor if engine is not None else governor
        self.executive = executive or StructuralExecutive(self.config)
        self.uncertainty_estimator = EnsembleUncertainty(
            self.executive, ensemble_size=ensemble_size, beta=beta,
        )
        self.calibrator = ConformalCalibrator(alpha=0.1)
        self.credit_tracker = MutationCreditTracker(
            gamma=gamma, horizons=credit_horizons or [16, 100, 1000],
        )
        self.consolidation = StabilityPlasticityController(
            max_budget=max_budget,
            tau_efficiency=tau_efficiency,
            probation_length=probation_length,
        )
        self.counterfactual = StructuralCounterfactualEngine(
            self.executive, max_candidates=max_candidates, no_op_penalty=no_op_penalty,
        )
        self.timescales = timescale_controller or MultiTimescaleController(
            equilibrium_delta_tol=(
                self.config.mutation.equilibrium_delta_tol
                if self.config.mutation.equilibrium_barrier_enabled else None
            ),
            equilibrium_required_steps=self.config.mutation.equilibrium_required_steps,
        )
        self.enforce_timescales = bool(enforce_timescales)
        self._step = 0
        self._credit_context: dict[int, tuple[StructuralObservation, StructuralAction]] = {}
        self._consumed_outcomes: set[int] = set()

    def _action_allowed_by_timescale(self, action: StructuralAction, step: int) -> bool:
        if not self.enforce_timescales or action == StructuralAction.NO_OP:
            return True
        if action == StructuralAction.CHANGE_GAUGE:
            return self.timescales.can_adapt_gauge(step)
        if action in (StructuralAction.REWEIGHT_AFFINITY, StructuralAction.SPAWN_FIBER, StructuralAction.PRUNE_FIBER):
            return self.timescales.can_adapt_affinity(step)
        if action in (StructuralAction.REWEIGHT_LENGTH, StructuralAction.COUPLED_REWEIGHT):
            return self.timescales.can_adapt_length(step)
        if action in (StructuralAction.ADD_EDGE, StructuralAction.PRUNE_EDGE):
            return self.timescales.can_adapt_topology(step)
        return True

    def _estimate_cost(self, action: StructuralAction, result: MutationResult | None) -> float:
        if action == StructuralAction.NO_OP:
            return 0.0
        if action in (StructuralAction.ADD_EDGE, StructuralAction.PRUNE_EDGE):
            return 1.0
        if action in (StructuralAction.REWEIGHT_AFFINITY, StructuralAction.REWEIGHT_LENGTH, StructuralAction.COUPLED_REWEIGHT):
            return 0.25
        if action in (StructuralAction.SPAWN_FIBER, StructuralAction.PRUNE_FIBER):
            channels = [] if result is None else result.metadata.get("channels", [])
            return float(max(1, len(channels)))
        if action == StructuralAction.CHANGE_GAUGE:
            return float(max(1, self.config.fiber.gauge_dim ** 2))
        return 1.0

    def _execute_engine_action(
        self,
        action: StructuralAction,
        target: dict[str, Any],
    ) -> MutationResult:
        assert self.engine is not None
        if action in (
            StructuralAction.ADD_EDGE, StructuralAction.PRUNE_EDGE,
            StructuralAction.REWEIGHT_AFFINITY, StructuralAction.REWEIGHT_LENGTH,
            StructuralAction.COUPLED_REWEIGHT,
        ):
            mutation = action_to_mutation(action, self.engine.graph, self.engine.fibers().detach(), **target)
            if mutation is None:
                return MutationResult(MutationDecision.REJECT, ["no_valid_concrete_mutation"], metadata={"action": action.value})
            return self.engine.evaluate_and_maybe_commit(mutation)
        if action in (StructuralAction.SPAWN_FIBER, StructuralAction.PRUNE_FIBER):
            return self.engine.evaluate_fiber_action(
                action.value, node=target.get("node"), width=target.get("width"),
            )
        if action == StructuralAction.CHANGE_GAUGE:
            return self.engine.evaluate_gauge_action(
                u=target.get("u"), v=target.get("v"), magnitude=float(target.get("magnitude", 0.01)),
            )
        return MutationResult(MutationDecision.ACCEPT, ["no_op"], metadata={"action": action.value})

    def _build_mutation(
        self,
        action: StructuralAction,
        target: dict[str, Any],
    ):
        """Build a mutation object from an action + target without executing it.

        v5.11 Phase 5: This separates mutation construction from mutation
        execution, enabling shadow-only evaluation.
        """
        assert self.engine is not None
        if action in (
            StructuralAction.ADD_EDGE, StructuralAction.PRUNE_EDGE,
            StructuralAction.REWEIGHT_AFFINITY, StructuralAction.REWEIGHT_LENGTH,
            StructuralAction.COUPLED_REWEIGHT,
        ):
            return action_to_mutation(
                action, self.engine.graph, self.engine.fibers().detach(), **target
            )
        # Fiber and gauge actions don't have a separate mutation object;
        # they're handled directly by the engine. Return None to indicate
        # the action should use the legacy path.
        return None

    def _apply_consolidation_gates(self) -> None:
        """Bind lifecycle gate values to actual spawned fiber channels."""
        if self.engine is None:
            return
        module = self.engine.fibers
        eps = 1e-5
        with torch.no_grad():
            for state in self.consolidation._fibers.values():
                node = state.metadata.get("node")
                channels = state.metadata.get("channels")
                if node is None or not channels:
                    continue
                ch = torch.as_tensor(channels, dtype=torch.long, device=module.gate_logits.device)
                if ch.numel() == 0:
                    continue
                g = min(1.0 - eps, max(eps, float(state.g_value)))
                logit = math.log(g / (1.0 - g))
                module.gate_logits[int(node), ch] = logit

    def _consume_long_term_credit(self) -> None:
        for outcome in self.credit_tracker.get_outcomes():
            if outcome.receipt_id in self._consumed_outcomes:
                continue
            ctx = self._credit_context.get(outcome.receipt_id)
            if ctx is not None:
                obs, action = ctx
                target_return = float(getattr(outcome, "advantage", outcome.discounted_return))
                self.executive.record_long_term_outcome(obs, action, target_return)
                self.uncertainty_estimator.update(
                    obs.to_vector(), ACTION_TO_IDX[action], target_return,
                    risk_target=0.0 if outcome.retained else 1.0,
                )
                self.calibrator.update(
                    float(outcome.metadata.get("predicted_delta_u", 0.0)),
                    float(outcome.discounted_return),
                )
            self._consumed_outcomes.add(outcome.receipt_id)

    def step(
        self,
        graph: GraphBuffers,
        z: Tensor,
        audit_snapshot: Any | None = None,
        task_loss: float = 0.0,
        task_loss_delta: float = 0.0,
        epistemic_uncertainty: float = 0.0,
        utility_fn: Callable[[GraphBuffers, Tensor], float] | None = None,
        shadow_simulator: Callable[[StructuralAction], float] | None = None,
    ) -> StructuralLoopResult:
        step = self._step
        if self.enforce_timescales:
            self.timescales.update(step)

        # The engine, when supplied, owns authoritative state.
        current_graph = self.engine.graph if self.engine is not None else graph
        current_z = self.engine.fibers().detach().clone() if self.engine is not None else z.detach().clone()
        fiber_state = self.engine.fibers if self.engine is not None else None
        if self.enforce_timescales:
            self.timescales.observe_latent(current_z)
        curvature_update_meta: dict[str, Any] | None = None
        if self.engine is not None and self.config.mutation.curvature_ema_enabled:
            curvature_update_meta = self.engine.update_curvature_history()
        if audit_snapshot is None and self.engine is not None:
            audit_snapshot = self.engine.audit()

        observation = self.executive.observe(
            current_graph, current_z, audit_snapshot, task_loss, task_loss_delta,
            epistemic_uncertainty, fiber_state=fiber_state,
        )
        counterfactual = self.counterfactual.evaluate(observation, shadow_simulator)
        chosen_action = counterfactual.winner if counterfactual.beats_no_op else StructuralAction.NO_OP
        target = self.executive.select_target(chosen_action, current_graph, current_z, fiber_state=fiber_state)

        hysteresis_meta: dict[str, Any] | None = None
        if (
            self.engine is not None
            and self.config.mutation.curvature_ema_enabled
            and chosen_action in (StructuralAction.ADD_EDGE, StructuralAction.PRUNE_EDGE)
        ):
            u = target.get("u")
            v = target.get("v")
            if u is None or v is None:
                hysteresis_meta = {"reason": "missing_edge_target"}
                chosen_action = StructuralAction.NO_OP
                target = {}
            else:
                allowed, hysteresis_meta = self.engine.curvature_hysteresis.allows(
                    "add" if chosen_action == StructuralAction.ADD_EDGE else "prune",
                    int(u), int(v),
                    add_threshold=self.config.mutation.add_curvature_threshold,
                    prune_threshold=self.config.mutation.prune_curvature_threshold,
                )
                if not allowed:
                    chosen_action = StructuralAction.NO_OP
                    target = {}

        obs_vec = observation.to_vector()
        action_idx = ACTION_TO_IDX.get(chosen_action, 0)
        unc_estimate = self.uncertainty_estimator.estimate(obs_vec, action_idx)
        conformal_interval = self.calibrator.interval(unc_estimate.mean) if self.calibrator.calibrated else None
        # v5.11 Phase 8: activate information gain, cost, and risk.
        # IG is derived from ensemble disagreement (uncertainty std).
        # Cost is proportional to the action's structural footprint.
        # Risk is derived from epistemic uncertainty and OOD score.
        ig = float(unc_estimate.std) * 0.1  # ensemble disagreement proxy
        cost = 0.01 if chosen_action != StructuralAction.NO_OP else 0.0
        risk = float(unc_estimate.std) * float(getattr(unc_estimate, 'ood_score', 0.0))
        score = unc_estimate.mean + 0.1 * ig - cost - 0.5 * risk
        uncertainty_decision = uncertainty_gated_decision(
            ActionProposal(
                action=chosen_action,
                expected_delta_utility=unc_estimate.mean,
                information_gain=ig,
                cost=cost,
                risk=risk,
                score=score,
                uncertainty=unc_estimate.std,
                lcb=unc_estimate.lcb,
            ),
            unc_estimate,
            conformal_interval=conformal_interval,
        )

        u_before = float(utility_fn(current_graph, current_z)) if utility_fn else 0.0
        # This advances every previously pending mutation even on NO_OP steps.
        self.credit_tracker.record_utility(step, u_before)
        self._consume_long_term_credit()

        governance_decision = "accept" if chosen_action == StructuralAction.NO_OP else "reject"
        mutation_result: MutationResult | None = None
        executed = False
        authority_before = self.engine.authority_hash() if self.engine is not None else current_graph.state_hash()

        if chosen_action != StructuralAction.NO_OP:
            if not self._action_allowed_by_timescale(chosen_action, step):
                governance_decision = "reject"
            elif uncertainty_decision != "accept":
                # Conservative gate: QUARANTINE is not execution.  We do not ask
                # the engine to commit when epistemic approval is absent.
                governance_decision = uncertainty_decision
            elif self.engine is not None:
                mutation_result = self._execute_engine_action(chosen_action, target)
                governance_decision = mutation_result.decision.value
                executed = mutation_result.decision == MutationDecision.ACCEPT
            elif self.governor is not None:
                bridge = certify_action_through_governor(
                    chosen_action, current_graph, current_z, self.governor, **target,
                )
                governance_decision = (
                    bridge.governor_result.decision.value if bridge.governor_result is not None else "reject"
                )
                # Certification without an engine is read-only by design.
                executed = False
            else:
                governance_decision = "quarantine"

        post_graph = self.engine.graph if self.engine is not None else current_graph
        post_z = self.engine.fibers().detach().clone() if self.engine is not None else current_z
        u_after = float(utility_fn(post_graph, post_z)) if (utility_fn and executed) else u_before
        delta_u = u_after - u_before
        authority_after = self.engine.authority_hash() if self.engine is not None else post_graph.state_hash()

        # v5.2: rejected/quarantined proposals are valuable supervision for
        # the risk head even though there is deliberately no task-utility label.
        if chosen_action != StructuralAction.NO_OP and not executed:
            self.executive.record_governance_outcome(
                observation,
                chosen_action,
                governance_decision,
                cost_target=self._estimate_cost(chosen_action, mutation_result),
                uncertainty_target=unc_estimate.std,
            )

        if executed:
            self.executive.record_mutation(chosen_action)
            predicted = (
                counterfactual.winner_proposal.expected_delta_utility
                if counterfactual.winner_proposal is not None else unc_estimate.mean
            )
            cost_target = self._estimate_cost(chosen_action, mutation_result)
            risk_target = 0.0
            unc_target = abs(float(predicted) - float(delta_u))

            # First assimilate the observed utility into the bootstrap ensemble.
            variance_before = float(unc_estimate.std ** 2)
            self.uncertainty_estimator.update(
                obs_vec, action_idx, delta_u,
                cost_target=cost_target, risk_target=risk_target,
            )
            unc_after = self.uncertainty_estimator.estimate(obs_vec, action_idx)
            variance_after = float(unc_after.std ** 2)
            # Realized epistemic-gain proxy: contraction of ensemble predictive
            # variance at the exact structural decision that was observed.
            ig_target = max(0.0, min(10.0, 0.5 * math.log((variance_before + 1e-8) / (variance_after + 1e-8))))
            self.executive.record_outcome(
                observation, chosen_action, delta_u,
                cost_target=cost_target, risk_target=risk_target,
                ig_target=ig_target, uncertainty_target=unc_target,
            )
            # Train the ensemble's auxiliary IG head without applying a second
            # delta-utility update to the posterior members.
            self.uncertainty_estimator.update(
                obs_vec, action_idx, delta_u,
                ig_target=ig_target, update_delta=False,
            )
            self.calibrator.update(float(predicted), float(delta_u))

            receipt = self.credit_tracker.record_mutation(
                action=chosen_action,
                step=step,
                predicted_delta_u=float(predicted),
                predicted_uncertainty=unc_estimate.std,
                governance_decision=governance_decision,
                governance_reasons=[] if mutation_result is None else list(mutation_result.reasons),
                graph_hash_before=authority_before,
                graph_hash_after=authority_after,
                config_governance_hash=config_governance_hash(self.config),
                metadata={"target": target, "authority_hash_before": authority_before, "authority_hash_after": authority_after},
                counterfactual_baseline=(
                    None if counterfactual.no_op_baseline is None
                    else float(counterfactual.no_op_baseline.expected_delta_utility)
                ),
            )
            self._credit_context[receipt.receipt_id] = (observation, chosen_action)
            # Establish the baseline for the newly-created pending mutation.
            self.credit_tracker.record_utility(step, u_before)

            if chosen_action == StructuralAction.SPAWN_FIBER and mutation_result is not None:
                channels = list(mutation_result.metadata.get("channels", []))
                if channels:
                    fs = self.consolidation.register_fiber(len(channels), step)
                    fs.metadata.update({"node": mutation_result.metadata.get("node"), "channels": channels})

        # Keep budget synchronized with authoritative state rather than side metadata.
        if self.engine is not None:
            self.consolidation.budget.total_edges = int(self.engine.graph.edge_count)
            self.consolidation.budget.total_fiber_dim = int(self.engine.fibers.capacity.sum().item())
        self.consolidation.update_lifecycle(step)
        self._apply_consolidation_gates()

        if len(self.executive._experience) >= 32:
            self.executive.train_step()

        self._step += 1
        return StructuralLoopResult(
            step=step,
            observation=observation,
            counterfactual=counterfactual,
            chosen_action=chosen_action,
            uncertainty_decision=uncertainty_decision,
            governance_decision=governance_decision,
            executed=executed,
            utility_before=u_before,
            utility_after=u_after,
            delta_utility=delta_u,
            metadata={
                "version": VERSION,
                "target": target,
                "authority_hash_before": authority_before,
                "authority_hash_after": authority_after,
                "uncertainty": {
                    "mean": unc_estimate.mean,
                    "std": unc_estimate.std,
                    "lcb": unc_estimate.lcb,
                    "ucb": unc_estimate.ucb,
                    "conformal_interval": conformal_interval,
                },
                "mutation_reasons": [] if mutation_result is None else list(mutation_result.reasons),
                "policy_supervision": {
                    "risk_target": None if chosen_action == StructuralAction.NO_OP else ({"accept": 0.0, "quarantine": 0.5, "reject": 1.0}.get(governance_decision, 1.0)),
                    "committed": bool(executed),
                },
                "consolidation": self.consolidation.summary(),
                "curvature_hysteresis": hysteresis_meta,
                "curvature_history": curvature_update_meta,
                "timescales": self.timescales.summary(),
            },
        )

    def summary(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "executive_experience": len(self.executive._experience),
            "credit": self.credit_tracker.summary(),
            "consolidation": self.consolidation.summary(),
            "timescales": self.timescales.summary(),
            "engine_bound": self.engine is not None,
            "version": VERSION,
        }
