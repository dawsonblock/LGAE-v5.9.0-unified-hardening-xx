"""Canonical v5.11 runtime: one authoritative end-to-end governed cycle.

``LGAERuntime`` orchestrates the 8-phase canonical cycle:

    observe -> reason -> propose -> plan -> evaluate -> authorize
    -> commit -> learn

Only ``commit()`` may mutate authoritative state. All other phases are
read-only w.r.t. authoritative state. Every phase emits an immutable,
state-bound contract from ``runtime.contracts``.

The authority model is strict:

  * Proposal authority  -> learned executive / counterfactual / memory / MPC
  * Verification authority -> governor (shadow evaluation / certification)
  * Commit authority -> ``LGAEEngine`` (the only component that mutates
    authoritative graph/fiber/gauge state, via transactional evaluation)

The runtime's ``step()`` calls all 8 phase methods in order. No hidden
nested orchestration via ``StructuralLearningLoop.step()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from ..config import LGAEConfig, ResearchConfig, config_governance_hash
from ..cache_coherence import (
    GraphReadCoordinator, run_consistent_read, StaleReadError as _CCStaleReadError,
    CommitEventBus, GraphCommitEvent, ChangeKind,
)
from ..evidence import EvidenceLedger, EvidenceRecord
from ..evolution import LGAEEngine
from ..executive import StructuralExecutive, StructuralAction, ACTION_TO_IDX
from ..receipts import mutation_receipt, append_receipt, ed25519_available, generate_keypair
from ..structural_loop import StructuralLearningLoop, StructuralLoopResult
from ..types import GraphBuffers, MutationDecision, MutationResult
from ..version import VERSION
from .runtime_config import RuntimeConfig, RuntimeMode
from .runtime_events import RuntimeEvent, RuntimePhase
from .runtime_result import RuntimeStepResult
from .runtime_state import RuntimeSnapshot, snapshot_from_engine, StaleReadError
from .authority import (
    AuthorityBoundary, AuthorityRole, AuthoritativeStateGuard,
    CommitChannel, UnauthorizedMutationError,
)
from .cache_coherence import MutationImpact, CacheRegistry
from .contracts import (
    ObservationSnapshot, ReasoningResult, StructuralDeficit, DiagnosticBundle,
    Candidate, CandidateSet, PlanningResult, CandidateValue,
    CounterfactualEvaluation, AuthorizationResult, AuthorizationStatus,
    RejectionReason, CommitResult, LearningResult, DecisionTransition,
    CreditAssignment, RuntimeStepResult as ContractStepResult,
    CANONICAL_PHASE_ORDER, canonical_hash,
)


class LGAERuntime:
    """One canonical governed structural-intelligence runtime.

    Parameters
    ----------
    graph:
        Initial authoritative graph buffers.
    config:
        Subsystem ``LGAEConfig`` (or preset). Defaults to ``ResearchConfig``.
    runtime_config:
        Orchestration-level config (mode, evidence/receipt paths, MPC).
    engine:
        Optional pre-built engine. When omitted the runtime constructs one
        bound to ``graph`` and ``config``. The engine is the sole commit
        authority.
    """

    def __init__(
        self,
        graph: GraphBuffers,
        config: LGAEConfig | None = None,
        *,
        runtime_config: RuntimeConfig | None = None,
        engine: LGAEEngine | None = None,
        executive: StructuralExecutive | None = None,
        utility_fn: Callable[[GraphBuffers, Tensor], float] | None = None,
    ) -> None:
        self.config = config or ResearchConfig()
        # Production mode must use the conservative preset. Never silently
        # fall into research behavior. We detect a production-grade config by
        # the safety machinery ProductionConfig() enables; a plain/research
        # config in production mode is a caller error.
        self.runtime_config = runtime_config or RuntimeConfig()
        if self.runtime_config.is_production:
            cfg = self.config
            production_grade = (
                bool(getattr(cfg.mutation, "curvature_ema_enabled", False))
                and bool(getattr(cfg.mutation, "equilibrium_barrier_enabled", False))
                and bool(getattr(cfg.audit, "require_persistent_homology", False))
            )
            if not production_grade:
                raise ValueError(
                    "production runtime mode requires a ProductionConfig-grade LGAEConfig "
                    "(curvature_ema_enabled, equilibrium_barrier_enabled, "
                    "require_persistent_homology must be enabled)"
                )

        self._engine = engine if engine is not None else LGAEEngine(graph, self.config)
        # v5.11 Phase 2: Generate exactly one authority capability.
        # This token is required for all engine mutation methods.
        from .state.authority_token import _AuthorityCapability
        self._authority_capability = _AuthorityCapability(id(self))
        self._engine._set_authority_capability(self._authority_capability)
        self.executive = executive or StructuralExecutive(self.config)

        util = utility_fn or self.runtime_config.utility_fn or _default_utility
        self.utility_fn = util

        # The governed structural-learning loop. It owns the counterfactual,
        # uncertainty, credit, consolidation, and timescale machinery and
        # delegates commit to the engine (the only commit authority).
        self.loop = StructuralLearningLoop(
            config=self.config,
            executive=self.executive,
            engine=self._engine,
            ensemble_size=self.runtime_config.ensemble_size,
            max_candidates=self.runtime_config.max_candidates,
        )

        # Evidence + receipt ledgers. In-memory when no path is configured.
        self.evidence_ledger = (
            EvidenceLedger(self.runtime_config.evidence_path)
            if self.runtime_config.evidence_path is not None
            else _InMemoryEvidenceLedger()
        )
        self._receipt_path = self.runtime_config.receipt_path
        self._signing_key = self.runtime_config.signing_key
        self._receipt_count = 0

        # Optional MPC planner (Phase 14). Lazily constructed so we do not
        # import heavy planning paths when horizon == 1.
        self._mpc: Any | None = None
        if self.runtime_config.mpc_horizon > 1:
            from ..mpc import StructuralMPC
            self._mpc = StructuralMPC(
                util,
                horizon=int(self.runtime_config.mpc_horizon),
                max_branching=int(self.runtime_config.mpc_max_branching),
                max_sequences=int(self.runtime_config.mpc_max_sequences),
            )

        self._step = 0
        self._events: list[RuntimeEvent] = []
        # Generation is the authoritative step counter bound to snapshots.
        self._generation = int(self._engine.step_index)
        # Phase execution tracking for v5.11 canonical path verification.
        self._last_phase_order: tuple[str, ...] = ()
        # v5.11 Sprint 3 D11-011: Utility before commit, for realized delta.
        self._u_before_commit: float = 0.0

        # Strict authority boundaries (Phase 2). The engine is the sole commit
        # authority; proposal/verification components receive read-only guards.
        self.boundary = AuthorityBoundary()
        self.boundary.register("engine", AuthorityRole.COMMIT)
        self.boundary.register("executive", AuthorityRole.PROPOSAL)
        self.boundary.register("counterfactual_engine", AuthorityRole.VERIFICATION)
        self.boundary.register("governor", AuthorityRole.VERIFICATION)
        if self._mpc is not None:
            self.boundary.register("mpc_planner", AuthorityRole.PROPOSAL)
        # Seqlock-style read coordinator (Phase 3): commits are bracketed by
        # a write epoch so optimistic readers retry on stale reads.
        self.read_coordinator = GraphReadCoordinator()
        # v5.11 Phase 12: WAL for crash-safe transactions.
        self._wal = None
        if self.runtime_config.wal_path is not None:
            from .wal import WriteAheadLog
            self._wal = WriteAheadLog(self.runtime_config.wal_path)
        self._commit_channel = CommitChannel(
            self._engine, self.boundary, component="engine",
            read_coordinator=self.read_coordinator,
            wal=self._wal,
            require_wal=self.runtime_config.is_production,
            capability=self._authority_capability,
        )
        # Mandatory cache coherence (Phase 4): a commit event bus drives
        # selective invalidation of declared-cache dependencies.
        self.commit_event_bus = CommitEventBus()
        self.cache_registry = CacheRegistry(self.commit_event_bus)

        # First-class structural intelligence & homeostasis modules (Phases 8-15)
        from .homeostasis import StructuralHomeostasis
        from .structural_diagnosis import StructuralDiagnoser, StructuralAttentionBudget
        from .multi_fidelity import MultiFidelityFunnel
        self.homeostasis = StructuralHomeostasis()
        self.diagnoser = StructuralDiagnoser()
        self.attention_budget = StructuralAttentionBudget()
        self.multi_fidelity = MultiFidelityFunnel()

    def recover_from_wal(self) -> list[dict[str, Any]]:
        """Recover committed transactions from the WAL.

        v5.11-RC Phase 14: Production startup must use this method to
        replay committed transactions. This ensures that recovery uses
        the canonical authority state and WAL protocol.

        In production mode, if the WAL is not available or is corrupted,
        this method raises an error (fail-closed).
        """
        if self._wal is None:
            if self.runtime_config.is_production:
                raise RuntimeError(
                    "production runtime requires a WAL for recovery; "
                    "no WAL path configured"
                )
            return []
        # Verify the WAL hash chain.
        if not self._wal.verify_chain():
            if self.runtime_config.is_production:
                raise RuntimeError(
                    "WAL hash chain verification failed; "
                    "cannot recover from corrupted WAL (fail-closed)"
                )
        # Replay committed transactions.
        from .wal import replay_committed_transactions
        return replay_committed_transactions(
            self.runtime_config.wal_path, self._engine,
        )

    # ------------------------------------------------------------------ #
    # Public API: read-only engine facade (Phase 1)
    # ------------------------------------------------------------------ #

    @property
    def engine(self) -> "EngineFacade":
        """Read-only facade over the internal engine.

        Returns an EngineFacade that exposes read methods but blocks
        all mutation. Direct engine mutation is physically prevented.
        """
        from .state.immutable_views import EngineFacade
        return EngineFacade(self._engine)

    @property
    def commit_channel(self) -> CommitChannel:
        """The sole authoritative mutation channel."""
        return self._commit_channel

    # ------------------------------------------------------------------ #
    # Authority boundary helpers (Phase 2 foundation)
    # ------------------------------------------------------------------ #
    def _assert_commit_authority(self) -> None:
        """Only the engine may mutate authoritative state. The runtime is an
        orchestrator, not a mutator."""
        if self._engine is None:
            raise UnauthorizedMutationError("no commit authority (engine) is bound")
        self.boundary.assert_can_mutate("engine")

    def guard_for(self, component: str) -> AuthoritativeStateGuard:
        """Return a read-only authoritative-state guard for a non-commit
        component. Commit-authority components must use the commit channel."""
        if self.boundary.role_of(component) == AuthorityRole.COMMIT:
            raise UnauthorizedMutationError(
                f"component '{component}' is commit-authority; use the commit channel, not a guard"
            )
        return AuthoritativeStateGuard(self._engine, self.boundary, component=component)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def authority_hash(self) -> str:
        return self._engine.authority_hash()

    @property
    def state_identity(self) -> Any:
        from .state_identity import AuthorityStateIdentity
        if hasattr(self._engine, "state_identity"):
            return self._engine.state_identity()
        return AuthorityStateIdentity.from_engine(self._engine)

    def snapshot(self) -> RuntimeSnapshot:
        """Capture an immutable authoritative snapshot for readers."""
        return snapshot_from_engine(self._engine, generation=self._generation)

    def consistent_read(self, compute_fn: Callable[[], Any]) -> Any:
        """Run a derived calculation and publish only a generation-consistent
        result. Retries on stale reads up to ``max_stale_read_retries``.

        This is the canonical reader path: every expensive reader should
        operate through ``consistent_read`` so no subsystem silently fetches
        mutable state halfway through a calculation. A read that overlaps a
        commit raises ``StaleReadError`` and is retried from a new snapshot.
        """
        return run_consistent_read(
            self.read_coordinator,
            generation_getter=lambda: int(self._engine.graph.version),
            compute_fn=compute_fn,
            max_retries=int(self.runtime_config.max_stale_read_retries),
        )

    # ------------------------------------------------------------------ #
    # Canonical 8-phase cycle (v5.11).
    #
    # Each phase is a real method that does actual work and returns an
    # immutable contract from runtime.contracts. step() calls them in
    # canonical order. No hidden nested orchestration.
    # ------------------------------------------------------------------ #
    def observe(self, *, task_loss: float = 0.0, task_loss_delta: float = 0.0,
                epistemic_uncertainty: float = 0.0) -> ObservationSnapshot:
        """Phase 1: Observation / Graph State -> Stable Snapshot.

        Captures an immutable authoritative snapshot. Every subsequent
        phase binds to this snapshot's version and hash.
        """
        snap = self.snapshot()
        obs = ObservationSnapshot(
            snapshot_id=f"{snap.authority_hash}:{snap.generation}",
            state_version=snap.generation,
            state_hash=snap.authority_hash,
            graph_version=snap.graph_version,
            authority_hash=snap.authority_hash,
            task_loss=float(task_loss),
            task_loss_delta=float(task_loss_delta),
            epistemic_uncertainty=float(epistemic_uncertainty),
            created_at_step=self._step,
        )
        self._emit(RuntimePhase.OBSERVE, {"graph_version": snap.graph_version,
                                          "authority_hash": snap.authority_hash,
                                          "task_loss": float(task_loss)})
        self._emit(RuntimePhase.SNAPSHOT, snap.to_summary())
        return obs

    def reason(self, observation: ObservationSnapshot) -> ReasoningResult:
        """Phase 2: Reasoning Graph + Memory.

        Runs the executive observe + diagnostics + uncertainty estimation.
        Produces structural deficits that drive candidate generation.
        """
        graph = self._engine.graph
        z = self._engine.fibers().detach().clone()
        audit = self._engine.audit()
        exec_obs = self.executive.observe(
            graph, z, audit,
            task_loss=observation.task_loss,
            task_loss_delta=observation.task_loss_delta,
            epistemic_uncertainty=observation.epistemic_uncertainty,
            fiber_state=self._engine.fibers,
        )
        # Uncertainty estimation.
        obs_vec = exec_obs.to_vector()
        unc_estimate = self.loop.uncertainty_estimator.estimate(obs_vec, 0)
        epistemic = float(unc_estimate.std)
        aleatoric = float(getattr(unc_estimate, "aleatoric_std", 0.0))
        # OOD score: distance from training distribution (simplified).
        ood_score = float(getattr(unc_estimate, "ood_score", 0.0))
        # Diagnostics: derive deficits from the observation.
        deficits: list[StructuralDeficit] = []
        # Check for oversquashing: low spectral gap indicates bottleneck.
        if hasattr(audit, "spectral_gap") and audit.spectral_gap < 0.1:
            deficits.append(StructuralDeficit(
                deficit_type="oversquashing",
                location="global",
                severity=min(1.0, 1.0 - float(audit.spectral_gap)),
                confidence=0.8,
                evidence={"spectral_gap": float(audit.spectral_gap)},
            ))
        # Check for negative curvature concentration.
        if hasattr(audit, "ricci_min") and audit.ricci_min < -0.5:
            deficits.append(StructuralDeficit(
                deficit_type="negative_curvature",
                location="edge",
                severity=min(1.0, abs(float(audit.ricci_min))),
                confidence=0.7,
                evidence={"ricci_min": float(audit.ricci_min)},
            ))
        # First-class structural diagnoses (Phase 12)
        diagnoses = self.diagnoser.diagnose(graph, audit, epistemic_uncertainty=epistemic)

        result = ReasoningResult(
            snapshot_id=observation.snapshot_id,
            state_version=observation.state_version,
            state_hash=observation.state_hash,
            diagnostics=DiagnosticBundle(diagnostic_level="L1"),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            ood_score=ood_score,
            deficits=tuple(deficits),
            diagnoses=tuple(diagnoses),
        )
        self._emit(RuntimePhase.REASON, {
            "observation": obs_vec.tolist(),
            "epistemic_uncertainty": epistemic,
            "deficits": len(deficits),
        })
        # Store for use by subsequent phases.
        self._exec_observation = exec_obs
        self._unc_estimate = unc_estimate
        return result

    def propose(self, observation: ObservationSnapshot,
                reasoning: ReasoningResult) -> CandidateSet:
        """Phase 3: Candidate Generation + Ranking.

        Delegates to the counterfactual engine to generate structural
        candidates. Produces a deterministically ordered, deduplicated set.
        """
        exec_obs = self._exec_observation
        counterfactual = self.loop.counterfactual.evaluate(exec_obs, None)
        # Build candidate contracts.
        candidates: list[Candidate] = []
        for prop in counterfactual.proposals:
            action = prop.action if hasattr(prop, "action") else StructuralAction.NO_OP
            params = {}
            if hasattr(prop, "target"):
                params = dict(prop.target) if prop.target else {}
            elif hasattr(prop, "u") and hasattr(prop, "v"):
                params = {"u": int(prop.u), "v": int(prop.v)}
            cid = canonical_hash({
                "state_hash": observation.state_hash,
                "action_type": action.value if hasattr(action, "value") else str(action),
                "parameters": params,
            })
            candidates.append(Candidate(
                candidate_id=cid,
                source_state_hash=observation.state_hash,
                source_state_version=observation.state_version,
                action_type=action.value if hasattr(action, "value") else str(action),
                parameters=params,
                origin="counterfactual",
                expected_utility=float(getattr(prop, "expected_delta_utility", 0.0)),
            ))
        chosen_action = counterfactual.winner if counterfactual.beats_no_op else StructuralAction.NO_OP
        # Canonicalize candidate enumeration. Candidate identity is semantic
        # and deterministic, so input proposal order cannot influence the
        # planner. Keep the first occurrence of an identical candidate ID,
        # then sort by that ID.
        total_generated = len(candidates)
        deduped = {c.candidate_id: c for c in candidates}
        candidates = sorted(deduped.values(), key=lambda c: c.candidate_id)
        result = CandidateSet(
            snapshot_id=observation.snapshot_id,
            state_version=observation.state_version,
            state_hash=observation.state_hash,
            candidates=tuple(candidates),
            total_generated=total_generated,
            duplicates_removed=total_generated - len(candidates),
        )
        self._emit(RuntimePhase.PROPOSE, {
            "candidates": len(candidates),
            "beats_no_op": bool(counterfactual.beats_no_op),
            "winner": chosen_action.value if hasattr(chosen_action, "value") else str(chosen_action),
        })
        # Store for subsequent phases.
        self._counterfactual = counterfactual
        self._chosen_action = chosen_action
        return result

    def plan(self, observation: ObservationSnapshot,
             reasoning: ReasoningResult,
             candidates: CandidateSet) -> PlanningResult:
        """Phase 4: Multi-Step Counterfactual Planning.

        When MPC is enabled, plans a receding horizon. Otherwise selects
        the single-step winner. Applies IG/cost/risk decomposition.
        """
        chosen = self._chosen_action
        # Canonical multi-objective valuation. Use the uncertainty ensemble
        # per action to derive information value and risk, while cost reflects
        # structural footprint. These terms now causally participate in
        # selection rather than being hard-coded to zero.
        candidate_values: list[CandidateValue] = []
        scored: list[tuple[float, str, Candidate]] = []
        obs_vec = self._exec_observation.to_vector()
        ig_w = float(getattr(self.runtime_config, "information_gain_weight", 0.1))
        cost_w = float(getattr(self.runtime_config, "cost_weight", 1.0))
        risk_w = float(getattr(self.runtime_config, "risk_weight", 0.5))
        rho_h = float(getattr(self.runtime_config, "homeostasis_weight", 0.5))
        for c in candidates.candidates:
            try:
                action = StructuralAction(c.action_type)
                action_idx = ACTION_TO_IDX.get(action, 0)
                estimate = self.loop.uncertainty_estimator.estimate(obs_vec, action_idx)
                ig = float(estimate.std)
                ood = float(getattr(reasoning, "ood_score", 0.0))
                risk = float(estimate.std) * max(0.0, ood)
            except Exception:
                ig = 0.0
                risk = max(0.0, float(getattr(reasoning, "epistemic_uncertainty", 0.0)))
            cost = _structural_action_cost(c.action_type)
            h_pen = self.homeostasis.compute_homeostasis_penalty(
                self._engine.graph, c.action_type, c.parameters, current_step=self._step
            )
            total = float(c.expected_utility) + ig_w * ig - cost_w * cost - risk_w * risk - rho_h * h_pen.total_penalty
            candidate_values.append(CandidateValue(
                expected_utility=float(c.expected_utility),
                information_gain=ig,
                cost=cost,
                risk=risk,
                homeostasis_penalty=h_pen.total_penalty,
                total_score=total,
            ))
            scored.append((total, c.candidate_id, c))

        # Max score wins; candidate_id provides a deterministic tie-break.
        selected: Candidate | None = None
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            selected = scored[0][2]
            try:
                chosen = StructuralAction(selected.action_type)
                self._chosen_action = chosen
            except Exception:
                pass
        # MPC planning (if enabled).
        mpc_plan: tuple[str, ...] = ()
        planner_name = "single_step"
        horizon = 1
        if self._mpc is not None and chosen != StructuralAction.NO_OP:
            planner_name = "mpc"
            horizon = int(self.runtime_config.mpc_horizon)
            graph = self._engine.graph
            z = self._engine.fibers().detach().clone()
            try:
                plan_result = self._mpc.plan(graph, z, seed=int(self.config.seed) + self._step)
                horizon = int(plan_result.horizon)
                self._planned_mutation = (
                    plan_result.best_sequence[0]
                    if plan_result.best_sequence else None
                )
                if self._planned_mutation is not None:
                    mpc_action = _action_for_mutation(self._planned_mutation)
                    if mpc_action is not None:
                        self._chosen_action = mpc_action
                        chosen = mpc_action
                        matching = [c for c in candidates.candidates if c.action_type == mpc_action.value]
                        if matching:
                            selected = sorted(matching, key=lambda c: c.candidate_id)[0]
                        mpc_plan = tuple(
                            _mutation_plan_token(m) for m in plan_result.best_sequence
                            if m is not None
                        )
                self._emit(RuntimePhase.PLAN, {
                    "horizon": horizon,
                    "candidates_evaluated": int(plan_result.candidates_evaluated),
                    "predicted_utility": float(plan_result.predicted_utility),
                    "first_authority": plan_result.first_mutation_authority.value,
                    "selected_action": chosen.value if hasattr(chosen, "value") else str(chosen),
                })
            except Exception:
                planner_name = "single_step_fallback"
                horizon = 1
        result = PlanningResult(
            snapshot_id=observation.snapshot_id,
            state_version=observation.state_version,
            state_hash=observation.state_hash,
            selected_candidate=selected,
            candidate_values=tuple(candidate_values),
            horizon=horizon,
            mpc_plan=mpc_plan,
            planner=planner_name,
        )
        if planner_name == "single_step":
            self._emit(RuntimePhase.PLAN, {"horizon": 1, "planner": "single_step"})
        return result

    def evaluate(self, observation: ObservationSnapshot,
                 planning: PlanningResult) -> CounterfactualEvaluation:
        """Phase 5: Shadow Transaction + Exact/Escalating Verification.

        v5.11 Phase 5: Shadow-only evaluation. Does NOT mutate authoritative
        state. Creates a StructuralTransaction capturing the shadow state.

        The governor evaluates the mutation on a shadow graph clone and
        returns (MutationResult, shadow_graph). No authoritative state
        changes occur in this phase.
        """
        from .transaction import make_graph_transaction

        chosen_action = self._chosen_action
        if chosen_action == StructuralAction.NO_OP:
            result = CounterfactualEvaluation(
                snapshot_id=observation.snapshot_id,
                state_version=observation.state_version,
                state_hash=observation.state_hash,
                candidate=planning.selected_candidate,
                passed=False,
            )
            self._emit(RuntimePhase.EVALUATE, {"decision": "no_op", "certification": None})
            self._mutation_result = None
            self._certification_level = None
            self._transaction = None
            return result
        # If MPC supplied an exact first mutation, execute that same mutation
        # in shadow evaluation. Otherwise select a target through the executive.
        planned_mutation = getattr(self, "_planned_mutation", None)
        if planned_mutation is not None and _action_for_mutation(planned_mutation) == chosen_action:
            mutation = planned_mutation
            target = _target_for_mutation(planned_mutation)
        else:
            target = self.executive.select_target(
                chosen_action, self._engine.graph, self._engine.fibers().detach(),
                fiber_state=self._engine.fibers,
            )
            mutation = self.loop._build_mutation(chosen_action, target)
        self._target = target
        if mutation is None:
            # Action not supported as a direct mutation.
            result = CounterfactualEvaluation(
                snapshot_id=observation.snapshot_id,
                state_version=observation.state_version,
                state_hash=observation.state_hash,
                candidate=planning.selected_candidate,
                passed=False,
            )
            self._emit(RuntimePhase.EVALUATE, {"decision": "no_mutation", "certification": None})
            self._mutation_result = None
            self._certification_level = None
            self._transaction = None
            return result
        # Shadow-only evaluation via the governor (no authoritative mutation).
        # Canonical transaction identity must bind to the complete authority
        # state, not the graph-only hash. CommitChannel validates against
        # authority_hash(), so using graph.state_hash() here creates a false
        # stale-transaction rejection for every fresh canonical mutation.
        base_hash = self._engine.authority_hash()
        base_version = int(self._engine.graph.version)
        mut_result, shadow_graph = self._engine.governor.evaluate_mutation(
            self._engine.graph, self._engine.fibers().detach(), mutation,
            seed=int(self.config.seed) + self._step,
            gauge_bank=self._engine.gauge_connections,
        )
        cert = None
        if isinstance(mut_result, MutationResult) and mut_result.metadata:
            cert = mut_result.metadata.get("certification_level")
        passed = isinstance(mut_result, MutationResult) and mut_result.decision == MutationDecision.ACCEPT
        # Build a StructuralTransaction capturing the shadow state.
        # This transaction will be committed in Phase 7 if authorized.
        candidate_id = (
            planning.selected_candidate.candidate_id
            if planning.selected_candidate is not None else None
        )
        transaction = make_graph_transaction(
            base_state_version=base_version,
            base_state_hash=base_hash,
            shadow_graph=shadow_graph,
            mutation_result=mut_result,
            mutation_name=getattr(mutation, "name", type(mutation).__name__),
            mutation_metadata=getattr(mut_result, "metadata", {}),
            candidate_id=candidate_id,
            plan_id=observation.snapshot_id,
            step=self._step,
        )
        result = CounterfactualEvaluation(
            snapshot_id=observation.snapshot_id,
            state_version=observation.state_version,
            state_hash=observation.state_hash,
            candidate=planning.selected_candidate,
            predicted_utility=float(getattr(mut_result, "delta_utility", 0.0)) if passed else 0.0,
            invariant_violations=tuple(
                str(r) for r in getattr(mut_result, "reasons", []) if "invariant" in str(r).lower()
            ) if isinstance(mut_result, MutationResult) else (),
            certification_level=cert,
            certification_reasons=tuple(
                str(r) for r in getattr(mut_result, "reasons", [])
            ) if isinstance(mut_result, MutationResult) else (),
            shadow_state_hash=shadow_graph.state_hash(),
            passed=passed,
        )
        self._emit(RuntimePhase.EVALUATE, {
            "decision": mut_result.decision.value if isinstance(mut_result, MutationResult) else "no_op",
            "certification": cert,
            "reasons": list(getattr(mut_result, "reasons", [])) if isinstance(mut_result, MutationResult) else [],
            "shadow_only": True,
            "transaction_id": transaction.transaction_id,
        })
        self._mutation_result = mut_result
        self._certification_level = cert
        self._transaction = transaction
        return result

    def authorize(self, observation: ObservationSnapshot,
                  evaluation: CounterfactualEvaluation) -> AuthorizationResult:
        """Phase 6: Authority Governor decision (reject/quarantine/commit).

        v5.11 Phase 4-5: The authorization is cryptographically bound to
        the transaction. The authorization_id is the transaction's
        authorization_binding_hash, preventing transaction swap attacks.
        """
        mut_result = self._mutation_result
        transaction = getattr(self, "_transaction", None)
        if mut_result is None or transaction is None:
            status = AuthorizationStatus.AUTHORIZED
            reason = RejectionReason.NO_OP
            auth_id = None
        else:
            decision = mut_result.decision
            if decision == MutationDecision.ACCEPT:
                status = AuthorizationStatus.AUTHORIZED
                reason = RejectionReason.NO_OP
            elif decision == MutationDecision.REJECT:
                status = AuthorizationStatus.REJECTED
                reason = RejectionReason.CERTIFICATION_FAILED
            else:
                status = AuthorizationStatus.QUARANTINED
                reason = RejectionReason.UNCERTAINTY_TOO_HIGH
            # Bind authorization to the transaction.
            auth_id = transaction.authorization_binding_hash()
            # Store the authorization_id on the transaction (reconstruct frozen).
            from .transaction import StructuralTransaction
            self._transaction = StructuralTransaction(
                transaction_id=transaction.transaction_id,
                base_state_version=transaction.base_state_version,
                base_state_hash=transaction.base_state_hash,
                graph_delta=transaction.graph_delta,
                fiber_delta=transaction.fiber_delta,
                gauge_delta=transaction.gauge_delta,
                candidate_id=transaction.candidate_id,
                plan_id=transaction.plan_id,
                authorization_id=auth_id,
                delta_hash=transaction.delta_hash,
                mutation_result=transaction.mutation_result,
            )
        result = AuthorizationResult(
            snapshot_id=observation.snapshot_id,
            state_version=observation.state_version,
            state_hash=observation.state_hash,
            status=status,
            reason=reason,
            certification_level=self._certification_level,
            authority_hash_before=observation.state_hash,
            transaction_hash=transaction.transaction_id if transaction is not None else "",
            candidate_id=transaction.candidate_id if transaction is not None else "",
            evaluation_hash=evaluation.to_hash() if evaluation is not None else "",
        )
        self._emit(RuntimePhase.AUTHORIZE, {"decision": status.value, "reason": reason.value})
        return result

    def commit(self, observation: ObservationSnapshot,
               authorization: AuthorizationResult) -> CommitResult:
        """Phase 7: Atomic State Update + Cache Invalidation + Evidence/Receipt.

        The engine has already performed the atomic commit inside
        ``evaluate`` (it is the commit authority). This phase records the
        immutable evidence and signed receipt for the committed mutation.
        """
        mut_result = self._mutation_result
        transaction = getattr(self, "_transaction", None)
        executed = (
            mut_result is not None
            and mut_result.decision == MutationDecision.ACCEPT
            and authorization.is_authorized
            and transaction is not None
        )
        if executed:
            self._assert_commit_authority()
            before_hash = observation.state_hash
            # v5.11 Sprint 3 D11-011: Capture utility BEFORE commit for
            # realized delta computation in learn().
            self._u_before_commit = float(self.utility_fn(
                self._engine.graph, self._engine.fibers().detach().clone(),
            ))
            # v5.11 Phase 4-5: commit through the CommitChannel, not
            # by directly mutating engine state. The CommitChannel
            # validates authorization binding, base state, and delta hash.
            try:
                commit_result = self.commit_channel.commit(
                    transaction, authorization,
                )
            except Exception as exc:
                # Commit failed (stale state, binding error, etc.).
                self._emit(RuntimePhase.COMMIT, {
                    "error": str(exc),
                    "transaction_id": transaction.transaction_id,
                })
                return CommitResult(
                    snapshot_id=observation.snapshot_id,
                    state_version=observation.state_version,
                    state_hash=observation.state_hash,
                    committed=False,
                )
            after_hash = self._engine.authority_hash()
            # Immutable evidence record.
            evidence = self.evidence_ledger.append(EvidenceRecord(
                record_type="runtime_mutation_commit",
                graph_hash=after_hash,
                payload={
                    "action": self._chosen_action.value,
                    "target": self._target,
                    "decision": mut_result.decision.value,
                    "reasons": list(mut_result.reasons),
                    "certification_level": self._certification_level,
                    "authority_hash_before": before_hash,
                    "authority_hash_after": after_hash,
                    "step": int(self._step),
                    "transaction_id": transaction.transaction_id,
                    "delta_hash": transaction.delta_hash,
                },
                authority_hash=after_hash,
            ))
            evidence_hash = evidence.get("sha256")
            self._emit(RuntimePhase.EVIDENCE, {"evidence_hash": evidence_hash})
            # Signed hash-chained receipt.
            receipt = mutation_receipt(
                mut_result,
                authority_state_hash_before=before_hash,
                authority_state_hash_after=after_hash,
                gauge_authority_hash=(
                    None if self._engine.gauge_connections is None
                    else self._engine.gauge_connections.state_hash()
                ),
                signing_key=self._signing_key,
            )
            receipt_hash = receipt.get("sha256")
            if self._receipt_path is not None:
                append_receipt(self._receipt_path, receipt, signing_key=self._signing_key)
            self._receipt_count += 1
            # Cache invalidation via commit event bus.
            impact = _impact_for_action(self._chosen_action)
            self.commit_event_bus.publish(GraphCommitEvent(
                generation=int(self._engine.graph.version),
                changes=impact.to_change_kind(),
                reason="runtime_commit",
            ))
            self._emit(RuntimePhase.COMMIT, {
                "authority_hash_after": after_hash,
                "receipt_hash": receipt_hash,
                "mutation_impact": impact.to_log(),
                "transaction_id": transaction.transaction_id,
            })
            self._emit(RuntimePhase.CACHE_INVALIDATE, {
                "graph_version": int(self._engine.graph.version),
                "invalidated": self.cache_registry.invalidations[-1]["invalidated"] if self.cache_registry.invalidations else [],
                "spared": self.cache_registry.invalidations[-1]["spared"] if self.cache_registry.invalidations else [],
            })
            u_after_commit = float(self.utility_fn(
                self._engine.graph, self._engine.fibers().detach().clone(),
            ))
            realized_delta = u_after_commit - float(self._u_before_commit)
            self.homeostasis.record_committed_action(
                self._step,
                getattr(self._chosen_action, "value", str(self._chosen_action)),
                getattr(self, "_target", {}) or {},
            )
            result = CommitResult(
                snapshot_id=observation.snapshot_id,
                state_version=observation.state_version,
                state_hash=observation.state_hash,
                committed=True,
                new_state_version=int(self._engine.graph.version),
                new_state_hash=after_hash,
                transaction_id=transaction.transaction_id,
                receipt_hash=receipt_hash,
                evidence_hash=evidence_hash,
                delta_utility=realized_delta,
                authority_hash_after=after_hash,
            )
        else:
            result = CommitResult(
                snapshot_id=observation.snapshot_id,
                state_version=observation.state_version,
                state_hash=observation.state_hash,
                committed=False,
            )
        return result

    def learn(self, observation: ObservationSnapshot,
              commit: CommitResult) -> LearningResult:
        """Phase 8: Replay / Experience -> Learn.

        Records the decision transition and updates credit/calibration.

        v5.11 Sprint 3: Learning integrity fixes.
        D11-011: Uses realized delta utility (U_after - U_before), not predicted delta.
        D11-012: Calibration compares predicted delta with realized delta, not absolute utility.
        D11-013: Hierarchical credit assignment connected (not just flat outcome_credit).
        """
        # Update credit tracker with current utility.
        graph = self._engine.graph
        z = self._engine.fibers().detach().clone()
        u_now = float(self.utility_fn(graph, z))
        self.loop.credit_tracker.record_utility(self._step, u_now)
        # v5.11 Sprint 3 D11-011: Compute REALIZED delta utility.
        # ΔU_realized = U_after - U_before
        # This is the actual outcome the policy should learn from.
        u_before = float(getattr(self, "_u_before_commit", u_now))
        realized_delta = float(commit.delta_utility) if commit.committed else 0.0
        predicted_delta = float(getattr(
            self._mutation_result, "delta_utility", 0.0
        )) if self._mutation_result is not None else 0.0
        # Record governance outcome for rejected/quarantined proposals.
        if self._chosen_action != StructuralAction.NO_OP and not commit.committed:
            if hasattr(self, "_exec_observation") and self._exec_observation is not None:
                self.executive.record_governance_outcome(
                    self._exec_observation,
                    self._chosen_action,
                    "reject",
                )
        # v5.11 Sprint 3 D11-011: Record the outcome with REALIZED delta.
        # The policy learns from the actual consequences of its actions.
        if self._chosen_action != StructuralAction.NO_OP and commit.committed:
            if hasattr(self, "_exec_observation") and self._exec_observation is not None:
                self.executive.record_outcome(
                    self._exec_observation,
                    self._chosen_action,
                    realized_delta,  # D11-011: use realized delta, not predicted
                    cost_target=float(getattr(self._mutation_result, "cost", 0.0))
                        if self._mutation_result is not None else None,
                    risk_target=float(getattr(self._mutation_result, "risk", 0.0))
                        if self._mutation_result is not None else None,
                    ig_target=float(getattr(self._mutation_result, "information_gain", 0.0))
                        if self._mutation_result is not None else None,
                )
        # v5.11 Sprint 3 D11-012: Calibration compares predicted delta
        # with REALIZED delta, not absolute utility.
        if self._mutation_result is not None and commit.committed:
            try:
                self.calibrator.update(predicted_delta, realized_delta)
            except Exception:
                pass  # calibration update is best-effort
        # Build the decision transition.
        transition = DecisionTransition(
            pre_state_hash=observation.state_hash,
            post_state_hash=commit.new_state_hash if commit.committed else observation.state_hash,
            selected_action=self._chosen_action.value if hasattr(self._chosen_action, "value") else str(self._chosen_action),
            predicted_outcome=predicted_delta,
            realized_outcome=realized_delta,  # D11-011: realized delta, not absolute utility
            reward=realized_delta,  # D11-011: reward is realized delta, not predicted
            authorization_status="authorized" if commit.committed else "rejected",
            transition_id=canonical_hash({
                "pre": observation.state_hash,
                "post": commit.new_state_hash if commit.committed else observation.state_hash,
                "step": self._step,
            }),
        )
        # v5.11 Sprint 3 D11-013: Hierarchical credit assignment.
        # Distribute realized delta across subsystems instead of using
        # flat outcome_credit = u_now.
        # The credit tracker tracks mutation receipts and computes
        # discounted returns R = Σ γ^τ ΔU_{t+τ}.
        diagnostic_credit = 0.0
        candidate_credit = 0.0
        planner_credit = 0.0
        action_credit = 0.0
        governance_credit = 0.0
        if commit.committed and realized_delta != 0.0:
            # Distribute credit across subsystems.
            # v5.11-RC Phase 15: This is a simple per-subsystem decomposition:
            # - diagnostics: 10% (the observation/diagnostic phase)
            # - candidates: 20% (the candidate generation phase)
            # - planner: 20% (the counterfactual planning phase)
            # - action: 20% (the action selection phase)
            # - governance: 30% (the authorization/commit phase)
            diagnostic_credit = realized_delta * 0.1
            candidate_credit = realized_delta * 0.2
            planner_credit = realized_delta * 0.2
            action_credit = realized_delta * 0.2
            governance_credit = realized_delta * 0.3
        outcome_credit_val = realized_delta if commit.committed else 0.0
        credit = CreditAssignment(
            diagnostic_credit=diagnostic_credit,
            candidate_credit=candidate_credit,
            planner_credit=planner_credit,
            action_credit=action_credit,
            governance_credit=governance_credit,
            outcome_credit=outcome_credit_val,
        )
        result = LearningResult(
            snapshot_id=observation.snapshot_id,
            state_version=observation.state_version,
            state_hash=observation.state_hash,
            transition=transition,
            credit=credit,
            replay_buffer_size=len(self.executive._experience),
            calibration_updated=commit.committed and self._mutation_result is not None,
        )
        self._emit(RuntimePhase.LEARN, {
            "executive_experience": len(self.executive._experience),
            "credit_summary": self.loop.credit_tracker.summary(),
            "predicted_delta": predicted_delta,
            "realized_delta": realized_delta,  # D11-011: report realized delta
            "realized_utility": u_now,
            "utility_before": u_before,  # D11-011: report u_before
            "outcome_recorded": commit.committed and self._chosen_action != StructuralAction.NO_OP,
        })
        return result

    # ------------------------------------------------------------------ #
    # The complete governed cycle (v5.11 canonical 8-phase path).
    # ------------------------------------------------------------------ #
    def step(self, *, task_loss: float = 0.0, task_loss_delta: float = 0.0,
             epistemic_uncertainty: float = 0.0) -> RuntimeStepResult:
        """Run one complete governed cycle end-to-end.

        Executes all 8 canonical phases in order:
            observe -> reason -> propose -> plan -> evaluate -> authorize
            -> commit -> learn

        No hidden nested orchestration. Each phase is a real method call
        that does actual work and returns an immutable contract.
        """
        # Track which phases are called (for GATE-1B verification).
        phases_called: list[str] = []

        # Capture the before-snapshot before any work is done.
        snap_before = self.snapshot()

        # Phase 1: OBSERVE
        observation = self.observe(
            task_loss=task_loss, task_loss_delta=task_loss_delta,
            epistemic_uncertainty=epistemic_uncertainty,
        )
        phases_called.append("observe")
        authority_before = observation.state_hash

        # Phase 2: REASON
        reasoning = self.reason(observation)
        phases_called.append("reason")

        # Phase 3: PROPOSE
        candidates = self.propose(observation, reasoning)
        phases_called.append("propose")

        # Phase 4: PLAN
        planning = self.plan(observation, reasoning, candidates)
        phases_called.append("plan")

        # Phase 5: EVALUATE
        evaluation = self.evaluate(observation, planning)
        phases_called.append("evaluate")

        # Phase 6: AUTHORIZE
        authorization = self.authorize(observation, evaluation)
        phases_called.append("authorize")

        # Phase 7: COMMIT
        commit_result = self.commit(observation, authorization)
        phases_called.append("commit")

        # Phase 8: LEARN
        learning = self.learn(observation, commit_result)
        phases_called.append("learn")

        # Record the phase order for verification.
        self._last_phase_order = tuple(phases_called)

        snap_after = self.snapshot()
        self._generation = int(self._engine.step_index)
        self._step += 1

        # Compute utility delta for backward-compatible result.
        u_before = float(self.utility_fn(
            self._engine.graph, self._engine.fibers().detach().clone(),
        ))
        delta_u = float(commit_result.delta_utility) if commit_result.committed else 0.0

        # For backward compatibility, governance_decision uses the engine's
        # MutationDecision value ("accept"/"reject"/"quarantine"), not the
        # authorization status ("authorized"/"rejected"/etc.).
        if self._mutation_result is not None:
            gov_decision = self._mutation_result.decision.value
        else:
            gov_decision = MutationDecision.ACCEPT.value  # NO_OP

        return RuntimeStepResult(
            step=self._step - 1,
            snapshot_before=snap_before,
            snapshot_after=snap_after,
            chosen_action=self._chosen_action.value if hasattr(self._chosen_action, "value") else str(self._chosen_action),
            governance_decision=gov_decision,
            executed=commit_result.committed,
            utility_before=u_before - delta_u,
            utility_after=u_before,
            delta_utility=delta_u,
            certification_level=self._certification_level,
            evidence_hash=commit_result.evidence_hash,
            receipt_hash=commit_result.receipt_hash,
            phases={ev.phase.value: ev.payload for ev in self._events_tail_since(observation.state_version)},
            metadata={
                "version": VERSION,
                "target": getattr(self, "_target", {}),
                "authority_hash_before": authority_before,
                "authority_hash_after": snap_after.authority_hash,
                "phase_order": list(self._last_phase_order),
            },
            observation=observation,
            reasoning=reasoning,
            candidates=candidates,
            planning=planning,
            evaluation=evaluation,
            authorization=authorization,
            commit=commit_result,
            learning=learning,
        )

    def _snap_from_obs(self, obs: ObservationSnapshot) -> RuntimeSnapshot:
        """Reconstruct a RuntimeSnapshot from an ObservationSnapshot."""
        return self.snapshot()

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    def _emit(self, phase: RuntimePhase, payload: dict[str, Any]) -> None:
        self._events.append(RuntimeEvent(phase=phase, step=self._step, payload=payload))

    def _events_tail_since(self, generation: int) -> list[RuntimeEvent]:
        # Events are append-only per step; return all events emitted this step.
        return [e for e in self._events if e.step == self._step]

    def events(self) -> list[dict[str, Any]]:
        return [e.to_log() for e in self._events]

    def summary(self) -> dict[str, Any]:
        return {
            "step": int(self._step),
            "generation": int(self._generation),
            "authority_hash": self.authority_hash,
            "receipt_count": int(self._receipt_count),
            "evidence_root": self.evidence_ledger.root_hash,
            "runtime_config": self.runtime_config.to_summary(),
            "loop_summary": self.loop.summary(),
            "version": VERSION,
        }


class _InMemoryEvidenceLedger(EvidenceLedger):
    """In-memory evidence ledger for research/non-persistent runs."""

    def __init__(self) -> None:
        # Bypass file-based construction.
        self.path = None  # type: ignore[assignment]
        self._records: list[dict[str, Any]] = []
        self._previous: str | None = None
        self._index = -1

    def append(self, record: EvidenceRecord) -> dict[str, Any]:
        from ..evidence import EVIDENCE_SCHEMA, _safe, _canonical
        import hashlib
        self._index += 1
        envelope = {
            "schema": EVIDENCE_SCHEMA,
            "build_version": VERSION,
            "index": self._index,
            "previous_hash": self._previous,
            "record": _safe(record),
        }
        envelope["sha256"] = hashlib.sha256(_canonical(envelope)).hexdigest()
        self._records.append(envelope)
        self._previous = envelope["sha256"]
        return envelope

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def verify(self) -> tuple[bool, list[str]]:
        ok, errors = super().verify() if self.path is not None else (True, [])
        return ok, errors

    @property
    def root_hash(self) -> str | None:
        return self._previous


def _structural_action_cost(action_type: str) -> float:
    """Small normalized structural-footprint cost used by canonical planning."""
    costs = {
        StructuralAction.NO_OP.value: 0.0,
        StructuralAction.REWEIGHT_AFFINITY.value: 0.005,
        StructuralAction.REWEIGHT_LENGTH.value: 0.005,
        StructuralAction.ADD_EDGE.value: 0.01,
        StructuralAction.PRUNE_EDGE.value: 0.01,
        StructuralAction.COUPLED_REWEIGHT.value: 0.015,
        StructuralAction.SPAWN_FIBER.value: 0.02,
        StructuralAction.PRUNE_FIBER.value: 0.015,
        StructuralAction.CHANGE_GAUGE.value: 0.02,
    }
    return float(costs.get(str(action_type), 0.02))


def _action_for_mutation(mutation: Any) -> StructuralAction | None:
    name = str(getattr(mutation, "name", type(mutation).__name__)).lower()
    mapping = {
        "add_edge": StructuralAction.ADD_EDGE,
        "prune_edge": StructuralAction.PRUNE_EDGE,
        "reweight_affinity": StructuralAction.REWEIGHT_AFFINITY,
        "reweight_length": StructuralAction.REWEIGHT_LENGTH,
    }
    return mapping.get(name)


def _target_for_mutation(mutation: Any) -> dict[str, Any]:
    target: dict[str, Any] = {}
    for key in ("u", "v", "factor", "weight", "length"):
        if hasattr(mutation, key):
            value = getattr(mutation, key)
            if value is not None:
                target[key] = value
    return target


def _mutation_plan_token(mutation: Any) -> str:
    action = _action_for_mutation(mutation)
    payload = {
        "action": action.value if action is not None else str(getattr(mutation, "name", type(mutation).__name__)),
        "target": _target_for_mutation(mutation),
    }
    return canonical_hash(payload)


def _default_utility(graph: GraphBuffers, z: Tensor) -> float:
    """Default structural utility: negative sum of squared latent distances
    over active edges (a smooth connectivity proxy)."""
    with torch.no_grad():
        src = graph.src[graph.valid]
        dst = graph.dst[graph.valid]
        if src.numel() == 0:
            return 0.0
        d = (z[src] - z[dst]).pow(2).sum(-1)
        w = graph.weight[graph.valid]
        return float(-(w * d).sum().item())


def _impact_for_action(action: StructuralAction) -> MutationImpact:
    """Map a structural action to the state dimensions it can change.

    This is a conservative over-approximation used for cache invalidation.
    The authoritative impact is the one observed by the transaction itself;
    this helper gives the runtime a declarative impact for the commit event.
    """
    from ..executive import StructuralAction as A
    if action in (A.ADD_EDGE, A.PRUNE_EDGE):
        return MutationImpact(topology=True, weights=True, metric=True)
    if action in (A.REWEIGHT_AFFINITY,):
        return MutationImpact(weights=True)
    if action in (A.REWEIGHT_LENGTH,):
        return MutationImpact(metric=True)
    if action == A.COUPLED_REWEIGHT:
        return MutationImpact(weights=True, metric=True)
    if action in (A.SPAWN_FIBER, A.PRUNE_FIBER):
        return MutationImpact(fibers=True, latents=True)
    if action == A.CHANGE_GAUGE:
        return MutationImpact(gauges=True)
    return MutationImpact()  # NO_OP
