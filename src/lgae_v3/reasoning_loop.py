"""v5.4 runtime bridge from learned structural reasoning to exact authority."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from .benchmark.tasks import StructuralAction
from .action_bridge import action_to_mutation
from .reasoning import StructuralReasoningExecutive, CandidateValue, ReasoningPlan, certify_ranked_candidates
from .types import MutationDecision
from .evidence import EvidenceLedger, EvidenceRecord
from .memory import StructuralExperienceMemory
from .reasoning_graph import ReasoningGraph


@dataclass(slots=True)
class StructuralReasoningStep:
    plan: ReasoningPlan
    proposed: CandidateValue
    executed: bool
    governance_decision: str
    mutation_result: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuralReasoningLoop:
    """One-step concrete reasoning loop.

    Learned code may rank candidates but cannot mutate authoritative state.
    The engine re-evaluates the winning concrete mutation transactionally.
    """
    def __init__(
        self, engine: Any, reasoner: StructuralReasoningExecutive, *, exact_top_k: int = 3,
        memory: StructuralExperienceMemory | None = None,
        evidence_ledger: EvidenceLedger | None = None,
        reasoning_graph: ReasoningGraph | None = None,
    ):
        self.engine = engine
        self.reasoner = reasoner
        self.exact_top_k = int(exact_top_k)
        self.memory = memory
        self.evidence_ledger = evidence_ledger
        self.reasoning_graph = reasoning_graph
        if memory is not None:
            self.reasoner.attach_memory(memory)

    def _append_evidence(self, record_type: str, graph_hash: str, payload: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any] | None:
        if self.evidence_ledger is None:
            return None
        authority_hash = None
        if hasattr(self.engine, "authority_hash"):
            try:
                authority_hash = self.engine.authority_hash()
            except Exception:
                authority_hash = None
        return self.evidence_ledger.append(EvidenceRecord(record_type, graph_hash, payload, authority_hash=authority_hash, reasoning_run_id=run_id))

    def step(self) -> StructuralReasoningStep:
        graph = self.engine.graph
        z = self.engine.fibers().detach().clone()
        reasoning_run = None
        if self.reasoning_graph is not None:
            reasoning_run = self.reasoning_graph.run({"graph": graph, "z": z, "engine": self.engine, "memory": self.memory})
        plan = self.reasoner.plan(graph, z)
        run_id = None if reasoning_run is None else reasoning_run.run_id
        self._append_evidence("reasoning_plan", graph.state_hash(), {
            "candidates_considered": plan.candidates_considered,
            "selected": plan.selected.candidate.key(),
            "ranked": [{"action": v.candidate.action.value, "target": v.candidate.target, "score": v.score, "mean": v.mean_delta_utility, "std": v.std_delta_utility, "risk": v.risk, "ig": v.information_gain, "memory_prior": v.memory_prior} for v in plan.ranked],
            "reasoning_evidence": {} if reasoning_run is None else {k: {"type": e.evidence_type, "payload": e.payload, "confidence": e.confidence, "hash": e.evidence_hash} for k, e in reasoning_run.outputs.items()},
        }, run_id=run_id)
        candidate, precheck = certify_ranked_candidates(
            plan.ranked, graph=graph, z=z, governor=self.engine.governor, top_k=self.exact_top_k,
            seed=int(getattr(self.engine.cfg, "seed", 0)) + int(getattr(self.engine, "step_index", 0)),
        )
        if candidate is None:
            noop = next(v for v in plan.ranked if v.candidate.action == StructuralAction.NO_OP)
            return StructuralReasoningStep(plan, noop, False, "reject", metadata={"reason":"no_candidate_certified"})
        if candidate.candidate.action == StructuralAction.NO_OP:
            return StructuralReasoningStep(plan, candidate, False, "accept", metadata={"reason":"no_op"})
        mutation = action_to_mutation(candidate.candidate.action, graph, z, **candidate.candidate.target)
        if mutation is None:
            return StructuralReasoningStep(plan, candidate, False, "reject", metadata={"reason":"unmappable"})
        before_hash = graph.state_hash()
        result = self.engine.evaluate_and_maybe_commit(mutation)
        executed = result.decision == MutationDecision.ACCEPT
        after_hash = self.engine.graph.state_hash()
        evidence = self._append_evidence("mutation_outcome", before_hash, {
            "action": candidate.candidate.action.value,
            "target": candidate.candidate.target,
            "predicted_delta_utility": candidate.mean_delta_utility,
            "predicted_std": candidate.std_delta_utility,
            "predicted_risk": candidate.risk,
            "predicted_information_gain": candidate.information_gain,
            "memory_prior": candidate.memory_prior,
            "decision": result.decision.value,
            "executed": executed,
            "graph_hash_after": after_hash,
        }, run_id=run_id)
        if self.memory is not None:
            from .reasoning import CounterfactualOutcome
            observed = float(getattr(result, "utility_delta", 0.0) or 0.0)
            outcome = CounterfactualOutcome(candidate.candidate, observed, 0.0, observed, executed, result.decision.value, before_hash, after_hash, {})
            self.memory.record_outcome(graph, z, outcome, evidence_hash=None if evidence is None else evidence.get("sha256"), prediction={"mean": candidate.mean_delta_utility, "std": candidate.std_delta_utility, "risk": candidate.risk, "ig": candidate.information_gain})
        return StructuralReasoningStep(
            plan, candidate, executed, result.decision.value, mutation_result=result,
            metadata={"precheck": None if precheck is None else precheck.decision.value, "reasoning_run_id": run_id, "evidence_hash": None if evidence is None else evidence.get("sha256")},
        )
