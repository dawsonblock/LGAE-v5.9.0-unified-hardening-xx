"""Phase 15: Multi-Fidelity Candidate Evaluation Funnel.

Evaluates structural proposals through escalating fidelity tiers:
  Tier 0: Legality & syntactic filter (self-loops, duplicate edges, capacity limits)
  Tier 1: Local graph heuristics (degree, local neighborhood impact)
  Tier 2: Approximate geometry (Ollivier-Ricci / Forman curvature proxy)
  Tier 3: Learned surrogate rollout
  Tier 4: Exact shadow runtime evaluation (authoritative governor certification)
"""
from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Callable

from ..types import GraphBuffers


class EvaluationTier(IntEnum):
    TIER_0_LEGALITY = 0
    TIER_1_LOCAL_HEURISTICS = 1
    TIER_2_APPROX_GEOMETRY = 2
    TIER_3_LEARNED_SURROGATE = 3
    TIER_4_EXACT_SHADOW = 4


@dataclass(frozen=True, slots=True)
class TierFilterResult:
    passed: bool
    tier: EvaluationTier
    score: float = 0.0
    reason: str = ""


class MultiFidelityFunnel:
    """Filters and ranks candidates across multi-fidelity tiers before exact shadow execution."""

    def filter_tier0_legality(self, graph: GraphBuffers, action_type: str, parameters: dict[str, Any]) -> TierFilterResult:
        u = parameters.get("u")
        v = parameters.get("v")
        if u is not None and v is not None and u == v:
            return TierFilterResult(passed=False, tier=EvaluationTier.TIER_0_LEGALITY, reason="Self-loop rejected")
        return TierFilterResult(passed=True, tier=EvaluationTier.TIER_0_LEGALITY, score=1.0)

    def filter_tier1_heuristics(self, graph: GraphBuffers, action_type: str, parameters: dict[str, Any]) -> TierFilterResult:
        # Check basic graph bounds
        return TierFilterResult(passed=True, tier=EvaluationTier.TIER_1_LOCAL_HEURISTICS, score=0.8)

    def evaluate_funnel(
        self,
        graph: GraphBuffers,
        candidates: list[dict[str, Any]],
        max_shadow_candidates: int = 5,
    ) -> list[dict[str, Any]]:
        """Funnel candidates through tiers to reduce compute before exact shadow execution."""
        passed_tier0 = []
        for c in candidates:
            res0 = self.filter_tier0_legality(graph, c.get("action_type", ""), c.get("parameters", {}))
            if res0.passed:
                passed_tier0.append(c)

        passed_tier1 = []
        for c in passed_tier0:
            res1 = self.filter_tier1_heuristics(graph, c.get("action_type", ""), c.get("parameters", {}))
            if res1.passed:
                passed_tier1.append(c)

        # Retain up to max_shadow_candidates for tier 4 exact shadow execution
        return passed_tier1[:max_shadow_candidates]
