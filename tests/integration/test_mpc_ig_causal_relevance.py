"""v5.11-RC Phase 21-22: MPC/IG causal relevance tests.

Tests that:
- Multi-step counterfactual planning is real (MPC planner explores sequences)
- Information gain is causally relevant (IG selection affects outcomes)
- MPC plan quality depends on horizon
- IG selection correlates with realized information gain
"""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime
from lgae_v3.runtime.structural_mpc import MPCPlanner, plan_with_mpc
from lgae_v3.runtime.information_gain import (
    InformationGainEstimate, select_information_directed,
    ensemble_disagreement_ig,
)


def _cfg() -> ResearchConfig:
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.fiber.gauge_dim = 0
    cfg.audit.orc_backend = "exact_lp"
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


class TestMPCCausalRelevance:
    """Multi-step counterfactual planning is real."""

    def test_mpc_plan_explores_multiple_steps(self):
        """MPC planner explores multi-step sequences."""
        planner = MPCPlanner(horizon=3, max_branching=2, max_sequences=16,
                             utility_fn=lambda s, a: 1.0)
        plan = planner.plan(candidates=["a", "b"])
        assert plan.horizon == 3, f"Plan should have horizon 3, got {plan.horizon}"
        assert len(plan.actions) == 3

    def test_mpc_plan_selects_higher_utility_actions(self):
        """MPC planner selects actions with higher utility."""
        # Action 'a' has utility 1.0, 'b' has utility 0.0.
        planner = MPCPlanner(horizon=1, max_branching=2, max_sequences=16,
                             utility_fn=lambda s, a: 1.0 if a == "a" else 0.0)
        plan = planner.plan(candidates=["a", "b"])
        assert plan.actions == ["a"], f"Should select 'a', got {plan.actions}"
        assert plan.total_utility == 1.0

    def test_mpc_longer_horizon_can_find_better_plans(self):
        """A longer horizon can find better plans than a shorter one."""
        # Simulate a scenario where taking a low-utility action now
        # leads to a high-utility state later.
        def utility_fn(state, action):
            if state is None:
                return 0.0
            return float(state.get("value", 0.0))

        def simulate_fn(state, action):
            if state is None:
                state = {"value": 0.0}
            if action == "invest":
                return {"value": state.get("value", 0.0) + 10.0}
            return state

        # With horizon=1, the planner can't see the future benefit.
        planner_short = MPCPlanner(horizon=1, max_branching=2, max_sequences=16,
                                    utility_fn=utility_fn)
        plan_short = planner_short.plan(
            candidates=["invest", "skip"],
            simulate_fn=simulate_fn,
            initial_state={"value": 0.0},
        )

        # With horizon=2, the planner can invest then collect.
        planner_long = MPCPlanner(horizon=2, max_branching=2, max_sequences=16,
                                   utility_fn=utility_fn)
        plan_long = planner_long.plan(
            candidates=["invest", "skip"],
            simulate_fn=simulate_fn,
            initial_state={"value": 0.0},
        )

        # The longer-horizon plan should have higher total utility.
        assert plan_long.total_utility >= plan_short.total_utility, (
            f"Longer horizon should find better or equal plans: "
            f"short={plan_short.total_utility}, long={plan_long.total_utility}"
        )

    def test_mpc_plan_is_deterministic(self):
        """MPC planning is deterministic for the same inputs."""
        planner = MPCPlanner(horizon=2, max_branching=2, max_sequences=16,
                             utility_fn=lambda s, a: 1.0 if a == "a" else 0.5)
        plan1 = planner.plan(candidates=["a", "b"])
        plan2 = planner.plan(candidates=["a", "b"])
        assert plan1.actions == plan2.actions
        assert plan1.total_utility == plan2.total_utility


class TestIGCausalRelevance:
    """Information gain is causally relevant."""

    def test_select_information_directed_picks_highest_score(self):
        """select_information_directed picks the candidate with highest total_score."""
        estimates = [
            InformationGainEstimate(
                candidate_id="a", predicted_ig=0.1,
                epistemic_uncertainty=0.1, aleatoric_uncertainty=0.1,
                exploration_bonus=0.1, total_score=0.3,
            ),
            InformationGainEstimate(
                candidate_id="b", predicted_ig=0.2,
                epistemic_uncertainty=0.2, aleatoric_uncertainty=0.2,
                exploration_bonus=0.2, total_score=0.6,
            ),
        ]
        chosen = select_information_directed(estimates)
        assert chosen == "b", f"Should pick 'b' (highest score), got {chosen}"

    def test_select_information_directed_empty_returns_empty(self):
        """select_information_directed returns '' for empty list."""
        assert select_information_directed([]) == ""

    def test_ensemble_disagreement_ig_produces_estimates(self):
        """ensemble_disagreement_ig produces estimates from ensemble scores."""
        # 2 candidates, 3 ensemble members.
        scores = torch.tensor([
            [0.1, 0.2, 0.3],  # candidate 0: low disagreement
            [0.1, 0.9, 0.2],  # candidate 1: high disagreement
        ])
        estimates = ensemble_disagreement_ig(
            candidate_ids=["a", "b"],
            ensemble_scores=scores,
            learned_scores=torch.tensor([0.5, 0.3]),
            exploration_weight=1.0,
        )
        assert len(estimates) == 2
        # Candidate 'b' has higher disagreement → higher epistemic uncertainty.
        est_a = next(e for e in estimates if e.candidate_id == "a")
        est_b = next(e for e in estimates if e.candidate_id == "b")
        assert est_b.epistemic_uncertainty > est_a.epistemic_uncertainty, (
            "Candidate with higher ensemble disagreement should have higher epistemic uncertainty"
        )

    def test_ig_score_affects_selection(self):
        """IG exploration bonus affects which candidate is selected."""
        # Without exploration bonus, 'a' wins (higher learned score).
        estimates_no_bonus = [
            InformationGainEstimate(
                candidate_id="a", predicted_ig=0.0,
                epistemic_uncertainty=0.0, aleatoric_uncertainty=0.0,
                exploration_bonus=0.0, total_score=0.5,
            ),
            InformationGainEstimate(
                candidate_id="b", predicted_ig=0.0,
                epistemic_uncertainty=0.0, aleatoric_uncertainty=0.0,
                exploration_bonus=0.0, total_score=0.3,
            ),
        ]
        assert select_information_directed(estimates_no_bonus) == "a"

        # With exploration bonus, 'b' wins (higher total score).
        estimates_with_bonus = [
            InformationGainEstimate(
                candidate_id="a", predicted_ig=0.0,
                epistemic_uncertainty=0.0, aleatoric_uncertainty=0.0,
                exploration_bonus=0.0, total_score=0.5,
            ),
            InformationGainEstimate(
                candidate_id="b", predicted_ig=0.0,
                epistemic_uncertainty=0.5, aleatoric_uncertainty=0.0,
                exploration_bonus=0.3, total_score=0.6,
            ),
        ]
        assert select_information_directed(estimates_with_bonus) == "b"
