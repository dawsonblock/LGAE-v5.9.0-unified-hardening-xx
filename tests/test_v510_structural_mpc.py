"""v5.10 Phase 14: multi-step structural MPC tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import MPCPlan, MPCPlanner, plan_with_mpc


def test_mpc_plan_first_action():
    plan = MPCPlan(actions=["a", "b", "c"], expected_utilities=[1, 2, 3], total_utility=6, horizon=3)
    assert plan.first_action == "a"


def test_mpc_plan_empty():
    plan = MPCPlan(actions=[], expected_utilities=[], total_utility=0, horizon=0)
    assert plan.first_action is None


def test_mpc_plan_to_log():
    plan = MPCPlan(actions=["a", "b"], expected_utilities=[1, 2], total_utility=3, horizon=2)
    log = plan.to_log()
    assert log["actions"] == ["a", "b"]
    assert log["total_utility"] == 3
    assert log["first_action"] == "a"


def test_mpc_planner_finds_best_sequence():
    candidates = ["c1", "c2", "c3"]
    # Utility: c1=0.5, c2=1.0, c3=0.1 per step
    def utility_fn(state, action_id):
        return {"c1": 0.5, "c2": 1.0, "c3": 0.1}.get(action_id, 0.0)

    planner = MPCPlanner(horizon=2, max_branching=3, max_sequences=20, utility_fn=utility_fn)
    plan = planner.plan(candidates=candidates)
    # Best 2-step sequence is c2, c2 with total utility 2.0.
    assert plan.first_action == "c2"
    assert plan.total_utility == pytest.approx(2.0)
    assert len(plan.actions) == 2


def test_mpc_planner_respects_max_sequences():
    candidates = [f"c{i}" for i in range(10)]
    def utility_fn(state, action_id):
        return 1.0

    planner = MPCPlanner(horizon=5, max_branching=10, max_sequences=5, utility_fn=utility_fn)
    plan = planner.plan(candidates=candidates)
    # Should stop after 5 sequences.
    assert plan.horizon <= 5


def test_mpc_planner_empty_candidates():
    def utility_fn(state, action_id):
        return 1.0
    planner = MPCPlanner(horizon=2, utility_fn=utility_fn)
    plan = planner.plan(candidates=[])
    assert plan.first_action is None
    assert plan.total_utility == 0.0  # empty plan has 0 utility


def test_mpc_planner_no_utility_fn_raises():
    planner = MPCPlanner(horizon=2)
    with pytest.raises(ValueError):
        planner.plan(candidates=["c1"])


def test_plan_with_mpc_convenience():
    candidates = ["a", "b"]
    def utility_fn(state, action_id):
        return 1.0 if action_id == "b" else 0.5
    plan = plan_with_mpc(
        candidates=candidates, horizon=2, max_branching=2, max_sequences=10,
        utility_fn=utility_fn,
    )
    assert plan.first_action == "b"


def test_mpc_planner_with_simulation():
    candidates = ["a", "b"]
    state = {"step": 0}
    def simulate_fn(s, action_id):
        return {"step": s["step"] + 1, "last_action": action_id}
    def utility_fn(s, action_id):
        return 1.0 if action_id == "b" else 0.5
    plan = plan_with_mpc(
        candidates=candidates, horizon=2, max_branching=2, max_sequences=10,
        utility_fn=utility_fn, simulate_fn=simulate_fn, initial_state=state,
    )
    assert plan.first_action == "b"
    assert len(plan.actions) == 2


def test_mpc_planner_to_log():
    planner = MPCPlanner(horizon=3, max_branching=4, max_sequences=64)
    log = planner.to_log()
    assert log["horizon"] == 3
    assert log["max_branching"] == 4
    assert log["max_sequences"] == 64
