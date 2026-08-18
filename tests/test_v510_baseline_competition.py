"""v5.10 Phase 23: baseline competition framework tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.executive import StructuralAction
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.runtime import (
    BaselineCompetition, CompetitionReport, PolicyResult,
    select_by_scores, learned_policy_from_scores,
)


def _cands():
    return [
        ConcreteAction(StructuralAction.NO_OP, channel="no_op"),
        ConcreteAction(StructuralAction.ADD_EDGE, {"u": 0, "v": 2}, channel="learned"),
        ConcreteAction(StructuralAction.ADD_EDGE, {"u": 1, "v": 3}, channel="fosr"),
        ConcreteAction(StructuralAction.PRUNE_EDGE, {"u": 2, "v": 3}, channel="er"),
    ]


def test_select_by_scores_picks_highest():
    assert select_by_scores([0.1, 0.9, 0.3, 0.7]) == 1


def test_select_by_scores_deterministic_tiebreak():
    # Ties broken by lowest index.
    assert select_by_scores([0.5, 0.5, 0.5]) == 0


def test_competition_oracle_has_zero_regret():
    comp = BaselineCompetition()
    deltas = [0.0, 0.08, 0.05, 0.02]
    res = comp.evaluate_state(_cands(), deltas)
    assert res["oracle"].regret == 0.0
    assert res["oracle"].chosen_delta == 0.08


def test_competition_no_op_regret_equals_oracle_delta():
    comp = BaselineCompetition()
    deltas = [0.0, 0.08, 0.05, 0.02]
    res = comp.evaluate_state(_cands(), deltas)
    assert res["no_op"].regret == pytest.approx(0.08, abs=1e-9)
    assert res["no_op"].chosen_delta == 0.0


def test_competition_learned_policy_regret():
    comp = BaselineCompetition()
    deltas = [0.0, 0.08, 0.05, 0.02]
    # Learned scores pick the FoSR candidate (index 2), not the oracle (index 1).
    learned = [0.0, 0.06, 0.07, 0.01]
    res = comp.evaluate_state(_cands(), deltas, learned_scores=learned)
    assert res["learned"].chosen_index == 2
    assert res["learned"].regret == pytest.approx(0.08 - 0.05, abs=1e-9)


def test_competition_aggregates_regret_distribution():
    comp = BaselineCompetition(catastrophic_threshold=0.05)
    # State 1: oracle delta 0.08 (index 1); learned picks index 2 (delta 0.05) -> regret 0.03
    comp.evaluate_state(_cands(), [0.0, 0.08, 0.05, 0.02], learned_scores=[0.0, 0.06, 0.07, 0.01])
    # State 2: oracle delta 0.10 (index 1); learned picks index 2 (delta 0.04) -> regret 0.06
    comp.evaluate_state(_cands(), [0.0, 0.10, 0.04, 0.01], learned_scores=[0.0, 0.03, 0.09, 0.01])
    summary = comp.summary()
    assert "learned" in summary and "oracle" in summary and "no_op" in summary
    assert summary["oracle"]["mean_regret"] == 0.0
    # Learned picked index 2 both times: regrets 0.03 and 0.06 -> mean 0.045.
    assert summary["learned"]["mean_regret"] == pytest.approx(0.045, abs=1e-9)
    assert summary["learned"]["count"] == 2
    # Catastrophic frequency: one of two regrets (0.06) >= 0.05 -> 0.5.
    assert summary["learned"]["catastrophic_regret_frequency"] == 0.5


def test_competition_extra_policies():
    comp = BaselineCompetition()
    deltas = [0.0, 0.08, 0.05, 0.02]
    # A "random" baseline that always picks index 3.
    extra = {"random": lambda c, d: 3}
    res = comp.evaluate_state(_cands(), deltas, extra_policies=extra)
    assert res["random"].chosen_index == 3
    assert res["random"].regret == pytest.approx(0.08 - 0.02, abs=1e-9)


def test_competition_rejects_length_mismatch():
    comp = BaselineCompetition()
    with pytest.raises(ValueError):
        comp.evaluate_state(_cands(), [0.0, 0.08])
    with pytest.raises(ValueError):
        comp.evaluate_state(_cands(), [0.0, 0.08, 0.05, 0.02], learned_scores=[0.0, 0.08])


def test_learned_policy_from_scores_helper():
    fn = learned_policy_from_scores([0.1, 0.9, 0.3])
    assert fn([], []) == 1
