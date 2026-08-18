"""v5.10 Phase 18: structural credit assignment tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    CreditAssignment, direct_credit, feature_based_credit,
    temporal_credit, baseline_credit,
)


def test_direct_credit_positive():
    c = direct_credit(action_id="a1", baseline_utility=0.5, realized_utility=0.8)
    assert c.credit == pytest.approx(0.3)
    assert c.advantage == pytest.approx(0.3)
    assert c.method == "direct"


def test_direct_credit_negative():
    c = direct_credit(action_id="a1", baseline_utility=0.5, realized_utility=0.3)
    assert c.credit == pytest.approx(-0.2)


def test_feature_based_credit_distributes_by_weight():
    credits = feature_based_credit(
        action_ids=["a1", "a2", "a3"],
        feature_weights=[0.5, 0.3, 0.2],
        baseline_utility=0.5, realized_utility=1.0,
    )
    assert len(credits) == 3
    assert credits[0].credit == pytest.approx(0.25)  # 0.5 * 0.5
    assert credits[1].credit == pytest.approx(0.15)  # 0.3 * 0.5
    assert credits[2].credit == pytest.approx(0.10)  # 0.2 * 0.5


def test_feature_based_credit_equal_when_zero_weights():
    credits = feature_based_credit(
        action_ids=["a1", "a2"],
        feature_weights=[0.0, 0.0],
        baseline_utility=0.0, realized_utility=1.0,
    )
    assert credits[0].credit == pytest.approx(0.5)
    assert credits[1].credit == pytest.approx(0.5)


def test_temporal_credit_discounts_distant_actions():
    credits = temporal_credit(
        action_ids=["a1", "a2", "a3"],
        utilities=[0.5, 0.6, 0.9],
        baseline_utility=0.5, discount=0.9,
    )
    assert len(credits) == 3
    # a3 (last) gets the most credit, a1 (first) gets the least.
    assert credits[2].credit > credits[1].credit > credits[0].credit


def test_temporal_credit_empty():
    credits = temporal_credit(action_ids=[], utilities=[], baseline_utility=0.5)
    assert credits == []


def test_baseline_credit():
    c = baseline_credit(action_id="a1", noop_utility=0.5, action_utility=0.8)
    assert c.credit == pytest.approx(0.3)
    assert c.method == "baseline"
    assert c.baseline_utility == 0.5


def test_baseline_credit_negative():
    c = baseline_credit(action_id="a1", noop_utility=0.8, action_utility=0.5)
    assert c.credit == pytest.approx(-0.3)


def test_credit_assignment_advantage():
    c = CreditAssignment(
        action_id="a1", credit=0.3, baseline_utility=0.5,
        realized_utility=0.8, method="direct",
    )
    assert c.advantage == pytest.approx(0.3)


def test_credit_assignment_to_log():
    c = direct_credit(action_id="a1", baseline_utility=0.5, realized_utility=0.8)
    log = c.to_log()
    assert log["action_id"] == "a1"
    assert log["credit"] == pytest.approx(0.3)
    assert log["method"] == "direct"
