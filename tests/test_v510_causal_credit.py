"""v5.10 Phase 19: causal structural credit tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    CausalCreditAssignment, CausalCreditAssigner,
    average_causal_effect, credit_concentration,
)


def test_causal_credit_assignment():
    # Counterfactual: do(a1) gives 0.8, do(noop) gives 0.5.
    def cf_fn(actual, counterfactual):
        return 0.8 if counterfactual == "a1" else 0.5
    assigner = CausalCreditAssigner(counterfactual_fn=cf_fn)
    result = assigner.assign_credit(action_id="a1", observational_utility=0.7)
    assert result.interventional_utility == pytest.approx(0.8)
    assert result.counterfactual_utility == pytest.approx(0.5)
    assert result.causal_effect == pytest.approx(0.3)
    assert result.credit == pytest.approx(0.3)


def test_causal_credit_no_counterfactual_fn_raises():
    assigner = CausalCreditAssigner()
    with pytest.raises(ValueError):
        assigner.assign_credit(action_id="a1", observational_utility=0.5)


def test_causal_credit_batch():
    def cf_fn(actual, counterfactual):
        return {"a1": 0.8, "a2": 0.6, "noop": 0.5}.get(counterfactual, 0.5)
    assigner = CausalCreditAssigner(counterfactual_fn=cf_fn)
    results = assigner.assign_credit_batch(
        action_ids=["a1", "a2"],
        observational_utilities=[0.7, 0.6],
    )
    assert len(results) == 2
    assert results[0].credit == pytest.approx(0.3)  # 0.8 - 0.5
    assert results[1].credit == pytest.approx(0.1)  # 0.6 - 0.5


def test_causal_credit_to_log():
    a = CausalCreditAssignment(
        action_id="a1", observational_utility=0.7,
        interventional_utility=0.8, counterfactual_utility=0.5,
        causal_effect=0.3, credit=0.3,
    )
    log = a.to_log()
    assert log["action_id"] == "a1"
    assert log["causal_effect"] == 0.3


def test_average_causal_effect():
    assignments = [
        CausalCreditAssignment("a1", 0.7, 0.8, 0.5, 0.3, 0.3),
        CausalCreditAssignment("a2", 0.6, 0.6, 0.5, 0.1, 0.1),
    ]
    avg = average_causal_effect(assignments)
    assert avg == pytest.approx(0.2)  # (0.3 + 0.1) / 2


def test_average_causal_effect_empty():
    assert average_causal_effect([]) == 0.0


def test_credit_concentration_uniform():
    assignments = [
        CausalCreditAssignment("a1", 0, 0, 0, 0.5, 0.5),
        CausalCreditAssignment("a2", 0, 0, 0, 0.5, 0.5),
    ]
    # Uniform credit -> low concentration.
    assert credit_concentration(assignments) < 0.5


def test_credit_concentration_concentrated():
    assignments = [
        CausalCreditAssignment("a1", 0, 0, 0, 1.0, 1.0),
        CausalCreditAssignment("a2", 0, 0, 0, 0.0, 0.0),
    ]
    # All credit on one action -> high concentration (Gini of [0, 1] = 0.5).
    assert credit_concentration(assignments) >= 0.5


def test_credit_concentration_empty():
    assert credit_concentration([]) == 0.0


def test_credit_concentration_single():
    assignments = [CausalCreditAssignment("a1", 0, 0, 0, 1.0, 1.0)]
    assert credit_concentration(assignments) == 0.0
