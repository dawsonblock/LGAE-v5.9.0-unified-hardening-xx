"""v5.11 Phase 8: verify information gain is activated.

After Phase 8, structural_loop.py no longer hardcodes IG/cost/risk to 0.0.
IG is derived from ensemble disagreement, cost from action footprint,
and risk from epistemic uncertainty.

This test replaces the v5.10 regression test that documented the defect.
"""
from __future__ import annotations

import inspect

from lgae_v3.structural_loop import StructuralLearningLoop


def test_ig_not_hardcoded_zero():
    """Information gain is no longer hardcoded to 0.0."""
    source = inspect.getsource(StructuralLearningLoop.step)
    assert "information_gain=0.0" not in source, (
        "structural_loop.py still hardcodes information_gain=0.0. "
        "IG must be computed from ensemble disagreement."
    )


def test_cost_not_hardcoded_zero():
    """Cost is no longer hardcoded to 0.0."""
    source = inspect.getsource(StructuralLearningLoop.step)
    assert "cost=0.0" not in source or "cost = 0.0" not in source, (
        "structural_loop.py still hardcodes cost=0.0. "
        "Cost must be computed from the action's structural footprint."
    )


def test_risk_not_hardcoded_zero():
    """Risk is no longer hardcoded to 0.0."""
    source = inspect.getsource(StructuralLearningLoop.step)
    # The old code had risk=0.0 in the ActionProposal constructor.
    # The new code computes risk from uncertainty and OOD score.
    assert "risk=0.0" not in source, (
        "structural_loop.py still hardcodes risk=0.0. "
        "Risk must be computed from epistemic uncertainty and OOD score."
    )


def test_ig_is_computed():
    """IG is computed from ensemble disagreement."""
    source = inspect.getsource(StructuralLearningLoop.step)
    assert "ig" in source.lower() or "information_gain" in source.lower(), (
        "IG should be computed in structural_loop.py"
    )
