"""v5.10 Phase 5: adaptive geometric diagnostics cascade tests."""
from __future__ import annotations

import pytest

from lgae_v3.mutations import MutationAuthorityLevel
from lgae_v3.runtime import (
    DiagnosticLevel,
    DiagnosticEscalationPolicy,
    DiagnosticCascade,
    DiagnosticResult,
)


def test_policy_low_signals_select_l0():
    p = DiagnosticEscalationPolicy()
    lvl = p.level_for(risk=0.0, uncertainty=0.0, disagreement=0.0,
                      authority=MutationAuthorityLevel.REVERSIBLE)
    assert lvl == DiagnosticLevel.L0_CHEAP


def test_policy_risk_escalates_through_levels():
    p = DiagnosticEscalationPolicy()
    assert p.level_for(risk=0.3) >= DiagnosticLevel.L1_LOCAL
    assert p.level_for(risk=0.6) >= DiagnosticLevel.L2_STRUCTURAL
    assert p.level_for(risk=0.85) == DiagnosticLevel.L3_EXACT


def test_policy_uncertainty_and_disagreement_escalate():
    p = DiagnosticEscalationPolicy()
    assert p.level_for(uncertainty=0.65) >= DiagnosticLevel.L2_STRUCTURAL
    assert p.level_for(disagreement=0.9) == DiagnosticLevel.L3_EXACT


def test_policy_authority_drives_minimum_level():
    p = DiagnosticEscalationPolicy()
    # Even with zero risk, a structural mutation requires at least L1.
    assert p.level_for(risk=0.0, authority=MutationAuthorityLevel.STRUCTURAL) >= DiagnosticLevel.L1_LOCAL
    # Irreversible requires L3 regardless of risk.
    assert p.level_for(risk=0.0, authority=MutationAuthorityLevel.IRREVERSIBLE) == DiagnosticLevel.L3_EXACT


def test_policy_rejects_out_of_range_signals():
    p = DiagnosticEscalationPolicy()
    with pytest.raises(ValueError):
        p.level_for(risk=1.5)
    with pytest.raises(ValueError):
        p.level_for(uncertainty=-0.1)


def test_cascade_runs_levels_up_to_selected():
    calls = []

    def mk(name):
        def fn():
            calls.append(name)
            return {f"{name}_metric": 1.0}
        return fn

    cascade = DiagnosticCascade({
        DiagnosticLevel.L0_CHEAP: mk("l0"),
        DiagnosticLevel.L1_LOCAL: mk("l1"),
        DiagnosticLevel.L2_STRUCTURAL: mk("l2"),
        DiagnosticLevel.L3_EXACT: mk("l3"),
    })
    res = cascade.evaluate(risk=0.0, authority=MutationAuthorityLevel.REVERSIBLE)
    assert res.level == DiagnosticLevel.L0_CHEAP
    assert calls == ["l0"]
    assert "l0_metric" in res.metrics
    assert not res.escalated


def test_cascade_escalates_and_accumulates_metrics():
    calls = []
    def mk(name):
        def fn():
            calls.append(name)
            return {f"{name}_metric": float(len(calls))}
        return fn

    cascade = DiagnosticCascade({
        DiagnosticLevel.L0_CHEAP: mk("l0"),
        DiagnosticLevel.L1_LOCAL: mk("l1"),
        DiagnosticLevel.L2_STRUCTURAL: mk("l2"),
        DiagnosticLevel.L3_EXACT: mk("l3"),
    })
    res = cascade.evaluate(risk=0.6, authority=MutationAuthorityLevel.STRUCTURAL)
    assert res.level >= DiagnosticLevel.L2_STRUCTURAL
    # All levels up to the selected one must have run (accumulation).
    assert calls == ["l0", "l1", "l2"] if res.level == DiagnosticLevel.L2_STRUCTURAL else calls == ["l0", "l1", "l2", "l3"]
    assert "l0_metric" in res.metrics and "l2_metric" in res.metrics
    assert res.escalated


def test_cascade_force_level_overrides_policy():
    cascade = DiagnosticCascade({DiagnosticLevel.L0_CHEAP: lambda: {"x": 1}})
    res = cascade.evaluate(force_level=DiagnosticLevel.L3_EXACT)
    assert res.level == DiagnosticLevel.L3_EXACT
    # Missing evaluators for L1..L3 are skipped without error.
    assert res.metrics == {"x": 1}


def test_diagnostic_result_is_exact_only_at_l3():
    r0 = DiagnosticResult(level=DiagnosticLevel.L0_CHEAP)
    r3 = DiagnosticResult(level=DiagnosticLevel.L3_EXACT)
    assert not r0.is_exact
    assert r3.is_exact
