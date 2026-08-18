"""v5.10 Phase 46: promotion gates tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    PromotionLevel, GateStatus, PromotionReport,
    PromotionGateError, evaluate_promotion, assert_promotion,
    SafetyQualificationReport, ScientificQualificationReport,
    PerformanceQualificationReport, SafetyCheckResult, SafetyCheckStatus,
    ScientificMetric, TierMeasurement, MeasurementStatus, ScaleTier,
)


def _passing_safety_report():
    return SafetyQualificationReport(checks=[
        SafetyCheckResult(name="unauthorized", status=SafetyCheckStatus.PASS, count=0),
    ])

def _failing_safety_report():
    return SafetyQualificationReport(checks=[
        SafetyCheckResult(name="unauthorized", status=SafetyCheckStatus.FAIL, count=1),
    ])

def _passing_scientific_report():
    return ScientificQualificationReport(
        regret_learned=ScientificMetric("rl", {0: 0.05}),
        regret_best_baseline=ScientificMetric("rb", {0: 0.10}),
        sigma_ood=ScientificMetric("so", {0: 0.5}),
        sigma_id=ScientificMetric("si", {0: 0.3}),
        ig_correlation=ScientificMetric("ig", {0: 0.1}, threshold=0.0, direction="higher"),
    )

def _passing_performance_report():
    return PerformanceQualificationReport(measurements=[
        TierMeasurement(tier=ScaleTier.S, n_nodes=100, status=MeasurementStatus.PASS),
        TierMeasurement(tier=ScaleTier.M, n_nodes=1000, status=MeasurementStatus.PASS),
    ])


def test_promotion_level_ordering():
    assert PromotionLevel.EXPERIMENTAL < PromotionLevel.CANDIDATE
    assert PromotionLevel.CANDIDATE < PromotionLevel.QUALIFIED
    assert PromotionLevel.QUALIFIED < PromotionLevel.PRODUCTION


def test_promotion_to_candidate_requires_safety():
    report = evaluate_promotion(
        current_level=PromotionLevel.EXPERIMENTAL,
        target_level=PromotionLevel.CANDIDATE,
        safety_report=_passing_safety_report(),
    )
    assert report.promotion_approved
    assert len(report.gates) == 1
    assert report.gates[0].name == "safety"


def test_promotion_to_candidate_fails_without_safety():
    report = evaluate_promotion(
        current_level=PromotionLevel.EXPERIMENTAL,
        target_level=PromotionLevel.CANDIDATE,
    )
    assert not report.promotion_approved
    assert report.gates[0].name == "safety"
    assert not report.gates[0].ran


def test_promotion_to_candidate_fails_with_failing_safety():
    report = evaluate_promotion(
        current_level=PromotionLevel.EXPERIMENTAL,
        target_level=PromotionLevel.CANDIDATE,
        safety_report=_failing_safety_report(),
    )
    assert not report.promotion_approved


def test_promotion_to_qualified_requires_all_three_gates():
    report = evaluate_promotion(
        current_level=PromotionLevel.CANDIDATE,
        target_level=PromotionLevel.QUALIFIED,
        safety_report=_passing_safety_report(),
        scientific_report=_passing_scientific_report(),
        performance_report=_passing_performance_report(),
    )
    assert report.promotion_approved
    assert len(report.gates) == 3


def test_promotion_to_qualified_fails_without_scientific():
    report = evaluate_promotion(
        current_level=PromotionLevel.CANDIDATE,
        target_level=PromotionLevel.QUALIFIED,
        safety_report=_passing_safety_report(),
        performance_report=_passing_performance_report(),
    )
    assert not report.promotion_approved
    sci = next(g for g in report.gates if g.name == "scientific")
    assert not sci.ran


def test_promotion_to_production_requires_signed_checkpoint():
    report = evaluate_promotion(
        current_level=PromotionLevel.QUALIFIED,
        target_level=PromotionLevel.PRODUCTION,
        safety_report=_passing_safety_report(),
        scientific_report=_passing_scientific_report(),
        performance_report=_passing_performance_report(),
        signed_checkpoint=None,
    )
    assert not report.promotion_approved  # no signed checkpoint


def test_promotion_to_production_passes_with_signed_checkpoint():
    report = evaluate_promotion(
        current_level=PromotionLevel.QUALIFIED,
        target_level=PromotionLevel.PRODUCTION,
        safety_report=_passing_safety_report(),
        scientific_report=_passing_scientific_report(),
        performance_report=_passing_performance_report(),
        signed_checkpoint="abc123signature",
    )
    assert report.promotion_approved


def test_promotion_to_same_or_lower_level_is_noop():
    report = evaluate_promotion(
        current_level=PromotionLevel.QUALIFIED,
        target_level=PromotionLevel.QUALIFIED,
    )
    assert report.promotion_approved


def test_assert_promotion_raises_on_denial():
    report = evaluate_promotion(
        current_level=PromotionLevel.EXPERIMENTAL,
        target_level=PromotionLevel.CANDIDATE,
    )
    with pytest.raises(PromotionGateError):
        assert_promotion(report)


def test_assert_promotion_returns_report_on_success():
    report = evaluate_promotion(
        current_level=PromotionLevel.EXPERIMENTAL,
        target_level=PromotionLevel.CANDIDATE,
        safety_report=_passing_safety_report(),
    )
    result = assert_promotion(report)
    assert result is report


def test_gate_status_not_run():
    g = GateStatus(name="x", ran=False, passed=False)
    assert g.status_str == "not_run"


def test_promotion_report_to_log():
    report = evaluate_promotion(
        current_level=PromotionLevel.EXPERIMENTAL,
        target_level=PromotionLevel.CANDIDATE,
        safety_report=_passing_safety_report(),
    )
    log = report.to_log()
    assert log["current_level"] == "EXPERIMENTAL"
    assert log["target_level"] == "CANDIDATE"
    assert log["promotion_approved"] is True
