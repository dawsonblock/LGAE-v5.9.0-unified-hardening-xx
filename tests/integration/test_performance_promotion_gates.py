"""v5.11-RC Phase 17-18: Performance measurement + promotion gates tests.

Tests that:
- measure_tier() returns INVALID when no benchmark functions ran
- Promotion requires PASS, not just MEASURED
- Empty benchmark execution returns INVALID
"""
from __future__ import annotations

import pytest

from lgae_v3.runtime.performance_qualification import (
    measure_tier, run_performance_qualification,
    ScaleTier, MeasurementStatus, TierMeasurement,
    PerformanceQualificationReport,
)
from lgae_v3.runtime.promotion import (
    evaluate_promotion, PromotionLevel,
)
from lgae_v3.runtime.qualification import SafetyQualificationReport, SafetyCheckResult, SafetyCheckStatus
from lgae_v3.runtime.scientific_qualification import ScientificQualificationReport, ScientificMetric


def _passing_safety_report():
    report = SafetyQualificationReport()
    report.add(SafetyCheckResult(name="authority", status=SafetyCheckStatus.PASS))
    report.add(SafetyCheckResult(name="atomicity", status=SafetyCheckStatus.PASS))
    report.add(SafetyCheckResult(name="crash_recovery", status=SafetyCheckStatus.PASS))
    report.add(SafetyCheckResult(name="determinism", status=SafetyCheckStatus.PASS))
    report.add(SafetyCheckResult(name="wal_integrity", status=SafetyCheckStatus.PASS))
    return report


def _passing_scientific_report():
    return ScientificQualificationReport(
        regret_learned=ScientificMetric("rl", {0: 0.1}, 0.2, "lower"),
        regret_best_baseline=ScientificMetric("rb", {0: 0.2}, 0.15, "lower"),
        sigma_ood=ScientificMetric("so", {0: 0.8}, 0.0, "higher"),
        sigma_id=ScientificMetric("si", {0: 0.3}, 0.0, "higher"),
        ig_correlation=ScientificMetric("ig", {0: 0.5}, 0.0, "higher"),
    )


class TestPerformanceMeasurementSemantics:
    """Phase 17: measure_tier() returns INVALID when nothing ran."""

    def test_no_functions_returns_invalid(self):
        """measure_tier with no functions returns INVALID, not MEASURED."""
        m = measure_tier(ScaleTier.S, n_nodes=50)
        assert m.status == MeasurementStatus.INVALID

    def test_with_functions_returns_measured(self):
        """measure_tier with functions returns MEASURED or PASS/FAIL."""
        def proposal_fn(graph):
            return 5
        m = measure_tier(ScaleTier.S, n_nodes=50, proposal_fn=proposal_fn)
        assert m.status in (MeasurementStatus.MEASURED, MeasurementStatus.PASS, MeasurementStatus.FAIL)

    def test_skip_returns_skipped(self):
        """measure_tier with skip=True returns SKIPPED."""
        m = measure_tier(ScaleTier.S, n_nodes=50, skip=True)
        assert m.status == MeasurementStatus.SKIPPED


class TestPerformancePromotionGates:
    """Phase 18: Promotion requires PASS, not just MEASURED."""

    def test_promotion_fails_with_measured_but_not_pass(self):
        """Promotion fails when performance is MEASURED but not PASS."""
        perf = PerformanceQualificationReport(measurements=[
            TierMeasurement(tier=ScaleTier.S, n_nodes=100, status=MeasurementStatus.MEASURED),
        ])
        report = evaluate_promotion(
            current_level=PromotionLevel.CANDIDATE,
            target_level=PromotionLevel.QUALIFIED,
            safety_report=_passing_safety_report(),
            scientific_report=_passing_scientific_report(),
            performance_report=perf,
        )
        assert not report.promotion_approved, (
            "Promotion should fail when performance is MEASURED but not PASS"
        )

    def test_promotion_passes_with_pass_status(self):
        """Promotion passes when performance is PASS."""
        perf = PerformanceQualificationReport(measurements=[
            TierMeasurement(tier=ScaleTier.S, n_nodes=100, status=MeasurementStatus.PASS),
            TierMeasurement(tier=ScaleTier.M, n_nodes=1000, status=MeasurementStatus.PASS),
        ])
        report = evaluate_promotion(
            current_level=PromotionLevel.CANDIDATE,
            target_level=PromotionLevel.QUALIFIED,
            safety_report=_passing_safety_report(),
            scientific_report=_passing_scientific_report(),
            performance_report=perf,
        )
        assert report.promotion_approved, (
            f"Promotion should pass with PASS status. Gates: {[(g.name, g.passed) for g in report.gates]}"
        )

    def test_promotion_to_production_requires_m_pass(self):
        """Promotion to PRODUCTION requires both S and M to PASS."""
        perf = PerformanceQualificationReport(measurements=[
            TierMeasurement(tier=ScaleTier.S, n_nodes=100, status=MeasurementStatus.PASS),
            # M is only MEASURED, not PASS.
            TierMeasurement(tier=ScaleTier.M, n_nodes=1000, status=MeasurementStatus.MEASURED),
        ])
        report = evaluate_promotion(
            current_level=PromotionLevel.QUALIFIED,
            target_level=PromotionLevel.PRODUCTION,
            safety_report=_passing_safety_report(),
            scientific_report=_passing_scientific_report(),
            performance_report=perf,
            signed_checkpoint="sha256:abc",
        )
        assert not report.promotion_approved, (
            "Promotion to PRODUCTION should fail when M is not PASS"
        )

    def test_promotion_to_production_passes_with_s_and_m_pass(self):
        """Promotion to PRODUCTION passes when S and M both PASS."""
        perf = PerformanceQualificationReport(measurements=[
            TierMeasurement(tier=ScaleTier.S, n_nodes=100, status=MeasurementStatus.PASS),
            TierMeasurement(tier=ScaleTier.M, n_nodes=1000, status=MeasurementStatus.PASS),
        ])
        report = evaluate_promotion(
            current_level=PromotionLevel.QUALIFIED,
            target_level=PromotionLevel.PRODUCTION,
            safety_report=_passing_safety_report(),
            scientific_report=_passing_scientific_report(),
            performance_report=perf,
            signed_checkpoint="sha256:abc",
        )
        assert report.promotion_approved

    def test_empty_benchmark_returns_invalid(self):
        """Running performance qualification with no functions returns INVALID."""
        report = run_performance_qualification(
            proposal_fn=None,
            diagnostic_fn=None,
            commit_fn=None,
            tiers=[ScaleTier.S],
        )
        for m in report.measurements:
            assert m.status == MeasurementStatus.INVALID, (
                f"Tier {m.tier} should be INVALID, got {m.status}"
            )
