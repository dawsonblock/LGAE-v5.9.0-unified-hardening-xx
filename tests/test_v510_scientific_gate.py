"""v5.10 Phase 47: scientific qualification gate tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    ScientificMetric, ScientificQualificationReport,
    ScientificGateError, assert_scientific_gate,
)


def test_scientific_metric_all_seeds_pass_lower():
    m = ScientificMetric("regret", {0: 0.1, 1: 0.08, 2: 0.12}, threshold=0.2, direction="lower")
    assert m.all_seeds_pass
    assert m.mean_passes
    assert m.n_seeds == 3


def test_scientific_metric_not_all_seeds_pass():
    m = ScientificMetric("regret", {0: 0.1, 1: 0.3, 2: 0.12}, threshold=0.2, direction="lower")
    assert not m.all_seeds_pass  # seed 1 exceeds threshold
    assert m.mean_passes  # but mean (0.173) is below threshold


def test_regret_gate_passes_when_learned_below_baseline_on_all_seeds():
    report = ScientificQualificationReport(
        regret_learned=ScientificMetric("regret_learned", {0: 0.05, 1: 0.04, 2: 0.06}),
        regret_best_baseline=ScientificMetric("regret_baseline", {0: 0.10, 1: 0.08, 2: 0.12}),
    )
    assert report.regret_gate_passed


def test_regret_gate_fails_when_learned_above_baseline_on_one_seed():
    report = ScientificQualificationReport(
        regret_learned=ScientificMetric("regret_learned", {0: 0.05, 1: 0.15, 2: 0.06}),
        regret_best_baseline=ScientificMetric("regret_baseline", {0: 0.10, 1: 0.08, 2: 0.12}),
    )
    assert not report.regret_gate_passed  # seed 1: 0.15 > 0.08


def test_regret_gate_fails_on_average_only_improvement():
    # Mean learned (0.083) < mean baseline (0.10), but seed 1 is worse.
    report = ScientificQualificationReport(
        regret_learned=ScientificMetric("regret_learned", {0: 0.02, 1: 0.15, 2: 0.08}),
        regret_best_baseline=ScientificMetric("regret_baseline", {0: 0.10, 1: 0.08, 2: 0.12}),
    )
    assert not report.regret_gate_passed


def test_ood_gate_passes_when_sigma_ood_above_id_on_all_seeds():
    report = ScientificQualificationReport(
        sigma_ood=ScientificMetric("sigma_ood", {0: 0.5, 1: 0.6}),
        sigma_id=ScientificMetric("sigma_id", {0: 0.3, 1: 0.4}),
    )
    assert report.ood_gate_passed


def test_ood_gate_fails_when_one_seed_has_id_above_ood():
    report = ScientificQualificationReport(
        sigma_ood=ScientificMetric("sigma_ood", {0: 0.5, 1: 0.3}),
        sigma_id=ScientificMetric("sigma_id", {0: 0.3, 1: 0.4}),
    )
    assert not report.ood_gate_passed  # seed 1: 0.3 < 0.4


def test_ig_gate_passes_when_correlation_positive_on_all_seeds():
    report = ScientificQualificationReport(
        ig_correlation=ScientificMetric("ig_rho", {0: 0.1, 1: 0.2, 2: 0.05}, threshold=0.0, direction="higher"),
    )
    assert report.ig_gate_passed


def test_ig_gate_fails_when_one_seed_has_negative_correlation():
    report = ScientificQualificationReport(
        ig_correlation=ScientificMetric("ig_rho", {0: 0.1, 1: -0.05, 2: 0.05}, threshold=0.0, direction="higher"),
    )
    assert not report.ig_gate_passed


def test_all_gates_passed_only_when_all_three_pass():
    report = ScientificQualificationReport(
        regret_learned=ScientificMetric("rl", {0: 0.05}),
        regret_best_baseline=ScientificMetric("rb", {0: 0.10}),
        sigma_ood=ScientificMetric("so", {0: 0.5}),
        sigma_id=ScientificMetric("si", {0: 0.3}),
        ig_correlation=ScientificMetric("ig", {0: 0.1}, threshold=0.0, direction="higher"),
    )
    assert report.all_gates_passed


def test_assert_scientific_gate_raises_on_failure():
    report = ScientificQualificationReport(
        regret_learned=ScientificMetric("rl", {0: 0.15}),
        regret_best_baseline=ScientificMetric("rb", {0: 0.10}),
    )
    with pytest.raises(ScientificGateError):
        assert_scientific_gate(report)


def test_assert_scientific_gate_passes_when_all_pass():
    report = ScientificQualificationReport(
        regret_learned=ScientificMetric("rl", {0: 0.05}),
        regret_best_baseline=ScientificMetric("rb", {0: 0.10}),
        sigma_ood=ScientificMetric("so", {0: 0.5}),
        sigma_id=ScientificMetric("si", {0: 0.3}),
        ig_correlation=ScientificMetric("ig", {0: 0.1}, threshold=0.0, direction="higher"),
    )
    assert_scientific_gate(report)  # does not raise


def test_report_to_log_structure():
    report = ScientificQualificationReport(
        regret_learned=ScientificMetric("rl", {0: 0.05}),
        regret_best_baseline=ScientificMetric("rb", {0: 0.10}),
    )
    log = report.to_log()
    assert "all_gates_passed" in log
    assert "regret_gate" in log
    assert log["regret_gate"]["learned"]["values_by_seed"] == {0: 0.05}
