"""v5.10 Phase 27: adversarial structural testing tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    AdversarialOutcome, AdversarialTestResult, AdversarialTestReport,
    run_adversarial_tests,
)


def test_adversarial_outcome_enum():
    assert AdversarialOutcome.ACCEPTED != AdversarialOutcome.REJECTED
    assert AdversarialOutcome.CRASHED != AdversarialOutcome.ACCEPTED


def test_adversarial_test_result_passed():
    r = AdversarialTestResult(
        name="x", outcome=AdversarialOutcome.ACCEPTED,
        expected_outcome=AdversarialOutcome.ACCEPTED,
    )
    assert r.passed


def test_adversarial_test_result_failed():
    r = AdversarialTestResult(
        name="x", outcome=AdversarialOutcome.CRASHED,
        expected_outcome=AdversarialOutcome.ACCEPTED,
    )
    assert not r.passed


def test_run_adversarial_tests_returns_report():
    report = run_adversarial_tests()
    assert isinstance(report, AdversarialTestReport)
    assert len(report.results) >= 10


def test_run_adversarial_tests_no_crashes():
    report = run_adversarial_tests()
    # No test should crash unexpectedly.
    assert report.crashed_count == 0, f"crashes: {[r.name for r in report.results if r.outcome == AdversarialOutcome.CRASHED]}"


def test_self_loop_is_rejected():
    report = run_adversarial_tests()
    self_loop = next(r for r in report.results if r.name == "self_loop_rejected")
    assert self_loop.outcome == AdversarialOutcome.REJECTED
    assert self_loop.passed


def test_edge_out_of_range_is_rejected():
    report = run_adversarial_tests()
    r = next(r for r in report.results if r.name == "edge_out_of_range_rejected")
    assert r.outcome == AdversarialOutcome.REJECTED
    assert r.passed


def test_single_node_is_accepted():
    report = run_adversarial_tests()
    r = next(r for r in report.results if r.name == "single_node_no_edges")
    assert r.outcome == AdversarialOutcome.ACCEPTED
    assert r.passed


def test_complete_graph_is_accepted():
    report = run_adversarial_tests()
    r = next(r for r in report.results if r.name == "complete_k10_accepted")
    assert r.outcome == AdversarialOutcome.ACCEPTED
    assert r.passed


def test_disconnected_graph_is_accepted():
    report = run_adversarial_tests()
    r = next(r for r in report.results if r.name == "disconnected_graph_accepted")
    assert r.outcome == AdversarialOutcome.ACCEPTED
    assert r.passed


def test_report_to_log_structure():
    report = run_adversarial_tests()
    log = report.to_log()
    assert "all_passed" in log
    assert "total" in log
    assert "crashed_count" in log
    assert "results" in log
    assert len(log["results"]) >= 10


def test_report_all_passed_property():
    report = AdversarialTestReport()
    report.add(AdversarialTestResult("a", AdversarialOutcome.ACCEPTED, AdversarialOutcome.ACCEPTED))
    report.add(AdversarialTestResult("b", AdversarialOutcome.REJECTED, AdversarialOutcome.REJECTED))
    assert report.all_passed

    report.add(AdversarialTestResult("c", AdversarialOutcome.CRASHED, AdversarialOutcome.ACCEPTED))
    assert not report.all_passed
