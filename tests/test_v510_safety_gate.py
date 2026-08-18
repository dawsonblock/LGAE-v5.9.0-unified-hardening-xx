"""v5.10 Phase 48: safety qualification gate tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    SafetyCheckStatus, SafetyCheckResult, SafetyQualificationReport,
    SafetyGateError, run_safety_qualification, assert_safety_gate,
)


def _check_zero():
    return (0, {"events": []}, "no violations observed")


def _check_two():
    return (2, {"events": ["e1", "e2"]}, "2 violations observed")


def test_safety_check_result_pass():
    r = SafetyCheckResult(name="x", status=SafetyCheckStatus.PASS, count=0, threshold=0)
    assert r.passed
    assert r.to_log()["status"] == "pass"


def test_safety_check_result_fail():
    r = SafetyCheckResult(name="x", status=SafetyCheckStatus.FAIL, count=1, threshold=0)
    assert not r.passed


def test_run_safety_qualification_all_pass():
    report = run_safety_qualification(
        unauthorized_mutation_count=_check_zero,
        receipt_chain_failures=_check_zero,
        silent_stale_read_count=_check_zero,
        nondeterministic_output_count=_check_zero,
        invariant_breaking_commit_count=_check_zero,
        unrecoverable_crash_count=_check_zero,
    )
    assert report.all_passed
    assert len(report.failed_checks) == 0
    assert len(report.checks) == 6


def test_run_safety_qualification_with_failure():
    report = run_safety_qualification(
        unauthorized_mutation_count=_check_two,  # 2 violations
        receipt_chain_failures=_check_zero,
        silent_stale_read_count=_check_zero,
        nondeterministic_output_count=_check_zero,
        invariant_breaking_commit_count=_check_zero,
        unrecoverable_crash_count=_check_zero,
    )
    assert not report.all_passed
    assert len(report.failed_checks) == 1
    assert report.failed_checks[0].name == "unauthorized_mutations"
    assert report.failed_checks[0].count == 2


def test_run_safety_qualification_skip_for_none_checks():
    report = run_safety_qualification(
        unauthorized_mutation_count=_check_zero,
    )
    # all_passed only checks non-skipped checks; with one passing check it's True.
    assert report.all_passed
    skipped = [c for c in report.checks if c.status == SafetyCheckStatus.SKIP]
    assert len(skipped) == 5


def test_assert_safety_gate_raises_on_failure():
    report = run_safety_qualification(
        unauthorized_mutation_count=_check_two,
        receipt_chain_failures=_check_zero,
        silent_stale_read_count=_check_zero,
        nondeterministic_output_count=_check_zero,
        invariant_breaking_commit_count=_check_zero,
        unrecoverable_crash_count=_check_zero,
    )
    with pytest.raises(SafetyGateError):
        assert_safety_gate(report)


def test_assert_safety_gate_passes_when_all_pass():
    report = run_safety_qualification(
        unauthorized_mutation_count=_check_zero,
        receipt_chain_failures=_check_zero,
        silent_stale_read_count=_check_zero,
        nondeterministic_output_count=_check_zero,
        invariant_breaking_commit_count=_check_zero,
        unrecoverable_crash_count=_check_zero,
    )
    assert_safety_gate(report)  # does not raise


def test_check_raising_is_treated_as_failure():
    def bad_check():
        raise RuntimeError("sensor broken")
    report = run_safety_qualification(unauthorized_mutation_count=bad_check)
    r = report.checks[0]
    assert r.status == SafetyCheckStatus.FAIL
    assert "check raised" in r.message


def test_report_to_log_structure():
    report = run_safety_qualification(
        unauthorized_mutation_count=_check_zero,
        receipt_chain_failures=_check_zero,
    )
    log = report.to_log()
    assert "all_passed" in log
    assert "check_count" in log
    assert "checks" in log
    assert len(log["checks"]) == 6
