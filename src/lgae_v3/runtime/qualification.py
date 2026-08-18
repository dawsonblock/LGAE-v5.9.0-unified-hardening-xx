"""Safety qualification gate (Phase 48).

The safety gate is independent of ML quality. It must verify, with evidence:

  0 unauthorized mutations
  0 receipt-chain failures
  0 silent stale reads
  0 nondeterministic qualification outputs
  0 invariant-breaking committed mutations
  0 unrecoverable transaction crashes

Each check produces a structured ``SafetyCheckResult`` with pass/fail and
evidence. The aggregate ``SafetyQualificationReport`` is the canonical
artifact for the safety gate. Production release requires all checks to pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable


class SafetyCheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # check not applicable / not configured


@dataclass(frozen=True, slots=True)
class SafetyCheckResult:
    """Outcome of one safety check."""
    name: str
    status: SafetyCheckStatus
    count: int = 0
    threshold: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.status == SafetyCheckStatus.PASS

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "count": int(self.count),
            "threshold": int(self.threshold),
            "passed": self.passed,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class SafetyQualificationReport:
    """Aggregate safety qualification report."""
    checks: list[SafetyCheckResult] = field(default_factory=list)

    def add(self, result: SafetyCheckResult) -> None:
        self.checks.append(result)

    @property
    def all_passed(self) -> bool:
        """True only if every non-skipped check passed."""
        return all(c.passed for c in self.checks if c.status != SafetyCheckStatus.SKIP)

    @property
    def failed_checks(self) -> list[SafetyCheckResult]:
        return [c for c in self.checks if c.status == SafetyCheckStatus.FAIL]

    def to_log(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "check_count": len(self.checks),
            "failed_count": len(self.failed_checks),
            "checks": [c.to_log() for c in self.checks],
        }


# A safety check function returns (count, evidence_dict, message).
SafetyCheckFn = Callable[[], tuple[int, dict[str, Any], str]]


def _zero_violation_check(name: str, fn: SafetyCheckFn) -> SafetyCheckResult:
    """Run a check that requires count == 0."""
    try:
        count, evidence, message = fn()
    except Exception as exc:
        return SafetyCheckResult(
            name=name, status=SafetyCheckStatus.FAIL, count=-1,
            evidence={}, message=f"check raised: {exc!r}",
        )
    passed = int(count) == 0
    return SafetyCheckResult(
        name=name,
        status=SafetyCheckStatus.PASS if passed else SafetyCheckStatus.FAIL,
        count=int(count), threshold=0,
        evidence=evidence, message=message,
    )


def run_safety_qualification(
    *,
    unauthorized_mutation_count: SafetyCheckFn | None = None,
    receipt_chain_failures: SafetyCheckFn | None = None,
    silent_stale_read_count: SafetyCheckFn | None = None,
    nondeterministic_output_count: SafetyCheckFn | None = None,
    invariant_breaking_commit_count: SafetyCheckFn | None = None,
    unrecoverable_crash_count: SafetyCheckFn | None = None,
) -> SafetyQualificationReport:
    """Run the six canonical safety checks and return an aggregate report.

    Any check function that is ``None`` is recorded as SKIP.
    """
    report = SafetyQualificationReport()
    checks = [
        ("unauthorized_mutations", unauthorized_mutation_count),
        ("receipt_chain_failures", receipt_chain_failures),
        ("silent_stale_reads", silent_stale_read_count),
        ("nondeterministic_outputs", nondeterministic_output_count),
        ("invariant_breaking_commits", invariant_breaking_commit_count),
        ("unrecoverable_crashes", unrecoverable_crash_count),
    ]
    for name, fn in checks:
        if fn is None:
            report.add(SafetyCheckResult(name=name, status=SafetyCheckStatus.SKIP, message="not configured"))
        else:
            report.add(_zero_violation_check(name, fn))
    return report


def assert_safety_gate(report: SafetyQualificationReport) -> None:
    """Raise if the safety gate did not pass. Used in production mode."""
    if not report.all_passed:
        failed = [c.name for c in report.failed_checks]
        raise SafetyGateError(f"safety gate failed: {failed}")


class SafetyGateError(RuntimeError):
    """Raised when the safety qualification gate does not pass."""
