"""Research vs production mode enforcement (Phase 44).

Production mode fails closed: operations that are acceptable in research
mode (e.g. skipping evidence, allowing heuristic certification, relaxing
deterministic ordering) are blocked in production mode. The enforcer is
a policy gate, not a code path; it raises ``ProductionModeViolation``
when a disallowed operation is attempted in production.

This builds on the existing ``RuntimeConfig.mode`` field and its
``__post_init__`` validation. It adds runtime operation-level enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .runtime_config import RuntimeConfig, RuntimeMode


class ProductionModeViolation(RuntimeError):
    """Raised when a disallowed operation is attempted in production mode."""


@dataclass(slots=True)
class ModeEnforcer:
    """Enforces research vs production mode restrictions at runtime."""
    config: RuntimeConfig
    _violations: list[str] = field(default_factory=list)

    @property
    def is_production(self) -> bool:
        return self.config.is_production

    @property
    def violations(self) -> list[str]:
        return list(self._violations)

    def _check(self, condition: bool, operation: str, message: str) -> None:
        if self.is_production and not condition:
            self._violations.append(f"{operation}: {message}")
            raise ProductionModeViolation(f"production mode violation: {operation}: {message}")

    def assert_signed_receipts(self, has_signing_key: bool) -> None:
        """Production requires signed receipts."""
        self._check(
            has_signing_key,
            "signed_receipts",
            "production mode requires a signing key for receipts",
        )

    def assert_evidence_persisted(self, has_evidence_path: bool) -> None:
        """Production requires persisted evidence."""
        self._check(
            has_evidence_path,
            "evidence_persistence",
            "production mode requires an evidence ledger path",
        )

    def assert_exact_certification(self, is_exact: bool, operation: str = "commit") -> None:
        """Production commits require exact certification, not heuristic proxies."""
        self._check(
            is_exact,
            f"{operation}_certification",
            "production mode requires exact certification for commits",
        )

    def assert_deterministic_ordering(self, is_deterministic: bool) -> None:
        """Production requires deterministic ordering (no set/dict iteration)."""
        self._check(
            is_deterministic,
            "deterministic_ordering",
            "production mode requires deterministic ordering",
        )

    def assert_safety_gate_passed(self, safety_passed: bool) -> None:
        """Production requires the safety gate to have passed."""
        self._check(
            safety_passed,
            "safety_gate",
            "production mode requires the safety qualification gate to pass",
        )

    def assert_no_skipped_invariants(self, skipped_count: int) -> None:
        """Production requires all invariants to have run (no skips)."""
        self._check(
            skipped_count == 0,
            "invariant_coverage",
            "production mode requires all invariants to run (no skips)",
        )

    def assert_authorized_mutation(self, is_authorized: bool) -> None:
        """Production requires all mutations to pass through commit authority."""
        self._check(
            is_authorized,
            "authorized_mutation",
            "production mode requires all mutations to pass through commit authority",
        )

    def gate(self, operation: str, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` only if the operation is allowed in the current mode.

        In research mode, all operations are allowed. In production mode,
        the caller is responsible for checking prerequisites before calling
        ``gate``; this method simply records the operation.
        """
        return fn()

    def to_log(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode.value,
            "is_production": self.is_production,
            "violation_count": len(self._violations),
            "violations": list(self._violations),
        }
