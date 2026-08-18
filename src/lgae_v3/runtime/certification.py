"""First-class certification for the canonical runtime (Phase 6).

Every verification result returns a structured ``CertificationResult`` that
makes the coverage and strength of evidence explicit. A heuristic proxy can
never masquerade as exact evidence: the level enum is ordered and the
``is_exact`` / ``is_global`` properties are derived from it.

Levels (ordered, weakest to strongest):

  HEURISTIC_PROXY    - fast proxy diagnostics only (AF3/WAF3)
  SAMPLED_LOCAL      - a subset of local edges/nodes audited
  SAMPLED_GLOBAL     - a subset audited but with global extrapolation
  CERTIFIED_LOCAL    - all local edges/nodes in scope audited exactly
  CERTIFIED_GLOBAL   - the entire graph audited exactly (small graphs)
  FORMALLY_VERIFIED  - invariant contracts proven, not just tested

This module extends the v5.3.2 ``CertificationLevel`` (which had only three
values) and provides a bidirectional mapping for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from ..types import CertificationLevel as LegacyCertificationLevel


class CertificationLevel(IntEnum):
    """Ordered certification strength."""
    HEURISTIC_PROXY = 0
    SAMPLED_LOCAL = 1
    SAMPLED_GLOBAL = 2
    CERTIFIED_LOCAL = 3
    CERTIFIED_GLOBAL = 4
    FORMALLY_VERIFIED = 5

    @classmethod
    def from_legacy(cls, lvl: LegacyCertificationLevel) -> "CertificationLevel":
        return {
            LegacyCertificationLevel.HEURISTIC_PROXY: cls.HEURISTIC_PROXY,
            LegacyCertificationLevel.SAMPLED_LOCAL: cls.SAMPLED_LOCAL,
            LegacyCertificationLevel.CERTIFIED_GLOBAL: cls.CERTIFIED_GLOBAL,
        }[lvl]

    def to_legacy(self) -> LegacyCertificationLevel:
        if self == CertificationLevel.HEURISTIC_PROXY:
            return LegacyCertificationLevel.HEURISTIC_PROXY
        if self in (CertificationLevel.SAMPLED_LOCAL, CertificationLevel.SAMPLED_GLOBAL,
                    CertificationLevel.CERTIFIED_LOCAL):
            return LegacyCertificationLevel.SAMPLED_LOCAL
        return LegacyCertificationLevel.CERTIFIED_GLOBAL


@dataclass(frozen=True, slots=True)
class CertificationResult:
    """Structured verification outcome.

    ``coverage`` is the fraction of the in-scope graph actually audited
    (1.0 = full coverage within the level's scope). ``assumptions`` lists the
    assumptions under which the certification holds. ``evidence_ids`` binds
    the result to immutable evidence ledger entries.
    """
    level: CertificationLevel
    passed: bool
    assumptions: tuple[str, ...] = ()
    coverage: float = 1.0
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.coverage) <= 1.0:
            raise ValueError("coverage must lie in [0,1]")

    @property
    def is_exact(self) -> bool:
        """True only for levels that use exact (non-proxy) evidence."""
        return self.level >= CertificationLevel.CERTIFIED_LOCAL

    @property
    def is_global(self) -> bool:
        """True only for globally-scoped certification."""
        return self.level in (CertificationLevel.SAMPLED_GLOBAL,
                              CertificationLevel.CERTIFIED_GLOBAL,
                              CertificationLevel.FORMALLY_VERIFIED)

    @property
    def is_formal(self) -> bool:
        return self.level == CertificationLevel.FORMALLY_VERIFIED

    def assert_exact(self) -> None:
        """Fail closed if this result is not exact evidence."""
        if not self.is_exact:
            raise CertificationError(
                f"certification level {self.level.name} is not exact; "
                "a heuristic/sampled result cannot substitute for exact evidence"
            )

    def to_log(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "level_name": self.level.name,
            "passed": bool(self.passed),
            "assumptions": list(self.assumptions),
            "coverage": float(self.coverage),
            "metrics": self.metrics,
            "evidence_ids": list(self.evidence_ids),
            "is_exact": self.is_exact,
            "is_global": self.is_global,
        }


class CertificationError(RuntimeError):
    """Raised when a non-exact certification is used where exact evidence is
    required (Phase 6: heuristics must not masquerade as exact)."""


def minimum_level_for(authority_level: Any) -> CertificationLevel:
    """Minimum certification strength required for a mutation authority level.

    Maps the Phase 7 mutation authority levels to a minimum certification
    level. Reversible mutations may use sampled validation; structural
    mutations require exact local verification; irreversible mutations
    require global certification.
    """
    from ..mutations import MutationAuthorityLevel
    return {
        MutationAuthorityLevel.REVERSIBLE: CertificationLevel.SAMPLED_LOCAL,
        MutationAuthorityLevel.STRUCTURAL: CertificationLevel.CERTIFIED_LOCAL,
        MutationAuthorityLevel.HIGH_IMPACT: CertificationLevel.SAMPLED_GLOBAL,
        MutationAuthorityLevel.IRREVERSIBLE: CertificationLevel.CERTIFIED_GLOBAL,
    }.get(authority_level, CertificationLevel.SAMPLED_LOCAL)


def meets_requirement(result: CertificationResult, required: CertificationLevel) -> bool:
    """True if ``result`` is at least as strong as ``required`` and passed."""
    return bool(result.passed) and result.level >= required
