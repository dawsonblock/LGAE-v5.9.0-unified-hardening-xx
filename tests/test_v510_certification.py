"""v5.10 Phase 6: first-class certification tests."""
from __future__ import annotations

import pytest

from lgae_v3.types import CertificationLevel as LegacyCertificationLevel
from lgae_v3.mutations import MutationAuthorityLevel
from lgae_v3.runtime import (
    CertificationLevel,
    CertificationResult,
    CertificationError,
    minimum_level_for,
    meets_requirement,
)


def test_level_ordering_is_monotonic():
    assert CertificationLevel.HEURISTIC_PROXY < CertificationLevel.SAMPLED_LOCAL
    assert CertificationLevel.SAMPLED_LOCAL < CertificationLevel.SAMPLED_GLOBAL
    assert CertificationLevel.SAMPLED_GLOBAL < CertificationLevel.CERTIFIED_LOCAL
    assert CertificationLevel.CERTIFIED_LOCAL < CertificationLevel.CERTIFIED_GLOBAL
    assert CertificationLevel.CERTIFIED_GLOBAL < CertificationLevel.FORMALLY_VERIFIED


def test_is_exact_only_for_certified_or_stronger():
    assert not CertificationResult(CertificationLevel.HEURISTIC_PROXY, True).is_exact
    assert not CertificationResult(CertificationLevel.SAMPLED_LOCAL, True).is_exact
    assert CertificationResult(CertificationLevel.CERTIFIED_LOCAL, True).is_exact
    assert CertificationResult(CertificationLevel.CERTIFIED_GLOBAL, True).is_exact
    assert CertificationResult(CertificationLevel.FORMALLY_VERIFIED, True).is_exact


def test_is_global_only_for_global_scopes():
    assert not CertificationResult(CertificationLevel.SAMPLED_LOCAL, True).is_global
    assert CertificationResult(CertificationLevel.SAMPLED_GLOBAL, True).is_global
    assert CertificationResult(CertificationLevel.CERTIFIED_GLOBAL, True).is_global
    assert CertificationResult(CertificationLevel.FORMALLY_VERIFIED, True).is_global


def test_assert_exact_rejects_heuristic():
    with pytest.raises(CertificationError):
        CertificationResult(CertificationLevel.HEURISTIC_PROXY, True).assert_exact()
    # Exact result does not raise.
    CertificationResult(CertificationLevel.CERTIFIED_LOCAL, True).assert_exact()


def test_coverage_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        CertificationResult(CertificationLevel.SAMPLED_LOCAL, True, coverage=1.5)
    with pytest.raises(ValueError):
        CertificationResult(CertificationLevel.SAMPLED_LOCAL, True, coverage=-0.1)


def test_legacy_level_roundtrip():
    for legacy in LegacyCertificationLevel:
        new = CertificationLevel.from_legacy(legacy)
        assert new.to_legacy() == legacy


def test_minimum_level_for_authority():
    assert minimum_level_for(MutationAuthorityLevel.REVERSIBLE) == CertificationLevel.SAMPLED_LOCAL
    assert minimum_level_for(MutationAuthorityLevel.STRUCTURAL) == CertificationLevel.CERTIFIED_LOCAL
    assert minimum_level_for(MutationAuthorityLevel.IRREVERSIBLE) == CertificationLevel.CERTIFIED_GLOBAL


def test_meets_requirement_respects_ordering_and_passed():
    req = CertificationLevel.CERTIFIED_LOCAL
    assert meets_requirement(CertificationResult(CertificationLevel.CERTIFIED_LOCAL, True), req)
    assert meets_requirement(CertificationResult(CertificationLevel.CERTIFIED_GLOBAL, True), req)
    # Failed result never meets requirement.
    assert not meets_requirement(CertificationResult(CertificationLevel.CERTIFIED_GLOBAL, False), req)
    # Weaker level does not meet.
    assert not meets_requirement(CertificationResult(CertificationLevel.SAMPLED_LOCAL, True), req)


def test_to_log_includes_strength_flags():
    r = CertificationResult(
        CertificationLevel.CERTIFIED_GLOBAL, True,
        assumptions=("small_graph",), coverage=1.0,
        metrics={"max_lly_deficit": 0.1}, evidence_ids=("e1",),
    )
    log = r.to_log()
    assert log["level_name"] == "CERTIFIED_GLOBAL"
    assert log["is_exact"] and log["is_global"]
    assert log["coverage"] == 1.0
    assert log["evidence_ids"] == ["e1"]
