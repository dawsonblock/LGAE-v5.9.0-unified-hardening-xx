"""v5.10 Phase 44: research vs production mode enforcement tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    RuntimeConfig, RuntimeMode, ModeEnforcer, ProductionModeViolation,
)


def _research_enforcer():
    return ModeEnforcer(config=RuntimeConfig(mode=RuntimeMode.RESEARCH))


def _production_enforcer():
    return ModeEnforcer(config=RuntimeConfig(
        mode=RuntimeMode.PRODUCTION,
        evidence_path="/tmp/evidence.jsonl",
        receipt_path="/tmp/receipts.jsonl",
        signing_key="test-key",
        require_signed_receipts=True,
        wal_path="/tmp/wal.jsonl",
    ))


def test_research_mode_allows_everything():
    e = _research_enforcer()
    e.assert_signed_receipts(False)  # no signing key
    e.assert_evidence_persisted(False)  # no evidence path
    e.assert_exact_certification(False)  # heuristic certification
    e.assert_deterministic_ordering(False)  # non-deterministic
    e.assert_safety_gate_passed(False)  # safety gate not passed
    e.assert_no_skipped_invariants(5)  # 5 skipped invariants
    e.assert_authorized_mutation(False)  # unauthorized
    assert len(e.violations) == 0


def test_production_mode_blocks_unsigned_receipts():
    e = _production_enforcer()
    with pytest.raises(ProductionModeViolation):
        e.assert_signed_receipts(False)


def test_production_mode_blocks_unpersisted_evidence():
    e = _production_enforcer()
    with pytest.raises(ProductionModeViolation):
        e.assert_evidence_persisted(False)


def test_production_mode_blocks_heuristic_certification():
    e = _production_enforcer()
    with pytest.raises(ProductionModeViolation):
        e.assert_exact_certification(False)


def test_production_mode_blocks_nondeterministic_ordering():
    e = _production_enforcer()
    with pytest.raises(ProductionModeViolation):
        e.assert_deterministic_ordering(False)


def test_production_mode_blocks_failed_safety_gate():
    e = _production_enforcer()
    with pytest.raises(ProductionModeViolation):
        e.assert_safety_gate_passed(False)


def test_production_mode_blocks_skipped_invariants():
    e = _production_enforcer()
    with pytest.raises(ProductionModeViolation):
        e.assert_no_skipped_invariants(1)


def test_production_mode_blocks_unauthorized_mutations():
    e = _production_enforcer()
    with pytest.raises(ProductionModeViolation):
        e.assert_authorized_mutation(False)


def test_production_mode_allows_when_conditions_met():
    e = _production_enforcer()
    e.assert_signed_receipts(True)
    e.assert_evidence_persisted(True)
    e.assert_exact_certification(True)
    e.assert_deterministic_ordering(True)
    e.assert_safety_gate_passed(True)
    e.assert_no_skipped_invariants(0)
    e.assert_authorized_mutation(True)
    assert len(e.violations) == 0


def test_violations_are_recorded():
    e = _production_enforcer()
    try:
        e.assert_signed_receipts(False)
    except ProductionModeViolation:
        pass
    try:
        e.assert_evidence_persisted(False)
    except ProductionModeViolation:
        pass
    assert len(e.violations) == 2
    assert "signed_receipts" in e.violations[0]
    assert "evidence_persistence" in e.violations[1]


def test_gate_runs_function_in_research_mode():
    e = _research_enforcer()
    result = e.gate("test_op", lambda: 42)
    assert result == 42


def test_gate_runs_function_in_production_mode():
    e = _production_enforcer()
    result = e.gate("test_op", lambda: 42)
    assert result == 42


def test_to_log_structure():
    e = _production_enforcer()
    log = e.to_log()
    assert log["mode"] == "production"
    assert log["is_production"] is True
    assert log["violation_count"] == 0
