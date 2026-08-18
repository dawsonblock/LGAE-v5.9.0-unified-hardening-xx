"""v5.11 Phase 1: canonical runtime contract tests.

Verifies that all 8 phase contracts are:
- immutable (frozen dataclasses)
- state-bound (carry state_version and state_hash)
- deterministically serializable
- hashable
"""
from __future__ import annotations

import pytest

from lgae_v3.runtime.contracts import (
    PhaseResult, canonical_json, canonical_hash,
    ObservationSnapshot, ReasoningResult, StructuralDeficit, DiagnosticBundle,
    Candidate, CandidateSet, PlanningResult, CandidateValue,
    CounterfactualEvaluation, AuthorizationResult, AuthorizationStatus,
    RejectionReason, CommitResult, LearningResult, DecisionTransition,
    CreditAssignment, RuntimeStepResult, CANONICAL_PHASE_ORDER,
)


class TestImmutability:
    """All phase contracts must be frozen dataclasses."""

    def test_observation_snapshot_immutable(self):
        snap = ObservationSnapshot(
            snapshot_id="s1", state_version=1, state_hash="h1",
        )
        with pytest.raises((AttributeError, TypeError)):
            snap.state_version = 2

    def test_reasoning_result_immutable(self):
        r = ReasoningResult(
            snapshot_id="s1", state_version=1, state_hash="h1",
        )
        with pytest.raises((AttributeError, TypeError)):
            r.epistemic_uncertainty = 0.5

    def test_candidate_immutable(self):
        c = Candidate(
            candidate_id="c1", source_state_hash="h1",
            source_state_version=1, action_type="add_edge",
            parameters={"u": 0, "v": 1},
        )
        with pytest.raises((AttributeError, TypeError)):
            c.action_type = "prune_edge"

    def test_authorization_result_immutable(self):
        a = AuthorizationResult(
            snapshot_id="s1", state_version=1, state_hash="h1",
            status=AuthorizationStatus.AUTHORIZED,
            transaction_hash="test_txn_hash",
        )
        with pytest.raises((AttributeError, TypeError)):
            a.status = AuthorizationStatus.REJECTED

    def test_commit_result_immutable(self):
        c = CommitResult(
            snapshot_id="s1", state_version=1, state_hash="h1",
        )
        with pytest.raises((AttributeError, TypeError)):
            c.committed = True

    def test_learning_result_immutable(self):
        l = LearningResult(
            snapshot_id="s1", state_version=1, state_hash="h1",
        )
        with pytest.raises((AttributeError, TypeError)):
            l.replay_buffer_size = 10


class TestStateBinding:
    """Every phase result must bind to its source state."""

    def test_observation_carries_state_version_and_hash(self):
        snap = ObservationSnapshot(
            snapshot_id="s1", state_version=42, state_hash="abc123",
        )
        assert snap.state_version == 42
        assert snap.state_hash == "abc123"

    def test_reasoning_carries_state_version_and_hash(self):
        r = ReasoningResult(
            snapshot_id="s1", state_version=42, state_hash="abc123",
        )
        assert r.state_version == 42
        assert r.state_hash == "abc123"

    def test_candidate_carries_source_state(self):
        c = Candidate(
            candidate_id="c1", source_state_hash="abc123",
            source_state_version=42, action_type="add_edge",
            parameters={"u": 0, "v": 1},
        )
        assert c.source_state_hash == "abc123"
        assert c.source_state_version == 42


class TestDeterministicSerialization:
    """All contracts must serialize deterministically."""

    def test_canonical_json_deterministic(self):
        snap1 = ObservationSnapshot(
            snapshot_id="s1", state_version=1, state_hash="h1",
        )
        snap2 = ObservationSnapshot(
            snapshot_id="s1", state_version=1, state_hash="h1",
        )
        assert canonical_json(snap1.to_dict()) == canonical_json(snap2.to_dict())

    def test_canonical_hash_deterministic(self):
        snap1 = ObservationSnapshot(
            snapshot_id="s1", state_version=1, state_hash="h1",
        )
        snap2 = ObservationSnapshot(
            snapshot_id="s1", state_version=1, state_hash="h1",
        )
        assert snap1.to_hash() == snap2.to_hash()

    def test_different_states_different_hash(self):
        snap1 = ObservationSnapshot(
            snapshot_id="s1", state_version=1, state_hash="h1",
        )
        snap2 = ObservationSnapshot(
            snapshot_id="s2", state_version=2, state_hash="h2",
        )
        assert snap1.to_hash() != snap2.to_hash()


class TestCanonicalPhaseOrder:
    """The canonical phase order must be exactly 8 phases."""

    def test_phase_order_has_8_phases(self):
        assert len(CANONICAL_PHASE_ORDER) == 8

    def test_phase_order_correct(self):
        assert CANONICAL_PHASE_ORDER == (
            "observe", "reason", "propose", "plan",
            "evaluate", "authorize", "commit", "learn",
        )


class TestRuntimeStepResult:
    """The aggregate step result must carry all 8 phase outputs."""

    def test_executed_all_phases_true(self):
        result = RuntimeStepResult(step=0)
        assert result.executed_all_phases is True

    def test_executed_all_phases_false_with_wrong_order(self):
        result = RuntimeStepResult(
            step=0,
            phase_order=("observe", "reason"),
        )
        assert result.executed_all_phases is False

    def test_step_result_hashable(self):
        result = RuntimeStepResult(step=0)
        assert isinstance(result.to_hash(), str)
        assert len(result.to_hash()) == 64  # SHA-256 hex

    def test_step_result_to_dict(self):
        result = RuntimeStepResult(step=0)
        d = result.to_dict()
        assert d["step"] == 0
        assert "phase_order" in d
        assert len(d["phase_order"]) == 8


class TestAuthorizationStatus:
    """Authorization status enum must have the 4 required values."""

    def test_has_authorized(self):
        assert AuthorizationStatus.AUTHORIZED.value == "authorized"

    def test_has_rejected(self):
        assert AuthorizationStatus.REJECTED.value == "rejected"

    def test_has_quarantined(self):
        assert AuthorizationStatus.QUARANTINED.value == "quarantined"

    def test_has_deferred(self):
        assert AuthorizationStatus.DEFERRED.value == "deferred"


class TestRejectionReason:
    """Rejection reason enum must have the required reason codes."""

    def test_has_stale_state(self):
        assert RejectionReason.STALE_STATE.value == "stale_state"

    def test_has_invariant_violation(self):
        assert RejectionReason.INVARIANT_VIOLATION.value == "invariant_violation"

    def test_has_uncertainty_too_high(self):
        assert RejectionReason.UNCERTAINTY_TOO_HIGH.value == "uncertainty_too_high"


class TestCreditAssignment:
    """Credit must be hierarchical (not a single scalar)."""

    def test_credit_has_six_dimensions(self):
        c = CreditAssignment()
        d = c.to_dict()
        assert "diagnostic_credit" in d
        assert "candidate_credit" in d
        assert "planner_credit" in d
        assert "action_credit" in d
        assert "governance_credit" in d
        assert "outcome_credit" in d

    def test_credit_immutable(self):
        c = CreditAssignment()
        with pytest.raises((AttributeError, TypeError)):
            c.diagnostic_credit = 1.0
