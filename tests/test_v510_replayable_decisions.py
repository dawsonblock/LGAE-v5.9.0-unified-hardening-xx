"""v5.10 Phase 29: replayable decisions tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    DecisionRecord, DecisionLedger, build_decision_record, verify_replay,
)
from lgae_v3.runtime.runtime_events import RuntimeEvent, RuntimePhase


def test_decision_record_hash_is_deterministic():
    r1 = build_decision_record(
        step=0, state_hash_before="abc", state_hash_after="def",
        chosen_action="add_edge", governance_decision="commit", committed=True,
        candidate_ids=["c1", "c2"], candidate_scores=[0.9, 0.1],
    )
    r2 = build_decision_record(
        step=0, state_hash_before="abc", state_hash_after="def",
        chosen_action="add_edge", governance_decision="commit", committed=True,
        candidate_ids=["c1", "c2"], candidate_scores=[0.9, 0.1],
    )
    assert r1.record_hash == r2.record_hash


def test_decision_record_hash_differs_for_different_actions():
    r1 = build_decision_record(
        step=0, state_hash_before="abc", state_hash_after="def",
        chosen_action="add_edge", governance_decision="commit", committed=True,
        candidate_ids=[], candidate_scores=[],
    )
    r2 = build_decision_record(
        step=0, state_hash_before="abc", state_hash_after="def",
        chosen_action="remove_edge", governance_decision="commit", committed=True,
        candidate_ids=[], candidate_scores=[],
    )
    assert r1.record_hash != r2.record_hash


def test_decision_ledger_append_and_latest():
    ledger = DecisionLedger()
    r = build_decision_record(
        step=0, state_hash_before="a", state_hash_after="b",
        chosen_action="x", governance_decision="commit", committed=True,
        candidate_ids=[], candidate_scores=[],
    )
    ledger.append(r)
    assert ledger.latest is r
    assert len(ledger.records) == 1


def test_decision_ledger_by_step():
    ledger = DecisionLedger()
    r0 = build_decision_record(step=0, state_hash_before="a", state_hash_after="b",
                               chosen_action="x", governance_decision="commit", committed=True,
                               candidate_ids=[], candidate_scores=[])
    r1 = build_decision_record(step=1, state_hash_before="b", state_hash_after="c",
                               chosen_action="y", governance_decision="reject", committed=False,
                               candidate_ids=[], candidate_scores=[])
    ledger.append(r0)
    ledger.append(r1)
    assert ledger.by_step(0) is r0
    assert ledger.by_step(1) is r1
    assert ledger.by_step(99) is None


def test_decision_ledger_by_state_hash():
    ledger = DecisionLedger()
    r1 = build_decision_record(step=0, state_hash_before="hash_a", state_hash_after="b",
                               chosen_action="x", governance_decision="commit", committed=True,
                               candidate_ids=[], candidate_scores=[])
    r2 = build_decision_record(step=1, state_hash_before="hash_a", state_hash_after="c",
                               chosen_action="y", governance_decision="commit", committed=True,
                               candidate_ids=[], candidate_scores=[])
    ledger.append(r1)
    ledger.append(r2)
    matches = ledger.by_state_hash("hash_a")
    assert len(matches) == 2


def test_decision_ledger_committed_and_rejected():
    ledger = DecisionLedger()
    ledger.append(build_decision_record(step=0, state_hash_before="a", state_hash_after="b",
                                        chosen_action="x", governance_decision="commit", committed=True,
                                        candidate_ids=[], candidate_scores=[]))
    ledger.append(build_decision_record(step=1, state_hash_before="b", state_hash_after="b",
                                        chosen_action="y", governance_decision="reject", committed=False,
                                        candidate_ids=[], candidate_scores=[]))
    assert len(ledger.committed_records()) == 1
    assert len(ledger.rejected_records()) == 1


def test_build_decision_record_with_events():
    events = [
        RuntimeEvent(RuntimePhase.OBSERVE, step=0, payload={"x": 1}),
        RuntimeEvent(RuntimePhase.COMMIT, step=0, payload={"hash": "abc"}),
    ]
    r = build_decision_record(
        step=0, state_hash_before="a", state_hash_after="b",
        chosen_action="x", governance_decision="commit", committed=True,
        candidate_ids=[], candidate_scores=[], events=events,
    )
    assert len(r.events) == 2
    assert r.events[0]["phase"] == "observe"
    assert r.events[1]["payload"]["hash"] == "abc"


def test_verify_replay_matches():
    r = build_decision_record(
        step=0, state_hash_before="a", state_hash_after="b",
        chosen_action="x", governance_decision="commit", committed=True,
        candidate_ids=[], candidate_scores=[],
    )
    assert verify_replay(r, expected_state_hash_after="b")


def test_verify_replay_mismatch():
    r = build_decision_record(
        step=0, state_hash_before="a", state_hash_after="b",
        chosen_action="x", governance_decision="commit", committed=True,
        candidate_ids=[], candidate_scores=[],
    )
    assert not verify_replay(r, expected_state_hash_after="wrong")


def test_decision_ledger_to_log():
    ledger = DecisionLedger()
    ledger.append(build_decision_record(step=0, state_hash_before="a", state_hash_after="b",
                                        chosen_action="x", governance_decision="commit", committed=True,
                                        candidate_ids=["c1"], candidate_scores=[0.9]))
    log = ledger.to_log()
    assert log["record_count"] == 1
    assert log["committed_count"] == 1
    assert log["records"][0]["candidate_ids"] == ["c1"]
