"""v5.10 Phase 8: candidate-union architecture tests."""
from __future__ import annotations

import pytest

from lgae_v3.executive import StructuralAction
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.runtime import (
    Candidate, CandidateUnion, candidate_id, build_candidate_union,
)


def _ca(action, **target) -> ConcreteAction:
    return ConcreteAction(action=action, target=target, channel="test")


def test_candidate_id_is_canonical_sha256():
    cid = candidate_id("state-abc", StructuralAction.ADD_EDGE, {"u": 3, "v": 1})
    assert len(cid) == 64
    # Endpoint order does not change the id.
    cid2 = candidate_id("state-abc", StructuralAction.ADD_EDGE, {"u": 1, "v": 3})
    assert cid == cid2
    # Different state -> different id.
    cid3 = candidate_id("state-xyz", StructuralAction.ADD_EDGE, {"u": 1, "v": 3})
    assert cid != cid3


def test_candidate_id_no_op_is_state_bound():
    a = candidate_id("s1", StructuralAction.NO_OP, {})
    b = candidate_id("s2", StructuralAction.NO_OP, {})
    assert a != b


def test_candidate_id_quantizes_continuous_values():
    a = candidate_id("s", StructuralAction.REWEIGHT_AFFINITY, {"u": 0, "v": 1, "factor": 1.1000001})
    b = candidate_id("s", StructuralAction.REWEIGHT_AFFINITY, {"u": 0, "v": 1, "factor": 1.1})
    # Quantized to 6 decimals -> identical.
    assert a == b


def test_build_candidate_union_merges_and_dedups():
    learned = [_ca(StructuralAction.ADD_EDGE, u=0, v=2)]
    fosr = [_ca(StructuralAction.ADD_EDGE, u=0, v=2), _ca(StructuralAction.ADD_EDGE, u=1, v=3)]
    er = [_ca(StructuralAction.PRUNE_EDGE, u=2, v=3)]
    union = build_candidate_union("state-1", channels={"learned": learned, "fosr": fosr, "er": er})
    # 3 unique structural candidates + NO_OP.
    assert union.size == 4
    counts = union.channel_counts()
    # Channels are processed in sorted name order ("er" < "fosr" < "learned"),
    # so "fosr" wins the duplicate ADD_EDGE(0,2); "learned" is deduped away.
    assert counts["fosr"] == 2  # ADD_EDGE(0,2) + ADD_EDGE(1,3)
    assert counts["er"] == 1    # PRUNE_EDGE(2,3)
    assert counts["no_op"] == 1
    assert "learned" not in counts  # fully deduped


def test_candidate_union_always_includes_no_op():
    union = build_candidate_union("s", channels={"learned": []})
    cands = union.candidates()
    assert any(c.is_no_op for c in cands)
    assert union.size == 1


def test_candidate_union_order_is_deterministic():
    # Add candidates in non-sorted id order; output must be sorted by id.
    union = CandidateUnion(state_id="s")
    union.add_channel("a", [_ca(StructuralAction.ADD_EDGE, u=5, v=0)])
    union.add_channel("b", [_ca(StructuralAction.ADD_EDGE, u=1, v=0)])
    union.add_no_op()
    cands = union.candidates()
    ids = [c.id for c in cands]
    assert ids == sorted(ids)


def test_candidate_union_order_independent_of_channel_insertion_order():
    state = "s"
    ch_a = [_ca(StructuralAction.ADD_EDGE, u=0, v=2)]
    ch_b = [_ca(StructuralAction.PRUNE_EDGE, u=1, v=3)]
    u1 = build_candidate_union(state, channels={"a": ch_a, "b": ch_b})
    u2 = build_candidate_union(state, channels={"b": ch_b, "a": ch_a})
    assert [c.id for c in u1.candidates()] == [c.id for c in u2.candidates()]


def test_candidate_from_concrete_roundtrip():
    ca = ConcreteAction(action=StructuralAction.ADD_EDGE, target={"u": 0, "v": 1}, channel="fosr", prior_score=0.5)
    cand = Candidate.from_concrete(ca, state_id="s")
    assert cand.channel == "fosr"
    assert cand.prior_score == 0.5
    back = cand.to_concrete()
    assert back.action == ca.action
    assert back.channel == "fosr"


def test_candidate_union_skips_empty_channels():
    union = build_candidate_union("s", channels={"learned": None, "fosr": [], "er": [_ca(StructuralAction.ADD_EDGE, u=0, v=1)]})
    assert union.size == 2  # one structural + NO_OP


def test_candidate_union_to_log():
    union = build_candidate_union("s", channels={"er": [_ca(StructuralAction.ADD_EDGE, u=0, v=1)]})
    log = union.to_log()
    assert log["state_id"] == "s"
    assert log["size"] == 2
    assert log["ids"] == sorted(log["ids"])
