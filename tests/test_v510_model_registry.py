"""v5.10 Phase 45: model registry tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    ModelRecord, PromotionTransition, ModelRegistry, PromotionLevel,
)


def test_register_model():
    reg = ModelRegistry()
    record = reg.register(name="policy_v1", version="0.1", content="model-bytes")
    assert record.name == "policy_v1"
    assert record.version == "0.1"
    assert record.maturity == PromotionLevel.EXPERIMENTAL
    assert len(record.content_hash) == 64
    assert record.model_id.startswith("policy_v1:0.1:")


def test_register_duplicate_raises():
    reg = ModelRegistry()
    reg.register(name="p", version="1", content="abc")
    with pytest.raises(ValueError):
        reg.register(name="p", version="1", content="abc")


def test_register_different_content_does_not_raise():
    reg = ModelRegistry()
    r1 = reg.register(name="p", version="1", content="abc")
    r2 = reg.register(name="p", version="1", content="xyz")
    assert r1.model_id != r2.model_id
    assert len(reg.models) == 2


def test_promote_increases_maturity():
    reg = ModelRegistry()
    record = reg.register(name="p", version="1", content="abc")
    t = reg.promote(record.model_id, PromotionLevel.CANDIDATE)
    assert t.from_level == PromotionLevel.EXPERIMENTAL
    assert t.to_level == PromotionLevel.CANDIDATE
    assert t.approved
    # Registry reflects the new maturity.
    assert reg.get(record.model_id).maturity == PromotionLevel.CANDIDATE


def test_promote_to_same_or_lower_raises():
    reg = ModelRegistry()
    record = reg.register(name="p", version="1", content="abc")
    reg.promote(record.model_id, PromotionLevel.CANDIDATE)
    with pytest.raises(ValueError):
        reg.promote(record.model_id, PromotionLevel.CANDIDATE)  # same
    with pytest.raises(ValueError):
        reg.promote(record.model_id, PromotionLevel.EXPERIMENTAL)  # lower


def test_models_at_maturity():
    reg = ModelRegistry()
    r1 = reg.register(name="a", version="1", content="x")
    r2 = reg.register(name="b", version="1", content="y")
    reg.promote(r1.model_id, PromotionLevel.CANDIDATE)
    exp = reg.models_at_maturity(PromotionLevel.EXPERIMENTAL)
    cand = reg.models_at_maturity(PromotionLevel.CANDIDATE)
    assert len(exp) == 1 and exp[0].name == "b"
    assert len(cand) == 1 and cand[0].name == "a"


def test_transitions_are_auditable():
    reg = ModelRegistry()
    r = reg.register(name="p", version="1", content="abc")
    reg.promote(r.model_id, PromotionLevel.CANDIDATE, gate_report={"safety": "pass"})
    reg.promote(r.model_id, PromotionLevel.QUALIFIED, gate_report={"safety": "pass", "scientific": "pass"})
    assert len(reg.transitions) == 2
    assert reg.transitions[0].to_level == PromotionLevel.CANDIDATE
    assert reg.transitions[1].to_level == PromotionLevel.QUALIFIED
    assert reg.transitions[1].gate_report["scientific"] == "pass"


def test_registry_to_log():
    reg = ModelRegistry()
    r = reg.register(name="p", version="1", content="abc", metadata={"arch": "gnn"})
    reg.promote(r.model_id, PromotionLevel.CANDIDATE)
    log = reg.to_log()
    assert log["model_count"] == 1
    assert log["transition_count"] == 1
    assert log["models"][0]["metadata"]["arch"] == "gnn"
    assert log["models"][0]["maturity"] == "CANDIDATE"


def test_models_sorted_by_id():
    reg = ModelRegistry()
    reg.register(name="zeta", version="1", content="a")
    reg.register(name="alpha", version="1", content="b")
    names = [m.name for m in reg.models]
    assert names == ["alpha", "zeta"]
