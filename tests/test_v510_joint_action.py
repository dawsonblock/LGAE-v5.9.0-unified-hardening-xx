"""v5.10 Phase 15: joint structural action v2 tests."""
from __future__ import annotations

import pytest

from lgae_v3.executive import StructuralAction
from lgae_v3.runtime import (
    SubAction, JointStructuralAction, make_joint_action,
    joint_action_authority_level,
)


def test_joint_action_has_deterministic_id():
    j1 = make_joint_action([
        (StructuralAction.ADD_EDGE, {"u": 0, "v": 1}),
        (StructuralAction.REWEIGHT_AFFINITY, {"u": 0, "v": 1, "factor": 1.5}),
    ])
    j2 = make_joint_action([
        (StructuralAction.ADD_EDGE, {"u": 0, "v": 1}),
        (StructuralAction.REWEIGHT_AFFINITY, {"u": 0, "v": 1, "factor": 1.5}),
    ])
    assert j1.joint_id == j2.joint_id


def test_joint_action_different_sub_actions_different_id():
    j1 = make_joint_action([(StructuralAction.ADD_EDGE, {"u": 0, "v": 1})])
    j2 = make_joint_action([(StructuralAction.ADD_EDGE, {"u": 0, "v": 2})])
    assert j1.joint_id != j2.joint_id


def test_joint_action_n_sub_actions():
    j = make_joint_action([
        (StructuralAction.ADD_EDGE, {"u": 0, "v": 1}),
        (StructuralAction.REWEIGHT_AFFINITY, {"u": 0, "v": 1, "factor": 1.5}),
        (StructuralAction.PRUNE_EDGE, {"u": 2, "v": 3}),
    ])
    assert j.n_sub_actions == 3


def test_joint_action_is_atomic():
    j = make_joint_action([(StructuralAction.ADD_EDGE, {"u": 0, "v": 1})])
    assert j.is_atomic is True


def test_joint_action_action_types():
    j = make_joint_action([
        (StructuralAction.ADD_EDGE, {"u": 0, "v": 1}),
        (StructuralAction.PRUNE_EDGE, {"u": 2, "v": 3}),
    ])
    assert j.action_types == [StructuralAction.ADD_EDGE, StructuralAction.PRUNE_EDGE]


def test_joint_action_authority_level_reversible():
    j = make_joint_action([
        (StructuralAction.REWEIGHT_AFFINITY, {"u": 0, "v": 1, "factor": 1.5}),
    ])
    assert joint_action_authority_level(j) == "reversible"


def test_joint_action_authority_level_structural():
    j = make_joint_action([
        (StructuralAction.ADD_EDGE, {"u": 0, "v": 1}),
        (StructuralAction.REWEIGHT_AFFINITY, {"u": 0, "v": 1, "factor": 1.5}),
    ])
    assert joint_action_authority_level(j) == "structural"


def test_joint_action_authority_level_irreversible():
    j = make_joint_action([
        (StructuralAction.ADD_EDGE, {"u": 0, "v": 1}),
        (StructuralAction.PRUNE_EDGE, {"u": 2, "v": 3}),
    ])
    assert joint_action_authority_level(j) == "irreversible"


def test_joint_action_to_log():
    j = make_joint_action([
        (StructuralAction.ADD_EDGE, {"u": 0, "v": 1}),
    ])
    log = j.to_log()
    assert log["n_sub_actions"] == 1
    assert log["action_types"] == ["add_edge"]
    assert len(log["joint_id"]) == 16


def test_sub_action_to_log():
    sa = SubAction(StructuralAction.ADD_EDGE, {"u": 0, "v": 1})
    log = sa.to_log()
    assert log["action_type"] == "add_edge"
    assert log["params"] == {"u": 0, "v": 1}


def test_joint_action_empty():
    j = make_joint_action([])
    assert j.n_sub_actions == 0
    assert joint_action_authority_level(j) == "reversible"
