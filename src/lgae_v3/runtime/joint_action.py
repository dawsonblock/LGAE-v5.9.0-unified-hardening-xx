"""Joint structural action v2 (Phase 15).

A joint structural action combines multiple primitive actions (add edge,
remove edge, reweight) into a single atomic proposal. This enables the
runtime to propose coordinated changes that are only beneficial when applied
together, such as:

  - adding an edge while rewiring another (structural swap)
  - reweighting multiple edges simultaneously (gauge adjustment)
  - adding an edge and updating the fiber bundle (consistent extension)

Joint actions are atomic: either all sub-actions are committed, or none.
This preserves the "all-or-nothing" semantics of authoritative mutations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..executive import StructuralAction


@dataclass(frozen=True, slots=True)
class SubAction:
    """One component of a joint action."""
    action_type: StructuralAction
    params: dict[str, Any]

    def to_log(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "params": self.params,
        }


@dataclass(frozen=True, slots=True)
class JointStructuralAction:
    """A composite action that applies multiple sub-actions atomically."""
    sub_actions: list[SubAction]
    joint_id: str = ""

    def __post_init__(self) -> None:
        if not self.joint_id:
            import hashlib
            import json
            payload = json.dumps({
                "sub_actions": [
                    {"type": sa.action_type.value, "params": sa.params}
                    for sa in self.sub_actions
                ],
            }, sort_keys=True, separators=(",", ":")).encode()
            object.__setattr__(self, "joint_id", hashlib.sha256(payload).hexdigest()[:16])

    @property
    def n_sub_actions(self) -> int:
        return len(self.sub_actions)

    @property
    def is_atomic(self) -> bool:
        """Joint actions are always atomic."""
        return True

    @property
    def action_types(self) -> list[StructuralAction]:
        return [sa.action_type for sa in self.sub_actions]

    def to_log(self) -> dict[str, Any]:
        return {
            "joint_id": self.joint_id,
            "n_sub_actions": int(self.n_sub_actions),
            "sub_actions": [sa.to_log() for sa in self.sub_actions],
            "action_types": [a.value for a in self.action_types],
        }


def make_joint_action(sub_actions: list[tuple[StructuralAction, dict[str, Any]]]) -> JointStructuralAction:
    """Create a joint action from a list of (action_type, params) tuples."""
    return JointStructuralAction(
        sub_actions=[SubAction(action_type=at, params=p) for at, p in sub_actions],
    )


def joint_action_authority_level(joint: JointStructuralAction) -> str:
    """Determine the authority level for a joint action.

    The joint action's authority level is the maximum of its sub-actions'
    levels. If any sub-action is IRREVERSIBLE, the joint action is
    IRREVERSIBLE. If any is HIGH_IMPACT, the joint is HIGH_IMPACT.
    """
    from ..mutations import MutationAuthorityLevel
    # Explicit severity ordering (not alphabetical).
    severity_order = [
        MutationAuthorityLevel.REVERSIBLE,
        MutationAuthorityLevel.STRUCTURAL,
        MutationAuthorityLevel.HIGH_IMPACT,
        MutationAuthorityLevel.IRREVERSIBLE,
    ]
    levels: list[MutationAuthorityLevel] = []
    for sa in joint.sub_actions:
        if sa.action_type == StructuralAction.ADD_EDGE:
            levels.append(MutationAuthorityLevel.STRUCTURAL)
        elif sa.action_type == StructuralAction.PRUNE_EDGE:
            levels.append(MutationAuthorityLevel.IRREVERSIBLE)
        elif sa.action_type == StructuralAction.REWEIGHT_AFFINITY:
            levels.append(MutationAuthorityLevel.REVERSIBLE)
        else:
            levels.append(MutationAuthorityLevel.STRUCTURAL)
    if not levels:
        return MutationAuthorityLevel.REVERSIBLE.value
    # Return the highest-severity level.
    max_severity = 0
    for level in levels:
        severity = severity_order.index(level)
        if severity > max_severity:
            max_severity = severity
    return severity_order[max_severity].value
