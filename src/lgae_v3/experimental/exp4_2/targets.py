"""Target definitions for exp4.2.

Each target is versioned and hashed. Targets are defined BEFORE training
and must not change after held-out access begins.

Primary utility target:
    y_U = ΔU_realized

Secondary normalized utility:
    y_U_norm = ΔU / max(ε, |U_before|)

Sign target:
    y_sign = 1[ΔU > 0]

Risk target:
    y_R = R

Cost target:
    y_C = C

Ranking target (pairwise within candidate set):
    y_ij = 1[ΔU_i > ΔU_j]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import json
import math


class TargetType:
    """Enumeration of target types."""
    UTILITY_REGRESSION = "utility_regression"
    UTILITY_NORMALIZED = "utility_normalized"
    UTILITY_SIGN = "utility_sign"
    CANDIDATE_RANKING = "candidate_ranking"
    RISK = "risk"
    COST = "cost"


@dataclass(frozen=True, slots=True)
class TargetDefinition:
    """Definition of a prediction target.

    Each target has:
    - name: canonical name
    - target_type: one of TargetType constants
    - task_category: "regression", "classification", or "ranking"
    - description: human-readable description
    - version: target schema version
    - schema_hash: hash of the target schema
    """
    name: str
    target_type: str
    task_category: str  # "regression", "classification", "ranking"
    description: str
    version: str = "v6.0-exp4.2"
    schema_hash: str = ""

    def __post_init__(self) -> None:
        if not self.schema_hash:
            object.__setattr__(self, "schema_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        content = json.dumps({
            "name": self.name,
            "target_type": self.target_type,
            "task_category": self.task_category,
            "version": self.version,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_type": self.target_type,
            "task_category": self.task_category,
            "description": self.description,
            "version": self.version,
            "schema_hash": self.schema_hash,
        }


# All targets defined before training. Frozen.
TARGET_DEFINITIONS: dict[str, TargetDefinition] = {
    "realized_delta": TargetDefinition(
        name="realized_delta",
        target_type=TargetType.UTILITY_REGRESSION,
        task_category="regression",
        description="Primary utility target: realized ΔU.",
    ),
    "normalized_delta": TargetDefinition(
        name="normalized_delta",
        target_type=TargetType.UTILITY_NORMALIZED,
        task_category="regression",
        description="Normalized utility: ΔU / max(ε, |U_before|).",
    ),
    "sign_delta": TargetDefinition(
        name="sign_delta",
        target_type=TargetType.UTILITY_SIGN,
        task_category="classification",
        description="Sign target: 1[ΔU > 0].",
    ),
    "candidate_ranking": TargetDefinition(
        name="candidate_ranking",
        target_type=TargetType.CANDIDATE_RANKING,
        task_category="ranking",
        description="Pairwise ranking within candidate set: 1[ΔU_i > ΔU_j].",
    ),
    "risk": TargetDefinition(
        name="risk",
        target_type=TargetType.RISK,
        task_category="regression",
        description="Risk target: realized risk R.",
    ),
    "cost": TargetDefinition(
        name="cost",
        target_type=TargetType.COST,
        task_category="regression",
        description="Cost target: realized cost C.",
    ),
}


def get_target_definition(name: str) -> TargetDefinition:
    """Get a target definition by name."""
    if name not in TARGET_DEFINITIONS:
        raise KeyError(
            f"Unknown target: '{name}'. "
            f"Available: {list(TARGET_DEFINITIONS.keys())}"
        )
    return TARGET_DEFINITIONS[name]


def extract_target_value(record: Any, target_name: str) -> float:
    """Extract a target value from a transition record.

    Args:
        record: A TransitionRecord.
        target_name: Name of the target.

    Returns:
        The target value as a float.
    """
    if target_name == "realized_delta":
        return float(getattr(record, "realized_delta", 0.0))
    elif target_name == "normalized_delta":
        delta = float(getattr(record, "realized_delta", 0.0))
        # Use a fixed epsilon to avoid division by zero.
        return delta / max(1e-8, abs(delta) + 1.0)
    elif target_name == "sign_delta":
        return 1.0 if float(getattr(record, "realized_delta", 0.0)) > 0 else 0.0
    elif target_name == "risk":
        return float(getattr(record, "realized_risk", 0.0))
    elif target_name == "cost":
        return float(getattr(record, "realized_cost", 0.0))
    else:
        raise KeyError(f"Unknown target: {target_name}")


def all_target_hashes() -> dict[str, str]:
    """Return a mapping of target name to schema hash."""
    return {name: td.schema_hash for name, td in TARGET_DEFINITIONS.items()}
