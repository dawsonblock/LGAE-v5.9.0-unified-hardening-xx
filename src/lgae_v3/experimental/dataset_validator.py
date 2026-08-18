"""Dataset validator for v6.0-exp2 structural transition datasets.

Validates that a dataset conforms to the schema and is free of common
data quality issues:

- Duplicate transition IDs
- Cross-split graph contamination
- Missing authority identity
- Nonfinite metrics
- Invalid action schema
- state_before == state_after for claimed mutation
- Post-state hash mismatch
- Duplicate episode/step identity
- Held-out leakage
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import json

from .transition_record import (
    TransitionRecord,
    TransitionProvenance,
    AuthorizationDecision,
)


@dataclass(slots=True)
class ValidationIssue:
    """A single validation issue."""
    severity: str  # "error", "warning"
    category: str  # "duplicate_id", "leakage", "nonfinite", etc.
    message: str
    record_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_log(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "record_id": self.record_id,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class ValidationResult:
    """Result of validating a dataset."""
    valid: bool
    n_records: int
    n_errors: int
    n_warnings: int
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "n_records": int(self.n_records),
            "n_errors": int(self.n_errors),
            "n_warnings": int(self.n_warnings),
            "issues": [i.to_log() for i in self.issues],
        }


class DatasetValidator:
    """Validates structural transition datasets.

    Usage::

        validator = DatasetValidator()
        result = validator.validate(records, expected_split="train")
        if not result.valid:
            for issue in result.issues:
                print(issue)
    """

    def __init__(
        self,
        *,
        held_out_families: set[str] | None = None,
        train_families: set[str] | None = None,
    ) -> None:
        self.held_out_families = held_out_families or set()
        self.train_families = train_families or set()

    def validate(
        self,
        records: list[TransitionRecord],
        *,
        expected_split: str | None = None,
    ) -> ValidationResult:
        """Validate a list of transition records.

        Args:
            records: The records to validate.
            expected_split: If set, all records must have this split.

        Returns:
            ValidationResult with all issues found.
        """
        issues: list[ValidationIssue] = []
        seen_ids: set[str] = set()
        seen_episode_steps: set[str] = set()

        for i, record in enumerate(records):
            # 1. Duplicate record IDs.
            if record.record_id in seen_ids:
                issues.append(ValidationIssue(
                    severity="error",
                    category="duplicate_id",
                    message=f"Duplicate record_id: {record.record_id}",
                    record_id=record.record_id,
                ))
            seen_ids.add(record.record_id)

            # 2. Duplicate episode/step identity.
            ep_step = f"{record.episode_id}:{record.step_id}:{record.provenance.value}"
            if ep_step in seen_episode_steps:
                issues.append(ValidationIssue(
                    severity="error",
                    category="duplicate_episode_step",
                    message=f"Duplicate episode/step: {ep_step}",
                    record_id=record.record_id,
                    details={"episode_id": record.episode_id, "step_id": record.step_id},
                ))
            seen_episode_steps.add(ep_step)

            # 3. Split consistency.
            if expected_split is not None and record.split != expected_split:
                issues.append(ValidationIssue(
                    severity="error",
                    category="split_mismatch",
                    message=f"Record split '{record.split}' != expected '{expected_split}'",
                    record_id=record.record_id,
                    details={"expected": expected_split, "actual": record.split},
                ))

            # 4. Held-out leakage: held-out families in train split.
            if (record.split == "train" and
                    self.held_out_families and
                    record.graph_family in self.held_out_families):
                issues.append(ValidationIssue(
                    severity="error",
                    category="held_out_leakage",
                    message=f"Held-out family '{record.graph_family}' found in train split",
                    record_id=record.record_id,
                    details={"graph_family": record.graph_family, "split": record.split},
                ))

            # 5. Train contamination in held-out split.
            if (record.split == "held_out" and
                    self.train_families and
                    record.graph_family in self.train_families):
                issues.append(ValidationIssue(
                    severity="error",
                    category="train_contamination",
                    message=f"Train family '{record.graph_family}' found in held-out split",
                    record_id=record.record_id,
                    details={"graph_family": record.graph_family, "split": record.split},
                ))

            # 6. Missing authority identity.
            if not record.authority_identity_before.state_hash:
                issues.append(ValidationIssue(
                    severity="error",
                    category="missing_authority",
                    message="Missing authority_identity_before.state_hash",
                    record_id=record.record_id,
                ))

            # 7. Nonfinite metrics.
            for field_name in [
                "predicted_delta", "predicted_risk", "predicted_cost", "predicted_ig",
                "realized_delta", "realized_cost", "realized_risk",
            ]:
                val = getattr(record, field_name)
                if not math.isfinite(val):
                    issues.append(ValidationIssue(
                        severity="error",
                        category="nonfinite",
                        message=f"Nonfinite {field_name}: {val}",
                        record_id=record.record_id,
                        details={"field": field_name, "value": str(val)},
                    ))

            # 8. Invalid action schema.
            if not record.action:
                issues.append(ValidationIssue(
                    severity="error",
                    category="invalid_action",
                    message="Empty action string",
                    record_id=record.record_id,
                ))

            # 9. state_before == state_after for claimed mutation.
            if (record.structural_state_after is not None and
                    record.success and
                    record.action != "NO_OP" and
                    record.structural_state_before.state_hash ==
                    record.structural_state_after.state_hash):
                issues.append(ValidationIssue(
                    severity="warning",
                    category="no_state_change",
                    message="state_before == state_after for claimed mutation",
                    record_id=record.record_id,
                    details={"action": record.action},
                ))

            # 10. Post-state hash mismatch.
            if (record.structural_state_after is not None and
                    record.authority_identity_after is not None and
                    record.structural_state_after.state_hash !=
                    record.authority_identity_after.state_hash):
                issues.append(ValidationIssue(
                    severity="warning",
                    category="hash_mismatch",
                    message="structural_state_after.state_hash != authority_identity_after.state_hash",
                    record_id=record.record_id,
                ))

            # 11. Provenance validity.
            if record.provenance not in TransitionProvenance:
                issues.append(ValidationIssue(
                    severity="error",
                    category="invalid_provenance",
                    message=f"Invalid provenance: {record.provenance}",
                    record_id=record.record_id,
                ))

            # 12. Authorization decision validity.
            if record.authorization_decision not in AuthorizationDecision:
                issues.append(ValidationIssue(
                    severity="error",
                    category="invalid_authorization",
                    message=f"Invalid authorization_decision: {record.authorization_decision}",
                    record_id=record.record_id,
                ))

        n_errors = sum(1 for i in issues if i.severity == "error")
        n_warnings = sum(1 for i in issues if i.severity == "warning")
        return ValidationResult(
            valid=n_errors == 0,
            n_records=len(records),
            n_errors=n_errors,
            n_warnings=n_warnings,
            issues=issues,
        )

    def validate_json(self, json_data: dict[str, Any]) -> ValidationResult:
        """Validate raw JSON data (before deserialization)."""
        issues: list[ValidationIssue] = []

        # Check required top-level keys.
        required_keys = {"metadata", "records"}
        if not required_keys.issubset(json_data.keys()):
            missing = required_keys - set(json_data.keys())
            issues.append(ValidationIssue(
                severity="error",
                category="schema",
                message=f"Missing top-level keys: {missing}",
            ))
            return ValidationResult(
                valid=False, n_records=0, n_errors=1, n_warnings=0, issues=issues,
            )

        # Check schema version.
        meta = json_data.get("metadata", {})
        schema = meta.get("schema_version", "")
        if "V6_0_EXP2" not in schema:
            issues.append(ValidationIssue(
                severity="error",
                category="schema",
                message=f"Unexpected schema_version: {schema}",
            ))

        n_errors = sum(1 for i in issues if i.severity == "error")
        return ValidationResult(
            valid=n_errors == 0,
            n_records=len(json_data.get("records", [])),
            n_errors=n_errors,
            n_warnings=0,
            issues=issues,
        )
