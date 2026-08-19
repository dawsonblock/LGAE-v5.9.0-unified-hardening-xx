"""Experiment state machine for held-out access control.

Enforces the protocol:
    PREPARATION -> TRAINING -> VALIDATION -> MODEL_LOCKED
        -> HELDOUT_OPENED -> FINALIZED

No backward transition from HELDOUT_OPENED. This provides methodological
discipline and auditability — the experiment registry permanently records
when held-out was accessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import time
import hashlib


class ExperimentStateError(Exception):
    """Raised when an invalid state transition is attempted."""


# Ordered states — transitions only allowed in this order.
STATES = [
    "PREPARATION",
    "TRAINING",
    "VALIDATION",
    "MODEL_LOCKED",
    "HELDOUT_OPENED",
    "FINALIZED",
]

# States where model selection / hyperparameter tuning is permitted.
SELECTION_STATES = {"PREPARATION", "TRAINING", "VALIDATION"}

# States where held-out data may be accessed.
HELDOUT_ACCESS_STATES = {"HELDOUT_OPENED", "FINALIZED"}


@dataclass
class ExperimentStateMachine:
    """Enforces the experiment lifecycle protocol.

    The state machine records:
    - current state
    - timestamp of each transition
    - locked model configuration hash
    - held-out access metadata

    Once HELDOUT_OPENED is entered, no backward transition to
    model selection is possible.
    """
    experiment_id: str = "LGAE_V6_EXP4_2_STRUCTURAL_PREDICTION_STUDY_001"
    _state: str = "PREPARATION"
    _transitions: list[dict[str, Any]] = field(default_factory=list)
    _locked_model_config_hash: str = ""
    _heldout_opened_at: str = ""
    _heldout_opened_by_run: str = ""

    def __post_init__(self) -> None:
        if self._state not in STATES:
            raise ExperimentStateError(f"Invalid initial state: {self._state}")
        if not self._transitions:
            self._transitions.append({
                "state": self._state,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "action": "initialized",
            })

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_in_selection_phase(self) -> bool:
        """True if model/hyperparameter selection is still permitted."""
        return self._state in SELECTION_STATES

    @property
    def heldout_accessible(self) -> bool:
        """True if held-out data may be accessed."""
        return self._state in HELDOUT_ACCESS_STATES

    @property
    def locked_model_config_hash(self) -> str:
        return self._locked_model_config_hash

    @property
    def heldout_opened_at(self) -> str:
        return self._heldout_opened_at

    def transition_to(self, new_state: str, *, action: str = "") -> None:
        """Transition to a new state.

        Raises:
            ExperimentStateError: If the transition is invalid.
        """
        if new_state not in STATES:
            raise ExperimentStateError(f"Invalid state: {new_state}")

        current_idx = STATES.index(self._state)
        new_idx = STATES.index(new_state)

        # No backward transitions at all.
        if new_idx < current_idx:
            raise ExperimentStateError(
                f"Cannot transition backward from {self._state} to {new_state}. "
                f"The experiment protocol is strictly forward-only."
            )

        # No skipping forward past MODEL_LOCKED without locking.
        if new_state == "HELDOUT_OPENED":
            if self._state != "MODEL_LOCKED":
                raise ExperimentStateError(
                    f"Cannot open held-out from {self._state}. "
                    f"Must be in MODEL_LOCKED state with locked finalists."
                )
            if not self._locked_model_config_hash:
                raise ExperimentStateError(
                    "Cannot open held-out: no locked model configuration. "
                    "Call lock_finalists() before opening held-out."
                )

        self._state = new_state
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._transitions.append({
            "state": new_state,
            "timestamp": ts,
            "action": action or f"transition_to_{new_state}",
        })

        if new_state == "HELDOUT_OPENED":
            self._heldout_opened_at = ts

    def lock_finalists(self, config_hash: str, *, run_id: str = "") -> None:
        """Lock the finalist configuration before opening held-out.

        Args:
            config_hash: Hash of the locked finalist configurations.
            run_id: Optional run identifier.
        """
        if self._state not in ("VALIDATION", "MODEL_LOCKED"):
            raise ExperimentStateError(
                f"Can only lock finalists during VALIDATION or MODEL_LOCKED, "
                f"not {self._state}"
            )
        if not config_hash:
            raise ExperimentStateError("Config hash must not be empty")
        self._locked_model_config_hash = config_hash
        self._heldout_opened_by_run = run_id
        if self._state == "VALIDATION":
            self.transition_to("MODEL_LOCKED", action="lock_finalists")

    def assert_selection_permitted(self) -> None:
        """Assert that model selection is still permitted."""
        if not self.is_in_selection_phase:
            raise ExperimentStateError(
                f"Model selection is not permitted in state {self._state}. "
                f"Held-out has been opened or finalists are locked."
            )

    def assert_heldout_accessible(self) -> None:
        """Assert that held-out data may be accessed."""
        if not self.heldout_accessible:
            raise ExperimentStateError(
                f"Held-out data is not accessible in state {self._state}. "
                f"Must lock finalists and transition to HELDOUT_OPENED first."
            )

    def to_log(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "state": self._state,
            "locked_model_config_hash": self._locked_model_config_hash,
            "heldout_opened_at": self._heldout_opened_at,
            "heldout_opened_by_run": self._heldout_opened_by_run,
            "transitions": list(self._transitions),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), sort_keys=True, indent=2)
