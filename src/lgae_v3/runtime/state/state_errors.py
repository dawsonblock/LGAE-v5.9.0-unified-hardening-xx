"""State errors (v5.11 Phase 1).

Error types for state ownership violations.
"""
from __future__ import annotations


class StateOwnershipError(RuntimeError):
    """Raised when code attempts to mutate state without authority capability."""


class StaleStateError(RuntimeError):
    """Raised when a transaction's base state doesn't match current state."""


class StateValidationError(RuntimeError):
    """Raised when a candidate state bundle fails validation."""


class DeterminismError(RuntimeError):
    """Raised when a deterministic path encounters a nondeterministic operation."""


class CapabilityError(RuntimeError):
    """Raised when a mutation method is called without a valid capability token."""
