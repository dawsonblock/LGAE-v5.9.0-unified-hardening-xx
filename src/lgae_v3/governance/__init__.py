"""v5.10 governance package: mutation authority policy and invariant contracts.

Phase 7 establishes mutation authority levels as policy rather than scattered
if-statements. Phase 37 will add the formal invariant layer here.
"""
from __future__ import annotations

from .authority_policy import (
    AuthorityRequirement, MutationAuthorityPolicy, DEFAULT_AUTHORITY_POLICY,
    requirement_for, classify_mutation_authority,
)
from .invariants import (
    InvariantRegistry, InvariantResult, InvariantSeverity,
    invariant, register_invariant, DEFAULT_REGISTRY,
)

__all__ = [
    "AuthorityRequirement",
    "MutationAuthorityPolicy",
    "DEFAULT_AUTHORITY_POLICY",
    "requirement_for",
    "classify_mutation_authority",
    "InvariantRegistry",
    "InvariantResult",
    "InvariantSeverity",
    "invariant",
    "register_invariant",
    "DEFAULT_REGISTRY",
]
