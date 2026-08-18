"""Mutation authority policy (Phase 7).

Maps each ``MutationAuthorityLevel`` to explicit minimum authorization
requirements, defined as policy rather than scattered if-statements:

  REVERSIBLE   -> sampled validation allowed
  STRUCTURAL   -> exact local verification
  HIGH_IMPACT  -> global invariant check
  IRREVERSIBLE -> global certification + cryptographic checkpoint + rollback plan

The policy is data, not control flow. The governor / runtime consults it to
decide what evidence a proposed mutation must produce before commit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..mutations import MutationAuthorityLevel, mutation_authority_level
from ..runtime.certification import CertificationLevel


@dataclass(frozen=True, slots=True)
class AuthorityRequirement:
    """Minimum authorization requirements for a mutation category."""
    min_certification_level: CertificationLevel
    allows_sampled_validation: bool
    requires_global_invariant_check: bool
    requires_cryptographic_checkpoint: bool
    requires_rollback_plan: bool
    # Maximum fraction of active edges a single mutation may touch.
    max_edge_fraction: float = 1.0


@dataclass(frozen=True, slots=True)
class MutationAuthorityPolicy:
    """Policy mapping authority levels to requirements."""
    requirements: dict[MutationAuthorityLevel, AuthorityRequirement] = field(default_factory=dict)

    def requirement_for(self, level: MutationAuthorityLevel) -> AuthorityRequirement:
        try:
            return self.requirements[level]
        except KeyError as exc:
            raise KeyError(f"no authority requirement registered for {level!r}") from exc

    def requirement_for_mutation(self, mutation: Any) -> AuthorityRequirement:
        return self.requirement_for(classify_mutation_authority(mutation))

    def to_summary(self) -> dict[str, Any]:
        return {
            lvl.value: {
                "min_certification_level": req.min_certification_level.name,
                "allows_sampled_validation": req.allows_sampled_validation,
                "requires_global_invariant_check": req.requires_global_invariant_check,
                "requires_cryptographic_checkpoint": req.requires_cryptographic_checkpoint,
                "requires_rollback_plan": req.requires_rollback_plan,
                "max_edge_fraction": float(req.max_edge_fraction),
            }
            for lvl, req in self.requirements.items()
        }


DEFAULT_AUTHORITY_POLICY = MutationAuthorityPolicy(
    requirements={
        MutationAuthorityLevel.REVERSIBLE: AuthorityRequirement(
            min_certification_level=CertificationLevel.SAMPLED_LOCAL,
            allows_sampled_validation=True,
            requires_global_invariant_check=False,
            requires_cryptographic_checkpoint=False,
            requires_rollback_plan=False,
            max_edge_fraction=0.05,
        ),
        MutationAuthorityLevel.STRUCTURAL: AuthorityRequirement(
            min_certification_level=CertificationLevel.CERTIFIED_LOCAL,
            allows_sampled_validation=False,
            requires_global_invariant_check=False,
            requires_cryptographic_checkpoint=False,
            requires_rollback_plan=False,
            max_edge_fraction=0.1,
        ),
        MutationAuthorityLevel.HIGH_IMPACT: AuthorityRequirement(
            min_certification_level=CertificationLevel.SAMPLED_GLOBAL,
            allows_sampled_validation=False,
            requires_global_invariant_check=True,
            requires_cryptographic_checkpoint=False,
            requires_rollback_plan=True,
            max_edge_fraction=0.25,
        ),
        MutationAuthorityLevel.IRREVERSIBLE: AuthorityRequirement(
            min_certification_level=CertificationLevel.CERTIFIED_GLOBAL,
            allows_sampled_validation=False,
            requires_global_invariant_check=True,
            requires_cryptographic_checkpoint=True,
            requires_rollback_plan=True,
            max_edge_fraction=1.0,
        ),
    }
)


def requirement_for(level: MutationAuthorityLevel) -> AuthorityRequirement:
    """Convenience: the default policy's requirement for a level."""
    return DEFAULT_AUTHORITY_POLICY.requirement_for(level)


def classify_mutation_authority(mutation: Any) -> MutationAuthorityLevel:
    """Classify a mutation, escalating STRUCTURAL to HIGH_IMPACT when the
    mutation touches a bridge or a large fraction of connectivity.

    The base classification comes from ``mutation_authority_level``. A
    structural mutation that targets a bridge edge (detected via the graph's
    bridge finder when a graph is attached to the mutation) is escalated to
    HIGH_IMPACT so the global invariant check is required.
    """
    base = mutation_authority_level(mutation)
    if base != MutationAuthorityLevel.STRUCTURAL:
        return base
    # Escalate structural -> high_impact when the mutation is known to target
    # a bridge. We check for an explicit ``is_high_impact`` flag or a
    # ``touches_bridge`` attribute set by the proposer. This keeps the
    # classifier side-effect-free (no graph traversal here).
    if getattr(mutation, "is_high_impact", False) or getattr(mutation, "touches_bridge", False):
        return MutationAuthorityLevel.HIGH_IMPACT
    return MutationAuthorityLevel.STRUCTURAL
