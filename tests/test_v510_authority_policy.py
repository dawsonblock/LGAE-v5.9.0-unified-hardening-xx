"""v5.10 Phase 7: mutation authority policy tests."""
from __future__ import annotations

import pytest

from lgae_v3.mutations import (
    MutationAuthorityLevel, mutation_authority_level,
    AddEdge, PruneEdge, ReweightAffinity, RicciFlowReweight,
)
from lgae_v3.governance import (
    AuthorityRequirement, MutationAuthorityPolicy, DEFAULT_AUTHORITY_POLICY,
    requirement_for, classify_mutation_authority,
)
from lgae_v3.runtime import CertificationLevel


def test_high_impact_level_exists_and_orders():
    levels = list(MutationAuthorityLevel)
    assert MutationAuthorityLevel.HIGH_IMPACT in levels
    # Ordering: REVERSIBLE < STRUCTURAL < HIGH_IMPACT < IRREVERSIBLE (by value string is not ordered;
    # policy mapping is explicit).
    assert requirement_for(MutationAuthorityLevel.REVERSIBLE).min_certification_level == CertificationLevel.SAMPLED_LOCAL
    assert requirement_for(MutationAuthorityLevel.STRUCTURAL).min_certification_level == CertificationLevel.CERTIFIED_LOCAL
    assert requirement_for(MutationAuthorityLevel.HIGH_IMPACT).min_certification_level == CertificationLevel.SAMPLED_GLOBAL
    assert requirement_for(MutationAuthorityLevel.IRREVERSIBLE).min_certification_level == CertificationLevel.CERTIFIED_GLOBAL


def test_reversible_allows_sampled_validation():
    req = requirement_for(MutationAuthorityLevel.REVERSIBLE)
    assert req.allows_sampled_validation
    assert not req.requires_global_invariant_check
    assert not req.requires_cryptographic_checkpoint


def test_structural_requires_exact_local():
    req = requirement_for(MutationAuthorityLevel.STRUCTURAL)
    assert not req.allows_sampled_validation
    assert req.min_certification_level == CertificationLevel.CERTIFIED_LOCAL


def test_high_impact_requires_global_invariant_check_and_rollback_plan():
    req = requirement_for(MutationAuthorityLevel.HIGH_IMPACT)
    assert req.requires_global_invariant_check
    assert req.requires_rollback_plan
    assert not req.requires_cryptographic_checkpoint


def test_irreversible_requires_cryptographic_checkpoint():
    req = requirement_for(MutationAuthorityLevel.IRREVERSIBLE)
    assert req.requires_cryptographic_checkpoint
    assert req.requires_rollback_plan
    assert req.requires_global_invariant_check
    assert req.min_certification_level == CertificationLevel.CERTIFIED_GLOBAL


def test_classify_reweight_is_reversible():
    assert classify_mutation_authority(ReweightAffinity(0, 1, factor=1.1)) == MutationAuthorityLevel.REVERSIBLE
    assert classify_mutation_authority(RicciFlowReweight(curvatures={}, target_curvature=0.0, dt=0.01)) == MutationAuthorityLevel.REVERSIBLE


def test_classify_add_edge_is_structural_by_default():
    mut = AddEdge(0, 2)
    assert classify_mutation_authority(mut) == MutationAuthorityLevel.STRUCTURAL


def test_classify_escalates_to_high_impact_when_flagged():
    class _FlaggedAdd:
        name = "add_edge"
        is_high_impact = True
    assert classify_mutation_authority(_FlaggedAdd()) == MutationAuthorityLevel.HIGH_IMPACT

    class _FlaggedPrune:
        name = "prune_edge"
        touches_bridge = True
    assert classify_mutation_authority(_FlaggedPrune()) == MutationAuthorityLevel.HIGH_IMPACT


def test_policy_requirement_for_mutation():
    pol = DEFAULT_AUTHORITY_POLICY
    req = pol.requirement_for_mutation(ReweightAffinity(0, 1, factor=1.1))
    assert req.min_certification_level == CertificationLevel.SAMPLED_LOCAL


def test_policy_summary_is_serializable():
    s = DEFAULT_AUTHORITY_POLICY.to_summary()
    assert "reversible" in s and "high_impact" in s and "irreversible" in s
    assert s["high_impact"]["requires_global_invariant_check"] is True


def test_policy_raises_on_unregistered_level():
    pol = MutationAuthorityPolicy(requirements={})
    with pytest.raises(KeyError):
        pol.requirement_for(MutationAuthorityLevel.STRUCTURAL)
