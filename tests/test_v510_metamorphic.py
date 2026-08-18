"""v5.10 Phase 39: metamorphic testing.

Metamorphic tests verify that a transformation of the input produces a
predictable transformation of the output. Unlike property-based tests
(which check invariants), metamorphic tests check *relations* between
outputs for related inputs.

Metamorphic relations tested:
  1. CandidateID is invariant under endpoint swap (symmetry).
  2. MutationImpact.to_change_kind is idempotent.
  3. CandidateUnion order is invariant under channel insertion order.
  4. DiagnosticEscalationPolicy level is monotonic under risk increase.
  5. CertificationLevel ordering is preserved under from_legacy/to_legacy.
  6. CandidateUnion dedup is idempotent (dedup twice = dedup once).
"""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st, assume

from lgae_v3.executive import StructuralAction
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.runtime import (
    candidate_id, build_candidate_union, MutationImpact,
    DiagnosticLevel, DiagnosticEscalationPolicy, CertificationLevel,
)
from lgae_v3.mutations import MutationAuthorityLevel
from lgae_v3.types import CertificationLevel as LegacyCertificationLevel


# MR1: CandidateID is invariant under endpoint swap.
@given(
    state_id=st.text(min_size=1, max_size=20),
    u=st.integers(min_value=0, max_value=500),
    v=st.integers(min_value=0, max_value=500),
)
def test_mr1_candidate_id_endpoint_swap_invariant(state_id, u, v):
    assume(u != v)
    cid_original = candidate_id(state_id, StructuralAction.ADD_EDGE, {"u": u, "v": v})
    cid_swapped = candidate_id(state_id, StructuralAction.ADD_EDGE, {"u": v, "v": u})
    assert cid_original == cid_swapped


# MR2: MutationImpact.to_change_kind is idempotent.
@given(
    topology=st.booleans(), weights=st.booleans(), metric=st.booleans(),
    gauges=st.booleans(), fibers=st.booleans(), latents=st.booleans(), roles=st.booleans(),
)
def test_mr2_mutation_impact_change_kind_idempotent(topology, weights, metric, gauges, fibers, latents, roles):
    mi = MutationImpact(topology=topology, weights=weights, metric=metric,
                        gauges=gauges, fibers=fibers, latents=latents, roles=roles)
    ck1 = mi.to_change_kind()
    mi2 = MutationImpact.from_change_kind(ck1)
    ck2 = mi2.to_change_kind()
    assert ck1 == ck2


# MR3: CandidateUnion order is invariant under channel insertion order.
@given(seed=st.integers(min_value=0, max_value=10000))
def test_mr3_candidate_union_order_invariant_under_channel_permutation(seed):
    import random
    rng = random.Random(seed)
    channels = {}
    for i in range(4):
        n = rng.randint(1, 5)
        channels[f"ch{i}"] = [
            ConcreteAction(StructuralAction.ADD_EDGE, {"u": rng.randint(0, 50), "v": rng.randint(51, 100)})
            for _ in range(n)
        ]
    # Build with original order.
    union1 = build_candidate_union("state", channels=channels)
    # Build with reversed channel order.
    reversed_channels = dict(reversed(list(channels.items())))
    union2 = build_candidate_union("state", channels=reversed_channels)
    # The output order (sorted by id) must be identical.
    assert [c.id for c in union1.candidates()] == [c.id for c in union2.candidates()]


# MR4: Diagnostic level is monotonic under risk increase.
@given(
    risk=st.floats(min_value=0.0, max_value=0.9, allow_nan=False),
    uncertainty=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    disagreement=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    delta=st.floats(min_value=0.01, max_value=0.1, allow_nan=False),
)
def test_mr4_diagnostic_level_monotonic_under_risk_increase(risk, uncertainty, disagreement, delta):
    p = DiagnosticEscalationPolicy()
    level_low = p.level_for(risk=risk, uncertainty=uncertainty, disagreement=disagreement)
    level_high = p.level_for(risk=min(1.0, risk + delta), uncertainty=uncertainty, disagreement=disagreement)
    assert level_high >= level_low


# MR5: CertificationLevel from_legacy/to_legacy is a valid roundtrip.
@given(
    lvl=st.sampled_from(LegacyCertificationLevel),
)
def test_mr5_certification_level_legacy_roundtrip(lvl):
    c = CertificationLevel.from_legacy(lvl)
    assert c.to_legacy() == lvl


# MR6: CandidateUnion dedup is idempotent.
@given(seed=st.integers(min_value=0, max_value=10000))
def test_mr6_candidate_union_dedup_idempotent(seed):
    import random
    rng = random.Random(seed)
    channels = {}
    for i in range(3):
        n = rng.randint(1, 4)
        channels[f"ch{i}"] = [
            ConcreteAction(StructuralAction.ADD_EDGE, {"u": rng.randint(0, 20), "v": rng.randint(21, 40)})
            for _ in range(n)
        ]
    union1 = build_candidate_union("state", channels=channels)
    # Re-adding the same candidates to a new union should not change the set.
    cands = union1.concrete_candidates()
    union2 = build_candidate_union("state", channels={"replay": cands})
    assert union1.size == union2.size
    assert set(c.id for c in union1.candidates()) == set(c.id for c in union2.candidates())


# MR7: Authority-driven minimum is always >= risk-driven level.
@given(
    risk=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    authority=st.sampled_from(MutationAuthorityLevel),
)
def test_mr7_authority_min_dominates_risk(risk, authority):
    p = DiagnosticEscalationPolicy()
    level_with_authority = p.level_for(risk=risk, authority=authority)
    level_without_authority = p.level_for(risk=risk, authority=MutationAuthorityLevel.REVERSIBLE)
    assert level_with_authority >= level_without_authority
