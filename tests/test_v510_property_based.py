"""v5.10 Phase 38: property-based testing with Hypothesis.

Property-based tests verify invariants hold for arbitrary valid inputs,
not just hand-picked examples. These complement the example-based tests.
"""
from __future__ import annotations

import hashlib
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from lgae_v3.executive import StructuralAction
from lgae_v3.runtime import (
    CandidateUnion, candidate_id, build_candidate_union,
    MutationImpact, DiagnosticLevel, DiagnosticEscalationPolicy,
    CertificationLevel, CertificationResult,
)
from lgae_v3.mutations import MutationAuthorityLevel


# ---------------------------------------------------------------------------
# CandidateID properties
# ---------------------------------------------------------------------------

@given(
    state_id=st.text(min_size=1, max_size=50),
    u=st.integers(min_value=0, max_value=1000),
    v=st.integers(min_value=0, max_value=1000),
)
def test_candidate_id_always_64_hex_chars(state_id, u, v):
    cid = candidate_id(state_id, StructuralAction.ADD_EDGE, {"u": u, "v": v})
    assert len(cid) == 64
    int(cid, 16)  # valid hex


@given(
    state_id=st.text(min_size=1, max_size=50),
    u=st.integers(min_value=0, max_value=1000),
    v=st.integers(min_value=0, max_value=1000),
)
def test_candidate_id_endpoint_order_invariant(state_id, u, v):
    assume(u != v)
    cid1 = candidate_id(state_id, StructuralAction.ADD_EDGE, {"u": u, "v": v})
    cid2 = candidate_id(state_id, StructuralAction.ADD_EDGE, {"u": v, "v": u})
    assert cid1 == cid2  # endpoint order does not change the id


@given(
    state_id=st.text(min_size=1, max_size=50),
    factor=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_candidate_id_quantization_invariance(state_id, factor):
    # Values that differ only beyond 6 decimal places produce the same id.
    # Use a delta small enough that round(x, 6) == round(x + delta, 6).
    cid1 = candidate_id(state_id, StructuralAction.REWEIGHT_AFFINITY, {"u": 0, "v": 1, "factor": factor})
    # Quantize first, then add a sub-quantization delta.
    q = round(factor, 6)
    cid2 = candidate_id(state_id, StructuralAction.REWEIGHT_AFFINITY, {"u": 0, "v": 1, "factor": q + 1e-9})
    assert cid1 == cid2 or round(factor, 6) == q + 1e-9  # either same id or same quantized value


# ---------------------------------------------------------------------------
# MutationImpact properties
# ---------------------------------------------------------------------------

@given(
    topology=st.booleans(),
    weights=st.booleans(),
    metric=st.booleans(),
    gauges=st.booleans(),
    fibers=st.booleans(),
    latents=st.booleans(),
    roles=st.booleans(),
)
def test_mutation_impact_roundtrip(topology, weights, metric, gauges, fibers, latents, roles):
    mi = MutationImpact(topology=topology, weights=weights, metric=metric,
                        gauges=gauges, fibers=fibers, latents=latents, roles=roles)
    ck = mi.to_change_kind()
    mi2 = MutationImpact.from_change_kind(ck)
    assert mi2 == mi


@given(
    topology=st.booleans(),
    weights=st.booleans(),
    metric=st.booleans(),
    gauges=st.booleans(),
    fibers=st.booleans(),
    latents=st.booleans(),
    roles=st.booleans(),
)
def test_mutation_impact_is_empty_iff_all_false(topology, weights, metric, gauges, fibers, latents, roles):
    mi = MutationImpact(topology=topology, weights=weights, metric=metric,
                        gauges=gauges, fibers=fibers, latents=latents, roles=roles)
    assert mi.is_empty == (not any([topology, weights, metric, gauges, fibers, latents, roles]))


# ---------------------------------------------------------------------------
# DiagnosticEscalationPolicy properties
# ---------------------------------------------------------------------------

@given(
    risk=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    uncertainty=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    disagreement=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_diagnostic_level_monotonic_in_risk(risk, uncertainty, disagreement):
    p = DiagnosticEscalationPolicy()
    level = p.level_for(risk=risk, uncertainty=uncertainty, disagreement=disagreement)
    # Higher risk should never produce a lower level.
    level_higher_risk = p.level_for(risk=min(1.0, risk + 0.1), uncertainty=uncertainty, disagreement=disagreement)
    assert level_higher_risk >= level


@given(
    risk=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_irreversible_authority_always_forces_l3(risk):
    p = DiagnosticEscalationPolicy()
    level = p.level_for(risk=risk, authority=MutationAuthorityLevel.IRREVERSIBLE)
    assert level == DiagnosticLevel.L3_EXACT


# ---------------------------------------------------------------------------
# CertificationResult properties
# ---------------------------------------------------------------------------

@given(
    level=st.sampled_from(CertificationLevel),
    passed=st.booleans(),
    coverage=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_certification_result_to_log_roundtrip(level, passed, coverage):
    r = CertificationResult(level=level, passed=passed, coverage=coverage)
    log = r.to_log()
    assert log["level_name"] == level.name
    assert log["passed"] == passed
    assert log["coverage"] == coverage
    assert log["is_exact"] == r.is_exact
    assert log["is_global"] == r.is_global


@given(
    level=st.sampled_from([l for l in CertificationLevel if l < CertificationLevel.CERTIFIED_LOCAL]),
)
def test_certification_result_assert_exact_rejects_non_exact(level):
    r = CertificationResult(level=level, passed=True)
    from lgae_v3.runtime import CertificationError
    with pytest.raises(CertificationError):
        r.assert_exact()


# ---------------------------------------------------------------------------
# CandidateUnion properties
# ---------------------------------------------------------------------------

@given(
    n_channels=st.integers(min_value=1, max_value=5),
    n_per_channel=st.integers(min_value=0, max_value=5),
)
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
def test_candidate_union_size_never_exceeds_total(n_channels, n_per_channel):
    from lgae_v3.reasoning import ConcreteAction
    channels = {}
    for i in range(n_channels):
        cands = [
            ConcreteAction(StructuralAction.ADD_EDGE, {"u": j, "v": j + 100}, channel=f"ch{i}")
            for j in range(n_per_channel)
        ]
        channels[f"ch{i}"] = cands
    union = build_candidate_union("state", channels=channels)
    # Union size <= total candidates + NO_OP.
    total = n_channels * n_per_channel
    assert union.size <= total + 1  # +1 for NO_OP
    # Union size >= 1 (always has NO_OP).
    assert union.size >= 1


@given(
    seed=st.integers(min_value=0, max_value=10000),
)
def test_candidate_union_order_is_always_sorted(seed):
    from lgae_v3.reasoning import ConcreteAction
    import random
    rng = random.Random(seed)
    channels = {}
    for i in range(4):
        n = rng.randint(0, 5)
        channels[f"ch{i}"] = [
            ConcreteAction(StructuralAction.ADD_EDGE, {"u": rng.randint(0, 50), "v": rng.randint(51, 100)}, channel=f"ch{i}")
            for _ in range(n)
        ]
    union = build_candidate_union("state", channels=channels)
    ids = [c.id for c in union.candidates()]
    assert ids == sorted(ids)
