"""Phases 12-15 tests: Structural Diagnosis, Attention Budget, and Multi-Fidelity Funnel."""
from __future__ import annotations

import pytest
import torch

from lgae_v3 import ResearchConfig, make_graph_buffers
from lgae_v3.runtime import LGAERuntime, RuntimeConfig
from lgae_v3.runtime.structural_diagnosis import (
    StructuralDiagnoser,
    DiagnosisType,
    StructuralDiagnosis,
    StructuralAttentionBudget,
)
from lgae_v3.runtime.multi_fidelity import (
    MultiFidelityFunnel,
    EvaluationTier,
    TierFilterResult,
)


def _graph():
    return make_graph_buffers(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], capacity=12)


def _cfg():
    cfg = ResearchConfig()
    cfg.fiber.d_base = 2
    cfg.fiber.d_max = 6
    cfg.fiber.spawn_width = 1
    cfg.audit.persistent_homology_enabled = False
    cfg.audit.entropic_nodes = 0
    cfg.audit.bakry_nodes = 0
    cfg.audit.cde_nodes = 0
    cfg.audit.exact_lly_top_k = 0
    cfg.audit.orc_top_k = 0
    cfg.mutation.shadow_horizons = [1, 2]
    cfg.mutation.curvature_ema_enabled = False
    return cfg


def test_structural_diagnoser_detects_deficits():
    diagnoser = StructuralDiagnoser()
    g = _graph()
    
    class MockAudit:
        spectral_gap = 0.05
        ricci_min = -0.7

    diagnoses = diagnoser.diagnose(g, MockAudit(), epistemic_uncertainty=0.6)
    diag_types = {d.diagnosis_type for d in diagnoses}
    
    assert DiagnosisType.OVERSQUASHING in diag_types
    assert DiagnosisType.STRUCTURAL_BOTTLENECK in diag_types
    assert DiagnosisType.HIGH_EPISTEMIC_REGION in diag_types
    
    for d in diagnoses:
        assert 0.0 <= d.severity <= 1.0
        assert 0.0 <= d.confidence <= 1.0
        assert isinstance(d.evidence, dict)


def test_structural_attention_budget():
    budget = StructuralAttentionBudget(alpha_severity=0.5, beta_uncertainty=0.3, gamma_utility=0.2)
    p_high = budget.compute_region_priority(severity=0.9, uncertainty=0.8, utility_impact=0.7)
    p_low = budget.compute_region_priority(severity=0.1, uncertainty=0.1, utility_impact=0.0)
    assert p_high > p_low
    assert p_high > 0.8
    assert p_low < 0.2


def test_multi_fidelity_funnel_rejects_illegal_tier0():
    funnel = MultiFidelityFunnel()
    g = _graph()
    
    res_self_loop = funnel.filter_tier0_legality(g, "ADD_EDGE", {"u": 2, "v": 2})
    assert not res_self_loop.passed
    assert res_self_loop.tier == EvaluationTier.TIER_0_LEGALITY
    
    res_valid = funnel.filter_tier0_legality(g, "ADD_EDGE", {"u": 0, "v": 3})
    assert res_valid.passed


def test_multi_fidelity_funnel_reduces_candidates():
    funnel = MultiFidelityFunnel()
    g = _graph()
    candidates = [
        {"action_type": "ADD_EDGE", "parameters": {"u": 0, "v": 0}},  # illegal self loop
        {"action_type": "ADD_EDGE", "parameters": {"u": 0, "v": 2}},
        {"action_type": "ADD_EDGE", "parameters": {"u": 0, "v": 3}},
        {"action_type": "ADD_EDGE", "parameters": {"u": 1, "v": 4}},
        {"action_type": "ADD_EDGE", "parameters": {"u": 1, "v": 1}},  # illegal self loop
        {"action_type": "ADD_EDGE", "parameters": {"u": 2, "v": 5}},
    ]
    filtered = funnel.evaluate_funnel(g, candidates, max_shadow_candidates=3)
    assert len(filtered) <= 3
    for c in filtered:
        assert c["parameters"]["u"] != c["parameters"]["v"]


def test_end_to_end_reasoning_emits_diagnoses():
    torch.manual_seed(42)
    rt = LGAERuntime(_graph(), _cfg())
    res = rt.step()
    assert res.reasoning is not None
    assert hasattr(res.reasoning, "diagnoses")
