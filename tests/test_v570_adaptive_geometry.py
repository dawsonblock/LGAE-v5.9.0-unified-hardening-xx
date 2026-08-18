import torch
from lgae_v3.adaptive_geometry import (
    DependencyRegistry, monitor_orthogonality, AdaptiveCurvatureCascade,
    CascadePolicy, CurvatureStage,
)


def test_operator_footprints_derive_local_and_global_cache_scope():
    r = DependencyRegistry()
    assert r.cache_dependency("forman").radius == 1
    assert r.cache_dependency("lly").radius == 2
    assert r.cache_dependency("ollivier_sinkhorn").radius == 2
    assert r.cache_dependency("spectral_gap").radius is None


def test_orthogonality_monitor_repairs_only_above_threshold():
    I = torch.eye(3)
    assert monitor_orthogonality(I).action == "healthy"
    W = I.clone(); W[0, 0] = 1.1
    health = monitor_orthogonality(W, warn_threshold=1e-6, repair_threshold=1e-4)
    assert health.action == "repaired"
    assert health.repaired is not None
    assert torch.allclose(health.repaired.T @ health.repaired, I, atol=1e-5)


def test_cascade_stops_at_cheap_operator_when_confident():
    called = []
    def ev(stage, value, ambiguity):
        def f(): called.append(stage); return value, ambiguity
        return f
    c = AdaptiveCurvatureCascade({
        CurvatureStage.FORM: ev("f", .2, .05),
        CurvatureStage.LLY: ev("l", .2, .05),
        CurvatureStage.SINKHORN: ev("s", .2, .05),
        CurvatureStage.EXACT: ev("e", .2, 0),
    })
    out = c.evaluate(risk=.1)
    assert out.selected.stage is CurvatureStage.FORM
    assert called == ["f"]


def test_cascade_escalates_ambiguity_and_high_risk_to_exact():
    def f(v, a): return lambda: (v, a)
    c = AdaptiveCurvatureCascade({
        CurvatureStage.FORM: f(.1, .8),
        CurvatureStage.LLY: f(.15, .5),
        CurvatureStage.SINKHORN: f(.17, .1),
        CurvatureStage.EXACT: f(.18, 0),
    }, CascadePolicy(exact_risk_threshold=.9))
    assert c.evaluate(risk=.2).selected.stage is CurvatureStage.SINKHORN
    assert c.evaluate(risk=.99).selected.stage is CurvatureStage.EXACT


def test_require_exact_always_reaches_reference():
    def f(): return .1, 0.0
    c = AdaptiveCurvatureCascade({s: f for s in CurvatureStage})
    out = c.evaluate(require_exact=True)
    assert out.selected.stage is CurvatureStage.EXACT
    assert len(out.evaluations) == 4
