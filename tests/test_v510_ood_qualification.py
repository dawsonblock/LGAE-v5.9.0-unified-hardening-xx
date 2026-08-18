"""v5.10 Phase 25: true OOD qualification tests."""
from __future__ import annotations

import pytest

from lgae_v3.runtime import (
    GraphFamily, CurriculumEntry, OODEvaluationResult, OODQualificationReport,
    evaluate_ood, to_scientific_report,
)


def test_ood_evaluation_result_to_log():
    r = OODEvaluationResult(family=GraphFamily.PATH, seed=0, n_nodes=10, metric_value=0.5, is_held_out=True)
    log = r.to_log()
    assert log["family"] == "path"
    assert log["is_held_out"] is True
    assert log["metric_value"] == 0.5


def test_ood_report_sigma_computation():
    report = OODQualificationReport()
    # ID results with low variance.
    for v in [0.1, 0.11, 0.12]:
        report.id_results.append(OODEvaluationResult(GraphFamily.PATH, 0, 10, v, False))
    # OOD results with higher variance.
    for v in [0.1, 0.3, 0.5]:
        report.ood_results.append(OODEvaluationResult(GraphFamily.BIPARTITE, 0, 10, v, True))
    assert report.sigma_ood > report.sigma_id
    assert report.ood_gate_passed


def test_ood_report_gate_fails_when_id_has_higher_variance():
    report = OODQualificationReport()
    for v in [0.1, 0.5, 0.9]:
        report.id_results.append(OODEvaluationResult(GraphFamily.PATH, 0, 10, v, False))
    for v in [0.1, 0.11, 0.12]:
        report.ood_results.append(OODEvaluationResult(GraphFamily.BIPARTITE, 0, 10, v, True))
    assert not report.ood_gate_passed  # sigma_id > sigma_ood


def test_evaluate_ood_with_simple_metric():
    # Metric: number of edges (a simple structural property).
    def metric_fn(graph, entry):
        return float(int(graph.valid.sum()))

    report = evaluate_ood(
        metric_fn=metric_fn,
        n_nodes=10,
        n_seeds=2,
    )
    assert len(report.id_results) > 0
    assert len(report.ood_results) > 0
    # All results should have positive metric values (graphs have edges).
    assert all(r.metric_value > 0 for r in report.id_results + report.ood_results)


def test_evaluate_ood_id_and_ood_are_disjoint_families():
    def metric_fn(graph, entry):
        return 1.0

    report = evaluate_ood(
        metric_fn=metric_fn,
        n_nodes=8,
        n_seeds=1,
    )
    id_families = set(r.family for r in report.id_results)
    ood_families = set(r.family for r in report.ood_results)
    assert id_families.isdisjoint(ood_families)


def test_to_scientific_report_converts_ood_to_scientific_format():
    report = OODQualificationReport()
    for seed in [0, 1, 2]:
        report.id_results.append(OODEvaluationResult(GraphFamily.PATH, seed, 10, 0.1 + seed * 0.01, False))
        report.ood_results.append(OODEvaluationResult(GraphFamily.BIPARTITE, seed, 10, 0.3 + seed * 0.05, True))
    sci = to_scientific_report(report)
    assert sci.sigma_ood is not None
    assert sci.sigma_id is not None
    assert sci.sigma_ood.n_seeds == 3
    assert sci.sigma_id.n_seeds == 3


def test_ood_report_to_log_structure():
    report = OODQualificationReport()
    report.id_results.append(OODEvaluationResult(GraphFamily.PATH, 0, 10, 0.1, False))
    report.ood_results.append(OODEvaluationResult(GraphFamily.BIPARTITE, 0, 10, 0.3, True))
    log = report.to_log()
    assert "sigma_id" in log
    assert "sigma_ood" in log
    assert "ood_gate_passed" in log
    assert len(log["id_results"]) == 1
    assert len(log["ood_results"]) == 1


def test_ood_report_mean_computation():
    report = OODQualificationReport()
    for v in [0.1, 0.2, 0.3]:
        report.id_results.append(OODEvaluationResult(GraphFamily.PATH, 0, 10, v, False))
    assert report.mean_id == pytest.approx(0.2, abs=1e-9)
