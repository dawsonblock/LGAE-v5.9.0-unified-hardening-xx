"""v5.10 Phase 13: information-directed exploration tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    InformationGainEstimate, InformationDirectedReport,
    ensemble_disagreement_ig, ucb_ig, posterior_variance_reduction_ig,
    compute_ig_correlation, select_information_directed,
)


def test_ensemble_disagreement_ig():
    ids = ["c1", "c2", "c3"]
    ensemble = torch.tensor([[0.9, 0.1, 0.5], [0.5, 0.5, 0.5], [0.1, 0.9, 0.5]])
    learned = torch.tensor([0.8, 0.5, 0.3])
    estimates = ensemble_disagreement_ig(
        candidate_ids=ids, ensemble_scores=ensemble, learned_scores=learned,
    )
    assert len(estimates) == 3
    # c1 and c3 have high disagreement; c2 has zero disagreement.
    assert estimates[0].epistemic_uncertainty > 0
    assert estimates[1].epistemic_uncertainty == pytest.approx(0.0, abs=1e-6)
    assert estimates[2].epistemic_uncertainty > 0


def test_ensemble_disagreement_ig_empty():
    estimates = ensemble_disagreement_ig(
        candidate_ids=[], ensemble_scores=torch.zeros(0, 3), learned_scores=torch.zeros(0),
    )
    assert estimates == []


def test_ucb_ig():
    ids = ["c1", "c2"]
    mean = torch.tensor([0.5, 0.3])
    uncertainty = torch.tensor([0.1, 0.5])
    estimates = ucb_ig(candidate_ids=ids, mean_scores=mean, uncertainty=uncertainty, exploration_weight=2.0)
    assert len(estimates) == 2
    # c2 has higher uncertainty, so higher bonus.
    assert estimates[1].exploration_bonus > estimates[0].exploration_bonus
    # total = mean + 2 * uncertainty
    assert estimates[0].total_score == pytest.approx(0.5 + 2 * 0.1)
    assert estimates[1].total_score == pytest.approx(0.3 + 2 * 0.5)


def test_posterior_variance_reduction_ig():
    ids = ["c1", "c2"]
    prior_var = torch.tensor([0.5, 0.8])
    post_var = torch.tensor([0.3, 0.2])  # c2 reduces more
    learned = torch.tensor([0.4, 0.6])
    estimates = posterior_variance_reduction_ig(
        candidate_ids=ids, prior_variance=prior_var,
        expected_posterior_variance=post_var, learned_scores=learned,
    )
    assert estimates[0].predicted_ig == pytest.approx(0.2)  # 0.5 - 0.3
    assert estimates[1].predicted_ig == pytest.approx(0.6)  # 0.8 - 0.2


def test_posterior_variance_reduction_clamps_negative():
    ids = ["c1"]
    prior_var = torch.tensor([0.3])
    post_var = torch.tensor([0.5])  # variance increases (negative IG)
    learned = torch.tensor([0.5])
    estimates = posterior_variance_reduction_ig(
        candidate_ids=ids, prior_variance=prior_var,
        expected_posterior_variance=post_var, learned_scores=learned,
    )
    assert estimates[0].predicted_ig == 0.0  # clamped to non-negative


def test_compute_ig_correlation_positive():
    pred = [1.0, 2.0, 3.0, 4.0]
    real = [1.5, 2.5, 3.5, 4.5]  # perfectly correlated
    rho = compute_ig_correlation(pred, real)
    assert rho > 0.99


def test_compute_ig_correlation_negative():
    pred = [1.0, 2.0, 3.0, 4.0]
    real = [4.0, 3.0, 2.0, 1.0]  # perfectly anti-correlated
    rho = compute_ig_correlation(pred, real)
    assert rho < -0.99


def test_compute_ig_correlation_zero():
    pred = [1.0, 2.0, 3.0, 4.0]
    real = [2.0, 2.0, 2.0, 2.0]  # no variance in real
    rho = compute_ig_correlation(pred, real)
    assert rho == 0.0


def test_compute_ig_correlation_short_lists():
    assert compute_ig_correlation([1.0], [1.0]) == 0.0
    assert compute_ig_correlation([], []) == 0.0


def test_select_information_directed():
    estimates = [
        InformationGainEstimate("c1", 0.1, 0.1, 0.0, 0.1, 0.9),
        InformationGainEstimate("c2", 0.5, 0.5, 0.0, 0.5, 1.2),
        InformationGainEstimate("c3", 0.0, 0.0, 0.0, 0.0, 0.5),
    ]
    chosen = select_information_directed(estimates)
    assert chosen == "c2"  # highest total_score


def test_select_information_directed_empty():
    assert select_information_directed([]) == ""


def test_information_gain_estimate_to_log():
    e = InformationGainEstimate("c1", 0.5, 0.3, 0.1, 0.2, 1.0)
    log = e.to_log()
    assert log["candidate_id"] == "c1"
    assert log["predicted_ig"] == 0.5
    assert log["total_score"] == 1.0
