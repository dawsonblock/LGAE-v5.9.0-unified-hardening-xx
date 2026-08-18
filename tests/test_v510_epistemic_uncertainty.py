"""v5.10 Phase 11: epistemic uncertainty correction tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    EpistemicUncertaintyEstimate,
    compute_ensemble_uncertainty, compute_distance_penalty,
    estimate_epistemic_uncertainty, verify_ood_uncertainty_property,
)


def test_ensemble_uncertainty_zero_for_single_member():
    outputs = torch.zeros(1, 5)
    assert compute_ensemble_uncertainty(outputs) == 0.0


def test_ensemble_uncertainty_positive_for_disagreement():
    outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sigma = compute_ensemble_uncertainty(outputs)
    assert sigma > 0


def test_ensemble_uncertainty_zero_for_agreement():
    outputs = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    sigma = compute_ensemble_uncertainty(outputs)
    assert sigma == 0.0


def test_distance_penalty_zero_without_training_embeddings():
    sigma, is_ood = compute_distance_penalty(torch.zeros(3), None)
    assert sigma == 0.0
    assert is_ood is False


def test_distance_penalty_zero_for_close_input():
    train = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    inp = torch.tensor([0.0, 0.0])
    sigma, is_ood = compute_distance_penalty(inp, train, threshold=0.5)
    assert sigma == 0.0  # distance is 0
    assert is_ood is False


def test_distance_penalty_positive_for_far_input():
    train = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    inp = torch.tensor([5.0, 5.0])
    sigma, is_ood = compute_distance_penalty(inp, train, threshold=2.0)
    assert sigma > 0
    assert is_ood is True


def test_estimate_epistemic_uncertainty_combines_both():
    ensemble = torch.tensor([[1.0], [2.0], [3.0]])  # some disagreement
    train = torch.tensor([[0.0, 0.0]])
    inp = torch.tensor([10.0, 10.0])  # far from training
    est = estimate_epistemic_uncertainty(
        ensemble_outputs=ensemble, input_embedding=inp,
        training_embeddings=train, ood_threshold=2.0,
    )
    assert est.sigma_ensemble > 0
    assert est.sigma_distance > 0
    assert est.sigma_total == max(est.sigma_ensemble, est.sigma_distance)
    assert est.is_ood is True


def test_estimate_epistemic_uncertainty_ensemble_only():
    ensemble = torch.tensor([[1.0], [2.0], [3.0]])
    est = estimate_epistemic_uncertainty(ensemble_outputs=ensemble)
    assert est.sigma_ensemble > 0
    assert est.sigma_distance == 0.0
    assert est.sigma_total == est.sigma_ensemble
    assert est.is_ood is False


def test_verify_ood_uncertainty_property_holds():
    id_estimates = [
        EpistemicUncertaintyEstimate(0.1, 0.0, 0.1, False),
        EpistemicUncertaintyEstimate(0.2, 0.0, 0.2, False),
    ]
    ood_estimates = [
        EpistemicUncertaintyEstimate(0.1, 0.5, 0.5, True),
        EpistemicUncertaintyEstimate(0.1, 0.8, 0.8, True),
    ]
    assert verify_ood_uncertainty_property(id_estimates, ood_estimates)


def test_verify_ood_uncertainty_property_fails():
    id_estimates = [
        EpistemicUncertaintyEstimate(0.5, 0.0, 0.5, False),
    ]
    ood_estimates = [
        EpistemicUncertaintyEstimate(0.1, 0.0, 0.1, True),  # OOD has lower sigma!
    ]
    assert not verify_ood_uncertainty_property(id_estimates, ood_estimates)


def test_verify_ood_uncertainty_property_empty():
    assert not verify_ood_uncertainty_property([], [])
    assert not verify_ood_uncertainty_property(
        [EpistemicUncertaintyEstimate(0.1, 0, 0.1, False)], [],
    )


def test_estimate_to_log():
    e = EpistemicUncertaintyEstimate(0.1, 0.5, 0.5, True)
    log = e.to_log()
    assert log["sigma_ensemble"] == 0.1
    assert log["sigma_distance"] == 0.5
    assert log["sigma_total"] == 0.5
    assert log["is_ood"] is True
