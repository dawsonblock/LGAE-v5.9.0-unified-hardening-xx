"""v5.10 Phase 12: uncertainty calibration tests."""
from __future__ import annotations

import pytest
import torch

from lgae_v3.runtime import (
    CalibrationMetrics, expected_calibration_error,
    negative_log_likelihood, brier_score,
    compute_calibration_metrics, is_well_calibrated,
)


def test_ece_perfect_calibration():
    # Perfectly calibrated: confidence = accuracy in every bin.
    confidences = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    correctness = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])  # 90% accuracy
    ece, _, _, _ = expected_calibration_error(confidences, correctness, n_bins=10)
    assert ece < 0.05  # well calibrated


def test_ece_poor_calibration():
    # Poorly calibrated: 90% confident but only 50% accurate.
    confidences = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    correctness = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 50% accuracy
    ece, _, _, _ = expected_calibration_error(confidences, correctness, n_bins=10)
    assert ece > 0.3  # poorly calibrated


def test_ece_empty():
    ece, accs, confs, counts = expected_calibration_error(torch.tensor([]), torch.tensor([]))
    assert ece == 0.0
    assert accs == []


def test_nll_low_for_confident_correct():
    probs = torch.tensor([0.99, 0.99, 0.99])
    nll = negative_log_likelihood(probs)
    assert nll < 0.05  # low NLL for confident correct predictions


def test_nll_high_for_confident_wrong():
    probs = torch.tensor([0.01, 0.01, 0.01])  # low probability for correct class
    nll = negative_log_likelihood(probs)
    assert nll > 4.0  # high NLL for confident wrong predictions


def test_brier_perfect():
    probs = torch.tensor([1.0, 1.0, 1.0])
    outcomes = torch.tensor([1.0, 1.0, 1.0])
    assert brier_score(probs, outcomes) == 0.0


def test_brier_worst():
    probs = torch.tensor([1.0, 1.0, 1.0])
    outcomes = torch.tensor([0.0, 0.0, 0.0])
    assert brier_score(probs, outcomes) == 1.0


def test_brier_partial():
    probs = torch.tensor([0.5, 0.5])
    outcomes = torch.tensor([1.0, 0.0])
    assert brier_score(probs, outcomes) == pytest.approx(0.25)


def test_compute_calibration_metrics_all():
    confidences = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    correctness = torch.tensor([1, 1, 0, 1, 0])
    metrics = compute_calibration_metrics(
        confidences=confidences, correctness=correctness, n_bins=5,
    )
    assert metrics.ece >= 0
    assert metrics.nll > 0
    assert metrics.brier >= 0
    assert metrics.bin_count == 5
    assert len(metrics.bin_accuracies) == 5


def test_compute_calibration_metrics_with_probs():
    confidences = torch.tensor([0.9, 0.8])
    correctness = torch.tensor([1, 0])
    probabilities = torch.tensor([0.9, 0.2])
    outcomes = torch.tensor([1.0, 0.0])
    metrics = compute_calibration_metrics(
        confidences=confidences, correctness=correctness,
        probabilities=probabilities, outcomes=outcomes,
    )
    assert metrics.nll > 0
    assert metrics.brier >= 0


def test_is_well_calibrated_true():
    metrics = CalibrationMetrics(ece=0.05, nll=0.1, brier=0.1, bin_count=10)
    assert is_well_calibrated(metrics, ece_threshold=0.1)


def test_is_well_calibrated_false():
    metrics = CalibrationMetrics(ece=0.2, nll=0.5, brier=0.3, bin_count=10)
    assert not is_well_calibrated(metrics, ece_threshold=0.1)


def test_calibration_metrics_to_log():
    metrics = CalibrationMetrics(
        ece=0.1, nll=0.3, brier=0.2, bin_count=5,
        bin_accuracies=[0.5, 0.6, 0.7, 0.8, 0.9],
        bin_confidences=[0.5, 0.6, 0.7, 0.8, 0.9],
        bin_counts=[10, 10, 10, 10, 10],
    )
    log = metrics.to_log()
    assert log["ece"] == 0.1
    assert log["bin_count"] == 5
    assert len(log["bin_accuracies"]) == 5
