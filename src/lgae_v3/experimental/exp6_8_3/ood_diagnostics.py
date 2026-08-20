"""OOD (out-of-distribution) diagnostics for exp6.8.3.

Measure whether states far from TRAIN/CALIBRATION distribution get
lower override coverage. If not, consider adding an explicit OOD gate.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def compute_ood_scores(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    """Compute OOD scores for test samples.

    Uses distance to nearest training sample as the OOD score.
    Higher score = more out-of-distribution.
    """
    if len(train_features) == 0 or len(test_features) == 0:
        return np.zeros(len(test_features))

    # Use normalized Euclidean distance to k-th nearest training sample.
    # Normalize features to unit variance.
    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0) + 1e-6
    train_norm = (train_features - train_mean) / train_std
    test_norm = (test_features - train_mean) / train_std

    # For each test sample, compute distance to nearest training sample.
    # Use chunked computation to avoid memory issues.
    ood_scores = []
    k = 5  # 5th nearest neighbor distance
    for i in range(len(test_norm)):
        diffs = train_norm - test_norm[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        sorted_dists = np.sort(dists)
        if len(sorted_dists) >= k:
            ood_scores.append(float(sorted_dists[k]))
        else:
            ood_scores.append(float(sorted_dists[-1]))

    return np.array(ood_scores)


def compute_ood_coverage_analysis(
    ood_scores: np.ndarray,
    used_learned: list[bool],
) -> dict:
    """Analyze whether OOD samples get lower override coverage.

    For a well-calibrated system, samples far from the training
    distribution should get lower coverage (more conservative).
    """
    if len(ood_scores) < 10:
        return {"deciles": [], "is_monotonic": True, "correlation": 0.0}

    used = np.array(used_learned)

    # Sort by OOD score (low = in-distribution, high = OOD).
    order = np.argsort(ood_scores)
    ood_sorted = ood_scores[order]
    used_sorted = used[order]

    n = len(ood_sorted)
    decile_size = max(1, n // 10)

    deciles = []
    for d in range(10):
        start = d * decile_size
        end = min((d + 1) * decile_size, n)
        if start >= n:
            break
        ood_d = ood_sorted[start:end]
        used_d = used_sorted[start:end]
        deciles.append({
            "decile": d,
            "mean_ood_score": float(np.mean(ood_d)),
            "coverage": float(np.mean(used_d)),
            "n": len(used_d),
        })

    # Check: coverage should decrease with OOD score.
    coverages = [d["coverage"] for d in deciles]
    is_monotonic = all(
        coverages[i] >= coverages[i + 1] - 1e-6
        for i in range(len(coverages) - 1)
    ) if len(coverages) > 1 else True

    # Correlation between OOD score and coverage.
    if np.std(ood_scores) > 1e-10 and np.std(used.astype(float)) > 1e-10:
        corr = float(np.corrcoef(ood_scores, used.astype(float))[0, 1])
    else:
        corr = 0.0

    return {
        "deciles": deciles,
        "is_monotonic": is_monotonic,
        "correlation": corr,
    }
