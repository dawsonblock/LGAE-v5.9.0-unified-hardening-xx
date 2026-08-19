"""Dynamics-OOD distance for exp5.3.

Instead of measuring OOD distance in state space (z_t), measure it
in (z_t, a_t, Δz_t) space — the actual transition dynamics.

A test transition is OOD if its (state, action, delta) triple is far
from the training distribution of transitions, not just if the state
is far from training states.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import math


def compute_dynamics_ood_distance(
    test_z: np.ndarray,
    test_a: np.ndarray,
    test_delta: np.ndarray,
    train_z: np.ndarray,
    train_a: np.ndarray,
    train_delta: np.ndarray,
) -> np.ndarray:
    """Compute per-sample dynamics-OOD distance.

    For each test transition (z, a, Δz), compute the minimum distance
    to any training transition in the joint (z, a, Δz) space.

    Uses normalized Euclidean distance per dimension.

    Returns:
        (n_test,) array of OOD distances.
    """
    if len(train_z) == 0 or len(test_z) == 0:
        return np.zeros(len(test_z))

    # Normalize each dimension by training std.
    z_std = np.std(train_z, axis=0)
    z_std[z_std < 1e-8] = 1.0
    a_std = np.std(train_a, axis=0)
    a_std[a_std < 1e-8] = 1.0
    d_std = np.std(train_delta, axis=0)
    d_std[d_std < 1e-8] = 1.0

    # Normalize.
    train_z_n = train_z / z_std
    train_a_n = train_a / a_std
    train_d_n = train_delta / d_std
    test_z_n = test_z / z_std
    test_a_n = test_a / a_std
    test_d_n = test_delta / d_std

    # Compute min distance for each test sample.
    # For efficiency, use batch computation.
    n_test = len(test_z)
    n_train = len(train_z)
    distances = np.zeros(n_test)

    # Process in chunks to avoid memory issues.
    chunk = 100
    for i in range(0, n_test, chunk):
        end = min(i + chunk, n_test)
        # (chunk, 1, dim) - (1, n_train, dim) -> (chunk, n_train, dim)
        dz = test_z_n[i:end, np.newaxis, :] - train_z_n[np.newaxis, :, :]
        da = test_a_n[i:end, np.newaxis, :] - train_a_n[np.newaxis, :, :]
        dd = test_d_n[i:end, np.newaxis, :] - train_d_n[np.newaxis, :, :]
        # Weight: delta is most important (it's what we're predicting).
        dist2 = np.sum(dz ** 2, axis=2) + np.sum(da ** 2, axis=2) + 2.0 * np.sum(dd ** 2, axis=2)
        distances[i:end] = np.sqrt(np.min(dist2, axis=1))

    return distances


def compute_family_dynamics_ood(
    test_z: np.ndarray,
    test_a: np.ndarray,
    test_delta: np.ndarray,
    test_families: list[str],
    train_z: np.ndarray,
    train_a: np.ndarray,
    train_delta: np.ndarray,
) -> dict[str, Any]:
    """Compute dynamics-OOD distances per family.

    Returns:
        dict with:
        - distances: per-sample distances
        - family_distances: mean distance per family
        - families: family labels per sample
    """
    distances = compute_dynamics_ood_distance(
        test_z, test_a, test_delta,
        train_z, train_a, train_delta,
    )

    family_distances: dict[str, float] = {}
    for fam in sorted(set(test_families)):
        mask = [f == fam for f in test_families]
        if any(mask):
            family_distances[fam] = float(np.mean(distances[mask]))

    return {
        "distances": distances.tolist(),
        "family_distances": family_distances,
        "families": test_families,
        "mean_distance": float(np.mean(distances)),
    }


def correlate_dynamics_ood_with_error(
    distances: np.ndarray,
    errors: np.ndarray,
) -> dict[str, float]:
    """Correlate dynamics-OOD distance with prediction error."""
    if len(distances) < 2:
        return {"corr": 0.0, "spearman": 0.0}

    if np.std(distances) < 1e-10 or np.std(errors) < 1e-10:
        return {"corr": 0.0, "spearman": 0.0}

    corr = float(np.corrcoef(distances, errors)[0, 1])

    try:
        from scipy.stats import spearmanr
        sp, _ = spearmanr(distances, errors)
        sp = float(sp) if not math.isnan(sp) else 0.0
    except Exception:
        sp = 0.0

    return {"corr": corr, "spearman": sp}


def correlate_dynamics_ood_with_uncertainty(
    distances: np.ndarray,
    uncertainties: np.ndarray,
) -> dict[str, float]:
    """Correlate dynamics-OOD distance with ensemble uncertainty."""
    return correlate_dynamics_ood_with_error(distances, uncertainties)
