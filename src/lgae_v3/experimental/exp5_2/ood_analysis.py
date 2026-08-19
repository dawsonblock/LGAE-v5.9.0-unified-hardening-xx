"""OOD distance computation for exp5.2.

Computes structural distance between graph families to understand
why cross-family generalization fails. Correlates OOD distance with
prediction error and ensemble uncertainty.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import math

from .state_encoding import encode_normalized_state, NORM_STATE_DIM


def compute_family_statistics(
    records: list[Any],
    *,
    graphs: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Compute mean state vector per graph family.

    Returns:
        dict mapping family name to mean normalized state vector.
    """
    family_vectors: dict[str, list[np.ndarray]] = {}
    for r in records:
        fam = getattr(r, "graph_family", "unknown")
        state = r.structural_state_before
        sv = encode_normalized_state(state)
        family_vectors.setdefault(fam, []).append(sv.vector)

    return {
        fam: np.mean(vecs, axis=0)
        for fam, vecs in family_vectors.items()
        if vecs
    }


def compute_ood_distance(
    test_vector: np.ndarray,
    train_vectors: np.ndarray,
) -> float:
    """Compute OOD distance from a test sample to the training distribution.

    Uses Mahalanobis-style distance with diagonal covariance.
    """
    if len(train_vectors) == 0:
        return 0.0
    mean = train_vectors.mean(axis=0)
    std = train_vectors.std(axis=0)
    std[std < 1e-8] = 1.0
    diff = test_vector - mean
    return float(np.sqrt(np.sum((diff / std) ** 2)))


def compute_family_ood_distances(
    test_records: list[Any],
    train_records: list[Any],
) -> dict[str, Any]:
    """Compute OOD distances for each test record.

    Returns:
        dict with:
        - distances: list of OOD distances per test record
        - errors: list of prediction errors (filled by caller)
        - uncertainties: list of uncertainties (filled by caller)
        - family_distances: mean OOD distance per family
    """
    # Get train state vectors.
    train_vectors = []
    for r in train_records:
        state = r.structural_state_before
        sv = encode_normalized_state(state)
        train_vectors.append(sv.vector)
    train_arr = np.array(train_vectors) if train_vectors else np.zeros((0, NORM_STATE_DIM))

    # Compute distance for each test record.
    distances = []
    families = []
    for r in test_records:
        state = r.structural_state_before
        sv = encode_normalized_state(state)
        d = compute_ood_distance(sv.vector, train_arr)
        distances.append(d)
        families.append(getattr(r, "graph_family", "unknown"))

    # Mean distance per family.
    family_distances: dict[str, float] = {}
    family_list = sorted(set(families))
    for fam in family_list:
        fam_dists = [d for d, f in zip(distances, families) if f == fam]
        family_distances[fam] = float(np.mean(fam_dists)) if fam_dists else 0.0

    return {
        "distances": distances,
        "families": families,
        "family_distances": family_distances,
    }


def correlate_ood_with_error(
    distances: list[float],
    errors: list[float],
) -> dict[str, float]:
    """Correlate OOD distance with prediction error."""
    if len(distances) < 2:
        return {"corr": 0.0, "spearman": 0.0}

    d_arr = np.array(distances)
    e_arr = np.array(errors)

    if np.std(d_arr) < 1e-10 or np.std(e_arr) < 1e-10:
        return {"corr": 0.0, "spearman": 0.0}

    corr = float(np.corrcoef(d_arr, e_arr)[0, 1])

    try:
        from scipy.stats import spearmanr
        sp, _ = spearmanr(d_arr, e_arr)
        sp = float(sp) if not math.isnan(sp) else 0.0
    except Exception:
        sp = 0.0

    return {"corr": corr, "spearman": sp}


def correlate_ood_with_uncertainty(
    distances: list[float],
    uncertainties: list[float],
) -> dict[str, float]:
    """Correlate OOD distance with ensemble uncertainty."""
    return correlate_ood_with_error(distances, uncertainties)
