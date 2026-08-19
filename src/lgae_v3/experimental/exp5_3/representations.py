"""State decomposition and representation ladder for exp5.3.

State is decomposed into:
- z_invariant: topology-invariant features that should transfer across families
  (density, normalized degree, clustering, graphlet frequencies, spectral gap normalized)
- z_context: family-identifying features that may not transfer
  (absolute node/edge counts, component count, fiber counts)
- z_derived: computed features that may or may not transfer
  (log transforms, per-node normalizations)

The representation ladder tests different subsets:
- R0: current normalized (all 20 features)
- R1: graphlet-only (triangle, wedge, 4-cycle, 3-star, clustering, transitivity)
- R2: spectral-only (spectral_gap_normalized, spectral_gap_per_node, log_spectral_gap)
- R3: curvature/resistance (diameter_proxy, avg_path_length_proxy, modularity_proxy)
- R4: graphlet + spectral
- R5: graphlet + geometric (graphlet + curvature)
- R6: invariant hybrid (graphlet + spectral + curvature, no absolute features)
- R7: learned invariant graph encoder (placeholder for future)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import math
import numpy as np

from ..exp5_2.state_encoding import encode_normalized_state, NORM_STATE_DIM


# ---------------------------------------------------------------------------
# State decomposition (indices into the 20-dim normalized vector).
# ---------------------------------------------------------------------------

# Full normalized vector layout (from exp5_2/state_encoding.py):
#  0: density
#  1: norm_degree_mean
#  2: norm_degree_std
#  3: spectral_gap_normalized
#  4: spectral_gap_per_node
#  5: n_components_normalized
#  6: avg_clustering
#  7: log_density
#  8: log_spectral_gap
#  9: degree_entropy
# 10: transitivity
# 11: modularity_proxy
# 12: diameter_proxy
# 13: avg_path_length_proxy
# 14: triangle_count_norm
# 15: wedge_count_norm
# 16: four_cycle_count_norm
# 17: three_star_count_norm
# 18: assortativity_proxy
# 19: fiber_count_normalized

# Invariant: features that characterize structural regime without family identity.
INVARIANT_INDICES = [
    0,   # density (scale-invariant)
    2,   # norm_degree_std (normalized)
    3,   # spectral_gap_normalized
    4,   # spectral_gap_per_node
    6,   # avg_clustering
    9,   # degree_entropy
    10,  # transitivity
    11,  # modularity_proxy
    12,  # diameter_proxy (normalized)
    13,  # avg_path_length_proxy (normalized)
    14,  # triangle_count_norm
    15,  # wedge_count_norm
    16,  # four_cycle_count_norm
    17,  # three_star_count_norm
    18,  # assortativity_proxy
]

# Context: features that may encode family identity.
CONTEXT_INDICES = [
    1,   # norm_degree_mean (can identify families like star vs regular)
    5,   # n_components_normalized
    19,  # fiber_count_normalized
]

# Derived: log transforms and per-node normalizations.
DERIVED_INDICES = [
    7,   # log_density
    8,   # log_spectral_gap
]


# ---------------------------------------------------------------------------
# Representation ladder.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RepresentationConfig:
    """Configuration for a representation in the ladder."""
    name: str
    indices: tuple[int, ...]
    description: str

    @property
    def dim(self) -> int:
        return len(self.indices)

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "description": self.description,
            "indices": list(self.indices),
        }


# Representation ladder R0-R7.
REPRESENTATION_LADDER: dict[str, RepresentationConfig] = {
    "R0_current": RepresentationConfig(
        name="R0_current",
        indices=tuple(range(NORM_STATE_DIM)),
        description="Current normalized representation (all 20 features)",
    ),
    "R1_graphlet": RepresentationConfig(
        name="R1_graphlet",
        indices=(6, 10, 14, 15, 16, 17, 9, 18),  # clustering, transitivity, graphlets, entropy, assortativity
        description="Graphlet-only: clustering, transitivity, triangle/wedge/4-cycle/3-star, degree entropy, assortativity",
    ),
    "R2_spectral": RepresentationConfig(
        name="R2_spectral",
        indices=(3, 4, 8),  # spectral_gap_normalized, spectral_gap_per_node, log_spectral_gap
        description="Spectral-only: normalized spectral gap, per-node, log",
    ),
    "R3_curvature": RepresentationConfig(
        name="R3_curvature",
        indices=(11, 12, 13),  # modularity_proxy, diameter_proxy, avg_path_length_proxy
        description="Curvature/resistance: modularity, diameter, path length proxies",
    ),
    "R4_graphlet_spectral": RepresentationConfig(
        name="R4_graphlet_spectral",
        indices=(3, 4, 8, 6, 10, 14, 15, 16, 17, 9, 18),
        description="Graphlet + spectral",
    ),
    "R5_graphlet_geometric": RepresentationConfig(
        name="R5_graphlet_geometric",
        indices=(6, 10, 14, 15, 16, 17, 9, 18, 11, 12, 13),
        description="Graphlet + geometric (curvature)",
    ),
    "R6_invariant_hybrid": RepresentationConfig(
        name="R6_invariant_hybrid",
        indices=tuple(INVARIANT_INDICES),
        description="Invariant hybrid: all topology-invariant features, no family-identifying features",
    ),
    "R7_learned_encoder": RepresentationConfig(
        name="R7_learned_encoder",
        indices=tuple(INVARIANT_INDICES),  # placeholder: same as R6 until learned encoder is built
        description="Learned invariant graph encoder (placeholder, uses R6 indices)",
    ),
}


def extract_representation(
    z: np.ndarray,
    rep_config: RepresentationConfig,
) -> np.ndarray:
    """Extract a sub-representation from the full normalized vector."""
    return z[:, rep_config.indices] if z.ndim == 2 else z[list(rep_config.indices)]


def extract_invariant(z: np.ndarray) -> np.ndarray:
    """Extract only the invariant dimensions."""
    return z[:, INVARIANT_INDICES] if z.ndim == 2 else z[INVARIANT_INDICES]


def extract_context(z: np.ndarray) -> np.ndarray:
    """Extract only the context dimensions."""
    return z[:, CONTEXT_INDICES] if z.ndim == 2 else z[CONTEXT_INDICES]


def extract_derived(z: np.ndarray) -> np.ndarray:
    """Extract only the derived dimensions."""
    return z[:, DERIVED_INDICES] if z.ndim == 2 else z[DERIVED_INDICES]


# ---------------------------------------------------------------------------
# Proper evaluation metrics.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RepresentationMetrics:
    """Metrics for a representation evaluation."""
    representation: str = ""
    # Delta metrics (primary).
    delta_r2: float = 0.0
    delta_rmse: float = 0.0
    delta_r2_invariant: float = 0.0  # delta R² on invariant dims only
    delta_rmse_invariant: float = 0.0
    # Absolute metrics (secondary, inflated by low variance).
    abs_r2: float = 0.0
    abs_rmse: float = 0.0
    # Baseline comparison.
    zero_delta_r2: float = 0.0  # R² of zero-delta predictor
    zero_delta_rmse: float = 0.0
    beats_zero_delta: bool = False
    # Ranking.
    spearman: float = 0.0
    # Calibration.
    calibration_corr: float = 0.0
    # Sample counts.
    n_samples: int = 0
    n_train: int = 0
    # Per-feature NRMSE.
    per_feature_nrmse: list[float] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "delta_r2": float(self.delta_r2),
            "delta_rmse": float(self.delta_rmse),
            "delta_r2_invariant": float(self.delta_r2_invariant),
            "delta_rmse_invariant": float(self.delta_rmse_invariant),
            "abs_r2": float(self.abs_r2),
            "abs_rmse": float(self.abs_rmse),
            "zero_delta_r2": float(self.zero_delta_r2),
            "zero_delta_rmse": float(self.zero_delta_rmse),
            "beats_zero_delta": bool(self.beats_zero_delta),
            "spearman": float(self.spearman),
            "calibration_corr": float(self.calibration_corr),
            "n_samples": int(self.n_samples),
            "n_train": int(self.n_train),
            "per_feature_nrmse": [float(x) for x in self.per_feature_nrmse],
        }


def compute_representation_metrics(
    pred_delta: np.ndarray,
    actual_delta: np.ndarray,
    *,
    pred_state: np.ndarray | None = None,
    actual_state: np.ndarray | None = None,
    uncertainties: np.ndarray | None = None,
    invariant_mask: np.ndarray | None = None,
    representation: str = "",
    n_train: int = 0,
) -> RepresentationMetrics:
    """Compute proper evaluation metrics.

    Primary metric: delta R² (how well does the model predict the CHANGE?)
    Baseline: zero-delta predictor (predict no change)
    Invariant: delta R² on invariant dimensions only
    """
    if len(pred_delta) == 0:
        return RepresentationMetrics(representation=representation)

    n = len(pred_delta)

    # --- Delta R² (primary) ---
    d_diff = pred_delta - actual_delta
    delta_rmse = float(np.sqrt(np.mean(d_diff ** 2)))
    d_ss_res = float(np.sum(d_diff ** 2))
    d_ss_tot = float(np.sum((actual_delta - actual_delta.mean(axis=0)) ** 2))
    delta_r2 = max(-10.0, min(1.0, 1.0 - d_ss_res / max(d_ss_tot, 1e-10)))

    # --- Invariant-only delta R² ---
    if invariant_mask is not None:
        inv_pred = pred_delta[:, invariant_mask]
        inv_actual = actual_delta[:, invariant_mask]
        inv_diff = inv_pred - inv_actual
        delta_rmse_inv = float(np.sqrt(np.mean(inv_diff ** 2)))
        inv_ss_res = float(np.sum(inv_diff ** 2))
        inv_ss_tot = float(np.sum((inv_actual - inv_actual.mean(axis=0)) ** 2))
        delta_r2_inv = max(-10.0, min(1.0, 1.0 - inv_ss_res / max(inv_ss_tot, 1e-10)))
    else:
        delta_r2_inv = delta_r2
        delta_rmse_inv = delta_rmse

    # --- Absolute R² (secondary) ---
    abs_r2 = 0.0
    abs_rmse = 0.0
    if pred_state is not None and actual_state is not None:
        a_diff = pred_state - actual_state
        abs_rmse = float(np.sqrt(np.mean(a_diff ** 2)))
        a_ss_res = float(np.sum(a_diff ** 2))
        a_ss_tot = float(np.sum((actual_state - actual_state.mean(axis=0)) ** 2))
        abs_r2 = max(-10.0, min(1.0, 1.0 - a_ss_res / max(a_ss_tot, 1e-10)))

    # --- Zero-delta baseline ---
    zero_delta = np.zeros_like(actual_delta)
    z_diff = zero_delta - actual_delta
    zero_delta_rmse = float(np.sqrt(np.mean(z_diff ** 2)))
    z_ss_res = float(np.sum(z_diff ** 2))
    z_ss_tot = d_ss_tot  # same denominator
    zero_delta_r2 = max(-10.0, min(1.0, 1.0 - z_ss_res / max(z_ss_tot, 1e-10)))

    # --- Per-feature NRMSE ---
    feat_std = np.std(actual_delta, axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    norm_d_diff = d_diff / feat_std
    per_feat = [float(np.sqrt(np.mean(norm_d_diff[:, j] ** 2)))
                for j in range(actual_delta.shape[1])]

    # --- Spearman ---
    spearman = 0.0
    try:
        from scipy.stats import spearmanr
        flat_pred = pred_delta.mean(axis=1)
        flat_actual = actual_delta.mean(axis=1)
        if len(flat_pred) > 1:
            sp, _ = spearmanr(flat_pred, flat_actual)
            spearman = float(sp) if not math.isnan(sp) else 0.0
    except Exception:
        pass

    # --- Calibration ---
    cal_corr = 0.0
    if uncertainties is not None and len(uncertainties) == n:
        errors = np.sqrt(np.sum(d_diff ** 2, axis=1))
        if np.std(uncertainties) > 1e-10 and np.std(errors) > 1e-10:
            cal_corr = float(np.corrcoef(uncertainties, errors)[0, 1])

    return RepresentationMetrics(
        representation=representation,
        delta_r2=delta_r2,
        delta_rmse=delta_rmse,
        delta_r2_invariant=delta_r2_inv,
        delta_rmse_invariant=delta_rmse_inv,
        abs_r2=abs_r2,
        abs_rmse=abs_rmse,
        zero_delta_r2=zero_delta_r2,
        zero_delta_rmse=zero_delta_rmse,
        beats_zero_delta=bool(delta_r2 > zero_delta_r2),
        spearman=spearman,
        calibration_corr=cal_corr,
        n_samples=n,
        n_train=n_train,
        per_feature_nrmse=per_feat,
    )
