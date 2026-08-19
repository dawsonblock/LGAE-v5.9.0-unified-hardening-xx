"""Phase 1: TopologyCalibration — a first-class immutable runtime object.

The calibration is:
    Δz_adapted = α_G ⊙ F_θ(z, a) + β_G

where α_G (scale) and β_G (offset) are fitted from a few exact
transitions on a new topology family.

Key properties:
- Immutable (frozen dataclass)
- Provenance-bound (records what data it was fitted on)
- Serializable (for storage and audit)
- Does NOT mutate the base world model parameters
- Regularized toward identity (α→1, β→0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import numpy as np


@dataclass(frozen=True, slots=True)
class TopologyCalibration:
    """Immutable topology-local calibration for the world model.

    Attributes:
        topology_signature: Hash identifying the topology family/context.
        scale: Per-dimension scale factor α_G.
        offset: Per-dimension offset β_G.
        n_samples: Number of transitions used for fitting.
        fitted_at_state_hash: State hash at fitting time.
        validation_delta_r2: Held-out delta R² after calibration.
        uncertainty_error_corr: Correlation between uncertainty and error.
        dynamics_ood_score: Dynamics-OOD distance for this topology.
        calibration_hash: Hash of the calibration parameters.
        regularization: Regularization strength used.
        n_fit: Number of samples used for fitting.
        n_validate: Number of samples used for validation.
    """
    topology_signature: str
    scale: tuple[float, ...]
    offset: tuple[float, ...]
    n_samples: int
    fitted_at_state_hash: str
    validation_delta_r2: float
    uncertainty_error_corr: float
    dynamics_ood_score: float
    calibration_hash: str
    regularization: float
    n_fit: int
    n_validate: int

    def apply(self, prediction: np.ndarray) -> np.ndarray:
        """Apply calibration to a raw delta prediction.

        Δz_adapted = α_G ⊙ prediction + β_G
        """
        scale = np.array(self.scale, dtype=np.float64)
        offset = np.array(self.offset, dtype=np.float64)
        return scale * prediction + offset

    def apply_batch(self, predictions: np.ndarray) -> np.ndarray:
        """Apply calibration to a batch of predictions."""
        scale = np.array(self.scale, dtype=np.float64)
        offset = np.array(self.offset, dtype=np.float64)
        return scale * predictions + offset

    def to_log(self) -> dict[str, Any]:
        return {
            "topology_signature": self.topology_signature,
            "scale": list(self.scale),
            "offset": list(self.offset),
            "n_samples": int(self.n_samples),
            "fitted_at_state_hash": self.fitted_at_state_hash,
            "validation_delta_r2": float(self.validation_delta_r2),
            "uncertainty_error_corr": float(self.uncertainty_error_corr),
            "dynamics_ood_score": float(self.dynamics_ood_score),
            "calibration_hash": self.calibration_hash,
            "regularization": float(self.regularization),
            "n_fit": int(self.n_fit),
            "n_validate": int(self.n_validate),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), indent=2)


def compute_calibration_hash(
    scale: np.ndarray,
    offset: np.ndarray,
    topology_signature: str,
) -> str:
    """Compute a hash for the calibration parameters."""
    content = json.dumps({
        "topology_signature": topology_signature,
        "scale": [float(x) for x in scale],
        "offset": [float(x) for x in offset],
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def fit_calibration(
    base_predictions: np.ndarray,
    actual_deltas: np.ndarray,
    *,
    topology_signature: str,
    state_hash: str = "",
    regularization: float = 1.0,
    dynamics_ood_score: float = 0.0,
    uncertainties: np.ndarray | None = None,
    validation_predictions: np.ndarray | None = None,
    validation_deltas: np.ndarray | None = None,
) -> TopologyCalibration:
    """Fit a regularized scale-offset calibration.

    Objective:
        L = ||Δz - (α ⊙ F(z,a) + β)||² + λ_α||α - 1||² + λ_β||β||²

    This regularizes toward identity (α=1, β=0), encoding the prior
    that the global model is correct and only small corrections are needed.

    Args:
        base_predictions: Raw delta predictions from the global model (n, d).
        actual_deltas: Actual delta values from exact transitions (n, d).
        topology_signature: Identifier for the topology family.
        state_hash: State hash at fitting time.
        regularization: λ for L2 regularization toward identity.
        dynamics_ood_score: Dynamics-OOD distance for this topology.
        uncertainties: Optional per-sample uncertainties.
        validation_predictions: Held-out predictions for validation.
        validation_deltas: Held-out actual deltas for validation.
    """
    n, d = actual_deltas.shape

    # Regularized least squares per dimension.
    # For each dimension j:
    #   min_α,β ||y - (α*x + β)||² + λ_α(α-1)² + λ_β*β²
    # This has a closed-form solution.
    scale = np.ones(d)
    offset = np.zeros(d)

    for j in range(d):
        x = base_predictions[:, j]
        y = actual_deltas[:, j]

        # Add regularization "pseudo-samples":
        # λ_α(α-1)² → pseudo-sample: x=1, y=1, weight=λ_α
        # λ_β*β² → pseudo-sample: x=0, y=0, weight=λ_β
        lam_alpha = regularization
        lam_beta = regularization

        # Augmented system: [x, 1] for each real sample, [1, 0] and [0, 1] for regularization.
        X_aug = np.vstack([
            np.column_stack([x, np.ones(n)]),  # real data
            np.array([[1.0, 0.0]]),  # α reg: x=1, intercept=0
            np.array([[0.0, 1.0]]),  # β reg: x=0, intercept=1
        ])
        y_aug = np.concatenate([y, [1.0], [0.0]])  # target: 1 for α, 0 for β
        w_aug = np.concatenate([np.ones(n), [lam_alpha], [lam_beta]])

        # Weighted least squares.
        W = w_aug.reshape(-1, 1)
        Xw = X_aug * W
        XtWX = X_aug.T @ Xw
        XtWy = X_aug.T @ (y_aug * w_aug)

        try:
            params = np.linalg.solve(XtWX, XtWy)
            scale[j] = float(params[0])   # α
            offset[j] = float(params[1])  # β
        except np.linalg.LinAlgError:
            scale[j] = 1.0
            offset[j] = float(np.mean(y) - np.mean(x))

    # Compute validation metrics.
    val_r2 = 0.0
    unc_corr = 0.0

    if validation_predictions is not None and validation_deltas is not None:
        val_pred = scale * validation_predictions + offset
        val_diff = val_pred - validation_deltas
        val_ss_res = float(np.sum(val_diff ** 2))
        val_ss_tot = float(np.sum(
            (validation_deltas - validation_deltas.mean(axis=0)) ** 2
        ))
        val_r2 = max(-10.0, min(1.0, 1.0 - val_ss_res / max(val_ss_tot, 1e-10)))

        if uncertainties is not None and len(uncertainties) == len(validation_predictions):
            val_errors = np.sqrt(np.sum(val_diff ** 2, axis=1))
            if np.std(uncertainties) > 1e-10 and np.std(val_errors) > 1e-10:
                unc_corr = float(np.corrcoef(uncertainties, val_errors)[0, 1])

    cal_hash = compute_calibration_hash(scale, offset, topology_signature)

    n_fit = n
    n_validate = len(validation_predictions) if validation_predictions is not None else 0

    return TopologyCalibration(
        topology_signature=topology_signature,
        scale=tuple(float(x) for x in scale),
        offset=tuple(float(x) for x in offset),
        n_samples=n + n_validate,
        fitted_at_state_hash=state_hash,
        validation_delta_r2=val_r2,
        uncertainty_error_corr=unc_corr,
        dynamics_ood_score=dynamics_ood_score,
        calibration_hash=cal_hash,
        regularization=regularization,
        n_fit=n_fit,
        n_validate=n_validate,
    )


def identity_calibration(dim: int, topology_signature: str = "unknown") -> TopologyCalibration:
    """Create an identity calibration (no correction)."""
    return TopologyCalibration(
        topology_signature=topology_signature,
        scale=tuple(1.0 for _ in range(dim)),
        offset=tuple(0.0 for _ in range(dim)),
        n_samples=0,
        fitted_at_state_hash="",
        validation_delta_r2=0.0,
        uncertainty_error_corr=0.0,
        dynamics_ood_score=0.0,
        calibration_hash=compute_calibration_hash(
            np.ones(dim), np.zeros(dim), topology_signature,
        ),
        regularization=0.0,
        n_fit=0,
        n_validate=0,
    )
