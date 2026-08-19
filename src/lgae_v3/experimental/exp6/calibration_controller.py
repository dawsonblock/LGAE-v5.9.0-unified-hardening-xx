"""Phase 2-4: Calibration acquisition controller.

Manages the process of collecting calibration transitions from a new
topology and fitting the local correction.

Key features:
- Incremental collection (1, 2, 3, 5, 8, 10 samples)
- Diversity-aware sample selection
- Leave-one-out validation on calibration set
- Early stopping when gates pass
- Separate fit/validate splits when possible
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import numpy as np

from .calibration import (
    TopologyCalibration, fit_calibration, identity_calibration,
)


class CalibrationState(Enum):
    """State of the calibration process."""
    UNTRUSTED = "untrusted"          # No calibration data yet
    CALIBRATING = "calibrating"      # Collecting samples
    LIMITED = "limited"              # Some calibration, gates not fully met
    CALIBRATED = "calibrated"        # Gates met, calibration active
    FAILED = "failed"                # Calibration attempted but gates not met


@dataclass
class CalibrationConfig:
    """Configuration for calibration acquisition."""
    # Sample schedule: try these k values incrementally.
    sample_schedule: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10])
    # Gate: minimum validation delta R² to pass.
    min_delta_r2: float = 0.0
    # Gate: minimum number of validation samples.
    min_validate: int = 1
    # Regularization toward identity.
    regularization: float = 1.0
    # Maximum samples before giving up.
    max_samples: int = 10
    # Diversity weight in sample selection.
    diversity_weight: float = 0.5
    # Risk penalty in sample selection.
    risk_penalty: float = 0.1


@dataclass
class CalibrationResult:
    """Result of a calibration attempt."""
    calibration: TopologyCalibration
    state: CalibrationState
    n_samples_collected: int
    n_fit: int
    n_validate: int
    validation_delta_r2: float
    passed_gate: bool
    sample_efficiency: int  # min k where gate passed, -1 if never

    def to_log(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "n_samples_collected": int(self.n_samples_collected),
            "n_fit": int(self.n_fit),
            "n_validate": int(self.n_validate),
            "validation_delta_r2": float(self.validation_delta_r2),
            "passed_gate": bool(self.passed_gate),
            "sample_efficiency": int(self.sample_efficiency),
            "calibration": self.calibration.to_log(),
        }


def compute_sample_diversity_score(
    candidate_z: np.ndarray,
    candidate_a: np.ndarray,
    selected_z: np.ndarray,
    selected_a: np.ndarray,
) -> float:
    """Compute diversity score for a candidate sample.

    Higher score = more diverse (farther from selected samples).
    """
    if len(selected_z) == 0:
        return 1.0  # first sample is always diverse

    # Distance to nearest selected sample in (z, a) space.
    dz = candidate_z[np.newaxis, :] - selected_z  # (n_selected, d_z)
    da = candidate_a[np.newaxis, :] - selected_a  # (n_selected, d_a)
    dist = np.sqrt(np.sum(dz ** 2, axis=1) + np.sum(da ** 2, axis=1))
    return float(np.min(dist))


def select_diverse_samples(
    available_z: np.ndarray,
    available_a: np.ndarray,
    available_deltas: np.ndarray,
    n_select: int,
    *,
    diversity_weight: float = 0.5,
) -> list[int]:
    """Select n_select diverse samples from available transitions.

    Greedy selection: at each step, pick the sample that maximizes
    diversity score.
    """
    n = len(available_z)
    if n <= n_select:
        return list(range(n))

    selected_indices: list[int] = []
    selected_mask = np.zeros(n, dtype=bool)

    # First sample: pick the one closest to the mean (representative).
    mean_z = available_z.mean(axis=0)
    dists = np.sqrt(np.sum((available_z - mean_z) ** 2, axis=1))
    first = int(np.argmin(dists))
    selected_indices.append(first)
    selected_mask[first] = True

    # Subsequent samples: maximize diversity.
    for _ in range(n_select - 1):
        best_idx = -1
        best_score = -1.0
        for i in range(n):
            if selected_mask[i]:
                continue
            score = compute_sample_diversity_score(
                available_z[i], available_a[i],
                available_z[selected_mask], available_a[selected_mask],
            )
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True

    return selected_indices


def loo_validate_calibration(
    base_predictions: np.ndarray,
    actual_deltas: np.ndarray,
    *,
    topology_signature: str,
    state_hash: str = "",
    regularization: float = 1.0,
    dynamics_ood_score: float = 0.0,
) -> tuple[float, TopologyCalibration]:
    """Leave-one-out validation of calibration.

    For each sample i:
    1. Fit calibration on all samples except i
    2. Predict sample i
    3. Record error

    Returns:
        (loo_delta_r2, calibration_fitted_on_all)
    """
    n = len(base_predictions)
    if n < 3:
        # Too few for LOO; fit on all and return R²=0.
        cal = fit_calibration(
            base_predictions, actual_deltas,
            topology_signature=topology_signature,
            state_hash=state_hash,
            regularization=regularization,
            dynamics_ood_score=dynamics_ood_score,
        )
        return 0.0, cal

    loo_preds = np.zeros_like(actual_deltas)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        cal = fit_calibration(
            base_predictions[mask], actual_deltas[mask],
            topology_signature=topology_signature,
            state_hash=state_hash,
            regularization=regularization,
            dynamics_ood_score=dynamics_ood_score,
        )
        loo_preds[i] = cal.apply(base_predictions[i])

    # Compute R² on LOO predictions.
    diff = loo_preds - actual_deltas
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((actual_deltas - actual_deltas.mean(axis=0)) ** 2))
    loo_r2 = max(-10.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-10)))

    # Fit final calibration on all samples.
    final_cal = fit_calibration(
        base_predictions, actual_deltas,
        topology_signature=topology_signature,
        state_hash=state_hash,
        regularization=regularization,
        dynamics_ood_score=dynamics_ood_score,
    )

    return loo_r2, final_cal


def run_calibration_acquisition(
    base_model: Any,
    available_z: np.ndarray,
    available_a: np.ndarray,
    available_z_next: np.ndarray,
    *,
    topology_signature: str,
    state_hash: str = "",
    config: CalibrationConfig | None = None,
    dynamics_ood_score: float = 0.0,
) -> CalibrationResult:
    """Run the full calibration acquisition process.

    1. Select diverse samples incrementally
    2. At each k, fit calibration with LOO validation
    3. Stop early when gate passes
    4. Return the best calibration

    Args:
        base_model: The global delta model (has predict_raw_batch).
        available_z: All available state vectors from the new topology.
        available_a: All available action vectors.
        available_z_next: All available next-state vectors.
        topology_signature: Identifier for this topology.
        state_hash: State hash at calibration time.
        config: Calibration configuration.
        dynamics_ood_score: Dynamics-OOD distance for this topology.
    """
    if config is None:
        config = CalibrationConfig()

    n_available = len(available_z)
    if n_available == 0:
        return CalibrationResult(
            calibration=identity_calibration(
                dim=base_model.state_dim,
                topology_signature=topology_signature,
            ),
            state=CalibrationState.UNTRUSTED,
            n_samples_collected=0,
            n_fit=0,
            n_validate=0,
            validation_delta_r2=0.0,
            passed_gate=False,
            sample_efficiency=-1,
        )

    # Compute all base predictions and actual deltas.
    all_base_preds = base_model.predict_raw_batch(available_z, available_a)
    all_actual_deltas = available_z_next - available_z

    # Select diverse samples in order of the schedule.
    # We select up to max_samples diverse samples, then evaluate at each k.
    n_to_select = min(config.max_samples, n_available)
    selected_idx = select_diverse_samples(
        available_z, available_a, all_actual_deltas,
        n_to_select,
        diversity_weight=config.diversity_weight,
    )

    best_cal: TopologyCalibration | None = None
    best_r2 = -999.0
    sample_efficiency = -1
    final_state = CalibrationState.CALIBRATING

    for k in config.sample_schedule:
        if k > n_to_select:
            break

        # Use first k selected samples.
        fit_idx = selected_idx[:k]

        fit_preds = all_base_preds[fit_idx]
        fit_deltas = all_actual_deltas[fit_idx]

        # LOO validation on the fit set.
        loo_r2, cal = loo_validate_calibration(
            fit_preds, fit_deltas,
            topology_signature=topology_signature,
            state_hash=state_hash,
            regularization=config.regularization,
            dynamics_ood_score=dynamics_ood_score,
        )

        # Also evaluate on remaining available samples (if any).
        remaining_idx = [i for i in range(n_available) if i not in fit_idx]
        if remaining_idx:
            rem_preds = all_base_preds[remaining_idx]
            rem_deltas = all_actual_deltas[remaining_idx]
            rem_adapted = cal.apply_batch(rem_preds)
            rem_diff = rem_adapted - rem_deltas
            rem_ss_res = float(np.sum(rem_diff ** 2))
            rem_ss_tot = float(np.sum(
                (rem_deltas - rem_deltas.mean(axis=0)) ** 2
            ))
            rem_r2 = max(-10.0, min(1.0, 1.0 - rem_ss_res / max(rem_ss_tot, 1e-10)))
            # Use the held-out R² as the primary metric.
            val_r2 = rem_r2
            n_val = len(remaining_idx)
        else:
            # Use LOO R².
            val_r2 = loo_r2
            n_val = k

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_cal = cal

        # Check gate.
        if val_r2 > config.min_delta_r2 and n_val >= config.min_validate:
            sample_efficiency = k
            final_state = CalibrationState.CALIBRATED
            break
    else:
        # Didn't pass gate at any k.
        if best_r2 > -999:
            final_state = CalibrationState.LIMITED
        else:
            final_state = CalibrationState.FAILED

    if best_cal is None:
        best_cal = identity_calibration(
            dim=base_model.state_dim,
            topology_signature=topology_signature,
        )

    return CalibrationResult(
        calibration=best_cal,
        state=final_state,
        n_samples_collected=min(config.sample_schedule[-1], n_to_select),
        n_fit=best_cal.n_fit,
        n_validate=best_cal.n_validate,
        validation_delta_r2=best_r2,
        passed_gate=(final_state == CalibrationState.CALIBRATED),
        sample_efficiency=sample_efficiency,
    )
