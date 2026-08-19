"""Phase 6: Multi-factor trust model with policy states.

Trust is not a single number. It's a collection of factors that
determine what the model is allowed to do.

Policy states:
    UNTRUSTED        → no model assistance
    CALIBRATING      → collecting calibration samples
    LIMITED          → calibration exists but gates not fully met
    TRUSTED_PREFILTER → model can filter candidates (exact verification still required)
    TRUSTED_ROLLOUT  → model can do multi-step rollout (not yet authorized)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import numpy as np


class TrustPolicyState(Enum):
    """Policy state determining what the model is allowed to do."""
    UNTRUSTED = "untrusted"
    CALIBRATING = "calibrating"
    LIMITED = "limited"
    TRUSTED_PREFILTER = "trusted_prefilter"
    TRUSTED_ROLLOUT = "trusted_rollout"


@dataclass(frozen=True, slots=True)
class TrustFactors:
    """Visible trust factors (not a black box)."""
    delta_r2: float               # Calibration validation delta R²
    calibration_corr: float       # Uncertainty-error correlation
    dynamics_ood: float           # Dynamics-OOD distance
    calibration_samples: int      # Number of calibration samples
    rollout_nrmse: float          # Rollout NRMSE at current horizon
    catastrophic_regret_rate: float  # Rate of catastrophic regret
    local_samples: int            # Samples from this topology

    def to_log(self) -> dict[str, Any]:
        return {
            "delta_r2": float(self.delta_r2),
            "calibration_corr": float(self.calibration_corr),
            "dynamics_ood": float(self.dynamics_ood),
            "calibration_samples": int(self.calibration_samples),
            "rollout_nrmse": float(self.rollout_nrmse),
            "catastrophic_regret_rate": float(self.catastrophic_regret_rate),
            "local_samples": int(self.local_samples),
        }


@dataclass(frozen=True, slots=True)
class TrustGates:
    """Thresholds for each policy state transition."""
    # To reach TRUSTED_PREFILTER:
    min_delta_r2: float = 0.0
    min_calibration_corr: float = 0.0  # 0 = no calibration correlation required initially
    min_calibration_samples: int = 3
    max_dynamics_ood: float = 10.0  # very permissive initially
    max_rollout_nrmse: float = 1.0
    max_catastrophic_regret_rate: float = 0.1

    # To reach TRUSTED_ROLLOUT (not yet authorized):
    min_delta_r2_rollout: float = 0.3
    min_calibration_corr_rollout: float = 0.3
    max_rollout_nrmse_rollout: float = 0.5


def compute_trust_state(
    factors: TrustFactors,
    gates: TrustGates | None = None,
) -> TrustPolicyState:
    """Determine the policy state from trust factors.

    The progression is:
    UNTRUSTED → CALIBRATING → LIMITED → TRUSTED_PREFILTER → TRUSTED_ROLLOUT

    Each transition requires meeting the corresponding gates.
    """
    if gates is None:
        gates = TrustGates()

    # No calibration data.
    if factors.calibration_samples == 0:
        return TrustPolicyState.UNTRUSTED

    # Some calibration but not enough samples.
    if factors.calibration_samples < gates.min_calibration_samples:
        return TrustPolicyState.CALIBRATING

    # Check TRUSTED_PREFILTER gates.
    prefilter_ok = (
        factors.delta_r2 > gates.min_delta_r2
        and factors.calibration_corr >= gates.min_calibration_corr
        and factors.dynamics_ood <= gates.max_dynamics_ood
        and factors.rollout_nrmse <= gates.max_rollout_nrmse
        and factors.catastrophic_regret_rate <= gates.max_catastrophic_regret_rate
    )

    if not prefilter_ok:
        return TrustPolicyState.LIMITED

    # Check TRUSTED_ROLLOUT gates.
    rollout_ok = (
        factors.delta_r2 >= gates.min_delta_r2_rollout
        and factors.calibration_corr >= gates.min_calibration_corr_rollout
        and factors.rollout_nrmse <= gates.max_rollout_nrmse_rollout
    )

    if rollout_ok:
        return TrustPolicyState.TRUSTED_ROLLOUT

    return TrustPolicyState.TRUSTED_PREFILTER


def compute_max_horizon(state: TrustPolicyState) -> int:
    """Determine the maximum allowed planning horizon.

    UNTRUSTED:       H = 0 (no model assistance)
    CALIBRATING:     H = 0
    LIMITED:         H = 1
    TRUSTED_PREFILTER: H = 1
    TRUSTED_ROLLOUT: H = 3
    """
    if state == TrustPolicyState.UNTRUSTED:
        return 0
    elif state == TrustPolicyState.CALIBRATING:
        return 0
    elif state == TrustPolicyState.LIMITED:
        return 1
    elif state == TrustPolicyState.TRUSTED_PREFILTER:
        return 1
    elif state == TrustPolicyState.TRUSTED_ROLLOUT:
        return 3
    return 0


@dataclass
class TrustReport:
    """Full trust assessment for a topology."""
    factors: TrustFactors
    state: TrustPolicyState
    max_horizon: int
    gates: TrustGates

    def to_log(self) -> dict[str, Any]:
        return {
            "factors": self.factors.to_log(),
            "state": self.state.value,
            "max_horizon": int(self.max_horizon),
            "gates": {
                "min_delta_r2": float(self.gates.min_delta_r2),
                "min_calibration_samples": int(self.gates.min_calibration_samples),
                "min_calibration_corr": float(self.gates.min_calibration_corr),
            },
        }


def assess_trust(
    delta_r2: float,
    calibration_corr: float,
    dynamics_ood: float,
    calibration_samples: int,
    rollout_nrmse: float = 0.0,
    catastrophic_regret_rate: float = 0.0,
    local_samples: int = 0,
    gates: TrustGates | None = None,
) -> TrustReport:
    """Full trust assessment."""
    factors = TrustFactors(
        delta_r2=delta_r2,
        calibration_corr=calibration_corr,
        dynamics_ood=dynamics_ood,
        calibration_samples=calibration_samples,
        rollout_nrmse=rollout_nrmse,
        catastrophic_regret_rate=catastrophic_regret_rate,
        local_samples=local_samples,
    )
    state = compute_trust_state(factors, gates)
    horizon = compute_max_horizon(state)
    return TrustReport(
        factors=factors,
        state=state,
        max_horizon=horizon,
        gates=gates or TrustGates(),
    )
