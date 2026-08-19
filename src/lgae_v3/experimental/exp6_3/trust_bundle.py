"""Separated trust channels for exp6.3.

Three independent trust channels:
- DynamicsTrust: can I trust predicted future state?
- ValueTrust: can I trust future-value ranking?
- RiskTrust: can I trust learned risk?

One global TrustState is too coarse. A topology can have
high dynamics trust but low value trust (as exp6.1 demonstrated).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrustLevel(Enum):
    UNTRUSTED = "untrusted"
    CALIBRATING = "calibrating"
    TRUSTED = "trusted"
    HIGH_CONFIDENCE = "high_confidence"


@dataclass(frozen=True)
class DynamicsTrust:
    """Trust in the structural dynamics model."""
    level: TrustLevel = TrustLevel.UNTRUSTED
    validation_delta_r2: float = 0.0
    n_calibration_samples: int = 0

    @property
    def can_rollout(self) -> bool:
        return self.level in (TrustLevel.TRUSTED, TrustLevel.HIGH_CONFIDENCE)


@dataclass(frozen=True)
class ValueTrust:
    """Trust in the future value model."""
    level: TrustLevel = TrustLevel.UNTRUSTED
    validation_spearman: float = 0.0
    n_calibration_samples: int = 0

    @property
    def can_plan(self) -> bool:
        return self.level in (TrustLevel.TRUSTED, TrustLevel.HIGH_CONFIDENCE)


@dataclass(frozen=True)
class RiskTrust:
    """Trust in the risk model."""
    level: TrustLevel = TrustLevel.UNTRUSTED
    n_risk_evaluations: int = 0

    @property
    def can_govern(self) -> bool:
        return self.level in (TrustLevel.TRUSTED, TrustLevel.HIGH_CONFIDENCE)


@dataclass(frozen=True)
class TrustBundle:
    """Combined trust state across all channels."""
    dynamics: DynamicsTrust
    value: ValueTrust
    risk: RiskTrust

    @property
    def max_horizon(self) -> int:
        """Maximum planning horizon allowed by trust state."""
        if self.dynamics.level == TrustLevel.HIGH_CONFIDENCE and self.value.can_plan:
            return 3
        elif self.dynamics.can_rollout and self.value.can_plan:
            return 2
        elif self.value.can_plan:
            return 1
        else:
            return 0

    def to_log(self) -> dict:
        return {
            "dynamics": {"level": self.dynamics.level.value, "delta_r2": self.dynamics.validation_delta_r2, "n_cal": self.dynamics.n_calibration_samples},
            "value": {"level": self.value.level.value, "spearman": self.value.validation_spearman, "n_cal": self.value.n_calibration_samples},
            "risk": {"level": self.risk.level.value, "n_risk": self.risk.n_risk_evaluations},
            "max_horizon": self.max_horizon,
        }


def compute_trust_bundle(
    *,
    dynamics_r2: float = 0.0,
    dynamics_n_cal: int = 0,
    value_spearman: float = 0.0,
    value_n_cal: int = 0,
) -> TrustBundle:
    """Compute trust bundle from validation metrics."""
    # Dynamics trust.
    if dynamics_r2 > 0.5 and dynamics_n_cal >= 5:
        dyn_level = TrustLevel.HIGH_CONFIDENCE
    elif dynamics_r2 > 0.0 and dynamics_n_cal >= 1:
        dyn_level = TrustLevel.TRUSTED
    elif dynamics_n_cal > 0:
        dyn_level = TrustLevel.CALIBRATING
    else:
        dyn_level = TrustLevel.UNTRUSTED

    # Value trust.
    if value_spearman > 0.7 and value_n_cal >= 5:
        val_level = TrustLevel.HIGH_CONFIDENCE
    elif value_spearman > 0.0 and value_n_cal >= 1:
        val_level = TrustLevel.TRUSTED
    elif value_n_cal > 0:
        val_level = TrustLevel.CALIBRATING
    else:
        val_level = TrustLevel.UNTRUSTED

    # Risk trust (always untrusted for now — no risk model yet).
    risk_level = TrustLevel.UNTRUSTED

    return TrustBundle(
        dynamics=DynamicsTrust(
            level=dyn_level,
            validation_delta_r2=dynamics_r2,
            n_calibration_samples=dynamics_n_cal,
        ),
        value=ValueTrust(
            level=val_level,
            validation_spearman=value_spearman,
            n_calibration_samples=value_n_cal,
        ),
        risk=RiskTrust(level=risk_level),
    )
