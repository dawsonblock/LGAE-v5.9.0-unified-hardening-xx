"""Horizon policy for exp6.3.

Selects planning horizon based on trust bundle.

UNTRUSTED          H=0 (greedy only)
CALIBRATING        H=0
TRUSTED_PREFILTER  H=1
TRUSTED_ROLLOUT    H=2
HIGH_CONFIDENCE    H=3
"""
from __future__ import annotations

from dataclasses import dataclass
from .trust_bundle import TrustBundle, TrustLevel


@dataclass(frozen=True)
class HorizonPolicy:
    """Policy for selecting planning horizon."""
    max_horizon: int = 3
    require_dynamics_for_h2: bool = True
    require_value_for_h1: bool = True

    def select(self, trust: TrustBundle) -> int:
        """Select planning horizon based on trust."""
        h = trust.max_horizon
        return min(h, self.max_horizon)
