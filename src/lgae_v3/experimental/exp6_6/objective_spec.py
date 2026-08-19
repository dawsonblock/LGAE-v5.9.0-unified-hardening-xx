"""Machine-readable objective specification for exp6.6.

An ObjectiveSpec encodes what the system is trying to accomplish
so the same topology can have different predicted future value
depending on the objective.

The spec is passed to objective-conditioned models as an encoded
vector, and to the causal-effect architecture as an evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ObjectiveSpec:
    """Machine-readable objective specification."""
    name: str
    # Which structural observable drives the bonus.
    observable: str  # "n_components", "redundancy", "hub_load", "spectral_gap"
    # Direction: minimize or maximize the observable.
    direction: str  # "minimize" or "maximize"
    # Target/threshold value.
    threshold: float
    # Reward magnitude (lambda).
    magnitude: float
    # Reward shape: "threshold" (step function) or "linear" (continuous).
    reward_shape: str  # "threshold" or "linear"
    # Planning horizon.
    horizon: int = 2

    @property
    def encoding(self) -> np.ndarray:
        """Encode the objective spec as a feature vector for models."""
        return encode_objective(self)


# The 4 mechanism objective specs.
OBJECTIVE_SPECS: dict[str, ObjectiveSpec] = {
    "connectivity_threshold": ObjectiveSpec(
        name="connectivity_threshold",
        observable="n_components",
        direction="minimize",
        threshold=1.0,
        magnitude=30.0,
        reward_shape="threshold",
    ),
    "redundancy_threshold": ObjectiveSpec(
        name="redundancy_threshold",
        observable="redundancy",
        direction="maximize",
        threshold=2.0,
        magnitude=25.0,
        reward_shape="threshold",
    ),
    "hub_load_threshold": ObjectiveSpec(
        name="hub_load_threshold",
        observable="hub_load",
        direction="minimize",  # minimize variance
        threshold=5.0,
        magnitude=25.0,
        reward_shape="threshold",
    ),
    "spectral_gap_threshold": ObjectiveSpec(
        name="spectral_gap_threshold",
        observable="spectral_gap",
        direction="maximize",
        threshold=0.5,
        magnitude=20.0,
        reward_shape="threshold",
    ),
}


def get_objective_spec(name: str) -> ObjectiveSpec:
    """Get an objective spec by name."""
    return OBJECTIVE_SPECS[name]


# Observable encoding: one-hot over 4 observables.
OBSERVABLE_NAMES = ["n_components", "redundancy", "hub_load", "spectral_gap"]
DIRECTION_NAMES = ["minimize", "maximize"]
REWARD_SHAPE_NAMES = ["threshold", "linear"]

OBJECTIVE_ENCODING_DIM = (
    len(OBSERVABLE_NAMES) +  # observable one-hot
    len(DIRECTION_NAMES) +   # direction one-hot
    1 +                      # threshold (normalized)
    1 +                      # magnitude (normalized)
    len(REWARD_SHAPE_NAMES) +  # reward shape one-hot
    1                        # horizon (normalized)
)


def encode_objective(spec: ObjectiveSpec) -> np.ndarray:
    """Encode an objective spec as a feature vector.

    This encoding does NOT leak the mechanism name. It encodes
    the structural observable, direction, threshold, magnitude,
    reward shape, and horizon.
    """
    # Observable one-hot.
    obs = np.zeros(len(OBSERVABLE_NAMES))
    if spec.observable in OBSERVABLE_NAMES:
        obs[OBSERVABLE_NAMES.index(spec.observable)] = 1.0

    # Direction one-hot.
    direction = np.zeros(len(DIRECTION_NAMES))
    if spec.direction in DIRECTION_NAMES:
        direction[DIRECTION_NAMES.index(spec.direction)] = 1.0

    # Scalar features.
    threshold_norm = spec.threshold / 10.0
    magnitude_norm = spec.magnitude / 50.0
    horizon_norm = spec.horizon / 5.0

    # Reward shape one-hot.
    shape = np.zeros(len(REWARD_SHAPE_NAMES))
    if spec.reward_shape in REWARD_SHAPE_NAMES:
        shape[REWARD_SHAPE_NAMES.index(spec.reward_shape)] = 1.0

    return np.concatenate([
        obs, direction, [threshold_norm], [magnitude_norm],
        shape, [horizon_norm],
    ])
