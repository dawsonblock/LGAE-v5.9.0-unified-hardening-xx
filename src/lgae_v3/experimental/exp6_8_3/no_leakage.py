"""No-leakage assertions for exp6.8.3.

Asserts the arbitrator never receives:
  - exact future Q
  - exact advantage
  - exact best continuation
  - exact MPC first action
  - TEST calibration statistics

These may exist only in label generation and evaluation.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def assert_no_future_oracle_leakage(
    features: np.ndarray,
    feature_names: list[str],
) -> None:
    """Assert that features do not contain any future-oracle information."""
    forbidden = [
        "exact_q", "exact_advantage", "exact_best_q", "exact_mpc_action",
        "oracle_q", "oracle_advantage", "future_q", "future_reward",
        "exact_continuation", "test_calibration",
    ]
    for name in feature_names:
        for f in forbidden:
            assert f not in name.lower(), (
                f"LEAKAGE DETECTED: feature '{name}' contains forbidden term '{f}'. "
                f"The arbitrator must not receive future-oracle information."
            )


def assert_no_test_statistics_leakage(
    calibration_stats: dict,
    test_stats: dict,
) -> None:
    """Assert that calibration statistics are not derived from test data."""
    # Check that calibration and test are physically separate.
    for key in calibration_stats:
        assert key not in test_stats or calibration_stats[key] != test_stats[key], (
            f"LEAKAGE DETECTED: calibration stat '{key}' matches test stat. "
            f"Calibration must be computed independently from test."
        )


def assert_train_calibration_test_isolation(
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """Assert that train/calibration/test splits are physically disjoint."""
    train_set = set(train_indices.tolist())
    cal_set = set(calibration_indices.tolist())
    test_set = set(test_indices.tolist())

    assert len(train_set & cal_set) == 0, (
        "LEAKAGE DETECTED: train and calibration splits overlap."
    )
    assert len(train_set & test_set) == 0, (
        "LEAKAGE DETECTED: train and test splits overlap."
    )
    assert len(cal_set & test_set) == 0, (
        "LEAKAGE DETECTED: calibration and test splits overlap."
    )


def assert_no_exact_mpc_in_features(
    feature_vector: np.ndarray,
    feature_names: list[str],
) -> None:
    """Assert that the exact MPC first action is not in the features."""
    forbidden_substrings = [
        "exact_mpc", "oracle_action", "best_action", "optimal_action",
    ]
    for name in feature_names:
        for f in forbidden_substrings:
            assert f not in name.lower(), (
                f"LEAKAGE DETECTED: feature '{name}' may contain exact MPC information."
            )
