"""Reward-formulation variants for exp6.7.

Tests whether the structural effect model generalizes across
different reward formulations for the same observable:

  Train: threshold reward (step function)
  Test:  linear reward (continuous)
  Test:  composite reward (multiple observables)

This distinguishes "learned the observable" from "memorized a
particular threshold reward."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_3.split_utility import compute_additive_utility
from ..exp6_4.structural_features import compute_component_info
from ..exp6_5.observable_features import _compute_degree_stats, _compute_spectral_gap


@dataclass(frozen=True)
class RewardVariant:
    """A reward formulation variant."""
    name: str
    mechanism: str  # base mechanism
    shape: str  # "threshold", "linear", "composite"
    # For threshold: same as original.
    # For linear: continuous reward proportional to the observable.
    # For composite: multiple observables combined.
    weights: dict[str, float] | None = None  # for composite


def _utility_connectivity_linear(graph, z, lambda_bonus=30.0, threshold=1):
    """Linear connectivity reward: proportional to component reduction."""
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    comp = compute_component_info(graph, n)
    # Linear: reward = lambda * (n - n_components) / n
    reward = lambda_bonus * (n - comp.n_components) / max(n, 1)
    return u_add + reward


def _utility_connectivity_composite(graph, z, lambda_bonus=30.0, threshold=1):
    """Composite: connectivity + spectral gap."""
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    comp = compute_component_info(graph, n)
    spec_gap = _compute_spectral_gap(graph, n)
    # Connectivity + spectral gap.
    reward = lambda_bonus * max(0, n - comp.n_components) / max(n, 1)
    reward += 0.5 * lambda_bonus * spec_gap
    return u_add + reward


def _utility_spectral_linear(graph, z, lambda_bonus=20.0, threshold=0.5):
    """Linear spectral gap reward: proportional to spectral gap."""
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    spec_gap = _compute_spectral_gap(graph, n)
    reward = lambda_bonus * spec_gap * 10
    return u_add + reward


def _utility_spectral_composite(graph, z, lambda_bonus=20.0, threshold=0.5):
    """Composite: spectral gap + efficiency."""
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    spec_gap = _compute_spectral_gap(graph, n)
    # Approximate efficiency via avg degree.
    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1
    eff = float(np.mean(degrees)) / max(n - 1, 1)
    reward = lambda_bonus * spec_gap * 10 + 0.5 * lambda_bonus * eff
    return u_add + reward


def _utility_redundancy_linear(graph, z, lambda_bonus=25.0, threshold=2):
    """Linear redundancy reward: proportional to min degree."""
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    _, _, _, _, min_deg = _compute_degree_stats(graph, n)
    reward = lambda_bonus * min_deg / max(threshold, 1)
    return u_add + reward


def _utility_redundancy_composite(graph, z, lambda_bonus=25.0, threshold=2):
    """Composite: redundancy + curvature."""
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    _, _, _, _, min_deg = _compute_degree_stats(graph, n)
    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1
    var_deg = float(np.var(degrees))
    reward = lambda_bonus * min_deg / max(threshold, 1) - 0.1 * var_deg
    return u_add + reward


def _utility_hub_linear(graph, z, lambda_bonus=30.0, threshold=1.0):
    """Linear hub-load reward: proportional to low variance."""
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1
    var_deg = float(np.var(degrees))
    target_var = (threshold ** 2) / max(n, 1)
    reward = lambda_bonus * max(0, target_var - var_deg) * n
    return u_add + reward


def _utility_hub_composite(graph, z, lambda_bonus=30.0, threshold=1.0):
    """Composite: hub-load + path length reduction."""
    u_add = compute_additive_utility(graph, z)
    n = int(graph.num_nodes)
    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s, d = int(graph.src[i].item()), int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1
    var_deg = float(np.var(degrees))
    target_var = (threshold ** 2) / max(n, 1)
    reward = lambda_bonus * max(0, target_var - var_deg) * n
    # Also reward lower max degree.
    max_deg = float(np.max(degrees))
    reward -= 0.5 * max_deg
    return u_add + reward


# Registry of reward variants.
REWARD_VARIANTS: dict[str, dict[str, Callable]] = {
    "connectivity_threshold": {
        "threshold": None,  # Use the original from test_f.py
        "linear": _utility_connectivity_linear,
        "composite": _utility_connectivity_composite,
    },
    "spectral_gap_threshold": {
        "threshold": None,
        "linear": _utility_spectral_linear,
        "composite": _utility_spectral_composite,
    },
    "redundancy_threshold": {
        "threshold": None,
        "linear": _utility_redundancy_linear,
        "composite": _utility_redundancy_composite,
    },
    "hub_load_threshold": {
        "threshold": None,
        "linear": _utility_hub_linear,
        "composite": _utility_hub_composite,
    },
}


def make_reward_variant_utility(
    mechanism: str, variant: str, lambda_bonus: float, threshold: float,
) -> Callable:
    """Create a utility function for a reward variant."""
    if variant == "threshold":
        from ..exp6_4.test_f import make_test_f_utility
        return make_test_f_utility(mechanism, lambda_bonus, int(threshold))

    fn = REWARD_VARIANTS.get(mechanism, {}).get(variant)
    if fn is None:
        raise ValueError(f"Unknown reward variant: {mechanism}/{variant}")

    return lambda g, z: fn(g, z, lambda_bonus=lambda_bonus, threshold=threshold)
