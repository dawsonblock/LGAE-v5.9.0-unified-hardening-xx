"""Structural state representation for exp6.8.

A compact but sufficient structural state that:
  1. Captures all objective-relevant observables
  2. Allows legal candidate generation at future states
  3. Can be predicted by a learned model

The key insight: we use EXACT graph transitions for topology changes
and LEARNED prediction only for expensive structural observables.

StructuralState contains:
  - The actual GraphBuffers (for exact candidate generation)
  - A feature vector z (learned consequential state)
  - Computed observables (can be exact or predicted)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_4.structural_features import compute_component_info
from ..exp6_5.observable_features import _compute_degree_stats, _compute_spectral_gap
from ..exp6_7.extended_effects import (
    _compute_avg_path_length, _compute_global_efficiency, _compute_curvature_proxy,
)


# Dimension of the structural observable vector.
STRUCTURAL_OBSERVABLE_DIM = 10


def compute_structural_observables(graph: GraphBuffers) -> np.ndarray:
    """Compute all structural observables from a graph.

    This is the ground-truth z that the learned model tries to predict.
    """
    n = int(graph.num_nodes)
    comp_info = compute_component_info(graph, n)
    _, mean_deg, std_deg, max_deg, min_deg = _compute_degree_stats(graph, n)
    spec_gap = _compute_spectral_gap(graph, n)
    path_len = _compute_avg_path_length(graph, n)
    efficiency = _compute_global_efficiency(graph, n)
    curvature = _compute_curvature_proxy(graph, n)

    return np.array([
        n / 30.0,                    # n_nodes (normalized)
        comp_info.n_components / 6.0,  # n_components
        mean_deg / 10.0,             # mean degree
        std_deg / 5.0,               # degree std
        max_deg / 20.0,              # max degree (hub load)
        min_deg / 5.0,               # min degree (redundancy)
        spec_gap,                    # spectral gap
        path_len / 10.0,             # avg path length
        efficiency,                  # global efficiency
        curvature,                   # curvature proxy
    ], dtype=np.float32)


@dataclass
class StructuralState:
    """A structural state at time t.

    Contains:
      - graph: the actual GraphBuffers (for exact candidate generation)
      - z: structural observable vector (STRUCTURAL_OBSERVABLE_DIM)
      - is_predicted: whether z was predicted (vs exact)
    """
    graph: GraphBuffers
    z: np.ndarray
    is_predicted: bool = False

    @classmethod
    def from_graph(cls, graph: GraphBuffers) -> "StructuralState":
        """Create a state with exact (ground-truth) observables."""
        return cls(graph=graph, z=compute_structural_observables(graph), is_predicted=False)

    @classmethod
    def from_predicted(cls, graph: GraphBuffers, z: np.ndarray) -> "StructuralState":
        """Create a state with predicted observables."""
        return cls(graph=graph, z=z, is_predicted=True)


def get_observable_value(z: np.ndarray, observable: str) -> float:
    """Extract a specific observable value from the z vector.

    Maps objective spec observable names to z vector indices.
    """
    mapping = {
        "n_components": z[1] * 6.0,   # un-normalize
        "redundancy": z[5] * 5.0,     # min_deg
        "hub_load": z[4] * 20.0,      # max_deg
        "spectral_gap": z[6],         # spec_gap
        "path_length": z[7] * 10.0,
        "efficiency": z[8],
        "curvature": z[9],
    }
    return float(mapping.get(observable, 0.0))
