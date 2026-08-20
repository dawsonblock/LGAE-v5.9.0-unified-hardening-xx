"""Split structural state for exp6.8.1.

StructuralState is divided into three tiers:
  - ExactState: computable exactly from graph topology
  - CertifiedApproxState: deterministic numerical approximation
  - LearnedState: genuinely uncertain, needs learning

This follows: Don't learn what you can calculate reliably.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_4.structural_features import compute_component_info
from ..exp6_5.observable_features import _compute_degree_stats
from ..exp6_7.extended_effects import (
    _compute_avg_path_length, _compute_global_efficiency,
)
from .deterministic_oracles import (
    compute_spectral_gap_deterministic,
    compute_effective_resistance,
    compute_curvature_estimate,
)


# Dimensions of each tier.
EXACT_STATE_DIM = 6      # n_nodes, n_components, mean_deg, std_deg, max_deg, min_deg
CERTIFIED_STATE_DIM = 3  # spectral_gap, eff_resistance, curvature
LEARNED_STATE_DIM = 3    # path_length, efficiency, future_opportunity
FULL_STATE_DIM = EXACT_STATE_DIM + CERTIFIED_STATE_DIM + LEARNED_STATE_DIM


@dataclass
class ExactState:
    """Exactly computable structural properties."""
    n_nodes: float
    n_components: float
    mean_degree: float
    std_degree: float
    max_degree: float
    min_degree: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.n_nodes, self.n_components, self.mean_degree,
            self.std_degree, self.max_degree, self.min_degree,
        ], dtype=np.float32)

    @classmethod
    def from_graph(cls, graph: GraphBuffers) -> "ExactState":
        n = int(graph.num_nodes)
        comp_info = compute_component_info(graph, n)
        _, mean_deg, std_deg, max_deg, min_deg = _compute_degree_stats(graph, n)
        return cls(
            n_nodes=n / 30.0,
            n_components=comp_info.n_components / 6.0,
            mean_degree=mean_deg / 10.0,
            std_degree=std_deg / 5.0,
            max_degree=max_deg / 20.0,
            min_degree=min_deg / 5.0,
        )


@dataclass
class CertifiedApproxState:
    """Deterministic numerical approximations (no learning)."""
    spectral_gap: float
    effective_resistance: float
    curvature: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.spectral_gap, self.effective_resistance, self.curvature,
        ], dtype=np.float32)

    @classmethod
    def from_graph(cls, graph: GraphBuffers) -> "CertifiedApproxState":
        n = int(graph.num_nodes)
        return cls(
            spectral_gap=compute_spectral_gap_deterministic(graph, n),
            effective_resistance=compute_effective_resistance(graph, n) / 10.0,
            curvature=compute_curvature_estimate(graph, n),
        )


@dataclass
class LearnedState:
    """Genuinely uncertain quantities that need learning."""
    path_length: float
    efficiency: float
    future_opportunity: float  # placeholder for delayed value signal

    def to_array(self) -> np.ndarray:
        return np.array([
            self.path_length, self.efficiency, self.future_opportunity,
        ], dtype=np.float32)

    @classmethod
    def from_graph(cls, graph: GraphBuffers) -> "LearnedState":
        """Compute exact values (used as training labels)."""
        n = int(graph.num_nodes)
        return cls(
            path_length=_compute_avg_path_length(graph, n) / 10.0,
            efficiency=_compute_global_efficiency(graph, n),
            future_opportunity=0.0,  # not directly observable
        )


@dataclass
class SplitStructuralState:
    """Full structural state with three tiers.

    The exact and certified tiers are always computed deterministically.
    The learned tier can be either exact (ground truth) or predicted.
    """
    graph: GraphBuffers
    exact: ExactState
    certified: CertifiedApproxState
    learned: LearnedState
    is_learned_predicted: bool = False

    @classmethod
    def from_graph(cls, graph: GraphBuffers) -> "SplitStructuralState":
        """Create state with all tiers computed exactly."""
        return cls(
            graph=graph,
            exact=ExactState.from_graph(graph),
            certified=CertifiedApproxState.from_graph(graph),
            learned=LearnedState.from_graph(graph),
            is_learned_predicted=False,
        )

    @classmethod
    def from_predicted(
        cls, graph: GraphBuffers, learned_z: np.ndarray,
    ) -> "SplitStructuralState":
        """Create state with exact+certified from graph, learned predicted."""
        return cls(
            graph=graph,
            exact=ExactState.from_graph(graph),
            certified=CertifiedApproxState.from_graph(graph),
            learned=LearnedState(
                path_length=float(learned_z[0]),
                efficiency=float(learned_z[1]),
                future_opportunity=float(learned_z[2]),
            ),
            is_learned_predicted=True,
        )

    def to_full_array(self) -> np.ndarray:
        """Concatenate all tiers into one vector."""
        return np.concatenate([
            self.exact.to_array(),
            self.certified.to_array(),
            self.learned.to_array(),
        ])

    def get_observable(self, name: str) -> float:
        """Get an observable value by name (un-normalized)."""
        if name == "n_components":
            return float(self.exact.n_components * 6.0)
        elif name == "redundancy":
            return float(self.exact.min_degree * 5.0)
        elif name == "hub_load":
            return float(self.exact.max_degree * 20.0)
        elif name == "spectral_gap":
            return float(self.certified.spectral_gap)  # already un-normalized
        elif name == "path_length":
            return float(self.learned.path_length * 10.0)
        elif name == "efficiency":
            return float(self.learned.efficiency)
        elif name == "curvature":
            return float(self.certified.curvature)
        else:
            return 0.0
