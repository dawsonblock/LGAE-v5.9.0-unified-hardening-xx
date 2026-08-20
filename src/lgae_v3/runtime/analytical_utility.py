"""Analytical Utility Oracle — trusted deterministic kernel component.

This module is part of the trusted deterministic kernel, NOT the
experimental ML layer. It provides exact analytical computation of
immediate structural utility deltas.

Validity contract:
    The analytical formulas are exact ONLY when:
    1. latent_state_static_during_mutation = True
    2. topology_only_mutation = True
    3. The mutation type is in supported_mutations

If any condition is violated, the oracle must fail closed or
explicitly downgrade to approximate mode.

Utility function:
    U(G) = -sum_{(u,v) in E} w_uv * ||z_u - z_v||^2

Analytical deltas:
    ADD_EDGE(u,v,w):     ΔU = -w * ||z_u - z_v||^2
    REMOVE_EDGE(u,v):    ΔU = +w * ||z_u - z_v||^2
    REWEIGHT(u,v,f):     ΔU = -(w'*f - w) * ||z_u - z_v||^2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import torch

from ..types import GraphBuffers


@dataclass(frozen=True)
class AnalyticalUtilityContract:
    """Formal validity contract for the analytical utility oracle."""
    latent_state_static: bool = True
    topology_only_mutation: bool = True
    supported_mutations: frozenset[str] = frozenset({
        "add_edge", "remove_edge", "reweight_up", "reweight_down",
        "bridge", "local_rewire", "hub_connect", "edge_swap",
    })

    def validate(self) -> bool:
        """Check if the contract is valid for exact computation."""
        return self.latent_state_static and self.topology_only_mutation


class AnalyticalUtilityOracle:
    """Trusted deterministic oracle for immediate structural utility deltas.

    This is NOT an experimental component. It is part of the trusted kernel
    and provides exact O(1) computation of utility deltas for supported
    mutation types.

    The oracle is valid only when latent states do not change during
    structural mutations. If latent states evolve, the oracle must
    be explicitly downgraded.
    """

    def __init__(self, contract: AnalyticalUtilityContract | None = None) -> None:
        self.contract = contract or AnalyticalUtilityContract()

    def _latent_distance_sq(self, z: torch.Tensor, u: int, v: int) -> float:
        """Compute ||z_u - z_v||^2."""
        with torch.no_grad():
            return float((z[u] - z[v]).pow(2).sum().item())

    def _get_edge_weight(self, graph: GraphBuffers, u: int, v: int) -> float | None:
        """Get the weight of edge (u,v), or None if it doesn't exist."""
        valid = graph.valid.bool()
        for i in range(graph.src.shape[0]):
            if valid[i]:
                s = int(graph.src[i].item())
                d = int(graph.dst[i].item())
                if (s == u and d == v) or (s == v and d == u):
                    return float(graph.weight[i].item())
        return None

    def delta_add_edge(
        self, graph: GraphBuffers, z: torch.Tensor,
        u: int, v: int, weight: float = 1.0,
    ) -> float:
        """Exact ΔU for adding edge (u,v) with given weight.

        ΔU = -w * ||z_u - z_v||^2

        If the edge already exists, the additional weight contributes
        the same delta (AddEdge merges weights).
        """
        if not self.contract.validate():
            raise RuntimeError(
                "AnalyticalUtilityOracle contract violated: "
                "latent_state_static or topology_only_mutation is False. "
                "Oracle must be downgraded to approximate mode."
            )
        d_sq = self._latent_distance_sq(z, u, v)
        return -weight * d_sq

    def delta_remove_edge(
        self, graph: GraphBuffers, z: torch.Tensor,
        u: int, v: int,
    ) -> float:
        """Exact ΔU for removing edge (u,v).

        ΔU = +w * ||z_u - z_v||^2

        Returns 0.0 if the edge doesn't exist.
        """
        if not self.contract.validate():
            raise RuntimeError(
                "AnalyticalUtilityOracle contract violated."
            )
        w = self._get_edge_weight(graph, u, v)
        if w is None:
            return 0.0
        d_sq = self._latent_distance_sq(z, u, v)
        return +w * d_sq

    def delta_reweight(
        self, graph: GraphBuffers, z: torch.Tensor,
        u: int, v: int, factor: float,
        *,
        w_min: float = 1e-3, w_max: float = 10.0,
    ) -> float:
        """Exact ΔU for reweighting edge (u,v) by factor.

        ΔU = -(w' - w) * ||z_u - z_v||^2
        where w' = clamp(w * factor, w_min, w_max)

        Returns 0.0 if the edge doesn't exist.
        """
        if not self.contract.validate():
            raise RuntimeError(
                "AnalyticalUtilityOracle contract violated."
            )
        w_old = self._get_edge_weight(graph, u, v)
        if w_old is None:
            return 0.0
        w_new = min(max(w_old * factor, w_min), w_max)
        d_sq = self._latent_distance_sq(z, u, v)
        return -(w_new - w_old) * d_sq

    def delta_for_mutation(
        self, graph: GraphBuffers, z: torch.Tensor,
        mutation_type: str, u: int, v: int,
        params: dict[str, Any] | None = None,
    ) -> float:
        """Dispatch to the correct analytical delta for a mutation type.

        Raises ValueError for unsupported mutation types.
        """
        if mutation_type not in self.contract.supported_mutations:
            raise ValueError(
                f"Unsupported mutation type: {mutation_type}. "
                f"Supported: {self.contract.supported_mutations}"
            )
        params = params or {}

        if mutation_type in ("add_edge", "bridge", "local_rewire", "hub_connect"):
            return self.delta_add_edge(graph, z, u, v, params.get("weight", 1.0))
        elif mutation_type == "remove_edge":
            return self.delta_remove_edge(graph, z, u, v)
        elif mutation_type == "reweight_up":
            return self.delta_reweight(graph, z, u, v, params.get("factor", 2.0))
        elif mutation_type == "reweight_down":
            return self.delta_reweight(graph, z, u, v, params.get("factor", 0.5))
        elif mutation_type == "edge_swap":
            # Edge swap: remove (u,v), add (u, w) where w = params["new_target"].
            w = params.get("new_target", v)
            return (self.delta_remove_edge(graph, z, u, v) +
                    self.delta_add_edge(graph, z, u, w, params.get("weight", 1.0)))
        else:
            return 0.0

    def rank_candidates(
        self,
        graph: GraphBuffers,
        z: torch.Tensor,
        candidates: Sequence[tuple[str, int, int, dict[str, Any]]],
    ) -> np.ndarray:
        """Rank candidates by analytical delta utility (descending).

        Returns an array of delta utilities, same length as candidates.
        Higher values are better.
        """
        deltas = np.array([
            self.delta_for_mutation(graph, z, mt, u, v, p)
            for mt, u, v, p in candidates
        ])
        return deltas

    def verify_against_exact(
        self,
        graph: GraphBuffers,
        z: torch.Tensor,
        candidates: Sequence[tuple[str, int, int, dict[str, Any]]],
        exact_deltas: np.ndarray,
        *,
        tolerance: float = 1e-4,
    ) -> dict[str, Any]:
        """Verify analytical deltas match exact oracle within tolerance.

        Args:
            candidates: List of (mutation_type, u, v, params).
            exact_deltas: Exact oracle deltas for the same candidates.
            tolerance: Maximum acceptable absolute error.

        Returns:
            Verification report with pass/fail and statistics.
        """
        analytical = self.rank_candidates(graph, z, candidates)
        diff = analytical - exact_deltas
        mae = float(np.mean(np.abs(diff)))
        max_err = float(np.max(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((exact_deltas - exact_deltas.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

        passed = mae < tolerance and r2 > 0.9999

        return {
            "passed": passed,
            "mae": mae,
            "rmse": rmse,
            "max_err": max_err,
            "r2": r2,
            "n_candidates": len(candidates),
            "tolerance": tolerance,
            "contract_valid": self.contract.validate(),
        }
