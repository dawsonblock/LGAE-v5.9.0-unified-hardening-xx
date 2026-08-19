"""Analytical utility delta computation for exp6.2.

The utility function is:
    U(G) = -sum_{(u,v) in E} w_uv * ||z_u - z_v||^2

For structural mutations where latent states z don't change:

ADD_EDGE(u, v, w):
    ΔU = -w * ||z_u - z_v||^2
    (If edge already exists, weight is added: ΔU = -w * d²)

REMOVE_EDGE(u, v):
    ΔU = +w_uv * ||z_u - z_v||^2

REWEIGHT(u, v, factor):
    w' = clamp(w * factor, min, max)
    ΔU = -(w' - w) * ||z_u - z_v||^2

These are EXACT and O(1) per candidate — no graph mutation needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6.candidate_generator import StructuralCandidate


def compute_latent_distance_sq(z: torch.Tensor, u: int, v: int) -> float:
    """Compute ||z_u - z_v||^2."""
    with torch.no_grad():
        return float((z[u] - z[v]).pow(2).sum().item())


def get_edge_weight(graph: GraphBuffers, u: int, v: int) -> float | None:
    """Get the weight of edge (u,v), or None if it doesn't exist."""
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if (s == u and d == v) or (s == v and d == u):
                return float(graph.weight[i].item())
    return None


def compute_analytical_delta_utility(
    graph: GraphBuffers,
    z: torch.Tensor,
    candidate: StructuralCandidate,
) -> float:
    """Compute the exact analytical delta utility for a candidate.

    This is O(1) — no graph mutation needed.

    For the utility function U = -sum(w * ||z_u - z_v||^2):
    - ADD_EDGE: ΔU = -w * d²
    - REMOVE_EDGE: ΔU = +w * d²
    - REWEIGHT_UP: ΔU = -(w' - w) * d² = -w*(factor-1) * d²
    - REWEIGHT_DOWN: same formula with factor < 1
    - BRIDGE/LOCAL_REWIRE/HUB_CONNECT: same as ADD_EDGE
    """
    u, v = candidate.u, candidate.v
    d_sq = compute_latent_distance_sq(z, u, v)

    if candidate.action_type in ("add_edge", "bridge", "local_rewire", "hub_connect"):
        w = candidate.params.get("weight", 1.0)
        # Check if edge already exists (weight would be added).
        existing_w = get_edge_weight(graph, u, v)
        if existing_w is not None:
            # Edge exists: adding weight w increases the existing term.
            # But the AddEdge mutation merges weights, so ΔU = -w * d²
            # (the additional weight contributes -w * d²)
            return -w * d_sq
        else:
            # New edge: ΔU = -w * d²
            return -w * d_sq

    elif candidate.action_type == "remove_edge":
        w = get_edge_weight(graph, u, v)
        if w is None:
            return 0.0  # edge doesn't exist, no change
        return +w * d_sq

    elif candidate.action_type == "reweight_up":
        w_old = get_edge_weight(graph, u, v)
        if w_old is None:
            return 0.0
        factor = candidate.params.get("factor", 2.0)
        w_new = min(max(w_old * factor, 1e-3), 10.0)
        return -(w_new - w_old) * d_sq

    elif candidate.action_type == "reweight_down":
        w_old = get_edge_weight(graph, u, v)
        if w_old is None:
            return 0.0
        factor = candidate.params.get("factor", 0.5)
        w_new = min(max(w_old * factor, 1e-3), 10.0)
        return -(w_new - w_old) * d_sq

    return 0.0


def compute_analytical_deltas_batch(
    graph: GraphBuffers,
    z: torch.Tensor,
    candidates: list[StructuralCandidate],
) -> np.ndarray:
    """Compute analytical delta utilities for all candidates (O(N) total)."""
    return np.array([
        compute_analytical_delta_utility(graph, z, c)
        for c in candidates
    ])


def verify_analytical_vs_oracle(
    graph: GraphBuffers,
    z: torch.Tensor,
    candidates: list[StructuralCandidate],
) -> dict[str, Any]:
    """Verify that analytical deltas match exact oracle deltas.

    This is the Phase 5 equivalence test.

    Returns:
        dict with MAE, RMSE, R², Spearman, max error, per-type breakdown.
    """
    from ..exp6.candidate_generator import evaluate_candidates_exact

    # Compute analytical deltas.
    analytical = compute_analytical_deltas_batch(graph, z, candidates)

    # Compute exact oracle deltas.
    evaluate_candidates_exact(graph, z, candidates)
    exact = np.array([c.exact_delta_utility for c in candidates])

    # Overall metrics.
    diff = analytical - exact
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(np.abs(diff)))

    # R².
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((exact - exact.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    # Spearman.
    try:
        from scipy.stats import spearmanr
        sp, _ = spearmanr(analytical, exact)
        sp = float(sp) if not np.isnan(sp) else 0.0
    except Exception:
        sp = 0.0

    # Per-type breakdown.
    per_type: dict[str, dict[str, float]] = {}
    for cand_type in set(c.action_type for c in candidates):
        mask = [c.action_type == cand_type for c in candidates]
        type_diff = diff[mask]
        type_exact = exact[mask]
        type_analytical = analytical[mask]
        if len(type_diff) > 0:
            type_ss_res = float(np.sum(type_diff ** 2))
            type_ss_tot = float(np.sum((type_exact - type_exact.mean()) ** 2))
            type_r2 = 1.0 - type_ss_res / max(type_ss_tot, 1e-10)
            per_type[cand_type] = {
                "mae": float(np.mean(np.abs(type_diff))),
                "rmse": float(np.sqrt(np.mean(type_diff ** 2))),
                "max_err": float(np.max(np.abs(type_diff))),
                "r2": float(type_r2),
                "n": int(len(type_diff)),
            }

    return {
        "mae": mae,
        "rmse": rmse,
        "max_err": max_err,
        "r2": r2,
        "spearman": sp,
        "n_candidates": len(candidates),
        "per_type": per_type,
    }
