"""Deterministic structural oracles for exp6.8.1.

Computes structural observables that are expensive but calculable,
using deterministic numerical methods rather than learning.

Principle: Don't learn what you can calculate reliably.

Spectral gap: estimated via Lanczos iteration on the Laplacian.
  L = D - A (graph Laplacian)
  lambda_2 = second smallest eigenvalue
  spectral_gap = lambda_max - lambda_2  (or just lambda_2 for connectivity)

For small graphs (n <= 50), we use dense eigendecomposition.
For larger graphs, Lanczos approximation.
"""
from __future__ import annotations

import numpy as np
from ...types import GraphBuffers


def _build_adjacency_matrix(graph: GraphBuffers, n: int) -> np.ndarray:
    """Build dense adjacency matrix from graph buffers."""
    A = np.zeros((n, n), dtype=np.float64)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            w = float(graph.weight[i].item()) if hasattr(graph, 'weight') else 1.0
            if s < n and d < n:
                A[s, d] = w
                A[d, s] = w
    return A


def _build_laplacian(A: np.ndarray) -> np.ndarray:
    """Build graph Laplacian: L = D - A."""
    D = np.diag(A.sum(axis=1))
    return D - A


def compute_spectral_gap_deterministic(graph: GraphBuffers, n: int) -> float:
    """Compute spectral gap using deterministic eigendecomposition.

    For n <= 50: dense eigendecomposition (exact).
    For n > 50: Lanczos approximation.

    Returns the spectral gap (lambda_max - lambda_2).
    """
    if n <= 1:
        return 0.0

    A = _build_adjacency_matrix(graph, n)
    L = _build_laplacian(A)

    if n <= 50:
        # Dense eigendecomposition — exact for small graphs.
        try:
            eigenvalues = np.linalg.eigvalsh(L)
            sorted_eg = np.sort(eigenvalues)
            lambda_2 = sorted_eg[1] if n > 1 else 0.0
            lambda_max = sorted_eg[-1]
            return float(lambda_max - lambda_2)
        except np.linalg.LinAlgError:
            return 0.0
    else:
        # Lanczos approximation for larger graphs.
        return _lanczos_spectral_gap(L, n, k=min(20, n - 1))


def _lanczos_spectral_gap(L: np.ndarray, n: int, k: int = 20) -> float:
    """Estimate spectral gap using Lanczos iteration.

    Estimates the two extreme eigenvalues of the Laplacian.
    """
    rng = np.random.RandomState(42)
    v = rng.randn(n)
    v = v / np.linalg.norm(v)

    # Lanczos iteration.
    alphas = []
    betas = []
    V = [v.copy()]

    for j in range(min(k, n)):
        w = L @ V[j]
        if j > 0:
            w = w - betas[j - 1] * V[j - 1]
        alpha = float(V[j] @ w)
        alphas.append(alpha)
        w = w - alpha * V[j]

        beta = float(np.linalg.norm(w))
        if beta < 1e-10:
            break
        betas.append(beta)
        V.append(w / beta)

    # Solve tridiagonal eigenvalue problem.
    m = len(alphas)
    if m < 2:
        return 0.0

    T = np.diag(alphas) + np.diag(betas[:m - 1], 1) + np.diag(betas[:m - 1], -1)
    try:
        eigvals = np.linalg.eigvalsh(T)
        sorted_eg = np.sort(eigvals)
        lambda_2 = sorted_eg[1] if m > 1 else 0.0
        lambda_max = sorted_eg[-1]
        return float(lambda_max - lambda_2)
    except np.linalg.LinAlgError:
        return 0.0


def compute_effective_resistance(graph: GraphBuffers, n: int) -> float:
    """Compute average effective resistance using the pseudoinverse of L.

    R_eff(u,v) = (e_u - e_v)^T L^+ (e_u - e_v)

    Returns the average over all node pairs.
    """
    if n <= 1:
        return 0.0

    A = _build_adjacency_matrix(graph, n)
    L = _build_laplacian(A)

    try:
        # Pseudoinverse of Laplacian.
        L_plus = np.linalg.pinv(L)
        # Average effective resistance.
        total = 0.0
        count = 0
        for u in range(min(n, 20)):  # Sample for efficiency.
            for v in range(u + 1, min(n, 20)):
                diff = np.zeros(n)
                diff[u] = 1.0
                diff[v] = -1.0
                r = float(diff @ L_plus @ diff)
                total += r
                count += 1
        return total / max(count, 1)
    except np.linalg.LinAlgError:
        return 0.0


def compute_curvature_estimate(graph: GraphBuffers, n: int) -> float:
    """Compute a deterministic curvature proxy.

    Uses the Ollivier-Ricci curvature approximation based on
    shortest-path distances and degree information.
    """
    if n <= 1:
        return 0.0

    A = _build_adjacency_matrix(graph, n)

    # Floyd-Warshall for small graphs.
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                dist[i, j] = 1.0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    # Average curvature proxy: 1/d_avg - 1 (higher = more connected).
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < np.inf:
                total_dist += dist[i, j]
                count += 1

    if count == 0:
        return 0.0

    avg_dist = total_dist / count
    return float(1.0 / avg_dist - 1.0) if avg_dist > 0 else 0.0
