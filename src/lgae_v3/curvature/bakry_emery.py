from __future__ import annotations

import math
import numpy as np
import torch
from torch import Tensor


def _gamma(Q: Tensor, f: Tensor, g: Tensor) -> Tensor:
    return 0.5 * (Q @ (f * g) - f * (Q @ g) - g * (Q @ f))


def _gamma2_at(Q: Tensor, f: Tensor, x: int) -> Tensor:
    gam = _gamma(Q, f, f)
    qf = Q @ f
    return 0.5 * ((Q @ gam)[x] - 2.0 * _gamma(Q, f, qf)[x])


def _matrix_from_quadratic(fn, m: int, *, dtype, device) -> Tensor:
    M = torch.zeros((m, m), dtype=dtype, device=device)
    basis = torch.eye(m, dtype=dtype, device=device)
    diag: list[Tensor] = []
    for i in range(m):
        qi = fn(basis[i])
        M[i, i] = qi
        diag.append(qi)
    for i in range(m):
        for j in range(i + 1, m):
            qij = fn(basis[i] + basis[j])
            v = 0.5 * (qij - diag[i] - diag[j])
            M[i, j] = v
            M[j, i] = v
    return 0.5 * (M + M.T)


def bakry_emery_curvature_matrix(
    Q: Tensor,
    x: int,
    dimension: float = float("inf"),
    tol: float = 1e-9,
) -> tuple[Tensor, Tensor]:
    """Return the Schur-complement curvature matrix and Γ metric at ``x``.

    The CD(K,N) inequality at a vertex is a generalized quadratic-form problem.
    Γ has a nullspace containing variables outside the one-step neighborhood. Those
    variables cannot simply be discarded: Γ2 couples them to neighbor variables.
    We therefore eliminate the Γ-null variables by minimizing the Γ2 quadratic form,
    i.e. by a Schur complement, before solving the generalized eigenvalue problem.

    Returns
    -------
    B_eff:
        Effective numerator matrix after nullspace elimination.
    A_eff:
        Positive Γ metric on the active subspace.

    If the quadratic form is unbounded below along Γ-null directions, ``B_eff``
    contains ``-inf`` as a 1x1 sentinel and curvature is ``-inf``.
    """
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")
    n = Q.shape[0]
    if not (0 <= int(x) < n):
        raise ValueError("vertex out of range")
    if not torch.allclose(Q.sum(dim=-1), torch.zeros(n, dtype=Q.dtype, device=Q.device), atol=1e-7, rtol=1e-7):
        raise ValueError("Q rows must sum to zero")

    coords = [i for i in range(n) if i != x]
    m = len(coords)
    if m == 0:
        z = torch.zeros((0, 0), dtype=Q.dtype, device=Q.device)
        return z, z

    def qA(vec: Tensor) -> Tensor:
        f = torch.zeros(n, dtype=Q.dtype, device=Q.device)
        f[coords] = vec
        return _gamma(Q, f, f)[x]

    def qB(vec: Tensor) -> Tensor:
        f = torch.zeros(n, dtype=Q.dtype, device=Q.device)
        f[coords] = vec
        val = _gamma2_at(Q, f, x)
        if math.isfinite(float(dimension)):
            if float(dimension) <= 0:
                raise ValueError("dimension must be positive or infinity")
            val = val - (Q @ f)[x].square() / float(dimension)
        return val

    A = _matrix_from_quadratic(qA, m, dtype=Q.dtype, device=Q.device)
    B = _matrix_from_quadratic(qB, m, dtype=Q.dtype, device=Q.device)

    avals, U = torch.linalg.eigh(A)
    pos = avals > tol
    if not bool(pos.any().item()):
        z = torch.zeros((0, 0), dtype=Q.dtype, device=Q.device)
        return z, z

    Up = U[:, pos]
    Aeff = torch.diag(avals[pos])
    Bpp = Up.T @ B @ Up

    null = ~pos
    if bool(null.any().item()):
        Un = U[:, null]
        Bpn = Up.T @ B @ Un
        Bnn = 0.5 * (Un.T @ B @ Un + (Un.T @ B @ Un).T)
        nvals, V = torch.linalg.eigh(Bnn)
        if float(nvals.min().item()) < -tol:
            sentinel = torch.full((1, 1), float("-inf"), dtype=Q.dtype, device=Q.device)
            return sentinel, torch.ones_like(sentinel)
        nz = nvals > tol
        if bool((~nz).any().item()):
            kernel = V[:, ~nz]
            # Any coupling to a zero-cost null direction makes the infimum unbounded.
            leaked = kernel.T @ Bpn.T
            if float(torch.linalg.matrix_norm(leaked).item()) > max(1e-7, 10.0 * tol):
                sentinel = torch.full((1, 1), float("-inf"), dtype=Q.dtype, device=Q.device)
                return sentinel, torch.ones_like(sentinel)
        if bool(nz.any().item()):
            pinv = (V[:, nz] / nvals[nz]) @ V[:, nz].T
            Beff = Bpp - Bpn @ pinv @ Bpn.T
        else:
            Beff = Bpp
    else:
        Beff = Bpp

    return 0.5 * (Beff + Beff.T), Aeff


def bakry_emery_curvature(
    Q: Tensor,
    x: int,
    dimension: float = float("inf"),
    tol: float = 1e-9,
) -> float:
    """Compute the Bakry–Émery curvature K_N(x) via Schur complement.

    This is the exact finite local quadratic-form calculation for the supplied
    graph/Markov generator, up to floating-point linear algebra.
    """
    B, A = bakry_emery_curvature_matrix(Q, x, dimension=dimension, tol=tol)
    if B.numel() == 0:
        return 0.0
    if bool(torch.isneginf(B).any().item()):
        return float("-inf")
    vals, U = torch.linalg.eigh(A)
    if bool((vals <= tol).any().item()):
        raise RuntimeError("internal error: effective Gamma metric is not positive definite")
    inv = vals.rsqrt()
    C = (inv[:, None] * (U.T @ B @ U)) * inv[None, :]
    return float(torch.linalg.eigvalsh(0.5 * (C + C.T)).min().item())


def stationary_measure_from_markov(P: Tensor, tol: float = 1e-10) -> Tensor:
    """Recover a stationary measure from detailed-balance ratios.

    This routine is intentionally specialized to the reversible Markov kernels required
    by the Bakry--Émery layer. It avoids a dense eigendecomposition and avoids slow
    mixing of power iteration. On each connected support component, detailed balance
    gives ``log m_j - log m_i = log P_ij - log P_ji``. Component scales are arbitrary
    for a reducible chain; equal component mass is chosen before global normalization.
    """
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square")
    n = P.shape[0]
    if n == 0:
        raise ValueError("P cannot be empty")
    if not bool(torch.isfinite(P).all().item()) or bool((P < -tol).any().item()):
        raise ValueError("P must be finite and nonnegative")
    if not torch.allclose(P.sum(-1), torch.ones(n, dtype=P.dtype, device=P.device), atol=1e-7, rtol=1e-7):
        raise ValueError("P must be row stochastic")

    work = P.to(torch.float64)
    positive = work > 0.0
    # Reversibility requires symmetric support (ignoring diagonal/self transitions).
    offdiag = ~torch.eye(n, dtype=torch.bool, device=P.device)
    asymmetric = (positive ^ positive.T) & offdiag
    if bool(asymmetric.any().item()):
        raise ValueError("Markov kernel has asymmetric support and cannot be reversible")

    visited = torch.zeros(n, dtype=torch.bool, device=P.device)
    log_m = torch.zeros(n, dtype=torch.float64, device=P.device)
    components: list[list[int]] = []
    consistency_tol = max(1e-6, 1000.0 * float(tol))

    for root in range(n):
        if bool(visited[root].item()):
            continue
        visited[root] = True
        log_m[root] = 0.0
        stack = [root]
        comp = [root]
        while stack:
            i = stack.pop()
            nbrs = torch.where(positive[i] & offdiag[i])[0].tolist()
            for j in nbrs:
                candidate = log_m[i] + torch.log(work[i, j]) - torch.log(work[j, i])
                if not bool(visited[j].item()):
                    visited[j] = True
                    log_m[j] = candidate
                    stack.append(j)
                    comp.append(j)
                elif abs(float((log_m[j] - candidate).item())) > consistency_tol:
                    raise ValueError("Markov kernel violates detailed-balance cycle consistency")
        components.append(comp)

    m = torch.zeros(n, dtype=torch.float64, device=P.device)
    for comp in components:
        ids = torch.tensor(comp, dtype=torch.long, device=P.device)
        vals = log_m[ids]
        weights = torch.exp(vals - vals.max())
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
        m[ids] = weights / max(len(components), 1)
    m = m / m.sum().clamp_min(torch.finfo(m.dtype).tiny)

    residual = float((m @ work - m).abs().max().item())
    if residual > max(1e-6, 10000.0 * float(tol)):
        raise ValueError(f"reconstructed measure is not stationary: residual={residual:.3e}")
    return m.to(dtype=P.dtype, device=P.device)


def validate_reversible_markov(P: Tensor, measure: Tensor | None = None, tol: float = 1e-7) -> Tensor:
    """Validate row-stochasticity and detailed balance; return stationary measure."""
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square")
    n = P.shape[0]
    if bool((P < -tol).any().item()) or not bool(torch.isfinite(P).all().item()):
        raise ValueError("P must be finite and nonnegative")
    if not torch.allclose(P.sum(-1), torch.ones(n, dtype=P.dtype, device=P.device), atol=tol, rtol=tol):
        raise ValueError("P must be row stochastic")
    m = stationary_measure_from_markov(P) if measure is None else measure.to(P)
    if m.ndim != 1 or m.numel() != n or bool((m <= 0).any().item()):
        raise ValueError("stationary measure must be strictly positive")
    balance = m[:, None] * P - m[None, :] * P.T
    if float(balance.abs().max().item()) > 10.0 * tol:
        raise ValueError("Markov kernel is not reversible under supplied measure")
    return m / m.sum()


def normalized_markov_generator(P: Tensor, measure: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Return a numerically conservative ``Δ=P-I`` and reversible volume measure.

    Diagnostic kernels are often constructed in float32. Re-normalizing rows in
    float64 before Γ/Γ2 work prevents tiny stochasticity error from becoming a false
    generator-row-sum failure after graph mutations.
    """
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square")
    work = P.to(torch.float64)
    row_mass = work.sum(dim=-1, keepdim=True)
    if bool((row_mass <= 0).any().item()) or not bool(torch.isfinite(row_mass).all().item()):
        raise ValueError("P has an invalid row mass")
    work = work / row_mass
    m = validate_reversible_markov(work, measure=None if measure is None else measure.to(work))
    Q = work - torch.eye(work.shape[0], dtype=work.dtype, device=work.device)
    # Remove the final floating-point row-sum ulp from the diagonal only.
    Q = Q - torch.diag_embed(Q.sum(dim=-1))
    return Q, m


def stationary_measure_general_markov(
    P: Tensor,
    *,
    tol: float = 1e-12,
    max_iter: int = 20000,
    teleport: float = 1e-10,
) -> Tensor:
    """Stationary distribution for a possibly directed row-stochastic kernel.

    A tiny uniform teleport is used only to make reducible/asymmetric kernels
    numerically ergodic for the purpose of constructing the *audit* symmetrized
    generator. The returned measure is strictly positive. Native reversible
    kernels should continue to use :func:`normalized_markov_generator`.
    """
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square")
    n = int(P.shape[0])
    if n == 0:
        raise ValueError("P cannot be empty")
    work = P.to(torch.float64)
    if not bool(torch.isfinite(work).all().item()) or bool((work < -1e-12).any().item()):
        raise ValueError("P must be finite and nonnegative")
    mass = work.sum(-1, keepdim=True)
    if bool((mass <= 0).any().item()):
        raise ValueError("P has zero-mass rows")
    work = work / mass
    eps = float(teleport)
    if not (0.0 <= eps < 1.0):
        raise ValueError("teleport must lie in [0,1)")
    if eps:
        work = (1.0 - eps) * work + eps / float(n)
    pi = torch.full((n,), 1.0 / float(n), dtype=work.dtype, device=work.device)
    for _ in range(int(max_iter)):
        nxt = pi @ work
        if float((nxt - pi).abs().sum().item()) <= float(tol):
            pi = nxt
            break
        pi = nxt
    pi = pi.clamp_min(torch.finfo(pi.dtype).tiny)
    pi = pi / pi.sum()
    residual = float((pi @ work - pi).abs().max().item())
    if residual > max(1e-9, 1000.0 * float(tol)):
        raise ValueError(f"stationary iteration did not converge: residual={residual:.3e}")
    return pi.to(dtype=P.dtype, device=P.device)


def stationary_symmetrized_markov_generator(
    P: Tensor,
    *,
    measure: Tensor | None = None,
    tol: float = 1e-12,
    teleport: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Symmetrize a directed Markov kernel in ``L²(pi)`` for Γ/Γ₂ audits.

    Let ``P* = Π^{-1} Pᵀ Π`` be the time-reversal under stationary measure
    ``pi``.  The additive reversiblization ``P_sym=(P+P*)/2`` is Markov and
    self-adjoint in ``L²(pi)``.  We return ``Q_sym=P_sym-I`` for Γ-calculus
    plus the Euclidean similarity representation

        S = Π^{1/2} Q_sym Π^{-1/2},

    which is explicitly symmetrized to remove floating-point skew.  ``S`` has
    a real spectrum by construction and is useful as a diagnostic oracle.
    """
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square")
    work = P.to(torch.float64)
    mass = work.sum(-1, keepdim=True)
    if bool((mass <= 0).any().item()) or not bool(torch.isfinite(mass).all().item()):
        raise ValueError("P has invalid row mass")
    work = (work / mass).clamp_min(0.0)
    pi = stationary_measure_general_markov(work, tol=tol, teleport=teleport) if measure is None else measure.to(work)
    if pi.ndim != 1 or pi.numel() != work.shape[0] or bool((pi <= 0).any().item()):
        raise ValueError("stationary measure must be strictly positive")
    pi = pi / pi.sum()
    p_rev = (work.T * pi[None, :]) / pi[:, None]
    p_sym = 0.5 * (work + p_rev)
    p_sym = p_sym.clamp_min(0.0)
    p_sym = p_sym / p_sym.sum(-1, keepdim=True).clamp_min(torch.finfo(p_sym.dtype).tiny)
    # Correct the last ulp so detailed balance remains stable numerically.
    q = p_sym - torch.eye(p_sym.shape[0], dtype=p_sym.dtype, device=p_sym.device)
    q = q - torch.diag_embed(q.sum(-1))
    sqrt_pi = pi.sqrt()
    inv_sqrt = sqrt_pi.reciprocal()
    similarity = sqrt_pi[:, None] * q * inv_sqrt[None, :]
    similarity = 0.5 * (similarity + similarity.T)
    return q, pi, similarity


def analytic_markov_generator(
    P: Tensor,
    *,
    directed_policy: str = "symmetrize",
) -> tuple[Tensor, Tensor, str]:
    """Return a Γ/Γ₂-safe generator for reversible or directed kernels.

    ``directed_policy='reject'`` preserves the older fail-closed behavior.
    ``'symmetrize'`` first attempts exact reversible validation, then falls back
    to stationary-measure additive reversiblization for asymmetric kernels.
    """
    try:
        q, pi = normalized_markov_generator(P)
        return q, pi, "reversible_normalized_markov"
    except ValueError:
        if directed_policy != "symmetrize":
            raise
        q, pi, _ = stationary_symmetrized_markov_generator(P)
        return q, pi, "stationary_measure_symmetrized_markov"
