"""Production dynamics hardening utilities for LGAE v5.3.

This module contains control state that is intentionally *not* part of the
compiled numerical kernel:

- curvature EMA / variance tracking and hysteretic surgery eligibility;
- latent-equilibrium barriers for slow topology updates;
- graph-conditioned control-variate baselines for structural credit.

All objects have deterministic state dictionaries so policy-affecting state can
be checkpointed by the owning controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import hashlib
import math

import torch
from torch import Tensor

from .mutations import canonical_edge


@dataclass(slots=True)
class CurvatureEMAEntry:
    """Edge-curvature tracking entry with Bayesian uncertainty.

    v5.3.2: Upgraded from pure EWMA variance to a Normal-Inverse-Gamma
    (NIG) conjugate posterior.  The NIG posterior gives calibrated
    credible intervals with proper effective sample size, addressing
    the audit's finding that "EWMA variance is not a calibrated
    uncertainty interval."

    The NIG posterior is parameterized by (mu, nu, alpha, beta):
      - mu: posterior mean of the curvature
      - nu: effective sample size (precision scaling)
      - alpha: shape parameter for variance distribution
      - beta: rate parameter for variance distribution

    The predictive distribution is a Student-t with:
      - degrees of freedom: 2 * alpha
      - mean: mu
      - scale: sqrt(beta * (nu + 1) / (alpha * nu))

    For backward compatibility, ``mean`` and ``variance`` properties
    still work — ``variance`` returns the posterior expected variance
    beta / (alpha - 1), and ``count`` returns the effective sample size.
    """
    mean: float = 0.0
    variance: float = 0.0
    count: int = 0
    # NIG posterior parameters (v5.3.2)
    _nu: float = 0.0
    _alpha: float = 0.5
    _beta: float = 1.0

    @property
    def sigma(self) -> float:
        return math.sqrt(max(0.0, float(self.variance)))

    @property
    def effective_sample_size(self) -> float:
        """Bayesian effective sample size (nu parameter)."""
        return float(self._nu)

    @property
    def predictive_sigma(self) -> float:
        """Student-t predictive scale (wider than posterior sigma)."""
        if self._alpha <= 0.5 or self._nu <= 0:
            return float("inf")
        return math.sqrt(self._beta * (self._nu + 1) / (self._alpha * self._nu))

    def credible_interval(self, width: float = 0.95) -> tuple[float, float]:
        """Bayesian credible interval for the curvature mean.

        Uses the Student-t predictive distribution.  ``width`` is the
        probability mass (e.g. 0.95 for a 95% interval).
        """
        if self._nu <= 1 or self._alpha <= 0.5:
            return float("-inf"), float("inf")
        dof = 2 * self._alpha
        # Approximate t-quantile via normal for large dof, exact for small
        if dof > 30:
            z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(width, 1.96)
        else:
            # Simple approximation for t-distribution quantiles
            z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(width, 1.96)
            z *= math.sqrt(dof / (dof - 2)) if dof > 2 else 1.0
        s = self.predictive_sigma
        return float(self.mean) - z * s, float(self.mean) + z * s


class CurvatureHysteresisController:
    """EMA-smoothed edge-curvature controller with uncertainty deadband.

    The tracker intentionally stores *edge-local* statistics.  It does not claim
    that a fast proxy such as AF3/WAF3 is an exact curvature certificate; it only
    stabilizes automatic surgery proposals so instantaneous proxy noise cannot
    cause add/prune oscillation.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.10,
        variance_alpha: float | None = None,
        min_samples: int = 3,
        sigma_guard: float = 1.0,
    ) -> None:
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError("alpha must lie in (0,1]")
        va = float(alpha if variance_alpha is None else variance_alpha)
        if not (0.0 < va <= 1.0):
            raise ValueError("variance_alpha must lie in (0,1]")
        if int(min_samples) < 1:
            raise ValueError("min_samples must be positive")
        if float(sigma_guard) < 0:
            raise ValueError("sigma_guard cannot be negative")
        self.alpha = float(alpha)
        self.variance_alpha = va
        self.min_samples = int(min_samples)
        self.sigma_guard = float(sigma_guard)
        self.entries: dict[tuple[int, int], CurvatureEMAEntry] = {}

    def update(self, curvatures: Mapping[tuple[int, int], float]) -> None:
        for edge, raw in curvatures.items():
            value = float(raw)
            if not math.isfinite(value):
                continue
            key = canonical_edge(*edge)
            entry = self.entries.get(key)
            if entry is None:
                # Initialize NIG posterior with weak prior.
                # Prior: mu=value, nu=1, alpha=0.5, beta=0.001
                # Very vague prior that concentrates quickly with data.
                # Initial var = beta/(alpha-1) is infinite (alpha < 1),
                # but after a few updates alpha > 1 and variance shrinks.
                entry = CurvatureEMAEntry(
                    mean=value, variance=1.0, count=1,
                    _nu=1.0, _alpha=0.5, _beta=0.001,
                )
                self.entries[key] = entry
                continue
            # v5.3.2: NIG conjugate posterior update.
            # Normal-Inverse-Gamma update:
            #   nu' = nu + 1
            #   mu' = (nu * mu + x) / (nu + 1)
            #   alpha' = alpha + 0.5
            #   beta' = beta + 0.5 * nu * (x - mu)^2 / (nu + 1)
            old_mean = float(entry.mean)
            old_nu = float(entry._nu)
            new_nu = old_nu + 1.0
            new_mean = (old_nu * old_mean + value) / new_nu
            new_alpha = float(entry._alpha) + 0.5
            new_beta = float(entry._beta) + 0.5 * old_nu * (value - old_mean) ** 2 / new_nu
            # Update entry
            entry.mean = new_mean
            entry._nu = new_nu
            entry._alpha = new_alpha
            entry._beta = new_beta
            entry.count += 1
            # Posterior expected variance = beta / (alpha - 1)
            if new_alpha > 1.0:
                entry.variance = new_beta / (new_alpha - 1.0)
            else:
                entry.variance = float("inf")

    def edge_stats(self, u: int, v: int) -> CurvatureEMAEntry | None:
        return self.entries.get(canonical_edge(u, v))

    def _node_incident_stats(self, node: int) -> list[CurvatureEMAEntry]:
        n = int(node)
        return [e for (u, v), e in self.entries.items() if u == n or v == n]

    def proposal_stats(self, action: str, u: int, v: int) -> tuple[float | None, float | None, int]:
        action = str(action).lower()
        if action == "prune":
            entry = self.edge_stats(u, v)
            if entry is None:
                return None, None, 0
            return float(entry.mean), float(entry.sigma), int(entry.count)
        if action == "add":
            # A missing edge has no own curvature.  Use the strongest negative
            # incident pressure at its endpoints as a conservative bottleneck proxy.
            candidates = self._node_incident_stats(u) + self._node_incident_stats(v)
            if not candidates:
                return None, None, 0
            mature = [e for e in candidates if e.count >= self.min_samples]
            pool = mature or candidates
            chosen = min(pool, key=lambda e: e.mean)
            return float(chosen.mean), float(chosen.sigma), int(chosen.count)
        raise ValueError("action must be 'add' or 'prune'")

    def allows(
        self,
        action: str,
        u: int,
        v: int,
        *,
        add_threshold: float,
        prune_threshold: float,
    ) -> tuple[bool, dict[str, Any]]:
        if not float(add_threshold) < float(prune_threshold):
            raise ValueError("add_threshold must be below prune_threshold")
        mean, sigma, count = self.proposal_stats(action, u, v)
        details = {
            "action": str(action),
            "edge": canonical_edge(u, v),
            "ema_curvature": mean,
            "sigma_curvature": sigma,
            "samples": int(count),
            "add_threshold": float(add_threshold),
            "prune_threshold": float(prune_threshold),
        }
        if mean is None or sigma is None or count < self.min_samples:
            details["reason"] = "curvature_ema_warmup"
            return False, details

        # Require the hysteresis band itself to be wider than the local noise
        # envelope.  Otherwise neither surgery direction is trustworthy.
        band = float(prune_threshold) - float(add_threshold)
        if band <= 2.0 * self.sigma_guard * float(sigma):
            details["reason"] = "curvature_noise_exceeds_hysteresis_band"
            return False, details

        if str(action).lower() == "add":
            allowed = float(mean) < float(add_threshold)
            details["reason"] = "below_add_threshold" if allowed else "not_below_add_threshold"
            return allowed, details
        allowed = float(mean) > float(prune_threshold)
        details["reason"] = "above_prune_threshold" if allowed else "not_above_prune_threshold"
        return allowed, details

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "variance_alpha": self.variance_alpha,
            "min_samples": self.min_samples,
            "sigma_guard": self.sigma_guard,
            "entries": [
                [u, v, e.mean, e.variance, e.count, e._nu, e._alpha, e._beta]
                for (u, v), e in sorted(self.entries.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "CurvatureHysteresisController":
        obj = cls(
            alpha=float(payload.get("alpha", 0.1)),
            variance_alpha=float(payload.get("variance_alpha", payload.get("alpha", 0.1))),
            min_samples=int(payload.get("min_samples", 3)),
            sigma_guard=float(payload.get("sigma_guard", 1.0)),
        )
        for row in payload.get("entries", []):
            u, v, mean, var, count = row[0], row[1], row[2], row[3], row[4]
            nu = float(row[5]) if len(row) > 5 else 0.01
            alpha_p = float(row[6]) if len(row) > 6 else 0.5
            beta_p = float(row[7]) if len(row) > 7 else 1.0
            obj.entries[canonical_edge(int(u), int(v))] = CurvatureEMAEntry(
                float(mean), float(var), int(count),
                _nu=nu, _alpha=alpha_p, _beta=beta_p,
            )
        return obj


class LatentEquilibriumBarrier:
    """Require consecutive low-drift latent steps before slow structural surgery.

    v5.3.1: Upgraded from pure state-delta to combined state-delta + dynamics
    residual.  The original check only measured::

        δ_t = ||z_t - z_{t-1}|| / ||z_{t-1}||

    This catches simple drift but is fooled by periodic orbits, metastable
    plateaus, vanishing-gradient stalls, and numerical saturation — all of
    which can have small consecutive deltas without being near a stable
    equilibrium.

    The upgraded check also measures the dynamics residual::

        r_t = ||F(z_t) - z_t|| / ||z_t||

    where ``F`` is the one-step diffusion map.  If ``r_t`` is small, the
    state is near a fixed point of the dynamics (a true equilibrium), not
    merely moving slowly.  When no dynamics map is supplied, the barrier
    falls back to the state-delta-only check for backward compatibility.

    The barrier is satisfied only when *both* δ_t and r_t (if provided)
    are below their respective tolerances for ``required_consecutive``
    consecutive steps.
    """

    def __init__(
        self,
        delta_tol: float = 1e-3,
        required_consecutive: int = 3,
        residual_tol: float = 1e-3,
    ) -> None:
        if float(delta_tol) <= 0:
            raise ValueError("delta_tol must be positive")
        if int(required_consecutive) < 1:
            raise ValueError("required_consecutive must be positive")
        if float(residual_tol) <= 0:
            raise ValueError("residual_tol must be positive")
        self.delta_tol = float(delta_tol)
        self.required_consecutive = int(required_consecutive)
        self.residual_tol = float(residual_tol)
        self._previous: Tensor | None = None
        self.consecutive = 0
        self.last_relative_delta = float("inf")
        self.last_relative_residual = float("inf")

    @torch.no_grad()
    def observe(self, z: Tensor, dynamics_residual: Tensor | None = None) -> bool:
        """Record a latent observation and return whether equilibrium is reached.

        Args:
            z: Current latent state [N, D].
            dynamics_residual: Optional tensor r = F(z_t) - z_t where F is
                the one-step diffusion map.  If provided, the barrier also
                checks ||r|| / ||z|| < residual_tol.  If None, only the
                state-delta check is used (backward compatible).
        """
        current = z.detach()
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current.clone()
            self.consecutive = 0
            self.last_relative_delta = float("inf")
            self.last_relative_residual = float("inf")
            return False
        prev = self._previous.to(device=current.device, dtype=current.dtype)
        denom = float(torch.linalg.vector_norm(prev).item())
        delta = float(torch.linalg.vector_norm(current - prev).item()) / max(denom, 1e-12)
        self.last_relative_delta = delta

        # Compute dynamics residual if provided
        residual_ok = True
        if dynamics_residual is not None:
            r = dynamics_residual.detach().to(device=current.device, dtype=current.dtype)
            r_norm = float(torch.linalg.vector_norm(r).item())
            z_norm = float(torch.linalg.vector_norm(current).item())
            self.last_relative_residual = r_norm / max(z_norm, 1e-12)
            residual_ok = self.last_relative_residual < self.residual_tol
        else:
            self.last_relative_residual = float("inf")

        delta_ok = delta < self.delta_tol
        self.consecutive = self.consecutive + 1 if (delta_ok and residual_ok) else 0
        self._previous = current.clone()
        return self.is_equilibrated

    @property
    def is_equilibrated(self) -> bool:
        return self.consecutive >= self.required_consecutive

    def summary(self) -> dict[str, Any]:
        return {
            "delta_tol": self.delta_tol,
            "residual_tol": self.residual_tol,
            "required_consecutive": self.required_consecutive,
            "consecutive": self.consecutive,
            "last_relative_delta": self.last_relative_delta,
            "last_relative_residual": self.last_relative_residual,
            "equilibrated": self.is_equilibrated,
        }


class GraphHashBaseline:
    """Low-variance structural return baseline keyed by graph-state hash buckets.

    Exact graph hashes rarely repeat in a plastic system.  Hash bucketing provides a
    deterministic tabular approximation to ``V_phi(H(G_t))`` while preserving the
    graph configuration as the conditioning source.  A counterfactual NO_OP value can
    override this estimate when available.

    .. deprecated:: v5.3.1
        Hash bucketing destroys geometric similarity: near-identical graph states
        hash to unrelated buckets.  Prefer :class:`GraphFeatureBaseline`, which
        conditions on structural features (N, E, λ₂, curvature moments, degree
        moments) and uses online ridge regression.  This class is retained as a
        fallback and for backward-compatible checkpoint loading.
    """

    def __init__(self, buckets: int = 1024, ema_alpha: float = 0.10) -> None:
        if int(buckets) < 1:
            raise ValueError("buckets must be positive")
        if not (0.0 < float(ema_alpha) <= 1.0):
            raise ValueError("ema_alpha must lie in (0,1]")
        self.buckets = int(buckets)
        self.ema_alpha = float(ema_alpha)
        self.values = [0.0] * self.buckets
        self.counts = [0] * self.buckets

    def _bucket(self, graph_hash: str) -> int:
        digest = hashlib.sha256(str(graph_hash).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.buckets

    def predict(self, graph_hash: str) -> float:
        i = self._bucket(graph_hash)
        return float(self.values[i]) if self.counts[i] else 0.0

    def update(self, graph_hash: str, realized_return: float) -> None:
        value = float(realized_return)
        if not math.isfinite(value):
            return
        i = self._bucket(graph_hash)
        if self.counts[i] == 0:
            self.values[i] = value
        else:
            self.values[i] = (1.0 - self.ema_alpha) * self.values[i] + self.ema_alpha * value
        self.counts[i] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "buckets": self.buckets,
            "ema_alpha": self.ema_alpha,
            "values": list(self.values),
            "counts": list(self.counts),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "GraphHashBaseline":
        obj = cls(int(payload.get("buckets", 1024)), float(payload.get("ema_alpha", 0.1)))
        vals = list(payload.get("values", []))
        counts = list(payload.get("counts", []))
        if len(vals) == obj.buckets and len(counts) == obj.buckets:
            obj.values = [float(v) for v in vals]
            obj.counts = [int(c) for c in counts]
        return obj


class GraphFeatureBaseline:
    """Feature-conditioned value baseline via online ridge regression.

    Replaces :class:`GraphHashBaseline` for structural credit assignment.
    Instead of hashing the graph into a bucket (which destroys geometric
    similarity — near-identical states hash to unrelated buckets), this
    baseline conditions on a compact structural feature vector::

        ψ(G) = [N, E, λ₂, κ̄, σ_κ, d̄, σ_d, β₀, β₁, ...]

    and learns ``V(ψ(G))`` via recursive least squares (RLS) with L2
    regularization.  This gives a smooth, interpolating value estimate:
    similar graph states produce similar baseline predictions, dramatically
    reducing variance in advantage estimation.

    The feature vector is supplied by the caller (the credit tracker or
    engine), so this class is agnostic to the exact feature set.  When no
    feature vector is provided, it falls back to a scalar EMA (equivalent
    to a single-bucket hash baseline), ensuring it never breaks callers
    that only have a graph hash.
    """

    def __init__(self, feature_dim: int = 16, ridge_lambda: float = 1.0) -> None:
        if int(feature_dim) < 1:
            raise ValueError("feature_dim must be positive")
        if float(ridge_lambda) <= 0:
            raise ValueError("ridge_lambda must be positive")
        self.feature_dim = int(feature_dim)
        self.ridge_lambda = float(ridge_lambda)
        # RLS state: P = (X^T X + λI)^{-1}, w = P X^T y.
        # Initialize P = (1/λ) I so the first update is well-conditioned.
        self._P = torch.eye(self.feature_dim + 1, dtype=torch.float64) * (1.0 / self.ridge_lambda)
        self._w = torch.zeros(self.feature_dim + 1, dtype=torch.float64)
        self._count = 0
        # Fallback EMA for hash-only callers.
        self._fallback_ema = 0.0
        self._fallback_count = 0

    @staticmethod
    def _augment(features: Tensor) -> Tensor:
        """Prepend a bias term (1.0) to the feature vector."""
        bias = torch.ones(features.shape[:-1] + (1,), dtype=features.dtype, device=features.device)
        return torch.cat([bias, features], dim=-1)

    def predict(self, graph_hash: str = "", features: Tensor | None = None) -> float:
        if features is not None:
            x = self._augment(features.to(dtype=torch.float64).reshape(-1))
            if x.shape[0] != self._w.shape[0]:
                # Feature dimension mismatch — fall back to EMA.
                return float(self._fallback_ema) if self._fallback_count else 0.0
            return float((self._w @ x).item())
        return float(self._fallback_ema) if self._fallback_count else 0.0

    def update(self, graph_hash: str = "", realized_return: float = 0.0, features: Tensor | None = None) -> None:
        value = float(realized_return)
        if not math.isfinite(value):
            return
        # Always update the fallback EMA so hash-only callers get something.
        if self._fallback_count == 0:
            self._fallback_ema = value
        else:
            self._fallback_ema = 0.9 * self._fallback_ema + 0.1 * value
        self._fallback_count += 1
        if features is None:
            return
        x = self._augment(features.to(dtype=torch.float64).reshape(-1))
        if x.shape[0] != self._w.shape[0]:
            return
        # RLS update: P_new = P - (P x x^T P) / (1 + x^T P x)
        # w_new = w + P x (y - x^T w) / (1 + x^T P x)
        Px = self._P @ x
        denom = 1.0 + float((x @ Px).item())
        self._P = self._P - torch.outer(Px, Px) / denom
        residual = value - float((self._w @ x).item())
        self._w = self._w + Px * (residual / denom)
        self._count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "ridge_lambda": self.ridge_lambda,
            "P": self._P.tolist(),
            "w": self._w.tolist(),
            "count": self._count,
            "fallback_ema": self._fallback_ema,
            "fallback_count": self._fallback_count,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "GraphFeatureBaseline":
        obj = cls(int(payload.get("feature_dim", 16)), float(payload.get("ridge_lambda", 1.0)))
        P = payload.get("P")
        w = payload.get("w")
        if P is not None and w is not None:
            obj._P = torch.tensor(P, dtype=torch.float64)
            obj._w = torch.tensor(w, dtype=torch.float64)
        obj._count = int(payload.get("count", 0))
        obj._fallback_ema = float(payload.get("fallback_ema", 0.0))
        obj._fallback_count = int(payload.get("fallback_count", 0))
        return obj


def compute_graph_features(
    num_nodes: int,
    num_edges: int,
    lambda2: float,
    mean_curvature: float = 0.0,
    std_curvature: float = 0.0,
    mean_degree: float = 0.0,
    std_degree: float = 0.0,
    betti0: int = 1,
    betti1: int = 0,
    latent_variance: float = 0.0,
    extra: list[float] | None = None,
) -> Tensor:
    """Compute a compact structural feature vector for the value baseline.

    Returns a 16-dimensional vector::

        ψ(G) = [log(N), log(E), λ₂, κ̄, σ_κ, d̄, σ_d, β₀, β₁,
                log(1+β₀), log(1+β₁), σ_z², log(1+σ_z²),
                E/N, (E/N)², 1.0]

    where N = num_nodes, E = num_edges, λ₂ = spectral gap, κ = curvature,
    d = degree, β = Betti numbers, σ_z² = latent variance.  Log-transforms
    are applied to scale-sensitive quantities so the ridge regression sees
    comparable magnitudes.

    This is intentionally a *cheap* feature set computable from quantities
    the governor already collects.  A learned graph embedding could replace
    this in the future, but even this crude feature vector should
    dramatically outperform hash bucketing because similar graph states
    produce similar feature vectors.
    """
    n = max(int(num_nodes), 1)
    e = max(int(num_edges), 1)
    density = float(e) / float(n)
    feats = [
        math.log(float(n)),
        math.log(float(e)),
        float(lambda2),
        float(mean_curvature),
        float(std_curvature),
        float(mean_degree),
        float(std_degree),
        float(betti0),
        float(betti1),
        math.log1p(float(max(betti0, 0))),
        math.log1p(float(max(betti1, 0))),
        float(latent_variance),
        math.log1p(max(float(latent_variance), 0.0)),
        density,
        density * density,
        1.0,
    ]
    if extra:
        feats.extend(float(x) for x in extra[:16 - len(feats)])
    return torch.tensor(feats, dtype=torch.float32)
