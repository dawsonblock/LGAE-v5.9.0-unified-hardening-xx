# LGAE v5.3.0 — Production Dynamics Hardening

## 1. Gauge/sheaf stability

Native LGAE connections remain in `SO(d)` by construction:

\[
A_{uv}=-A_{uv}^{\top},\qquad U_{uv}=\exp(A_{uv})\ \text{or Cayley}(A_{uv}).
\]

v5.3 adds an orthogonality monitor/penalty for arbitrary external restriction maps and clips transported stalk messages to a configurable source-relative norm before aggregation. The clip is defensive; it does not replace the exact Lie-group parameterization used by the native gauge bank.

## 2. Curvature hysteresis

Automatic edge surgery may optionally use an EMA state

\[
\bar\kappa_t=(1-\alpha)\bar\kappa_{t-1}+\alpha\kappa_t
\]

plus EMA variance. Surgery is disabled during warm-up and when measured curvature noise is too large relative to the add/prune hysteresis band. The production preset combines this with the existing edge cooldown, deadband, bridge gate, beta0 guard and spectral-gap governor.

## 3. Directed Γ₂ diagnostics

For a non-reversible row-stochastic kernel `P`, the production diagnostic may construct its stationary distribution `pi`, time reversal

\[
P^*=\Pi^{-1}P^\top\Pi,
\]

and additive reversiblization

\[
P_{\rm sym}=\frac12(P+P^*).
\]

Bakry–Émery/CDE analysis then uses `Q = P_sym - I`. The audit records that symmetrization was used. Config may instead select `directed_gamma2_policy="reject"` to fail closed on a non-reversible kernel.

## 4. Atomic acceleration-cache lifecycle

Neighbor indexes are explicitly non-authoritative acceleration state. They are attached to `LGAEEngine` and carry dirty/generation metadata. Every authoritative graph/fiber/gauge commit invalidates them; queries rebuild lazily. Rejected/quarantined shadow states never mutate the authoritative cache. `graph_transaction()` provides the same generation-safe behavior for external in-place workflows.

## 5. Multi-timescale equilibrium barrier

Slow structural mutation can be blocked until latent dynamics satisfy a relative-drift criterion for a configured number of consecutive observations:

\[
\frac{\|Z_t-Z_{t-1}\|}{\|Z_{t-1}\|+\epsilon}<\delta_{\rm tol}.
\]

This prevents topology from changing while fast stalk/gauge dynamics are still in transient motion.

## 6. Variance-reduced structural credit

Mutation credit now supports graph-conditioned baselines. For a realized return `R` under graph-state hash `H(G_t)`, the learning target is an advantage

\[
A_t=R_t-b(H(G_t)),
\]

or an explicit no-op/counterfactual baseline when supplied. The graph-hash baseline is updated as an EMA and is persistent across credit-state save/restore.

## 7. Production preset

`configs/v5_3_production.yaml` enables the new controls while code defaults remain backward-compatible. This keeps old experiments replayable while giving production-style runs a stricter preset.

## Qualification

- pytest: 559/559 passed in bounded batches
- geometry qualification: 9/9
- production dynamics: 8/8
- held-out structural-policy gate: 86.67% accuracy, 0.0274 mean regret
