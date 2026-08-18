# v5.1.1 Closed-Loop Authority Integration

## Authority invariant

The learned executive is advisory. Only `LGAEEngine` may alter authoritative structural state.

```
observation
  -> learned action + target proposal
  -> counterfactual ranking against NO_OP
  -> epistemic/conformal gate
  -> engine shadow transaction
  -> ACCEPT | QUARANTINE | REJECT
  -> durable receipt/outcome
  -> long-horizon learning
```

`ACCEPT` commits. `QUARANTINE` stores an exact shadow and does not execute. `REJECT` discards the shadow.

## Transaction classes

- graph topology / affinity / metric mutations: engine graph transaction path;
- fiber birth/death: full `FiberStateSnapshot` transaction path;
- gauge perturbation: raw Lie-generator transaction path while exposed matrices remain in SO(d).

All three can be restored from safe checkpoints. Quarantine acceptance is protected by state hashes so stale shadows cannot overwrite newer authority.

## Uncertainty

v5.1.0 temporarily swapped ensemble weights into the authoritative executive and used a shallow `state_dict()` backup. v5.1.1 uses independent ensemble modules, so an uncertainty query is read-only. Members are updated online by bootstrap-supervised structural outcomes.

Split-conformal residual calibration uses the finite-sample index

`k = ceil((n+1)(1-alpha))`

clipped to the calibration sample count, and the resulting interval is combined conservatively with the ensemble interval.

## Long-horizon credit

Utility traces advance on every controller step, including NO_OP periods. Finalized discounted structural outcomes are returned to the executive and uncertainty ensemble rather than remaining receipt-only telemetry.

## Geometry corrections

- sheaf Laplacian: `delta = A_F z - D z = -L_F z`, so diffusion is `z <- z + eta*delta`;
- isolated rows in sheaf-adjacency diffusion preserve state;
- Γ2/BE/CDE local certification needs a complete two-hop neighborhood;
- PH bottleneck uses minimax perfect matching with diagonal copies;
- reverse dynamic-gauge pairs use antisymmetric generators so `U_ji=U_ij^T`.

## Re-verified inherited hardening

This release retains the v3/v4 gates for SO(d) manifold integrity, log-domain Sinkhorn, reversible Markov Γ2, positive log-conformal Ricci flow, hysteretic surgery, sparse LOBPCG, transactional rollback, and fixed-capacity compiled numerical kernels.

## Explicit boundaries

The package includes experimental causal, hypergraph, dynamic-gauge, ANN, and timescale modules. Presence in the package is not equivalent to being part of the default authoritative engine state. Where a module is not checkpointed, hashed, and governed by the engine, documentation labels it experimental rather than claiming autonomous authority.
