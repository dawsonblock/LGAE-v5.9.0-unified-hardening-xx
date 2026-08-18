# LGAE v5.2.0 — Structural Policy Qualification

## Objective

v5.2.0 advances the learned structural executive from a safe proposal mechanism to a policy layer that is explicitly qualified on held-out structural interventions. The engine/governor remains the sole commit authority.

## Major changes

### 1. Richer structural observation
The executive now observes compact latent diagnostics in addition to graph/fiber statistics:
- mean/std/max latent norm;
- mean/max latent mismatch across active graph edges.

These features let the policy distinguish topological bottlenecks, local representational overload, spurious edges, gauge/frame mismatch, distribution shift, and healthy NO_OP states.

### 2. Bounded learned mutation magnitude
The executive proposes conservative magnitudes in addition to action type and target:
- edge reweight factors are bounded to `[0.5, 2.0]`;
- added-edge affinity is bounded to `[0.5, 2.0]` and length is initialized from latent distance;
- fiber birth/death proposes a width in `[1, spawn_width]`;
- gauge perturbation magnitude is bounded to `[0.0025, 0.1]`.

The governor still validates every concrete proposal transactionally.

### 3. Governance outcomes train risk
REJECT and QUARANTINE are now explicit risk supervision rather than discarded proposals:
- ACCEPT → risk target `0.0`;
- QUARANTINE → risk target `0.5`;
- REJECT → risk target `1.0`.

Non-committed proposals never receive a task-utility target or mutation credit receipt.

### 4. Realized information-gain proxy
For committed mutations the bootstrap ensemble is updated with the observed outcome and the controller measures predictive-variance contraction at that exact decision:

`IG_realized = 0.5 * log((Var_before + eps)/(Var_after + eps))`, clipped to `[0, 10]`.

This provides a measured epistemic-gain training signal instead of the prior neutral `IG=0` target.

### 5. Policy-prior head
The executive now includes a supervised structural-action prior in addition to value/cost/risk/IG heads. In controlled benchmark environments with known-optimal structural interventions, this head can be trained from oracle labels. Live execution does not assume such labels exist.

The final proposal score remains task-grounded and risk-aware, with a centered policy-prior bonus rather than allowing classification logits to bypass utility/risk/cost estimates.

### 6. Long-horizon credit persistence
Pending mutation credit now survives restart, including:
- baseline utility;
- partial horizon samples;
- receipt metadata;
- generic horizon→utility maps.

A mutation being tracked at horizons such as `{2,4}` or `{16,100,1000}` can be checkpointed and later finalized without losing credit state.

## Policy qualification

`scripts/qualify_policy.py` trains the proposal model on synthetic full-information structural outcomes across 16 training seeds and evaluates on five held-out seeds.

Release result:
- training structural outcome samples: **864**;
- held-out diagnosis accuracy: **96.67%**;
- mean mutation regret: **0.00685**;
- held-out seeds: `101..105`;
- per-seed diagnosis accuracy: `100%, 100%, 100%, 83.33%, 100%`.

The benchmark remains synthetic. It demonstrates that the proposal architecture can learn the intended intervention classes from state features; it is not evidence of general intelligence or deployment safety.

## Inherited release gates

v5.2.0 also re-runs all inherited geometry and numerical gates:
- SO(d) gauge invariants;
- stabilized log-domain Sinkhorn;
- reversible Markov Gamma2 / Bakry-Emery Schur complement;
- positive log-conformal Ricci-flow updates;
- cooldown/hysteresis surgery;
- sparse LOBPCG spectral checks;
- bridge protection;
- exact LLY cross-qualification;
- fixed-capacity compile boundary.
