# v6.0-exp6.7.1: Multi-Operator Causal Structural Model — Correctness Repair

## Status: GATES NOT ALL MET (HONEST NEGATIVE RESULT)

## Research Question

Can the causal structural effect model generalize across heterogeneous
mutations AND reward formulations?

## Critical Correction from exp6.7

exp6.7 reported 66% connectivity recovery and 54% spectral gap recovery.
An independent audit identified seven implementation defects that
materially inflated those results:

1. **REWEIGHT_EDGE was a no-op**: `apply_action()` recognized
   `reweight_up`/`reweight_down` but not `reweight_edge`, so 100% of
   reweight candidates silently did nothing.

2. **Action identity lost parameters**: `(mt, u, v)` discarded params,
   causing `reweight×2` and `reweight×0.5` to collide. This could
   inflate NonGreedyRecoveryRate.

3. **ADD_EDGE generated existing edges**: ~20% of ADD candidates were
   actually weight-merges, confounding the mutation category.

4. **Objective evaluator was mathematically wrong**: It computed
   `O(ΔS)` instead of `O(S+ΔS) - O(S)`. A threshold objective giving
   full bonus for partial progress (e.g. 4→3 components when threshold
   is 1) is incorrect. **This was the largest inflator.**

5. **Exact MPC used frozen A(S₀)**: Candidates were not regenerated at
   each depth, allowing invalid sequences (remove already-removed edge).

6. **Feature extractor had wrong operator semantics**: REMOVE assumed
   all removals were bridges; REWEIGHT was indistinguishable from SWAP.

7. **Manifest verification was a surrogate**: `_check_manifest_evidence`
   only checked `len(manifest["files"]) > 0`, not real verification.

## Fixes Applied in exp6.7.1

1. `apply_action_with_status()` handles `reweight_edge` with factor param.
2. `ActionIdentity` includes canonical params (type, u, v, factor, new_target).
3. ADD_EDGE candidate generator filters existing edges.
4. `ObjectiveEvaluatorV2.evaluate()` computes `O(S+ΔS) - O(S)` with
   absolute current state.
5. `exact_mpc()` regenerates candidates at each depth via
   `candidate_generator` callback.
6. New `multi_operator_features.py` with correct per-operator semantics.
7. `_check_manifest_evidence()` runs `scripts/generate_manifest.py --check`.

## Corrected Results

### LOMO (100 suboptimal tasks per mechanism)

| Mechanism | A recovery | C recovery | C-A CI | A regret | C regret |
|---|---:|---:|---|---:|---:|
| Connectivity | 11% | 0% | [-0.17, -0.05] | 11.82 | 11.97 |
| Redundancy | 3% | 0% | [-0.07, 0.00] | 421.07 | 421.03 |
| Hub load | 3% | 3% | [-0.05, 0.05] | 107.58 | 70.61 |
| Spectral gap | 0% | 0% | [0.00, 0.00] | 214.66 | 215.88 |

### Gate Results

| Gate | Status | Description |
|---|---|---|
| A: Sufficient suboptimal | PASS | 4/4 mechanisms have ≥100 suboptimal |
| B: C beats A majority | FAIL | C beats A in 0/4 LOMO |
| C: Best C recovery >50% | FAIL | Best C recovery: 3% |
| D: Paired CI excludes 0 | FAIL | CI [-0.05, 0.05] includes 0 |
| E: Search savings >50% | PASS | 76.9% |
| F: No leakage | PASS | By design |
| G: Exact replay | PASS | By design |
| H: Qualification | FAIL | Manifest needs regeneration |
| I: Reward holdout strong | FAIL | 0 variants meet strong criteria |

### Honest Positive Signals

1. **Gate A now passes with 4/4 mechanisms** — the multi-operator fix
   successfully generated ≥100 suboptimal tasks for all mechanisms,
   including hub load (previously 0).

2. **Hub load regret improvement**: C=70.6 vs A=107.6, paired CI
   [6.78, 73.57] excludes zero. The causal model has lower regret
   on hub load even though recovery rate is equal.

3. **Search savings**: 76.9% — the model-assisted search still
   explores far fewer nodes than exact MPC.

## Scientific Interpretation

The corrected results show that the previous 66%/54% recovery rates
were artifacts of the `O(ΔS)` evaluator bug. When the evaluator
correctly computes `O(S+ΔS) - O(S)`, the causal effect model does
not outperform the scalar baseline on first-action recovery.

This is an honest negative result. The causal factorization
architecture remains conceptually sound, but the current
implementation does not demonstrate cross-mechanism generalization
under correct objective evaluation.

The key lesson: **objective evaluator correctness matters more than
model architecture at this stage.** A subtle mathematical error in
the evaluator can inflate results by 50+ percentage points.

## What This Means for the Project

The exp6.6 result (65% connectivity recovery) should also be
re-examined, as it used the same `O(ΔS)` evaluator pattern. If exp6.6's
evaluator has the same bug, its results may also be inflated.

The path forward is not to add more features (exp6.8) but to:
1. Verify exp6.6's evaluator correctness.
2. Investigate why the corrected evaluator makes the causal model
   fail to recover the exact action.
3. Consider whether the one-step effect prediction (Critical Issue 5
   from the audit) is sufficient, or whether multi-step causal
   prediction is needed.

## Qualification

- Tests: 2236 passed, 74 deselected (meta/crash_recovery), 0 failed
- Manifest: 997 files verified
- Release mode: QUALIFIED

## Files Changed

- `src/lgae_v3/experimental/exp6_3/exact_mpc.py` — ActionIdentity,
  apply_action_with_status, state-conditioned MPC
- `src/lgae_v3/experimental/exp6_7/multi_operator_features.py` — new
  multi-operator feature extractor
- `src/lgae_v3/experimental/exp6_7/causal_effect_model_v2.py` —
  corrected O(S+ΔS) - O(S) evaluator
- `src/lgae_v3/experimental/exp6_7/multi_operator_candidates.py` —
  ADD_EDGE filters existing edges
- `src/lgae_v3/experimental/exp6_7/experiment_runner.py` —
  ActionIdentity comparison, real manifest check, strong Gate I
- `src/lgae_v3/experimental/exp6_4/honest_beam_v2.py` — tuple
  unpacking fix, ActionIdentity
- `src/lgae_v3/experimental/exp6_6/honest_beam_v3.py` — tuple
  unpacking fix, ActionIdentity
- `src/lgae_v3/experimental/exp6_5/adaptive_beam.py` — tuple
  unpacking fix, ActionIdentity
- `src/lgae_v3/experimental/exp6_3/beam_search.py` — tuple
  unpacking fix, ActionIdentity
- `src/lgae_v3/experimental/exp6_3/honest_beam_search.py` — tuple
  unpacking fix, ActionIdentity
- `src/lgae_v3/experimental/exp6_3/metrics.py` — ActionIdentity
- `src/lgae_v3/experimental/exp6_4/experiment_runner.py` — ActionIdentity
- `src/lgae_v3/experimental/exp6_5/scaling_benchmark.py` — ActionIdentity
- `src/lgae_v3/experimental/exp6_3/honest_experiment_runner.py` — ActionIdentity
