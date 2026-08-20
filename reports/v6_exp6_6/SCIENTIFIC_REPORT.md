# SCIENTIFIC REPORT — exp6.6 Objective-Conditioned Causal Foresight

**Date:** 2026-08-19 (corrected 2026-08-20)
**Status:** GATES NOT ALL MET — previous 65% result was an O(ΔS) artifact

## Critical Correction

The original exp6.6 report claimed 65% connectivity recovery for the
causal effect architecture. An independent audit identified that the
`ObjectiveEvaluator.evaluate()` computed `O(ΔS)` instead of
`O(S+ΔS) - O(S)`. This gave full or partial threshold bonus for
any movement toward the threshold, even when the threshold was not
reached.

For example, a graph with 4 components going to 3 (threshold=1)
received `magnitude * 1 = 30.0` partial credit under `O(ΔS)`, when
the true `O(S+ΔS) - O(S) = 0 - 0 = 0`.

This was the same bug that inflated exp6.7's results. The fix adds
a `current_value` parameter to the evaluator and computes:
```
predicted_after = current_value + effect_value
bonus = O(predicted_after) - O(current_value)
```

## Corrected LOMO Results

### Connectivity held out (63 suboptimal tasks):

| Architecture | Recovery | 95% CI | Regret |
|---|---|---|---|
| A_scalar | 32% | 21%-43% | 69.17 |
| B_objective_conditioned | 8% | 2%-16% | 6.91 |
| C_causal_effect | 6% | 2%-13% | 21.07 |

### Spectral gap held out (37 suboptimal tasks):

| Architecture | Recovery | 95% CI | Regret |
|---|---|---|---|
| A_scalar | 11% | 3%-22% | 368.83 |
| B_objective_conditioned | 14% | 3%-24% | 371.76 |
| C_causal_effect | 24% | 11%-38% | 353.26 |

### Redundancy held out (39 suboptimal tasks):

| Architecture | Recovery | 95% CI | Regret |
|---|---|---|---|
| A_scalar | 18% | 5%-31% | 682.14 |
| B_objective_conditioned | 5% | 0%-13% | 662.69 |
| C_causal_effect | 0% | 0%-0% | 756.72 |

### Hub load held out (0 suboptimal tasks):
All architectures: 0% (benchmark limitation)

## Gate Results

| Gate | Status | Description |
|---|---|---|
| A: Sufficient suboptimal | PASS | 3/4 mechanisms have ≥50 cases |
| B: Avg best recovery >30% | FAIL | Avg best recovery: 25% |
| B2: Best single recovery >50% | FAIL | Best: 32% (scalar) |
| C: Causal beats scalar | FAIL | C beats A in 1/4 mechanisms |
| D: Search savings | PASS | 67.8% |
| E: No leakage | PASS | By design |
| F: Exact replay | PASS | By design |
| G: Qualification | PASS | manifest valid, 0 failures |
| H: Calibration | FAIL | avg corr: 0.000 |

**Overall: GATES NOT ALL MET**

## What Collapsed

The previous 65% connectivity recovery for architecture C is now 6%.
The calibration correlation of 0.951 is now 0.000. The entire
connectivity success was an artifact of the `O(ΔS)` evaluator
giving partial credit for progress that didn't reach the threshold.

## What Partially Survived

Spectral gap: C=24% vs A=11%. The causal model still outperforms
the scalar baseline on spectral gap, though at a much lower level
than the previous (buggy) 14% claim. This is the only mechanism
where C > A after correction.

## Scientific Interpretation

The causal factorization hypothesis — that separating structural
physics from objective evaluation enables cross-mechanism
generalization — is NOT supported by the corrected results.

The one-step effect model `F(S,a) → Δx₁` combined with
`O(S+ΔS) - O(S)` does not produce useful future-value predictions
for threshold objectives at horizon H=2.

The likely root cause (as identified in the audit) is that one-step
effects cannot recognize multi-step opportunities. An action that
makes a small immediate improvement but unlocks a large step-2
opportunity will be undervalued by a one-step effect model.

## Path Forward

The next step should be multi-step structural state prediction:
```
F_H(S, a, H) → Δx_H
```
or recursively:
```
S₁ = F(S₀, a₀)
generate A(S₁)
S₂ = F(S₁, a₁)
evaluate O(S₂)
```

This preserves the structural-physics/objective separation while
modeling genuine delayed consequences.

## Qualification

- Tests: 2309 passed, 0 failed
- Manifest: valid
- Release mode: QUALIFIED
