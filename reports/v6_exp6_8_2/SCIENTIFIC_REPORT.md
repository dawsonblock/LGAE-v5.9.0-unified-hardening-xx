# v6.0-exp6.8.2: Calibrated Selective Planning

## Status: GATES NOT ALL MET (4/10 PASS) — HONEST NEGATIVE ON UNCERTAINTY

## Research Question

Can ensemble-based LCB-margin arbitration with calibrated kappa
preserve non-greedy gains while eliminating tail risk?

## Architecture

- Ensemble of M=5 LearnedStateModel instances
- LCB(margin) = mu_margin - kappa * sigma_margin
- Use learned only if LCB(margin) > 0
- kappa chosen on calibration split, evaluated on locked test
- future_opportunity trained on actual best second-step gain

## Key Result: The Ensemble Abstains

The hybrid planner achieves **0% coverage** on all mechanisms.
The calibrated kappa is high (2.0–5.0), meaning the LCB is very
conservative. The planner never confidently beats greedy, so it
correctly falls back to greedy on every task.

This is **safe behavior**: the system abstains rather than making
unreliable decisions. But it means the non-greedy gains from exp6.8
are not captured.

## Root Cause: Uncalibrated Ensemble Uncertainty

The uncertainty-error correlation is near zero (-0.034 average):
the ensemble std does not predict which predictions are wrong.

| Mechanism | Unc-Error Corr |
|---|---:|
| Connectivity | -0.106 |
| Redundancy | 0.008 |
| Hub load | 0.007 |
| Spectral | -0.047 |

The ensemble members (simple MLPs with different random seeds)
converge to similar predictions. Their disagreement doesn't
correlate with actual error. This is a known limitation of
naive ensembles of simple models.

## Gate Results

| Gate | Status | Description |
|---|---|---|
| 1: Calibration split | **PASS** | By design |
| 2: Connectivity full | FAIL | hybrid=greedy (0% coverage) |
| 3: Redundancy tail | FAIL | hybrid=greedy (0% coverage) |
| 4: Spectral no regression | **PASS** | 54% = 54% (abstained) |
| 5: Hub load no regression | **PASS** | 4% = 4% (abstained) |
| 6: Unc-error correlation | FAIL | avg corr: -0.034 |
| 7: Monotonic risk deciles | FAIL | 0/4 monotonic |
| 8: Coverage > 10% on 2 mechanisms | FAIL | 0 mechanisms |
| 9: Search savings | **PASS** | 61.5% |
| 10: Qualification | **PASS** | manifest valid, 2344 tests |

**Overall: 4/10 PASS**

## What This Tells Us

1. **The LCB framework is correct in principle**: it abstains when
   it can't confidently beat greedy. This is the right behavior for
   a safety-critical structural planner.

2. **The ensemble of simple MLPs doesn't produce calibrated uncertainty**:
   The members converge to similar predictions, and their disagreement
   doesn't predict error. This is a known limitation of naive ensembles.

3. **The calibration split works**: it correctly identifies that the
   learned model isn't reliably better than greedy at any kappa, and
   chooses a conservative operating point.

4. **The recursive planner still finds non-greedy actions** (30%
   connectivity, 32% redundancy) but the LCB arbitration doesn't
   trust these predictions enough to use them.

## What Would Fix This

The uncertainty quality is the bottleneck. Options:

1. **Better ensemble diversity**: Use bootstrap sampling (bagging)
   or different architectures, not just different random seeds.

2. **Direct Q-uncertainty**: Train M models to directly predict Q,
   not z. The ensemble Q disagreement is more directly relevant to
   the arbitration decision.

3. **Conformal prediction**: Use conformal prediction to calibrate
   the uncertainty against actual regret on the calibration set.

4. **Bayesian neural network**: Use MC dropout or variational
   inference for better-calibrated uncertainty.

5. **Hybrid approach**: Use the exp6.8.1 coverage curve approach
   but with the calibration split to select tau_sigma. The coverage
   curve showed useful results at tau_sigma=5.0; the calibration
   split would make this principled.

## Comparison Across Experiments

| Experiment | Connectivity | Spectral | Risk-aware? | Calibrated? |
|---|---:|---:|---|---|
| exp6.8 | 26% | 11% | No | No |
| exp6.8.1 | 25%* | 52% | Yes | No (feature norm) |
| exp6.8.2 | 0%** | 54% | Yes | Yes (but abstains) |

*At tau_sigma=5.0 on evaluation set (not calibrated).
**0% because LCB abstains on all tasks.

## Scientific Interpretation

This is an honest negative result on uncertainty calibration. The
LCB-margin arbitration framework is correct, but the ensemble of
simple MLPs doesn't produce useful uncertainty estimates. The system
correctly abstains rather than making unreliable decisions.

The path forward is not more architecture changes but better
uncertainty estimation. The most promising approach is probably
conformal prediction, which can calibrate any uncertainty metric
against actual regret on a calibration set without requiring
ensemble diversity.

## Qualification

- Tests: 2344 passed, 0 failed
- Manifest: 1039 files valid
- Release mode: QUALIFIED
