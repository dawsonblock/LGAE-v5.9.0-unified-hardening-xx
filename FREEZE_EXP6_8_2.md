# FREEZE: exp6.8.2 Baseline — Commit e29dbf4

**Frozen:** 2026-08-21
**Commit:** e29dbf4
**Branch:** main

## Qualification at Freeze

- Tests: 2360 passed, 0 failed
- Manifest: 1045 files
- Status: QUALIFIED
- Release gate: PASS

## Experiment Results at Freeze

### exp6.8 (40be90c) — Exact-transition model-based planning

| Mechanism | Greedy | Recursive | Paired CI |
|---|---:|---:|---|
| Connectivity | 0% | 26% | [0.18, 0.35] |
| Redundancy | 7% | 28% | [0.12, 0.30] |
| Hub load | 8% | 3% | [-0.11, 0.01] |
| Spectral | 52% | 11% | [-0.52, -0.29] |

Gates: 6/10 PASS

### exp6.8.1 (10562c8) — Selective hybrid + deterministic spectral oracle

Coverage curve at tau_sigma=5.0 (connectivity):
- Coverage: 59%, Recovery: 25%, Median regret: 0.52 (vs 12.12 greedy)
- P95 regret: 4.63 (vs 30.87 greedy)
- Spectral: 0% coverage at all thresholds (deterministic oracle works)

Gates: 8/10 PASS

### exp6.8.2 (e29dbf4) — Calibrated LCB arbitration

- Ensemble of M=5, LCB-margin arbitration
- kappa chosen on calibration split, evaluated on locked test
- Result: 0% coverage on all mechanisms (ensemble abstains)
- Uncertainty-error correlation: -0.034 (uncalibrated)
- Spectral preserved at 54% (abstained correctly)

Gates: 4/10 PASS

## Known Negative Result

The ensemble of simple MLPs with different random seeds does not
produce calibrated uncertainty. The uncertainty-error correlation
is near zero. The LCB framework correctly abstains rather than
making unreliable decisions.

**The bottleneck is uncertainty calibration, not architecture.**

## Benchmark Seeds

- Training data: seed=42, per-mechanism offset *1000
- Eval tasks: seed=777, max_attempts=800-1200
- Calibration/test split: shuffle seed=123
- Ensemble member seeds: 42 + i*100

## Architecture at Freeze

```
StructuralState
├── ExactState: components, degrees, topology (exact)
├── CertifiedApproxState: spectral gap, resistance, curvature (deterministic)
└── LearnedState: path length, efficiency, future opportunity (learned, 3 dims)

G_{t+1} = T_exact(G_t, a_t)     [exact graph transition]
z_{t+1} = F(G_t, z_t, a_t)      [learned consequential state]

LCB(margin) = mu_margin - kappa * sigma_margin
Use learned only if LCB(margin) > 0
```

## What exp6.8.3 Must Improve

Replace the naive ensemble uncertainty with conformal calibration
of an advantage predictor. The decision variable changes from
"how uncertain is Q?" to "is there calibrated evidence that the
learned action is better than baseline?"

```
A(S) = Q_H(S, a_learned) - Q_H(S, a_baseline)
LCB_A = A_hat - q_{1-alpha}
override only if LCB_A > 0
```

## Do Not Modify After Freeze

- exp6.8, exp6.8.1, exp6.8.2 results
- Benchmark seeds
- Existing deterministic baseline (greedy + certified spectral)
- Existing recursive learned planner (exp6.8)
- v5.11 authority boundary
- CommitChannel
