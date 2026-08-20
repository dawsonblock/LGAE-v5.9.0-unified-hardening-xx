# FREEZE: exp6.8.3 Baseline — Commit 62f0a40

**Frozen:** 2026-08-19
**Commit:** 62f0a40
**Branch:** main

## Qualification at Freeze

- Tests: 2394 passed, 0 failed
- Manifest: 1074 files
- Status: QUALIFIED
- Release gate: PASS

## Experiment Results at Freeze

### exp6.8.3 (62f0a40) — Conformal Structural Advantage

Architecture:
  A* = Q_H(S, a_learned) - Q_H(S, a_baseline)  [exact advantage]
  LCB_A = A_hat - q_{1-alpha}                   [conformal LCB]
  override only if LCB_A > 0
  alpha chosen on calibration, evaluated on test

Key results (linear model, 1000 train/mech):
  Redundancy: P95 regret 517 vs 633 (-18%), CVaR95 709 vs 773 (-8%)
  Connectivity: median regret 5.85 vs 6.71, coverage 2.5%, precision 50%
  Spectral: 0% coverage (correctly abstains)
  Hub load: 1.4% coverage (nearly abstains)

Gates: 7/11 PASS

## Known Bottleneck

The advantage A* has std~200 with 50-dim features and ~400 examples/mech.
The conformal framework works but the advantage model is too weak.
The question for exp6.8.4: is A* actually learnable?

## Do Not Modify After Freeze

- Conformal calibration logic
- Arbitration rule (LCB_A > 0)
- Exact transition mechanics
- Candidate generation
- v5.11 authority boundary
- CommitChannel
- exp6.8/6.8.1/6.8.2/6.8.3 results
