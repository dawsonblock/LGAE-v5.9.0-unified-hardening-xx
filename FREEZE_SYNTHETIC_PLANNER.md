# FREEZE: Synthetic Planner Baseline — Commit 92b4ce1

**Frozen:** 2026-08-19
**Commit:** 92b4ce1
**Branch:** main

## Qualification at Freeze

- Tests: 2421 passed, 0 failed
- Manifest: 1102 files
- Status: QUALIFIED

## Synthetic Stack Complete

```
exp6.8    exact-transition planning          ✓
exp6.8.1  selective hybrid + spectral oracle  ✓
exp6.8.2  calibrated LCB arbitration          ✓
exp6.8.3  conformal structural advantage      ✓
exp6.8.4  advantage model identification      ✓
exp6.8.5  full structural advantage features  ✓
```

## Frozen Conformal Arbitrator Configuration

- Model: GBT (gradient-boosted trees)
- Target: T2_normalized (A / |Q_B|)
- Features: F4_full (state + action effects + topology + global)
- Calibration: split-conformal with alpha=0.20

## Key Results

- Advantage is partially learnable (Spearman 0.354 at small scale)
- F4 features show learning curve where F1 saturates
- Tail-risk reduction: P95 -77% (small scale), -12% (larger scale)
- Selective override with conformal abstention works
- Deterministic/certified fallback preserves strong baselines

## Do Not Modify After Freeze

- exp6.8 through exp6.8.5 results
- Conformal calibration logic
- v5.11 authority boundary
- CommitChannel
- Exact transition mechanics
- Candidate generation

## Next: exp7-real-ai-topology

The synthetic planner is frozen. The next question is whether
structural adaptation improves a real AI system per unit compute.
