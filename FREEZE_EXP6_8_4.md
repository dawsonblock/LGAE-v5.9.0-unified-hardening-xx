# FREEZE: exp6.8.4 Baseline — Commit a5e9042

**Frozen:** 2026-08-19
**Commit:** a5e9042
**Branch:** main

## Qualification at Freeze

- Tests: 2415 passed, 0 failed
- Manifest: 1091 files
- Status: QUALIFIED

## Key Result at Freeze

Advantage is partially learnable:
  GBT x normalized x F1 x N=250:
    Spearman 0.354, precision 75%, coverage 22%
    P95 regret: 4.59 vs 20.31 (-77%)
    CVaR95: 13.18 vs 22.70 (-42%)

Learning curves: 0/8 improve with F1 features.
Bottleneck: representation, not data or model capacity.

## exp6.8.5 Must Answer

Does F4 (full structural features) break the ceiling?
If Spearman improves with N under F4, representation was the issue.
If not, freeze and move to exp7-real-ai-topology.
