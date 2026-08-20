# FREEZE: exp7.1 Baseline — Commit 663be14

**Frozen:** 2026-08-19
**Commit:** 663be14

## Status at Freeze

- Tests: 2445 passed, 0 failed
- Manifest: 1127 files
- Status: QUALIFIED

## exp7.1 Result

Infrastructure validated with mock LLM:
- 5-node topology, 3 conditions (Fixed, Dynamic, LGAE)
- Topology mutations work (ADD/REMOVE/REWEIGHT/BYPASS)
- Shadow evaluation, Pareto analysis, rollback all functional
- LGAE applied 0 mutations (mock LLM is topology-invariant)
- Dynamic router achieved 5% cost reduction via task-specific rules

## Limitation Identified

The mock LLM produces deterministic outputs that don't vary with
topology. Topology changes only affect cost, not quality. This
means LGAE has no causal signal to learn from.

## exp7.2 Must Address

1. Topology must change actual context/prompts per node
2. ModelBackend protocol for pluggable backends
3. Researcher node (6 nodes total)
4. Deterministic quality evaluation per task class
5. Normalized objective with budget normalization
6. Real shadow evaluation
7. Quality-adjusted compute efficiency (Q/Tokens)
8. 300-600 tasks across 6 families
9. LGAE must not receive task labels directly
10. Pre-defined gates before evaluation
