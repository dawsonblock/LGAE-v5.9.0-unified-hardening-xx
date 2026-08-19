# SCIENTIFIC REPORT — exp6.5 Cross-Mechanism Foresight Generalization

**Date:** 2026-08-19
**Status:** MIXED RESULT — negative for cross-mechanism, strongly positive for scaling

## 1. Research Question

Can the learned future-residual model generalize across different
forms of non-additive structural value?

## 2. Method

Leave-one-mechanism-out (LOMO) evaluation:
  For each mechanism M_i:
    Train on {M_j : j != i}
    Test on M_i

4 mechanisms: connectivity, redundancy, hub_load, spectral_gap
~10k training samples, 25 eval tasks per mechanism

## 3. LOMO Results (NEGATIVE for cross-mechanism)

| Held-out mechanism | Suboptimal | Recovery | Regret |
|---|---|---|---|
| connectivity_threshold | 3/25 | 0% | 28.2 |
| redundancy_threshold | 0/25 | N/A | 0.0 |
| hub_load_threshold | 1/25 | 0% | 0.02 |
| spectral_gap_threshold | 9/25 | 0% | 37.6 |

The model trained on 3 mechanisms cannot recover non-greedy actions
on the held-out 4th mechanism. The learned foresight is
mechanism-specific, not transferable.

The decomposed multi-head model (intermediate causal quantities)
also fails, showing no improvement over the scalar model.

## 4. Scaling Results (STRONGLY POSITIVE)

Connectivity-trained model on connectivity scaling tasks:

| n_nodes | n_cands | Exact MPC | Model | Speedup | Savings | Regret |
|---|---|---|---|---|---|---|
| 20 | 25 | 0.50s | 0.06s | 8.5x | 88.0% | 0.000 |
| 20 | 50 | 1.95s | 0.11s | 18.1x | 94.0% | 0.000 |
| 50 | 100 | 14.3s | 0.42s | 34.0x | 97.0% | 0.000 |
| 50 | 250 | 92.7s | 1.07s | 86.7x | 98.8% | 0.000 |

**86.7x wall-clock speedup with zero regret** on 50-node, 250-candidate
problems. This far exceeds the 5x-20x target.

## 5. Interpretation

The learned foresight is:
- **Mechanism-specific**: does not transfer across delayed-value mechanisms
- **Extremely efficient within-mechanism**: 86.7x speedup, zero regret
- **Not a general structural value function**: it learns one mechanism well

This is consistent with the hypothesis that different non-additive
utilities create fundamentally different decision landscapes that
require different prediction patterns.

## 6. Implications

For practical deployment:
- Use mechanism-specific models (train on the target mechanism)
- The within-mechanism speedup is large enough for real-time planning
- Cross-mechanism transfer remains an open problem

For future research:
- Try richer features that explicitly encode mechanism-relevant observables
- Try meta-learning across mechanisms
- Try much larger training sets
- Consider that some mechanisms may not share transferable structure

## 7. Gates

| Gate | Status | Detail |
|------|--------|--------|
| A — All mechanisms suboptimal | FAIL | redundancy has 0 suboptimal |
| B — Avg recovery > 20% | FAIL | 0% across LOMO |
| C — Best LOMO recovery > 0% | FAIL | 0% |
| D — Search savings > 50% | PASS | 69.6% |
| E — No leakage | PASS | |
| F — Speedup > 2x | PASS | **86.7x** |
| G — Exact replay | PASS | |
| H — Qualification | PASS | |
| I — Calibration corr > 0 | PASS | 0.087 |

## 8. Conclusion

LGAE has demonstrated:
1. Learned structural foresight works within a mechanism (exp6.4: 91% recovery)
2. Within-mechanism scaling gives 86.7x speedup with zero regret
3. Cross-mechanism generalization does NOT work with current features

The honest scientific claim is:

"LGAE learns mechanism-specific structural foresight that enables
86.7x planning speedup with zero regret, but this foresight does
not transfer across unseen delayed-value mechanisms."
