# RELEASE FREEZE — v6.0-exp6.2: Direct Utility Alignment

**Frozen at commit:** `c83d824`
**Qualification commit:** `8d899d6`
**Date:** 2026-08-19

## 1. Scientific Result

**Question:** Is one-step utility analytically derivable cheaply enough that learned utility prediction is unnecessary?

**Answer:** YES.

The utility function `U = -sum(w * ||z_u - z_v||^2)` has exact closed-form deltas:

| Mutation | Analytical ΔU |
|----------|---------------|
| ADD_EDGE(u,v,w) | `-w * ||z_u - z_v||^2` |
| REMOVE_EDGE(u,v) | `+w * ||z_u - z_v||^2` |
| REWEIGHT(u,v,f) | `-(w'*f - w) * ||z_u - z_v||^2` |

These are O(1) per candidate — no graph mutation needed.

## 2. Frozen Results

### Analytical vs Oracle Equivalence
- R² = 1.0000000000
- MAE < 3×10⁻⁵ (floating-point precision)
- Spearman = 1.000000
- Verified on TEST-B (4 families) and TEST-C (5 families)

### Prefilter on TEST-D (21 untouched families)
| K/N | Saved | Oracle Recall | Near-Oracle@0.05 | Regret |
|-----|-------|---------------|-------------------|--------|
| 50% | 50% | 100% | 100% | 0 |
| 25% | 74% | 100% | 100% | 0 |
| 10% | 89% | 100% | 100% | 0 |
| 5% | 94% | 100% | 100% | 0 |

### Gates (ALL PASSED)
- analytical_equivalence: PASS
- prefilter_test_d: PASS
- analytical_vs_random: PASS (21/21)
- safety: PASS

## 3. TEST-D Generators (Untouched)
- strongly_regular
- caveman
- connected_caveman
- windmill
- multipartite
- knn_geometric
- random_intersection

7 generators, 3 configs each = 21 configs

## 4. Architectural Implication

Learned utility prediction is UNNECESSARY for one-step planning.
The analytical formula is exact and O(1) per candidate.

Learning should be reserved for:
1. Multi-step value estimation
2. Risk prediction
3. Long-term structural value

## 5. Authority Boundary

The analytical utility is a scoring function, not an authority.
Every final action is exactly verified and committed exclusively
through the v5.11 CommitChannel.

## 6. Validity Assumption

AnalyticalUtilityValidity:
    latent_state_static_during_mutation = True

If latent states evolve during mutations, the closed-form delta
no longer captures the total realized effect and qualification
must rerun.
