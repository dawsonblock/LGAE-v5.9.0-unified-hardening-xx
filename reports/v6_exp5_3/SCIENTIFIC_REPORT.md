# v6.0-exp5.3 — Topology-Invariant Representation Study

## 1. Purpose

Determine whether structural dynamics can be represented in a
topology-invariant way that transfers across unseen graph families.

## 2. Methodology Corrections from exp5.2

1. **Realized-only evaluation**: Train and evaluate on realized records only,
   not counterfactual. The exp5.2 one-step R²=-2.609 was contaminated by
   360 counterfactual records with 35x larger delta magnitudes.
2. **Delta R² as primary metric**: The absolute R² is inflated by low
   variance of state changes (zero-delta baseline already gets R²=0.927).
3. **Zero-delta baseline**: Always compare against predicting no change.
4. **State decomposition**: Split into invariant/context/derived dimensions.

## 3. Representation Ladder (R0-R7)

| Representation | Dim | Δ R² | Δ R² (invariant) | Zero-Δ R² | Beats zero? |
|---------------|-----|------|-------------------|-----------|-------------|
| R0_current | 20 | -10.0000 | -10.0000 | -0.4266 | False |
| R1_graphlet | 8 | -0.0458 | -0.0458 | -0.4551 | True |
| R2_spectral | 3 | -0.2578 | -1.2080 | -0.1697 | False |
| R3_curvature | 3 | -10.0000 | -10.0000 | -0.2125 | False |
| R4_graphlet_spectral | 11 | -0.3488 | -0.3561 | -0.4437 | True |
| R5_graphlet_geometric | 11 | -10.0000 | -10.0000 | -0.3429 | False |
| R6_invariant_hybrid | 15 | -10.0000 | -10.0000 | -0.3428 | False |
| R7_learned_encoder | 15 | -10.0000 | -10.0000 | -0.3428 | False |

Best: **R1_graphlet** with delta R²=-0.0458

## 4. Leave-One-Family-Out (R1_graphlet)

| Held-out family | Δ R² | Δ R² (invariant) | Zero-Δ R² | Beats? | N |
|-----------------|------|-------------------|-----------|--------|---|
| barbell | -2.6544 | -2.6544 | -2.3032 | False | 15 |
| cycle | 0.0468 | 0.0468 | -1.4335 | True | 15 |
| grid | 0.1689 | 0.1689 | -0.4730 | True | 15 |
| path | -0.1985 | -0.1985 | -0.1626 | False | 15 |
| random_ba | -1.8549 | -1.8549 | -0.1660 | False | 15 |
| random_er | -1.6383 | -1.6383 | -0.2100 | False | 15 |
| star | -1.6245 | -1.6245 | -0.5645 | False | 15 |

## 5. Component-Wise Adaptation

| Family | k | Adaptation | Δ R² | Δ R² (inv) | Zero-Δ R² | Beats? |
|--------|---|------------|------|------------|-----------|--------|
| circular_ladder | 0 | none | 0.3043 | 0.3043 | -1.5506 | True |
| circular_ladder | 5 | none | 0.3043 | 0.3043 | -1.5506 | True |
| circular_ladder | 5 | bias_only | 0.3415 | 0.3415 | -1.5506 | True |
| circular_ladder | 5 | scale_offset | 0.8046 | 0.8046 | -1.5506 | True |
| circular_ladder | 5 | low_rank_r2 | 0.7255 | 0.7255 | -1.5506 | True |
| circular_ladder | 5 | full_retrain | 0.3550 | 0.3550 | -1.5506 | True |
| circular_ladder | 10 | none | 0.3043 | 0.3043 | -1.5506 | True |
| circular_ladder | 10 | bias_only | 0.1557 | 0.1557 | -1.5506 | True |
| circular_ladder | 10 | scale_offset | 0.8145 | 0.8145 | -1.5506 | True |
| circular_ladder | 10 | low_rank_r2 | 0.5749 | 0.5749 | -1.5506 | True |
| circular_ladder | 10 | full_retrain | 0.3566 | 0.3566 | -1.5506 | True |
| circular_ladder | 25 | none | 0.3043 | 0.3043 | -1.5506 | True |
| circular_ladder | 25 | bias_only | 0.1557 | 0.1557 | -1.5506 | True |
| circular_ladder | 25 | scale_offset | 0.8145 | 0.8145 | -1.5506 | True |
| circular_ladder | 25 | low_rank_r2 | 0.5749 | 0.5749 | -1.5506 | True |
| circular_ladder | 25 | full_retrain | 0.3566 | 0.3566 | -1.5506 | True |
| circular_ladder | 50 | none | 0.3043 | 0.3043 | -1.5506 | True |
| circular_ladder | 50 | bias_only | 0.1557 | 0.1557 | -1.5506 | True |
| circular_ladder | 50 | scale_offset | 0.8145 | 0.8145 | -1.5506 | True |
| circular_ladder | 50 | low_rank_r2 | 0.5749 | 0.5749 | -1.5506 | True |
| circular_ladder | 50 | full_retrain | 0.3566 | 0.3566 | -1.5506 | True |
| hypercube | 0 | none | -0.1748 | -0.1748 | -0.1697 | False |
| hypercube | 5 | none | -0.1748 | -0.1748 | -0.1697 | False |
| hypercube | 5 | bias_only | 0.3284 | 0.3284 | -0.1697 | True |
| hypercube | 5 | scale_offset | 0.6236 | 0.6236 | -0.1697 | True |
| hypercube | 5 | low_rank_r2 | -1.3806 | -1.3806 | -0.1697 | False |
| hypercube | 5 | full_retrain | 0.1704 | 0.1704 | -0.1697 | True |
| hypercube | 10 | none | -0.1748 | -0.1748 | -0.1697 | False |
| hypercube | 10 | bias_only | 0.2752 | 0.2752 | -0.1697 | True |
| hypercube | 10 | scale_offset | 0.6405 | 0.6405 | -0.1697 | True |
| hypercube | 10 | low_rank_r2 | -3.2020 | -3.2020 | -0.1697 | False |
| hypercube | 10 | full_retrain | 0.2388 | 0.2388 | -0.1697 | True |
| hypercube | 25 | none | -0.1748 | -0.1748 | -0.1697 | False |
| hypercube | 25 | bias_only | 0.2752 | 0.2752 | -0.1697 | True |
| hypercube | 25 | scale_offset | 0.6405 | 0.6405 | -0.1697 | True |
| hypercube | 25 | low_rank_r2 | -3.2020 | -3.2020 | -0.1697 | False |
| hypercube | 25 | full_retrain | 0.2388 | 0.2388 | -0.1697 | True |
| hypercube | 50 | none | -0.1748 | -0.1748 | -0.1697 | False |
| hypercube | 50 | bias_only | 0.2752 | 0.2752 | -0.1697 | True |
| hypercube | 50 | scale_offset | 0.6405 | 0.6405 | -0.1697 | True |
| hypercube | 50 | low_rank_r2 | -3.2020 | -3.2020 | -0.1697 | False |
| hypercube | 50 | full_retrain | 0.2388 | 0.2388 | -0.1697 | True |
| ladder | 0 | none | -0.0323 | -0.0323 | -0.2851 | True |
| ladder | 5 | none | -0.0323 | -0.0323 | -0.2851 | True |
| ladder | 5 | bias_only | 0.1882 | 0.1882 | -0.2851 | True |
| ladder | 5 | scale_offset | 0.2215 | 0.2215 | -0.2851 | True |
| ladder | 5 | low_rank_r2 | 0.4547 | 0.4547 | -0.2851 | True |
| ladder | 5 | full_retrain | 0.0449 | 0.0449 | -0.2851 | True |
| ladder | 10 | none | -0.0323 | -0.0323 | -0.2851 | True |
| ladder | 10 | bias_only | 0.1351 | 0.1351 | -0.2851 | True |
| ladder | 10 | scale_offset | 0.1443 | 0.1443 | -0.2851 | True |
| ladder | 10 | low_rank_r2 | 0.2055 | 0.2055 | -0.2851 | True |
| ladder | 10 | full_retrain | 0.0865 | 0.0865 | -0.2851 | True |
| ladder | 25 | none | -0.0323 | -0.0323 | -0.2851 | True |
| ladder | 25 | bias_only | 0.1351 | 0.1351 | -0.2851 | True |
| ladder | 25 | scale_offset | 0.1443 | 0.1443 | -0.2851 | True |
| ladder | 25 | low_rank_r2 | 0.2055 | 0.2055 | -0.2851 | True |
| ladder | 25 | full_retrain | 0.0865 | 0.0865 | -0.2851 | True |
| ladder | 50 | none | -0.0323 | -0.0323 | -0.2851 | True |
| ladder | 50 | bias_only | 0.1351 | 0.1351 | -0.2851 | True |
| ladder | 50 | scale_offset | 0.1443 | 0.1443 | -0.2851 | True |
| ladder | 50 | low_rank_r2 | 0.2055 | 0.2055 | -0.2851 | True |
| ladder | 50 | full_retrain | 0.0865 | 0.0865 | -0.2851 | True |
| wheel | 0 | none | -1.1395 | -1.1395 | -1.7994 | True |
| wheel | 5 | none | -1.1395 | -1.1395 | -1.7994 | True |
| wheel | 5 | bias_only | 0.0888 | 0.0888 | -1.7994 | True |
| wheel | 5 | scale_offset | 0.9069 | 0.9069 | -1.7994 | True |
| wheel | 5 | low_rank_r2 | -2.1689 | -2.1689 | -1.7994 | False |
| wheel | 5 | full_retrain | 0.0349 | 0.0349 | -1.7994 | True |
| wheel | 10 | none | -1.1395 | -1.1395 | -1.7994 | True |
| wheel | 10 | bias_only | -0.2029 | -0.2029 | -1.7994 | True |
| wheel | 10 | scale_offset | 0.9415 | 0.9415 | -1.7994 | True |
| wheel | 10 | low_rank_r2 | -5.5901 | -5.5901 | -1.7994 | False |
| wheel | 10 | full_retrain | -0.0588 | -0.0588 | -1.7994 | True |
| wheel | 25 | none | -1.1395 | -1.1395 | -1.7994 | True |
| wheel | 25 | bias_only | -0.2029 | -0.2029 | -1.7994 | True |
| wheel | 25 | scale_offset | 0.9415 | 0.9415 | -1.7994 | True |
| wheel | 25 | low_rank_r2 | -5.5901 | -5.5901 | -1.7994 | False |
| wheel | 25 | full_retrain | -0.0588 | -0.0588 | -1.7994 | True |
| wheel | 50 | none | -1.1395 | -1.1395 | -1.7994 | True |
| wheel | 50 | bias_only | -0.2029 | -0.2029 | -1.7994 | True |
| wheel | 50 | scale_offset | 0.9415 | 0.9415 | -1.7994 | True |
| wheel | 50 | low_rank_r2 | -5.5901 | -5.5901 | -1.7994 | False |
| wheel | 50 | full_retrain | -0.0588 | -0.0588 | -1.7994 | True |

## 6. Dynamics-OOD Distance

- circular_ladder: 2.4978
- hypercube: 3.0447
- ladder: 2.4097
- wheel: 4.9707

Dynamics-OOD vs error: corr=0.3045

## 7. Extended Rollout (Realized, Delta)

### Validation

| Horizon | NRMSE | R² | N |
|---------|-------|-----|---|
| h=1 | 122729.3800 | -10.0000 | 24 |
| h=10 | 0.0000 | 0.0000 | 0 |
| h=2 | 115384.0841 | -10.0000 | 18 |
| h=3 | 112679.6304 | -10.0000 | 12 |
| h=5 | 0.0000 | 0.0000 | 0 |

### TEST-B

| Horizon | NRMSE | R² | N |
|---------|-------|-----|---|
| h=1 | 173907.2011 | -10.0000 | 48 |
| h=10 | 0.0000 | 0.0000 | 0 |
| h=2 | 160897.5671 | -10.0000 | 36 |
| h=3 | 139571.8748 | -10.0000 | 24 |
| h=5 | 0.0000 | 0.0000 | 0 |

## 8. exp6 Authorization Gates

- loo_r2_positive: ✓
- test_b_r2_positive: ✗
- beats_zero_delta: ✓
- calibration_above_0_3: ✗
- rollout_h3_bounded: ✓

**NOT READY** for exp6 MPC.

## 9. Authority Boundary

The world model is **advisory-only**. The v5.11 CommitChannel remains
the sole authority boundary.
