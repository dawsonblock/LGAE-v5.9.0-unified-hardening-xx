# v6.0-exp5.2 — Cross-Family Generalization

## 1. Purpose

Determine whether structural dynamics can be represented in a
topology-invariant way that transfers across unseen graph families.

## 2. Baseline (from exp5.1, commit 693cc2f)

- TEST-B one-step R² = -2.6443 (negative = worse than mean predictor)
- Calibration corr = 0.4213 (ensemble uncertainty useful)
- Trust = 0.0

## 3. Representation Ablation

| Representation | Mode | Train R² | TEST-B R² | Δ R² | RMSE | Spearman |
|---------------|------|----------|-----------|------|------|----------|
| raw | absolute | 0.1821 | -2.6443 | 0.0000 | 2.1824 | 0.0901 |
| normalized | absolute | 0.1806 | -2.6091 | 0.0000 | 0.1839 | 0.2040 |
| normalized | delta | 0.1806 | -2.6091 | 0.5743 | 0.1839 | 0.2040 |

## 4. Leave-One-Family-Out Generalization Matrix

| Held-out family | R² | Δ R² | RMSE | Spearman | N_train | N_test |
|-----------------|-----|------|------|----------|---------|--------|
| barbell | -10.0000 | 0.0000 | 0.3568 | 0.2264 | 630 | 105 |
| cycle | -9.3367 | 0.0000 | 0.1247 | 0.2027 | 630 | 105 |
| grid | -0.2139 | 0.0000 | 0.0475 | 0.4095 | 630 | 105 |
| path | -10.0000 | 0.0000 | 0.1384 | 0.3273 | 630 | 105 |
| random_ba | -10.0000 | 0.0000 | 0.1845 | 0.3349 | 630 | 105 |
| random_er | -5.0838 | 0.0000 | 0.1676 | 0.2392 | 630 | 105 |
| star | -5.2443 | 0.0000 | 0.1442 | 0.6443 | 630 | 105 |

## 5. Family-Bootstrap Ensemble (TEST-B)

- TEST-B R²: -2.6053
- TEST-B Δ R²: 0.0000
- Calibration corr: 0.4667
- Calibration ρ: 0.2700
- N members: 8

## 6. OOD Distance Analysis

- circular_ladder: 1.4123
- hypercube: 1.4642
- ladder: 1.3593
- wheel: 1.9185

OOD distance vs error: corr=-0.5059
OOD distance vs uncertainty: corr=-0.8155

## 7. Adaptation Curves

| Family | k | R² before | R² after | RMSE before | RMSE after |
|--------|---|-----------|----------|-------------|------------|
| circular_ladder | 0 | -10.0000 | -10.0000 | 0.1275 | 0.1275 |
| circular_ladder | 5 | -10.0000 | -10.0000 | 0.1275 | 0.1273 |
| circular_ladder | 10 | -10.0000 | -10.0000 | 0.1275 | 0.1268 |
| circular_ladder | 25 | -10.0000 | -10.0000 | 0.1275 | 0.1246 |
| circular_ladder | 50 | -10.0000 | -9.4750 | 0.1275 | 0.1213 |
| hypercube | 0 | -10.0000 | -10.0000 | 0.1792 | 0.1792 |
| hypercube | 5 | -10.0000 | -10.0000 | 0.1792 | 0.1786 |
| hypercube | 10 | -10.0000 | -10.0000 | 0.1792 | 0.1778 |
| hypercube | 25 | -10.0000 | -10.0000 | 0.1792 | 0.1746 |
| hypercube | 50 | -10.0000 | -10.0000 | 0.1792 | 0.1694 |
| ladder | 0 | -5.4238 | -5.4238 | 0.0954 | 0.0954 |
| ladder | 5 | -5.4238 | -5.3876 | 0.0954 | 0.0951 |
| ladder | 10 | -5.4238 | -5.3501 | 0.0954 | 0.0949 |
| ladder | 25 | -5.4238 | -5.1805 | 0.0954 | 0.0936 |
| ladder | 50 | -5.4238 | -4.8816 | 0.0954 | 0.0913 |
| wheel | 0 | -10.0000 | -10.0000 | 0.2997 | 0.2997 |
| wheel | 5 | -10.0000 | -10.0000 | 0.2997 | 0.2980 |
| wheel | 10 | -10.0000 | -10.0000 | 0.2997 | 0.2965 |
| wheel | 25 | -10.0000 | -10.0000 | 0.2997 | 0.2910 |
| wheel | 50 | -10.0000 | -10.0000 | 0.2997 | 0.2821 |

## 8. Extended Rollout (Normalized Delta)

### Validation

| Horizon | NRMSE | R² | N samples |
|---------|-------|-----|-----------|
| h=1 | 0.2103 | 0.9968 | 24 |
| h=10 | 0.0000 | 0.0000 | 0 |
| h=2 | 0.2842 | 0.9943 | 18 |
| h=3 | 0.3504 | 0.9913 | 12 |
| h=5 | 0.0000 | 0.0000 | 0 |

### TEST-B

| Horizon | NRMSE | R² | N samples |
|---------|-------|-----|-----------|
| h=1 | 0.2044 | 0.9784 | 48 |
| h=10 | 0.0000 | 0.0000 | 0 |
| h=2 | 0.2396 | 0.9715 | 36 |
| h=3 | 0.2695 | 0.9627 | 24 |
| h=5 | 0.0000 | 0.0000 | 0 |

## 9. Conclusion

No representation achieves positive TEST-B R².
Best: normalized with R²=-2.6091

No adaptation achieves positive R².

## 10. exp6 Authorization

**NOT READY** for exp6 MPC.
Gates not met: TEST-B R² must be > 0, calibration > 0.3.

## 11. Authority Boundary

The world model is **advisory-only**. The v5.11 CommitChannel remains
the sole authority boundary.
