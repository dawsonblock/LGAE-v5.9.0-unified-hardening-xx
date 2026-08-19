# v6.0-exp6 — Adaptive Model-Assisted MPC

## 1. Architecture

```
Δz = α_G ⊙ F_θ(z, a) + β_G
```

where F_θ is the global structural prior (trained on all training families)
and (α_G, β_G) is a tiny topology-local calibration fitted from a few
exact transitions.

## 2. Pipeline

```
New topology → Dynamics-OOD → Calibration → Trust → Prefilter → Exact verification → CommitChannel
```

## 3. Per-Family Results

| Family | Cal State | k* | ΔR² | Trust | N_cand | Saved | Recall@25 | Learned Regret | Random Regret |
|--------|-----------|-----|------|-------|--------|-------|-----------|----------------|---------------|
| circular_ladder | calibrated | 1 | 0.2842 | trusted_prefilter | 225 | 56% | 100% | 0.000000 | 0.000000 |
| hypercube | calibrated | 1 | 0.0365 | trusted_prefilter | 225 | 56% | 100% | 0.000000 | 0.000000 |
| ladder | calibrated | 1 | 0.1149 | trusted_prefilter | 225 | 56% | 100% | 0.000000 | 0.000000 |
| wheel | calibrated | 5 | 0.1650 | trusted_prefilter | 225 | 56% | 100% | 0.000000 | 0.000000 |

## 4. Adaptation Curves

### circular_ladder (k* = 0)

| k | Δ R² |
|---|------|
| 0 | 0.3492 |
| 2 | -0.2707 |
| 3 | 0.0752 |
| 5 | 0.3481 |
| 8 | 0.1994 |
| 10 | 0.1994 |

### hypercube (k* = 2)

| k | Δ R² |
|---|------|
| 0 | -0.2024 |
| 2 | 0.3217 |
| 3 | 0.3407 |
| 5 | 0.3528 |
| 8 | 0.3469 |
| 10 | 0.3469 |

### ladder (k* = 2)

| k | Δ R² |
|---|------|
| 0 | -0.1571 |
| 2 | 0.1382 |
| 3 | 0.1925 |
| 5 | 0.2017 |
| 8 | 0.1740 |
| 10 | 0.1740 |

### wheel (k* = 5)

| k | Δ R² |
|---|------|
| 0 | -0.8649 |
| 2 | -0.8984 |
| 3 | -0.3232 |
| 5 | 0.0899 |
| 8 | -0.1119 |
| 10 | -0.1119 |

## 5. Success Gates

| Gate | Status | Description |
|------|--------|-------------|
| A_adaptation | ✓ | 4/4 families reach ΔR² > 0 |
| B_pruning | ✗ | avg 56% evaluations saved, 100% recall@25 |
| C_regret | ✗ | 0/4 families beat random regret |
| D_calibration | ✓ | 4/4 families calibrated |
| E_safety | ✓ | All actions verified through v5.11 CommitChannel |

## 6. Authority Boundary

The learned model is **advisory-only**. Every final action is
exactly verified and committed exclusively through the v5.11
CommitChannel. The learned model assists candidate reduction
and trajectory prioritization only.
