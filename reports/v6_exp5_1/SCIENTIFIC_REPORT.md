# v6.0-exp5.1 — Scientific Repair & Rollout Qualification

## 1. Purpose

Fix methodological defects identified in the audit of exp4.2/exp5.
This is a scientific repair release, not exp6 MPC.

## 2. Fixes Applied

- Multi-factor trust score (replaces 0.5 + R²/2)
- Fresh TEST-B split (wheel, ladder, circular_ladder, hypercube)
- Multi-step rollout with realized-only trajectories
- Per-feature normalized RMSE
- R² clipping for near-zero variance dimensions
- Realistic risk distribution (not constant zero)
- Realistic cost (graph complexity, fragmentation)
- Mutation type diversity (ADD_EDGE + REMOVE_EDGE)
- Separate task conclusions per target
- Scientific controls always evaluated on held-out

## 3. Mutation Type Distribution

- ADD_EDGE: 450
- REMOVE_EDGE: 495
- add_edge: 165

## 4. Results

### One-Step Prediction (TEST-B, fresh)

| Variant | Dynamics R² | Outcome R² | Risk RMSE | Cost RMSE |
|---------|-------------|------------|-----------|-----------|
| linear | 0.6932 | -1.1464 | 0.033640 | 0.076029 |
| mlp | -1.2112 | -1.1464 | 0.033640 | 0.076029 |

### Multi-Step Rollout (Validation, Normalized RMSE)

| Horizon | Linear RMSE | MLP RMSE | Linear R² | MLP R² |
|---------|-------------|----------|-----------|--------|
| h=1 | 0.0003 | 2.1420 | 0.0000 | 0.0000 |
| h=2 | 0.0003 | 2.7071 | 0.0000 | 0.0000 |
| h=3 | 0.0003 | 7.9871 | 0.0000 | 0.0000 |

### Multi-Factor Trust Scores

| Factor | Linear | MLP |
|--------|--------|-----|
| one_step_quality | 0.6932 | 0.0000 |
| rollout_quality | 0.0000 | 0.0000 |
| calibration_quality | 0.0000 | 0.0000 |
| tail_safety | 0.7600 | 0.7600 |
| ood_safety | 1.0000 | 1.0000 |

| **Trust score** | **0.0000** | **0.0000** |

## 5. Decision

**Best variant:** `linear`
**Trust score:** 0.0000
**Recommended planning horizon:** 1
**Exact verification fraction:** 1.0

## 6. Readiness for exp6 MPC

The multi-factor trust score is 0.0000.
**NOT READY for exp6 MPC.** Trust is too low.

## 7. Authority Boundary

The world model is **advisory-only**. The v5.11 CommitChannel remains
the sole authority boundary. All predictions must pass through
governance and exact shadow execution.
