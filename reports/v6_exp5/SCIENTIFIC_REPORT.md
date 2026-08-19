# v6.0-exp5 — Lightweight Latent World Model

**Architecture:** `lightweight_latent_dynamics`
**Authorized by:** v6.0-exp4.2 (QUALIFIED_SIMPLE)

## 1. Question

Can a lightweight latent dynamics model predict the next structural
state and outcome of a mutation, generalizing to unseen graph families?

## 2. Models

### Linear Dynamics
- Parameters: 459
- Architecture: z_{t+1} = A·z_t + B·a_t + c

### MLP Dynamics
- Parameters: 1407
- Architecture: z_{t+1} = MLP([z_t, a_t]) with hidden_dim=32

## 3. Results

### Held-Out Performance

| Variant | Dynamics RMSE | Dynamics R² | Outcome RMSE | Outcome R² |
|---------|--------------|-------------|--------------|------------|
| Linear  | 1.171566 | 0.9155 | 0.016950 | 0.0453 |
| MLP     | 2.935240 | 0.4693 | 0.016950 | 0.0453 |

### Multi-Step Rollout (Validation)

| Horizon | Linear RMSE | MLP RMSE |
|---------|-------------|----------|
| h=1 | 8.026787 | 3.877515 |
| h=2 | 8.246740 | 4.751046 |
| h=3 | 8.175402 | 5.137857 |

## 4. Decision

**Best variant:** `linear`

**Trust score:** 0.9577
**Recommended planning horizon:** 2
**Exact verification fraction:** 1.0

## 5. Authority Boundary

The world model is **advisory-only**. It does not mutate authoritative
runtime state. The v5.11 CommitChannel remains the sole authority
boundary. All world model predictions used for planning must still
pass through governance and exact shadow execution.

## 6. Next Steps

The trained world model can now be used as a proposal/prediction
layer in future MPC planning experiments (exp6+). The trust report
informs how much to rely on model predictions vs exact verification.
