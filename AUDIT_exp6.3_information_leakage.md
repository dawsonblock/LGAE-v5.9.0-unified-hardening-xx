# INFORMATION LEAKAGE AUDIT — exp6.3 Beam Search

**Date:** 2026-08-19
**Audited commit:** `2914498`
**Finding:** SEARCH COMPRESSION, NOT LEARNED FORESIGHT

## 1. What was claimed

exp6.3 reported:
- 100% first-action agreement with exact MPC
- 50% search savings
- Model beats greedy on 4/4 suboptimal cases

These results were presented as evidence that learned future value
helps the planner recover exact MPC decisions at lower search cost.

## 2. What actually happened

The beam search scoring function at each depth uses:

```python
u_curr = utility_fn(current_graph, z)    # EXACT non-additive utility
u_next = utility_fn(next_graph, z)       # EXACT non-additive utility
delta = u_next - u_curr                  # EXACT delta including bonus
```

The `utility_fn` is the FULL non-additive utility, including the
threshold connectivity bonus. This bonus is the non-additive
future-value signal that creates the delayed-value structure.

At depth < horizon-1, the value model contributes v=0.0.
At depth = horizon-1, the value model contributes a small term.

The exact utility dominates the score at every step.

## 3. Why this is search compression

The beam search is doing exact MPC with pruning:
- It expands branches using exact utility deltas
- It prunes using exact utility rankings
- The learned model is essentially irrelevant to the decision

With beam_width >= n_actions, beam search = exact MPC (no pruning).
With beam_width < n_actions, exact utility guides correct pruning.

The 100% agreement is expected and uninformative.

## 4. What genuine learned foresight requires

The planner must NOT have access to `utility_fn` during search.
It should use only:

1. **Analytical additive ΔU** (exact, O(1), from AnalyticalUtilityOracle)
   - This is the -w * ||z_u - z_v||^2 term
   - This is cheap and exact

2. **Learned non-additive bonus prediction**
   - This predicts the threshold connectivity bonus
   - This is the quantity that CANNOT be computed analytically
   - This is the true "residual" that requires learning

The non-additive bonus depends on global graph structure
(component count, spectral radius, diameter) which cannot be
derived from local edge deltas.

## 5. Architectural fix for exp6.4

Split the utility into two components:

    U(G) = U_additive(G) + U_bonus(G)

Where:
    U_additive(G) = -sum(w * ||z_u - z_v||^2)    [exact, analytical]
    U_bonus(G) = lambda * max(0, threshold + 1 - n_components)  [non-additive, learned]

During search:
    Q_hat(S, a) = delta_U_additive(S, a) + gamma * V_bonus_hat(S')

Where V_bonus_hat is the learned prediction of future bonus.

The exact utility_fn is used ONLY for:
- Training label generation (exact enumeration)
- Finalist replay and verification
- NOT for beam search scoring

## 6. Implication for exp6.3 results

The exp6.3 results are valid as:
- A demonstration that non-additive utility creates delayed-value structure
- A demonstration that beam search can compress exact MPC
- A demonstration that the benchmark tasks are genuinely non-greedy

The exp6.3 results are NOT valid as:
- Evidence that learned future value improves planning
- Evidence that the value model ladder contributes to decisions

## 7. Status

exp6.3 benchmark validity: REAL (greedy is genuinely suboptimal)
exp6.3 learned foresight claim: NOT YET PROVEN (information leakage)
exp6.3 search compression: REAL but not the intended claim

Next step: exp6.4 must separate exact additive from learned bonus.
