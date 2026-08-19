# SCIENTIFIC REPORT — exp6.3 Post-Audit Honest Result

**Date:** 2026-08-19
**Commit:** post-audit
**Status:** NEGATIVE RESULT for learned foresight, POSITIVE for benchmark validity

## 1. Research Question

Can learned bonus prediction recover exact MPC decisions while
evaluating far fewer future branches?

## 2. Architecture (Post-Audit, No Leakage)

    Q_hat(S,a) = delta_U_additive(S,a) + gamma * V_bonus_hat(S')

Where:
- delta_U_additive: exact analytical O(1) from AnalyticalUtilityOracle
- V_bonus_hat: learned Ridge regression prediction of non-additive bonus
- utility_fn NOT accessible during search

## 3. Results

### Benchmark validity (POSITIVE)
- 4/5 tasks have greedy suboptimal at H=2 (80%)
- 5/5 tasks have greedy suboptimal at H=3 (100%)
- The threshold connectivity utility creates genuine delayed-value structure

### Search compression (POSITIVE)
- Beam width=2 achieves 50% node expansion savings
- This is real but is search compression, not learned foresight

### Learned foresight (NEGATIVE)
- RidgeBonus predictor: 0% first-action agreement on suboptimal cases
- ZeroBonus predictor: 0% first-action agreement (expected — it's greedy)
- RidgeBonus does NOT beat ZeroBonus
- RidgeBonus does NOT beat greedy

## 4. Why the Learned Model Failed

The Ridge regression on simple structural features (density, degree
statistics, n_isolated, edge_ratio) cannot predict the threshold
connectivity bonus because:

1. The bonus depends on n_components, which requires global graph traversal
2. Simple features like n_isolated are a weak proxy for n_components
3. The model was trained on only 150 samples
4. Ridge regression is linear — the relationship is highly non-linear
   (bonus jumps from 0 to lambda when n_components crosses threshold)

## 5. What This Means

The pre-audit "100% agreement" was an artifact of the beam search
having access to the exact utility function (including the bonus)
at every step. This was search compression, not learned foresight.

The post-audit honest result shows that:
- The benchmark is valid (greedy IS suboptimal)
- Search compression works (beam pruning saves nodes)
- But the learned component does NOT yet provide foresight
- Better bonus prediction is needed (more data, better features, non-linear models)

## 6. Path Forward (exp6.4)

1. Better bonus prediction models (MLP, GNN, ensemble)
2. More training data from procedural task generation
3. Scale to larger graphs and candidate sets
4. Test on unseen delayed-value mechanisms (TEST-F)
5. Measure wall-clock speedup, not just node expansion
6. Pareto frontier: beam width vs regret vs evaluations vs latency
