# v6.0-exp6.1 — Real Candidate Prefilter Qualification

## 1. Scientific Question

Can 1-5 exact calibration transitions enable a learned filter to
eliminate most candidate evaluations while preserving near-oracle
decisions on unseen topology generators?

## 2. Protocol

The protocol was frozen before inspecting TEST-C:
- Representation: R1_graphlet (8-dim)
- Calibration: 1-5 transitions, regularized scale-offset
- Pruning ratios: K/N = 50%, 25%, 10%, 5%
- UCB kappa: 0, 0.5, 1, 2
- Near-oracle epsilon: 0.01, 0.05, 0.1, 0.5
- Candidate types: ADD_EDGE, REMOVE_EDGE, REWEIGHT_UP, REWEIGHT_DOWN, BRIDGE, LOCAL_REWIRE, HUB_CONNECT

## 3. TEST-B Results (Development)

| Family | N_cand | Cal | ΔR² | Oracle best |
|--------|--------|-----|------|-------------|
| wheel | 40 | limited | 0.0000 | 6.529823 |
| ladder | 41 | limited | 0.0000 | 4.284134 |
| circular_ladder | 42 | limited | 0.0000 | 4.284126 |
| hypercube | 44 | limited | 0.0000 | 4.163147 |

## 4. TEST-C Results (Untouched)

| Family | N_cand | Cal | ΔR² | Oracle best | Utility std |
|--------|--------|-----|------|-------------|-------------|
| test_c_sbm_0 | 42 | limited | 0.0000 | 3.245041 | 1.446681 |
| test_c_sbm_1 | 41 | limited | 0.0000 | 3.358456 | 2.136090 |
| test_c_sbm_2 | 44 | limited | 0.0000 | 3.335632 | 2.099725 |
| test_c_geometric_3 | 39 | limited | 0.0000 | 3.471336 | 1.762223 |
| test_c_geometric_4 | 40 | limited | 0.0000 | 3.676971 | 1.812618 |
| test_c_geometric_5 | 34 | limited | 0.0000 | 2.823418 | 1.780425 |
| test_c_regular_6 | 43 | limited | 0.0000 | 3.770111 | 1.950932 |
| test_c_regular_7 | 46 | limited | 0.0000 | 5.474636 | 1.974970 |
| test_c_regular_8 | 44 | limited | 0.0000 | 6.902267 | 3.143214 |
| test_c_powerlaw_cluster_9 | 41 | limited | 0.0000 | 4.120598 | 1.743883 |
| test_c_powerlaw_cluster_10 | 47 | limited | 0.0000 | 3.679291 | 2.886183 |
| test_c_powerlaw_cluster_11 | 47 | limited | 0.0000 | 5.527996 | 2.600512 |
| test_c_lollipop_12 | 43 | limited | 0.0000 | 3.965088 | 2.147689 |
| test_c_lollipop_13 | 38 | limited | 0.0000 | 2.771851 | 1.715334 |
| test_c_lollipop_14 | 40 | limited | 0.0000 | 3.520309 | 1.930513 |
| test_c_lobster_15 | 46 | limited | 0.0000 | 2.772001 | 2.373724 |
| test_c_lobster_16 | 48 | limited | 0.0000 | 7.071976 | 2.538639 |
| test_c_lobster_17 | 46 | limited | 0.0000 | 5.450935 | 2.216537 |
| test_c_highdim_grid_18 | 45 | limited | 0.0000 | 2.762810 | 2.384826 |
| test_c_highdim_grid_19 | 43 | limited | 0.0000 | 2.344795 | 1.855423 |
| test_c_highdim_grid_20 | 46 | limited | 0.0000 | 4.296867 | 2.215929 |

## 5. Success Gates

| Gate | Status | Description |
|------|--------|-------------|
| A_adaptation | FAIL | 0/21 TEST-C families reach ΔR² > 0 |
| B_pruning | FAIL | avg 89% saved, 0% near-oracle@0.05 (K/N=10%) |
| C_regret | FAIL | 3/21 adapted UCB beats heuristic |
| D_calibration | FAIL | 0/21 TEST-C families calibrated |
| E_safety | PASS | All actions verified through v5.11 CommitChannel |

## 6. Authority Boundary

The learned model is advisory-only. Every final action is
exactly verified and committed exclusively through the v5.11
CommitChannel.
