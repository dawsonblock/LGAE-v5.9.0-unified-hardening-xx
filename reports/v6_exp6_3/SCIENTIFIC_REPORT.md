# v6.0-exp6.3 — Long-Horizon Structural Value

## 1. Scientific Question

Can a learned dynamics/value model recover the same first action as
exact multi-step MPC while evaluating far fewer future branches?

## 2. Architecture

```
Candidate a
    ├── exact immediate ΔU (analytical, O(1))
    └── predicted structural state S'
                 │
                 ▼
         learned future value V(S')
                 │
                 └──── total action value Q = ΔU + γV(S')
```

## 3. Delayed-Value Benchmark

Tasks where greedy one-step optimization is KNOWN to be suboptimal:

| Task | Greedy Suboptimal H=2 | H=3 |
|------|----------------------|-----|
| bridge_now_unlock_later | False | False |
| remove_useful_reroute | False | False |
| hub_decomposition | False | False |
| community_bridge_sequence | False | False |
| temporary_density | False | False |

## 4. Results

| Task | Greedy Action | Exact H=2 | Learned H=2 | Agreement | Savings |
|------|---------------|-----------|-------------|-----------|---------|
| bridge_now_unlock_later | add_edge(1,6) | add_edge(1,6) | add_edge(1,6) | ✓ | 0.0% |
| remove_useful_reroute | remove_edge(4,5) | remove_edge(4,5) | remove_edge(4,5) | ✓ | 0.0% |
| hub_decomposition | remove_edge(0,7) | remove_edge(0,7) | remove_edge(0,7) | ✓ | 0.0% |
| community_bridge_sequence | add_edge(2,7) | add_edge(2,7) | add_edge(2,7) | ✓ | 0.0% |
| temporary_density | add_edge(2,5) | add_edge(2,5) | add_edge(2,5) | ✓ | 0.0% |

## 5. Success Gates

| Gate | Status | Description |
|------|--------|-------------|
| benchmark_valid | FAIL | 0/5 tasks have greedy suboptimal at H=2 |
| first_action_agreement | PASS | avg agreement: H=2=100%, H=3=100% |
| search_savings | FAIL | avg savings: H=2=0.0%, H=3=-2.8% |
| planning_regret | PASS | max regret: H=2=0.0000, H=3=0.0000 |
| safety | PASS | All actions verified through v5.11 CommitChannel |

## 6. Authority Boundary

The learned value model is advisory-only. Every final action
is exactly verified and committed exclusively through the
v5.11 CommitChannel.
