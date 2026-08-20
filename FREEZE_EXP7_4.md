# FREEZE: exp7.4 — Learned Routing Policy

**Commit:** `09fb845`
**Date:** 2025-01-24
**Status:** FROZEN — do not modify routing architecture until exp7.5 completes

## Frozen Results

```
Condition                  Quality   Tokens    J        Cost
A_fixed                     0.8270    1036    -0.3468   1.471
B_dynamic                   0.8333     414     0.5114   0.614
D_lgae_task_conditioned     0.8312     784    -0.0981   1.123
D_lgae_node_necessity       0.8333     389     0.5031   0.576
```

## Frozen Routing Policy

```
93% minimal topology (verifier only)
 7% full topology (researcher + critic + memory + verifier)
```

Token ratio vs dynamic: **0.94x** (LGAE uses fewer tokens than hand-designed router)

## Frozen Configuration

### Objective weights (DO NOT CHANGE)

```python
ObjectiveWeights(
    w_quality=1.0,
    lambda_tokens=0.3,
    lambda_latency=0.2,
    lambda_calls=0.2,
    lambda_failures=0.5,
    token_budget=2000,
    latency_budget_ms=5000.0,
    call_budget=6,
)
```

### Task splits

```
Benchmark seed: 42
Tasks per class: 50
Total tasks: 300
Task classes: simple_factual, research_synthesis, coding_debugging,
              multi_step_reasoning, verification_sensitive, memory_dependent
```

### Task embedding

```
Dimensions: 34
Components:
  - 18 manual structural features (token count, complexity, keywords)
  - 6 category scores (factual, research, coding, reasoning, verification, memory)
  - 4 structural features (length, complexity, difficulty, avg word length)
  - 6 hash-based dense features (discriminability)
Frozen: deterministic, no training required
```

### k-NN configuration

```python
MarginalValueEstimator(
    k=5,
    min_samples=3,
)
```

### Marginal-value training protocol

```
Calibration interval: 20 tasks
Shadow batch size: 5 tasks
For each shadow task × each optional node:
  Run with node → J_with
  Run without node → J_without
  Record ΔJ = J_with - J_without
Optional nodes: researcher, critic, verifier, memory
```

### Topology definition

```
Six nodes: Planner, Worker, Researcher, Critic, Verifier, Memory
Default edges:
  planner → worker (weight 1.0)
  planner → researcher (weight 0.5)
  researcher → worker (weight 0.8)
  worker → critic (weight 0.7)
  worker → verifier (weight 0.3)
  critic → verifier (weight 1.0)
  memory → planner (weight 0.4)
  planner → memory (weight 0.2)
Edge selection: weighted random
```

### Authority/governance rules

```
1. LGAE proposes topology changes only through NodeNecessityRouter
2. Proposals go through shadow evaluation
3. Conformal advantage gate (alpha=0.2) must approve
4. Governor approves/rejects
5. CommitChannel is sole authoritative mutation path
6. KNOWN_GOOD_TOPOLOGY preserved for rollback
7. Online rollback: rolling J < baseline - ε triggers revert
8. No direct LLM modification of routing graph
```

### Benchmark seeds

```
Benchmark generation: seed=42
Mock backend: seed=42
```

## What Changes in exp7.5

```
ONLY the model backend changes.
  MockModelBackend → OpenAIBackend

Everything else remains frozen:
  - Task splits
  - Task embeddings
  - Node prompts (will be versioned v1)
  - Objective coefficients
  - Topology definitions
  - k-NN configuration
  - Marginal-value training protocol
  - Authority/governance rules
  - Benchmark seeds
```

## Scientific Question for exp7.5

```
Does LGAE's learned sparse routing policy still beat fixed and
hand-designed routing when every cognitive node is backed by a
real LLM?
```

## Success Gates for exp7.5 (predeclared)

```
A. Real backend executes every role
B. Topology measurably changes context/output
C. Fixed/Dynamic/LGAE use identical model and prompts
D. Deterministic graders validate benchmark where possible
E. LGAE quality >= Fixed - tolerance
F. LGAE all-in token cost < Fixed
G. LGAE J > Fixed
H. LGAE quality approximately matches Dynamic
I. LGAE tokens <= Dynamic
J. Nonzero meaningful adaptive routing
K. No catastrophic failure-rate regression
L. Rollback works on degraded topology
M. Test split untouched during policy tuning
N. Authority/CommitChannel boundary preserved
O. Full release qualification passes
```
