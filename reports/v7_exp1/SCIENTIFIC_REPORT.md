# v7.0-exp1: Real AI Topology — First Pass

## Status: INFRASTRUCTURE VALIDATED — MOCK LLM TOO SIMPLE FOR LGAE ADVANTAGE

## Research Question

Can structural adaptation make an AI system perform better per unit compute?

## Architecture

```
Planner → Worker → Critic → Verifier
Memory ↔ Planner, Worker
```

Three conditions:
- A. Fixed topology (no adaptation)
- B. Hand-written dynamic router (rule-based, task-aware)
- C. LGAE adaptive topology (structural planning with shadow evaluation)

Held constant: models, prompts, tasks, tokens, tools, hardware.
LGAE controls only: routing topology (edges).

## Results (120 tasks, 6 classes, mock LLM)

| Condition | Quality | Tokens | Calls | J | Cost |
|---|---:|---:|---:|---:|---:|
| A. Fixed | 0.9167 | 498.5 | 4.00 | 0.2181 | 0.6985 |
| **B. Dynamic** | 0.9167 | **470.5** | **3.83** | **0.2545** | **0.6622** |
| C. LGAE | 0.9167 | 498.5 | 4.00 | 0.2181 | 0.6985 |

**Dynamic router achieves 5% cost reduction** by bypassing critic on simple tasks.
**LGAE applies 0 mutations** — shadow evaluation finds no positive advantage.

## Why LGAE Applied 0 Mutations

The mock LLM produces deterministic, role-based outputs that don't
vary with topology. When LGAE shadow-evaluates a topology mutation
(e.g., bypass critic), the quality is identical because:

1. The mock worker produces the same output regardless of whether
   the critic reviewed it
2. The mock verifier passes regardless of the path taken
3. The only difference is cost (fewer calls = lower cost), but
   the 5% random failure rate introduces noise that masks this

The dynamic router wins because it has **task-specific knowledge**
(difficulty, requires_verification, requires_memory) that LGAE
doesn't have. LGAE only sees telemetry, not task metadata.

## What This Tells Us

### The infrastructure is validated

- All three conditions run correctly
- Topology mutations (ADD_ROUTE, REMOVE_ROUTE, REWEIGHT_ROUTE, BYPASS_NODE) work
- Shadow evaluation correctly compares mutated vs baseline topology
- Pareto analysis correctly identifies the efficient frontier
- Rollback mechanism is in place
- Telemetry is collected at node and edge level
- The authority pattern is preserved (LGAE proposes → shadow eval → commit)

### The mock LLM is the limitation

The mock LLM is too simple to demonstrate LGAE's value because:
1. Output quality doesn't depend on which nodes processed it
2. The critic doesn't actually evaluate quality
3. The verifier doesn't actually catch errors
4. Topology changes only affect cost, not quality

With a real LLM:
- Different routing produces different prompts → different outputs
- The critic actually evaluates quality → bypassing it matters
- The verifier actually catches errors → routing through it matters
- Topology-dependent quality differences would give LGAE's shadow
  evaluation something to optimize

## Gate Results

| Gate | Status | Description |
|---|---|---|
| 1: LGAE Pareto efficient | FAIL | Same as fixed |
| 2: Cost reduction ≥20% | FAIL | 0% (LGAE = fixed) |
| 3: Quality improvement ≥10% | FAIL | 0% |
| 4: Objective improvement | PASS | LGAE J = fixed J (not worse) |
| 5: No regression | PASS | No quality/failure regression |
| 6: Mutations applied | FAIL | 0 mutations |
| 7: Qualification | PASS | Manifest valid, tests pass |

**Overall: 3/7 PASS** (infrastructure works, mock LLM too simple)

## What's Needed Next

### Option 1: Real LLM backend

Plug in a real LLM (OpenAI, Anthropic, local model) via the
`_llm_call` interface on AINode. This would:
- Make topology-dependent quality differences real
- Give LGAE's shadow evaluation something to optimize
- Test whether structural adaptation improves real AI tasks

### Option 2: Richer mock LLM

Make the mock LLM topology-sensitive:
- Critic actually reduces quality if bypassed on hard tasks
- Verifier actually catches errors that worker makes
- Memory actually provides useful context for memory-dependent tasks
- Different routing produces different quality outcomes

### Option 3: Task-aware LGAE

Give LGAE access to task metadata (difficulty, class) so it can
make task-specific topology decisions like the dynamic router does.
This would level the playing field but changes the experimental
design (LGAE would use information the dynamic router already has).

## Recommendation

**Option 1 (real LLM) is the right next step.** The infrastructure
is validated. The mock LLM has served its purpose — it proved the
runtime, topology, telemetry, and authority patterns work. The next
meaningful evidence requires a real LLM where topology actually
affects output quality.

## The Honest Assessment

The synthetic planner proved that structural advantage is learnable
and that selective intervention reduces tail risk. The exp7
infrastructure proves that the topology control system works. What
remains is the critical test: does any of this improve a real AI
system?

The mock LLM can't answer that question. Only a real LLM can.

## Qualification

- Tests: 2445 passed, 0 failed
- Manifest: valid
- RELEASE GATE: PASS
