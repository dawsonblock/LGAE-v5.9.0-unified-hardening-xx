<div align="center">

# LGAE v5.11.0

### Autonomous Structural Intelligence Runtime

**One canonical end-to-end cycle: observe, reason, propose, verify, commit, learn.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10+](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-1604%20passing-brightgreen.svg)]()
[![Phases](https://img.shields.io/badge/phases-50%20complete-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gauge: SO(d)](https://img.shields.io/badge/gauge-SO(d)%20%7C%20SU(2)%20%7C%20GL(d)-purple.svg)]()
[![Runtime: autonomous](https://img.shields.io/badge/runtime-autonomous-brightgreen.svg)]()

</div>

---

## What is LGAE v5.11.0?

LGAE v5.11.0 is an **autonomous structural intelligence runtime** that unifies graph learning, geometric diagnostics, deterministic governance, cryptographic evidence, and offline RL into one authoritative cycle.

The governing principle:

> **Learned models propose. Deterministic governance authorizes. Evidence proves.**

Every structural mutation passes through a strict pipeline: a learned model generates candidates, a deterministic authority governor approves or rejects them, and cryptographic evidence proves what happened. The runtime never allows learned models to directly mutate state.

### The canonical cycle

```
Observation → Stable Snapshot → Adaptive Diagnostics → Reasoning Graph + Memory
    → Candidate Generation → Learned + Algorithmic Ranking → Epistemic/Aleatoric Uncertainty
    → Information Gain + Risk → Multi-Step MPC → Joint Topology + Gauge Action
    → Shadow Transaction → Exact/Escalating Verification → Authority Governor
    → Reject / Quarantine / Commit → Atomic State Update → Selective Cache Invalidation
    → Local Structural Credit → Signed Evidence / Receipt → Replay / Experience → Learn
```

---

## What's new in v5.10.0

v5.10.0 adds **50 phases** (495 new tests, 719 → 1214 passing) on top of the v5.9.0 unified-hardening baseline.

### Runtime core

| Phase | Subsystem | Description |
|-------|-----------|-------------|
| 0 | Baseline freeze | v5.9.0 reference artifact (719 tests) |
| 1 | Canonical runtime | `LGAERuntime` orchestrator over existing engines |
| 2 | Authority boundaries | Strict proposal / verification / commit separation |
| 3 | Seqlock state | Unified runtime state with seqlock enforcement |
| 4 | Cache coherence | Mandatory cache coherence with selective invalidation |
| 5 | Adaptive diagnostics | L0–L3 geometric diagnostics cascade |
| 6 | Certification | 6 ordered certification levels |
| 7 | Authority policy | `HIGH_IMPACT` mutation authority classification |
| 8 | Candidate union | Canonical `CandidateID`, deterministic deduplication |
| 9 | Retrieval evaluation | recall@K, oracle recall, regret metrics |

### Learning & intelligence

| Phase | Subsystem | Description |
|-------|-----------|-------------|
| 11 | Epistemic uncertainty | `sigma_OOD > sigma_ID` via distance penalty |
| 12 | Calibration | ECE, NLL, Brier score metrics |
| 13 | Information gain | Ensemble disagreement, UCB, posterior variance reduction |
| 14 | Multi-step MPC | Receding-horizon planning with bounded branching |
| 15 | Joint actions | Atomic composite actions (add + reweight + prune) |
| 16 | Lie-group geometry | SO(3), SU(2), GL(d) exponential maps |
| 17 | Sheaf consistency | Cycle flatness certification |
| 18 | Credit assignment | Direct, feature-based, temporal, baseline |
| 19 | Causal credit | Do-calculus counterfactual credit |
| 20 | Replay buffer | Prioritized, stratified, deduplicated, FIFO |
| 21 | Hard-negative replay | Overconfident wrong prediction mining |
| 22 | Offline RL | Conservative Q-Learning (CQL) |

### Evidence & persistence

| Phase | Subsystem | Description |
|-------|-----------|-------------|
| 28 | Merkle evidence | SHA-256 Merkle tree aggregation |
| 29 | Replayable decisions | Decision ledger with deterministic hashes |
| 30 | WAL | Crash-safe transactions (ARIES-style) |
| 31 | Checkpointing | Signed Merkle checkpoint chain |

### Testing & qualification

| Phase | Subsystem | Description |
|-------|-----------|-------------|
| 23 | Baseline competition | Regret distribution vs FoSR, Forman, effective resistance |
| 24 | Curriculum | 11 diverse graph families (BA, WS, ER, SBM, path, cycle, etc.) |
| 25 | OOD qualification | Held-out graph family evaluation |
| 27 | Adversarial testing | Structural perturbation stress tests |
| 37 | Formal invariants | `@invariant` contracts |
| 38 | Property-based tests | Hypothesis strategies |
| 39 | Metamorphic tests | Metamorphic relations |
| 40 | Observability | JSONL metrics sink |
| 41 | Decision trace | Human-readable reasoning chain |
| 42 | CLI | `inspect`, `diagnose`, `propose`, `step`, `run`, `qualify` |
| 43 | Config presets | Research / production / benchmark |
| 44 | Mode enforcement | Research vs production fail-closed |
| 45 | Model registry | Append-only, auditable promotion tracking |
| 46 | Promotion gates | EXPERIMENTAL → PRODUCTION |
| 47 | Scientific gate | Regret < baseline, OOD, IG > 0 |
| 48 | Safety gate | Zero-violation checks |
| 49 | Performance gate | S / M / L / XL tiers |

### Performance

| Phase | Subsystem | Description |
|-------|-----------|-------------|
| 32 | Profiling | Performance profiling harness |
| 33 | Tensor graph ops | NetworkX-free hot paths (10–100x speedup) |
| 34 | Sparse graphs | CSR format, O(1) neighbor lookup |
| 35 | GPU path | CUDA / MPS auto-select |
| 36 | Batched counterfactuals | Parallel candidate evaluation |

### Benchmarks

| Phase | Subsystem | Description |
|-------|-----------|-------------|
| 26 | Real graph benchmarks | Karate Club, Dolphin, Les Misérables, PolBooks, Football |

---

## Quickstart

```bash
# Install
pip install -e .

# Run the demo
python -m lgae_v3.cli demo --nodes 20 --steps 5

# Inspect runtime state
python -m lgae_v3.cli inspect

# Run one step
python -m lgae_v3.cli step

# Run qualification
python -m lgae_v3.cli qualify
```

## Code example

```python
from lgae_v3.runtime import LGAERuntime, RuntimeConfig, RuntimeMode

# Create a runtime in research mode
config = RuntimeConfig(mode=RuntimeMode.RESEARCH)
runtime = LGAERuntime(config=config)

# Run one autonomous step
result = runtime.step()
print(f"Committed: {result.snapshot_after.authority_hash}")
print(f"Certification: {result.certification.level}")

# Inspect the decision trace
print(result.decision_trace.to_text())
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LGAERuntime                          │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Observe  │→ │ Diagnose │→ │ Reason   │→ │Propose │ │
│  │ Snapshot │  │ L0–L3    │  │ Graph+M  │  │Cand.   │ │
│  └──────────┘  └──────────┘  └──────────┘  └────┬───┘ │
│                                                   │     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────▼───┐ │
│  │  Learn   │← │ Evidence │← │  Commit  │← │Verify  │ │
│  │  Replay  │  │  Merkle  │  │  Atomic  │  │Shadow  │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│         ↑                              ↑               │
│  ┌──────┴──────┐              ┌────────┴────────┐      │
│  │ Credit + RL │              │ Authority Gov.  │      │
│  │ CQL + Causal│              │ Reject/Quarantine│     │
│  └─────────────┘              └─────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## Key design principles

1. **Learned models propose. Deterministic governance authorizes. Evidence proves.**
2. Every mutation passes through shadow transaction → verification → authority → commit.
3. Production mode fails closed when signed receipts are required without a signing key.
4. Cache invalidation is selective and tied to committed mutations.
5. Candidate IDs are canonical and deterministically deduplicated.
6. Uncertainty is higher on OOD inputs than ID inputs (`sigma_OOD > sigma_ID`).
7. Evidence is append-only and tamper-evident (Merkle tree + Ed25519 signatures).
8. Decisions are replayable from deterministic records.

## Testing

```bash
# Run all 1214 tests
python -m pytest

# Run only v5.10 tests
python -m pytest tests/test_v510_*.py

# Run with coverage
python -m pytest --cov=lgae_v3
```

### Test categories

- **Unit tests:** 719 baseline tests (v5.9.0)
- **Runtime tests:** 495 new tests across 50 phases
- **Property-based tests:** Hypothesis strategies for invariant verification
- **Metamorphic tests:** Metamorphic relations for equivalence checking
- **Adversarial tests:** Structural perturbation stress tests
- **Qualification tests:** Scientific, safety, and performance gates

## CLI commands

| Command | Description |
|---------|-------------|
| `demo` | Run a demo with synthetic graphs |
| `inspect` | Inspect current runtime state |
| `diagnose` | Run adaptive diagnostics |
| `propose` | Generate candidates without committing |
| `step` | Run one autonomous step |
| `run` | Run N autonomous steps |
| `qualify` | Run qualification gates |
| `qualify-lly` | Run LLY qualification |

## Configuration

```python
from lgae_v3.runtime import RuntimeConfig, RuntimeMode

# Research mode (permissive)
research = RuntimeConfig(mode=RuntimeMode.RESEARCH)

# Production mode (fail-closed)
production = RuntimeConfig(mode=RuntimeMode.PRODUCTION)

# Benchmark mode
benchmark = RuntimeConfig.load_preset("benchmark")
```

## Mathematical foundations

- **Fiber bundles:** Latent vectors on graph nodes with SO(d) gauge connections
- **Ricci curvature:** Ollivier, Forman, and Bakry-Émery curvature oracles
- **Sheaf theory:** Local-to-global consistency via cycle flatness
- **Lie groups:** SO(3), SU(2), GL(d) exponential maps for gauge transformations
- **Information theory:** Epistemic uncertainty, information gain, calibration
- **Control theory:** Model Predictive Control with receding horizon
- **Causal inference:** Do-calculus for counterfactual credit assignment
- **Cryptography:** SHA-256 Merkle trees, Ed25519 signatures, hash-chained receipts

## Requirements

- Python 3.12+
- PyTorch 2.10+
- NumPy, SciPy, NetworkX, Hypothesis

## License

MIT

## Citation

```bibtex
@software{lgae_v511,
  title  = {LGAE v5.11.0: Autonomous Structural Intelligence Runtime},
  author = {Dawson Block},
  year   = {2026},
  url    = {https://github.com/dawsonblock/LGAE-v5.9.0-unified-hardening}
}
```

---

<div align="center">

**50 phases · 1604+ tests · 0 regressions · Autonomous by design**

</div>
