# LGAE v5.11.0 Build Report

Release: **Authority and Durability Closure**

## Summary

LGAE v5.11.0 is the authority and durability closure release. The
objective is transactional correctness, crash-safety, determinism,
self-verification, and scientific honesty.

**Test suite: 2309 passed, 0 failed**

## Defects repaired (19 total)

### Transactional foundation (Phases 1-8)

- **D11-001**: Direct engine mutation bypasses CommitChannel → Fixed: engine is private (`self._engine`), `rt.engine` returns read-only `EngineFacade`
- **D11-002**: CommitChannel only logically exclusive → Fixed: capability-gated mutation primitives (`_AuthorityCapability` token)
- **D11-003**: Graph/fiber/gauge apply is non-atomic → Fixed: exception-atomic commit with rollback to pre-state
- **D11-004**: WAL records graph but not complete transaction state → Fixed: WAL serializes graph + fiber + gauge deltas
- **D11-005**: WAL COMMIT occurs after live mutation → Fixed: COMMIT written BEFORE APPLY (COMMIT-before-APPLY ordering), ABORT invalidates on rollback
- **D11-006**: WAL counters reset on reopen → Fixed: `_restore_counters()` scans existing records
- **D11-007**: Crash tests do not kill inside transaction stages → Fixed: subprocess SIGKILL crash matrix at 4 stages (before BEGIN, after BEGIN, after WRITE, after COMMIT)
- **D11-008**: FiberDelta fallback uses Python `hash()` → Fixed: raises `DeterminismError`, `FiberStateSnapshot.state_hash()` added
- **D11-009**: Authorization binding is optional/incomplete → Fixed: mandatory, non-nullable, `transaction_hash` binding
- **D11-010**: Fiber/gauge evaluation still mutate-and-restore → Fixed: shadow-only evaluation, restore before evaluation

### Learning integrity (Sprint 3)

- **D11-011**: `learn()` uses predicted delta as realized reward → Fixed: `realized_delta = U_after - U_before`
- **D11-012**: Calibration compares delta prediction against absolute utility → Fixed: `calibrator.update(predicted, realized_delta)`
- **D11-013**: Hierarchical credit not connected → Fixed: 6-field per-subsystem credit assignment (diagnostic/candidate/planner/action/governance/outcome). Renamed from "hierarchical" to "per-subsystem" in v5.11-RC Phase 15.

### Qualification (Sprint 4)

- **D11-014**: Performance `MEASURED` can mean nothing executed → Fixed: NOT_RUN/INVALID/MEASURED/PASS/FAIL with thresholds
- **D11-018**: Final qualification asserts symbols instead of invariants → Fixed: behavioral invariant tests
- **D11-019**: Four "real graph" benchmarks remain synthetic surrogates → Verified: `is_real_data` flag, synthetic descriptions

### Release integrity (Sprint 5)

- **D11-015**: Hypothesis missing from dev dependencies → Fixed: added to `pyproject.toml`
- **D11-016**: Release manifest still stale → Updated
- **D11-017**: BUILD_REPORT remains v5.9 / 719 tests → Updated to v5.11.0 / 1458 tests

### Authority and durability closure (v5.11-RC Phases 5-22)

- **D11-020**: StateBundle.state_hash doesn't cover complete authoritative state → Fixed: `canonical_hash` property covers graph, fiber, gauge, calibration, model_ref, version, and 12 extended state fields
- **D11-021**: WAL doesn't write complete transaction records → Fixed: `TX_PREPARE` record type with transaction_id, base_state_hash, base_state_version, delta_hash, authorization_id, delta presence flags
- **D11-022**: Normal commit and recovery use different apply paths → Fixed: shared `apply_wal_mutation()` function used by both CommitChannel and `replay_committed_transactions()`
- **D11-023**: No concurrency/CAS qualification tests → Added: 7 tests (stale version/hash rejected, concurrent commits only one succeeds, CAS atomic, CAS hash changes, failed CAS preserves state, sequential commits)
- **D11-024**: Production startup doesn't verify WAL integrity → Fixed: `recover_from_wal()` method verifies hash chain, fails closed on corruption in production mode
- **D11-025**: Credit assignment falsely claimed as "hierarchical" → Fixed: renamed to "per-subsystem credit attribution" in docstrings and comments
- **D11-026**: MPC/IG causal relevance not tested → Added: 8 tests (multi-step exploration, utility selection, horizon effect, determinism, IG selection, ensemble disagreement, exploration bonus effect)
- **D11-027**: Golden multi-domain transaction scenario → Added: 8 tests (joint graph/fiber/gauge transaction, crash matrix at every internal commit boundary, recovery produces exactly pre-state or post-state)

## Architecture

```
LGAERuntime
    │
    ├── immutable public APIs (rt.engine → EngineFacade)
    │
    └── _engine (private)
           │
           ├── _authority_capability (mutation token)
           │
           ├── graph
           ├── fibers
           ├── gauges
           ├── calibration
           ├── model
           ├── state_version
           └── state_hash
```

## Runtime invariant

```
S_{t+1} = Commit(S_t, T_t, A_t)
```

- `S_t`: immutable authoritative state
- `T_t`: deterministic structural transaction
- `A_t`: authorization bound to that exact transaction

## Crash invariant

```
S_restart ∈ {S_t, S_{t+1}}
```

Never: `S_restart = S_t + partial(T_t)`

## Deterministic replay invariant

```
F(S_t, O_t, C, M, R) = S_{t+1}
```

for identical state, observation, configuration, models, and deterministic randomness.

## Governing principle

> Learned models propose. Deterministic governance authorizes. Evidence proves.

## Test breakdown

- Unit tests: ~1300
- Integration tests: ~889
- Total: 2189 passed, 0 failed

## Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- Dev: pytest, pytest-cov, hypothesis
