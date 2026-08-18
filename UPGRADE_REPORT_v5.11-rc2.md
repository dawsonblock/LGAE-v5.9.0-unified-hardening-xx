# LGAE v5.11 Canonical Runtime Convergence Upgrade

This archive is an implementation-focused repair of the canonical runtime defects identified in the deep audit of the prior build.

## Implemented repairs

1. Canonical transaction base identity now uses `engine.authority_hash()` rather than the graph-only hash. Fresh accepted transactions therefore bind to the same state identity validated by `CommitChannel`.
2. Commit exception rollback now restores graph, fiber state, gauge state, and invalidates derived neighbor indices. An ordinary exception after a live state swap appends WAL ABORT after restoring pre-state, while a hard crash still leaves durable COMMIT replay semantics intact.
3. Candidate generation now performs deterministic candidate-ID deduplication and canonical sorting.
4. Canonical planning now activates information-gain, cost, and risk terms instead of hard-coding them to zero. RuntimeConfig exposes explicit weights for the multi-objective score.
5. MPC now hands its exact first mutation into canonical shadow evaluation when the mutation class maps to a supported structural action; the canonical chosen action is updated accordingly.
6. Commit computes realized utility delta once and places it in `CommitResult`; learning consumes that same value as its outcome truth.
7. Added regression coverage for fresh authority-bound commits and full-state exception rollback at state-swap failpoints.

## Validation performed in this build environment

- `tests/test_v511_closed_loop_hardening.py`: PASS
- `tests/test_v511_contracts.py`: PASS
- `tests/test_v511_rc2_canonical_commit_repairs.py`: PASS
- `tests/integration/test_state_bundle_commit.py`: PASS
- `tests/integration/test_wal_complete_transactions.py`: PASS
- Selected v5.10 regression suites for hash determinism, guard mutability, and information-gain activation: PASS

A larger v5.10/v5.11 run progressed through hundreds of tests before the execution window ended. Its first encountered failure was `test_v510_release.py::test_phase_count`, which requires Git commit history via `git log`; source ZIPs do not contain `.git`, so that release-provenance test is not executable from the archive alone. The Hypothesis property/metamorphic files also require the declared development dependency `hypothesis`, which was not installed in the execution environment.

## Remaining work

This build intentionally does not claim completion of the longer v6+ research roadmap. Remaining high-value work includes stronger causal subsystem credit assignment, complete removal of dual orchestration authority, scientific held-out graph-family qualification, structural world-model acceleration, and application branches for adaptive memory and agent topology.
