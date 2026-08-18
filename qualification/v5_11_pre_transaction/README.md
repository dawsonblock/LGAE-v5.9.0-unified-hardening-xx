# v5.11 Pre-Transaction Repair Qualification

This directory freezes the state of the v5.11-dev build **before** the
transactional convergence repair prescribed by the 40-phase plan.

## Purpose

- Record the exact source tree hash at the freeze point.
- Document all 19 known defects (D11-001 through D11-019).
- Preserve pre-repair behavior as historical defect reproductions.
- Provide a baseline against which post-repair verification can compare.

## Freeze point

- **Commit**: `95c527c528652f52408c636c6f8dde0bba6f7789`
- **Version**: `5.11.0-dev`
- **Source files**: 133 Python files in `src/lgae_v3/`
- **Test files**: 114 Python files in `tests/`
- **Test count**: 1404 passed (last measured)
- **Branch**: `v5.10-canonical-runtime`

## Known defects (19)

| ID | Name | Severity | Phase |
|----|------|----------|-------|
| D11-001 | direct_engine_mutation_bypasses_commit_channel | P0 | 1 |
| D11-002 | commit_channel_only_logically_exclusive | P0 | 2 |
| D11-003 | graph_fiber_gauge_apply_is_non_atomic | P0 | 7 |
| D11-004 | wal_records_graph_but_not_complete_transaction_state | P0 | 9 |
| D11-005 | wal_commit_occurs_after_live_mutation | P0 | 10 |
| D11-006 | wal_counters_reset_on_reopen | P0 | 11 |
| D11-007 | crash_tests_do_not_kill_inside_transaction_stages | P0 | 13 |
| D11-008 | fiber_delta_fallback_uses_python_hash | P0 | 4 |
| D11-009 | authorization_binding_is_optional_incomplete | P0 | 6 |
| D11-010 | fiber_gauge_evaluation_still_mutate_and_restore | P0 | 5 |
| D11-011 | learn_uses_predicted_delta_as_realized_reward | P1 | 17 |
| D11-012 | calibration_compares_delta_prediction_against_absolute_utility | P1 | 17 |
| D11-013 | hierarchical_credit_not_connected | P1 | 18 |
| D11-014 | performance_measured_can_mean_nothing_executed | P1 | 21 |
| D11-015 | hypothesis_missing_from_dev_dependencies | P1 | 23 |
| D11-016 | release_manifest_stale | P1 | 36 |
| D11-017 | build_report_remains_v5_9_719_tests | P1 | 34 |
| D11-018 | final_qualification_asserts_symbols_not_invariants | P1 | 24 |
| D11-019 | four_real_graph_benchmarks_remain_synthetic_surrogates | P2 | 29 |

## Repair order

Per the 40-phase plan:

- **Sprint 1** (Phases 0-8): Authority closure — no state mutation outside authority layer
- **Sprint 2** (Phases 9-16): Durability — every transaction survives kill-at-any-point
- **Sprint 3** (Phases 17-20, 25-28): Learning integrity — learn from actual outcomes
- **Sprint 4** (Phases 21-24, 29-32): Qualification — real performance gates, honest tests
- **Sprint 5** (Phases 33-40): Release — clean manifest, reproducible artifact

## Files

- `build_inventory.json` — build metadata at freeze point
- `source_hashes.json` — SHA-256 of every source file
- `test_inventory.json` — test suite inventory
- `known_defects.json` — all 19 defects with locations and fix approaches
- `architecture_trace.json` — canonical path and bypass paths
- `authority_bypass_report.json` — all direct mutation sites
- `wal_behavior_report.json` — WAL ordering, content, and restart issues
- `determinism_report.json` — Python hash() usage and determinism status
- `learning_semantics_report.json` — learning signal, calibration, and credit issues
- `release_integrity_report.json` — manifest, build report, and qualification issues

## Gate

- [x] current source tree hash frozen
- [x] all known defects recorded
- [x] pre-repair behavior reproducible (tests pass at freeze point)
