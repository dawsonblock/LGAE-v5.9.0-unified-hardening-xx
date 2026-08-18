# v5.10 Baseline Freeze

This directory contains the frozen baseline of LGAE v5.10.0 before the
v5.11.0 Canonical Runtime Convergence repair.

## Purpose

To preserve exactly what exists before refactoring, so that:
1. v5.10 behavior can be reproduced
2. Every known defect has a regression test
3. Current behavior is frozen before refactor

## Files

- `source_inventory.json` — SHA-256 hashes of all source files
- `package_metadata.json` — version, Python version, platform
- `test_inventory.json` — test count and file breakdown
- `known_defects.json` — catalog of 7 known defects with regression test mappings

## Known Defects

| ID | Name | Severity | Description |
|----|------|----------|-------------|
| DEFECT-001 | fake_canonical_path | P0 | step() delegates to loop.step(), 8 phase methods never called |
| DEFECT-002 | guard_mutability | P0 | guard.graph returns raw mutable GraphBuffers |
| DEFECT-003 | production_failopen | P0 | Production mode starts without WAL/evidence/checkpointing |
| DEFECT-004 | hash_seed_nondeterminism | P1 | curriculum.py uses hash() which is PYTHONHASHSEED-dependent |
| DEFECT-005 | mpc_not_called | P1 | MPC planner instantiated but never used in step() |
| DEFECT-006 | information_gain_not_used | P1 | IG/cost/risk hardcoded to 0.0 in structural_loop.py |
| DEFECT-007 | manifest_mismatch | P2 | MANIFEST.sha256.json references v5.9.0, missing v5.10 files |

## Regression Tests

Each defect has a regression test in `tests/test_v510_*_regression.py` that
PASSES against v5.10, proving the defect exists. These tests will FAIL after
the corresponding v5.11 phase fixes the defect.

## Legacy Tag

The existing runtime is tagged internally as `legacy_v510_structural_loop`.
It is not deleted during v5.11; it is refactored.
