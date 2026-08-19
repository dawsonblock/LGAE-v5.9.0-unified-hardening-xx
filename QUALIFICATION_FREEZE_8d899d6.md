# QUALIFICATION FREEZE — Commit 8d899d6

**Date:** 2026-08-19
**Commit:** `8d899d6`
**Description:** Qualification tiers with explicit markers, mode metadata, fast/release paths

## 1. Qualification Infrastructure

### Markers
- `meta`: self-referential tests that re-execute other tests (excluded from all parallel runs)
- `crash_recovery`: tests exercising crash recovery, cross-process determinism, subprocess isolation

### Tiers
| Tier | Script | Marker Expression | Tests | Runtime |
|------|--------|-------------------|-------|---------|
| Fast | `qualification-fast.sh` | `not meta and not crash_recovery` | 2115 | ~36s |
| Release | `qualification-release.sh` | `not meta` | 2188 | ~103s |

### Metadata Fields
- `qualification_mode`: "release" or "fast"
- `parallel_workers`: 12
- `meta_tests_excluded`: 1
- `marker_expression`: recorded for auditability

## 2. Frozen Test Counts
- Release: 2188 collected, 2188 passed, 0 failed, 0 skipped
- Fast: 2115 collected, 2115 passed, 0 failed, 0 skipped
- Excluded from fast: 73 crash_recovery tests

## 3. Performance Targets
- Full release qualification: ≤120 seconds on 12-core machine
- Fast qualification: ≤60 seconds on 12-core machine
- Current: release=103s, fast=36s

## 4. Release Gate Requirements
For any frozen experimental release:
- `qualification_mode == "release"`
- `tests_failed == 0`
- `manifest_valid == True`

Fast-mode metadata must NOT be used for release promotion.

## 5. Optimization Provenance
- Original: 437s (single-threaded, redundant meta-test)
- After xdist + meta deselect: 109s
- After crash_recovery marker: 103s
- Speedup: 4.2x from execution architecture, not reduced coverage
