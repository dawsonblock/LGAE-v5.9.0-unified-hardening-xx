# LGAE v5.11 Canonical Convergence Baseline Marker

**Tag**: `v5.11.0-convergence-baseline`
**Status**: Frozen upgraded baseline before v5.11.0 formalization and phase execution

## Environment & Build Metadata
- **Release Version**: `5.11.0` (upgraded baseline)
- **Manifest SHA-256**: `e78782f0972a79dceb9b8247bde457406f7bb9022f7b238b474fd48675ddd54d`
- **Python Version**: `3.12.0 (CPython)`
- **Core Dependencies**:
  - `torch`: 2.10.0
  - `numpy`: 2.4.6
  - `scipy`: 1.17.0
  - `networkx`: 3.6.1
  - `PyYAML`: 6.0.3
  - `safetensors`: 0.8.0

## Permanent Invariants Frozen in Baseline
1. **Authority-Bound Transactions**: Transactions bind to full `authority_hash` across graph, fiber, and gauge state.
2. **Full State Rollback**: Exception atomic commits restore graph, fiber state, and gauge generators on any failure.
3. **Candidate Deduplication & Canonical Ordering**: Candidate proposals are deterministically deduplicated and sorted by canonical candidate ID.
4. **Active Multi-Objective Scoring**: Active information-gain ($\nu$), cost ($\lambda$), and risk ($\mu$) weights in planning.
5. **MPC First-Action Integration**: Canonical planner evaluates and executes the exact first action of receding-horizon MPC plans.
6. **Single Realized Utility Truth**: Realized utility delta calculated once during commit and propagated to learning.
7. **Complete WAL Serialization**: Full transaction metadata, delta presence, and counter restoration preserved.

## Initial Test Results
- Total Tests: 1604
- Passing: 1599
- Provenance / Git Log Blockers: 1 (`test_v510_release.py::test_phase_count` requiring `.git` in source archive)

## Known Environmental Limitations
- Source ZIP archives omit `.git` directory; release provenance must validate embedded `BUILD_PROVENANCE.json`.

## Known Remaining Architectural Inconsistencies (Addressed in Execution Plan)
1. Conflicting release identities across files (`5.11.0-dev`, `5.10.0`, `5.11.0-RC`).
2. Dual orchestration authority in `StructuralLearningLoop` vs `LGAERuntime`.
3. Separate string-based state hashes/versions instead of `AuthorityStateIdentity`.
4. Informal WAL transitions requiring explicit state machine lifecycle (`NEW -> PREPARED -> COMMIT_INTENT -> APPLIED -> VERIFIED -> FINALIZED`).
5. Need for formal homeostasis penalties, anti-oscillation controller, and diagnosis-conditioned candidate search.
