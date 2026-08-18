# v5.11.0 — Canonical Runtime Convergence Release

- Unified release truth across pyproject.toml, version.py, README, and verification artifacts.
- Enforced single phase-ordering authority in LGAERuntime.
- Bound transactions and authorizations to complete AuthorityStateIdentity.
- Formalized WAL transaction lifecycle state machine.
- Comprehensive exception, SIGTERM, and SIGKILL crash matrix qualification.
- Proved active MPC horizon and IG/risk/cost terms directly change committed transactions.
- Embedded build provenance for standalone source archives.

# v5.8.3

- Added low-rank Lie-algebra gauge heads for joint topology/gauge edge proposals.
- Added shadow-only gauge overrides to GeometryGovernor mutation certification.
- Added post-commit gauge-slot initialization with optimizer-state reset and generation synchronization.
- Added normalized connection-Dirichlet local credit with multi-scale graph-distance attribution.
- Added v5.8.3 joint-action qualification and regression coverage.
- Scientific generalization remains NOT_YET_QUALIFIED.

# v5.8.2
- ANN-backed learned candidate inference.
- First-order Fiedler/FoSR production proposals.
- Online epistemic scale calibration.
- Contextual LCB arbitration.
- 1-WL replay redundancy gating.
- Dedicated scalable structural-intelligence qualification.

# v5.8.1 — Structural Intelligence Hardening

- Added randomized-prior Q ensembles.
- Added structural-regime stratified replay.
- Added trainable contrastive candidate retrieval.
- Added effective-resistance, exact spectral reference, and BORF-style Forman-flow proposal channels.
- Added conservative LCB arbitration.
- Added v5.8.1 hardening qualification with candidate recall, regret, and ID/OOD uncertainty reporting.
- Scientific generalization remains separately gated.

# v5.6.3 — Localized cache dependency hardening

- Added radius-aware spatial cache dependencies for local graph-derived data.
- Added selective dirty-region invalidation keyed by changed node IDs.
- Preserves unaffected cache partitions while advancing authoritative generation stamps.
- Global dependencies remain conservatively invalidated; missing locality metadata fails closed.
- Added `test_v563_localized_cache.py` and localized-cache qualification.

# v5.6.2 — Concurrent snapshot hardening

- Added `GraphReadCoordinator` with a seqlock-style mutation epoch: even generations are stable, odd epochs mark an in-place authoritative writer.
- Added `GraphReadView` and `run_consistent_read()` so long-running curvature, spectral, candidate, or reasoning computations reject results that overlap any commit or rollback.
- Integrated both snapshot and delta graph transactions with the read coordinator without blocking readers on normal stable generations.
- Made the commit event bus thread-safe while retaining callbacks outside the internal lock.
- Added adversarial concurrent-read/write tests and a dedicated concurrent snapshot qualification report.

# v5.6.1 — Commit-boundary cache coherence

- Added authoritative `GraphCommitEvent` / `CommitEventBus` infrastructure.
- Added generation-stamped derived-cache validation with fail-closed stale access.
- Added dependency-aware selective invalidation by topology, weights, metric lengths, latents, fibers, gauges, and roles.
- Delta transactions publish precise changed-node/edge metadata after commit only.
- Rollback remains invisible to the commit bus; no authoritative generation is emitted.
- Neighbor-index backends can subscribe to commit events and invalidate only on configured dependencies.
- Added adversarial cache-coherence qualification and regression tests.

# v5.6.1 — Numerical and runtime hardening

- Projects arbitrary external sheaf restriction maps to SO(d) by default before transport.
- Adds cached Ollivier support/shortest-path neighborhoods for repeated audit calls; hot-loop audit remains Sinkhorn/AFRC-prioritized.
- Adds delta-journal graph transactions for mutation paths that opt into journaled slot writes, while preserving legacy snapshot transactions for arbitrary external direct tensor mutation.
- Adds optional k-edge-connectivity preservation before edge pruning in addition to bridge and global spectral-gap checks.
- Adds evidence-authority weighted memory replacement/consolidation so low-authority traces cannot overwrite stronger grounded experience.

# v5.5.0 — Evidence-grounded reasoning memory

- Added append-only hash-chained `EvidenceLedger`.
- Added derived `StructuralExperienceMemory` with associative structural retrieval and action priors.
- Added typed parallel `ReasoningGraph` fan-out/reduce execution.
- Integrated optional memory priors and evidence capture into the v5.4 structural reasoning loop without weakening governor authority.
- Replaced `O(N^2)` random non-edge materialization with bounded direct sampling.
- Added v5.5 qualification and regression tests.

# Changelog

## v5.4.0 — Counterfactual Structural Reasoning Executive

### Added

- `src/lgae_v3/reasoning.py`: concrete structural action representation, bounded multi-channel candidate generator, dependency-free permutation-equivariant graph encoder, probabilistic candidate `Q(S,a)` network, counterfactual replay buffer, governor-grounded counterfactual factory, and exact-candidate certification helper.
- `src/lgae_v3/reasoning_loop.py`: runtime bridge that lets the learned reasoner rank proposals while preserving `LGAEEngine`/`GeometryGovernor` as the sole commit authority.
- Heteroscedastic utility training with pairwise ranking loss so the model learns relative intervention quality instead of only action labels.
- Explicit `NO_OP` baseline and random exploration channel to prevent forced mutation and candidate-distribution collapse.
- `scripts/qualify_reasoning.py` and five v5.4 reasoning tests.

### Changed

- Canonical package/release/schema identity advanced to 5.4.0.
- Test contract advanced to 578 tests.
- Structural reasoning is now scoped to concrete edge mutations for the default exact counterfactual factory; fiber/gauge planning remains behind the older governed paths until dedicated counterfactual evaluators are qualified.

### Qualification boundary

The v5.4 reasoning smoke verifies architecture, finite scoring, bounded candidate generation, permutation equivariance, replay training, and governor grounding. It does **not** supersede the held-out generalization result from v5.3.x. Production qualification requires training on large procedural counterfactual data and demonstrating lower regret than the spectral heuristic on graph families excluded from training.


## v5.3.1 — Integrity & baseline-comparison fixes (unreleased)

This is a correctness/integrity patch on v5.3.0. It does not change the
governor, geometry oracles, or numerical kernels. It fixes four issues
identified in a review of v5.3.0's release artifacts and adds the missing
baseline comparison.

### Fixed

- **Nondeterministic policy qualification (release-gate integrity bug).**
  `qualify_structural_policy` constructed the `StructuralExecutive` (and its
  network/target-scorer weight initialization) *before* calling
  `torch.manual_seed`. Three runs of `scripts/qualify_policy.py` on the same
  code produced diagnosis accuracy of 100%, 90%, and 86.7% — i.e. the
  release-gate number was a random draw and a bad draw could fail the 80%
  threshold. The seed is now set before network construction, making the
  qualification deterministic (83.3% / 0.0176 regret on the current code).
  `src/lgae_v3/benchmark/policy_qualification.py`

- **Stale `example_output.txt`.** It reported `"version": "3.2.0"` from a
  pre-v5 release. Regenerated from the current CLI (`5.3.0`).

- **Stale qualification reports.** `policy_qualification_report.json`
  (86.7% / 0.0274), `qualification_report.json`, and
  `production_qualification_report.json` were regenerated from the current
  deterministic code.

- **Circular benchmark utility in Task A.** `TaskA_Bottleneck.utility`
  added `+0.1 * inter_cluster_edge_count`, a term that directly encoded the
  correct action's structural signature (the correct action *is* "add an
  inter-cluster edge"). The utility is now the pure spectral gap λ₂ — a
  physical graph invariant. The correct action still maximizes λ₂ because
  of the physics of bottlenecks, not because the utility was written to
  reward it. `src/lgae_v3/benchmark/tasks.py`

### Added

- **Baseline controllers** (`src/lgae_v3/benchmark/baselines.py`):
  `RandomActionController` (lower bound), `SpectralHeuristicController`
  (non-learned threshold rules on cheap observables — not tuned per task),
  and `OracleController` (upper bound). These let the learned executive be
  compared on the same axis instead of in a vacuum.

- **Baseline comparison script** (`scripts/compare_baselines.py`): runs
  random / spectral-heuristic / learned / oracle on both the in-distribution
  tasks and truly held-out structurally-distinct tasks, reporting diagnosis
  accuracy and mean regret for each.

  Findings on the current code (seed 0, 500 gradient steps):
  ```
  controller             split                  accuracy     regret
  random                 in_distribution          0.1000     2.6918
  spectral_heuristic     in_distribution          0.4667     2.2669
  learned                in_distribution          0.8333     0.0176
  oracle                 in_distribution          1.0000     0.0000
  random                 held_out_structure       0.1000     3.0092
  spectral_heuristic     held_out_structure       0.6000     2.4000
  learned                held_out_structure       0.3000     0.6175
  oracle                 held_out_structure       1.0000     0.0000
  ```
  The learned executive beats random and the spectral heuristic
  in-distribution, but **loses to the spectral heuristic on held-out
  diagnosis accuracy** (30% vs 60%), though with lower regret (0.62 vs
  2.40), suggesting it defaults to safe NO_OP on unseen structures rather
  than misdiagnosing. This is the first honest generalization signal the
  benchmark has produced.

- **Truly held-out task variants** (`src/lgae_v3/benchmark/tasks.py`):
  `HeldOutBottleneck` (variable cluster size / bridge position) and
  `HeldOutSpuriousEdge` (variable graph size). The original "held-out seeds
  101–105" produced *identical* graph structures to seed 42 for all six
  tasks (only latent noise differed); these parametric variants generate
  structurally different graphs so held-out evaluation measures something
  beyond seed noise.

### Changed

- **README tone.** Removed the hardcoded "559/559 passing" badge (the count
  is code-dependent and shouldn't be hardcoded into a badge URL), added a
  "validation: synthetic-only" badge, and added a "Validation boundaries"
  section that states explicitly what is and is not claimed. The stale
  "96.7%" v5.2 claim and "86.7%" v5.3 claim were corrected.

- **Naming note.** README now documents that the release is LGAE-v5.3.0,
  the repo is `1LR`, and the Python dist/module is `lgae-v3` / `lgae_v3`
  (kept for import stability). `docs/ARCHITECTURE.md` and
  `docs/READING_LIST.md` updated to stop implying the current version is
  3.2.

### Not fixed (deliberately)

- The package/module name `lgae_v3` is **not** renamed. Renaming would
  touch every import in 53 source + 31 test files for no functional gain
  and high regression risk. The naming note in the README documents the
  historical reason instead.

- The remaining benchmark tasks (B–F) still have utilities that are
  constructed around the correct action's effect. Fully decoupling them
  requires defining independent downstream objectives (reconstruction,
  diffusion mixing) per task, which is a larger redesign left for a future
  release. The README's Validation boundaries section states this.

## v5.8.0 — Structural Intelligence Qualification

- Added state-grouped counterfactual replay so competing interventions from the same graph remain together for pairwise/listwise ranking supervision.
- Added shared graph encoder + multi-head Q ensemble with explicit epistemic/aleatoric uncertainty decomposition.
- Added effective-resistance candidate retrieval as a global bottleneck proposal channel.
- Added procedural graph families with topology-disjoint train/held-out splits.
- Added exact candidate-regret and uncertainty-calibration metrics.
- Added `structural_intelligence_qualification_report.json`; infrastructure qualification is separated from scientific generalization claims.
- The bundled smoke-scale held-out run does not yet beat the effective-resistance baseline, and is explicitly reported as `NOT_YET_QUALIFIED` rather than promoted to a false success.
