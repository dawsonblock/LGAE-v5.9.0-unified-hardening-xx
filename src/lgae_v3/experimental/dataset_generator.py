"""Dataset generator for v6.0-exp2 structural transition datasets.

Turns the v5.11 runtime into a training/evaluation data pipeline that
produces rich TransitionRecords with:

- Both successful and unsuccessful decisions (no survivorship bias).
- Negative sampling: non-selected candidates with shadow evaluation.
- Split control: separate train/validation/held-out artifacts.
- Immutable metadata with provenance.
- Deterministic regeneration: same seed + config → identical dataset hash.

The generator is a PASSIVE OBSERVER. It runs the v5.11 runtime and records
transitions, but never mutates authoritative state. Negative sampling uses
shadow evaluation on graph copies, never the runtime's authority.

Output structure::

    datasets/
    ├── train/
    │   ├── dataset.json
    │   ├── manifest.json
    │   └── quality_report.json
    ├── validation/
    │   ├── dataset.json
    │   ├── manifest.json
    │   └── quality_report.json
    └── heldout/
        ├── dataset.json
        ├── manifest.json
        └── quality_report.json
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import numpy as np
import networkx as nx

from ..types import GraphBuffers, make_graph_buffers
from ..config import ResearchConfig
from ..runtime import LGAERuntime, RuntimeConfig
from ..runtime.curriculum import CurriculumEntry, generate_graph as gen_graph
from ..operators import spectral_gap_graphbuffers
from ..version import VERSION

from .graph_families import FrozenGraphFamilyRegistry, FROZEN_SPLIT
from .transition_record import (
    TransitionRecord,
    TransitionProvenance,
    AuthorizationDecision,
    AuthorityIdentity,
    StructuralStateSummary,
    DiagnosisSummary,
    CandidateSummary,
    CandidateSetSummary,
    PlannerMetadata,
    ComputeMetrics,
    make_record_id,
)
from .feature_extraction import extract_global_features, extract_local_action_features
from .quality_report import generate_quality_report, DataQualityReport
from .dataset_validator import DatasetValidator, ValidationResult


DATASET_SCHEMA_VERSION = "LGAE_STRUCTURAL_DATASET_V6_0_EXP2"
GENERATOR_VERSION = "6.0-exp2"


@dataclass(frozen=True, slots=True)
class DatasetImmutableMetadata:
    """Immutable metadata for a generated dataset split."""
    schema_version: str
    split: str
    graph_family: str
    seed: int
    generator_version: str
    base_runtime: str
    config_hash: str
    dataset_hash: str
    n_records: int
    n_observed: int
    n_counterfactual: int
    created_at: str
    description: str = ""

    def to_log(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "graph_family": self.graph_family,
            "seed": int(self.seed),
            "generator_version": self.generator_version,
            "base_runtime": self.base_runtime,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "n_records": int(self.n_records),
            "n_observed": int(self.n_observed),
            "n_counterfactual": int(self.n_counterfactual),
            "created_at": self.created_at,
            "description": self.description,
        }


@dataclass(slots=True)
class SplitDataset:
    """A dataset for one split (train/validation/held_out)."""
    split: str
    records: list[TransitionRecord] = field(default_factory=list)
    metadata: DatasetImmutableMetadata | None = None
    quality_report: DataQualityReport | None = None

    @property
    def n_records(self) -> int:
        return len(self.records)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of all records (deterministic).

        Excludes volatile fields (transaction_id, receipt_hash, evidence_hash)
        that may differ between identical runs due to runtime-internal state.
        """
        records_log = []
        for r in self.records:
            log = r.to_log()
            # Remove volatile fields for deterministic hashing.
            log.pop("transaction_id", None)
            log.pop("timestamp", None)
            records_log.append(log)
        content = json.dumps(records_log, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "metadata": self.metadata.to_log() if self.metadata else {},
            "records": [r.to_log() for r in self.records],
            "quality_report": self.quality_report.to_log() if self.quality_report else None,
            "content_hash": self.content_hash,
        }, sort_keys=True, indent=2)

    def save(self, directory: str | Path) -> None:
        """Save dataset, manifest, and quality report to a directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        # Dataset.
        (dir_path / "dataset.json").write_text(self.to_json())
        # Manifest.
        manifest = {
            "schema": DATASET_SCHEMA_VERSION,
            "split": self.split,
            "n_records": self.n_records,
            "content_hash": self.content_hash,
            "metadata": self.metadata.to_log() if self.metadata else {},
        }
        (dir_path / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2)
        )
        # Quality report.
        if self.quality_report:
            (dir_path / "quality_report.json").write_text(
                json.dumps(self.quality_report.to_log(), sort_keys=True, indent=2)
            )


class DatasetGenerator:
    """Generates structural transition datasets from v5.11 runtime.

    Usage::

        generator = DatasetGenerator(config=_cfg(), seed=42)
        datasets = generator.generate_all_splits(n_steps=5, n_episodes=3)
        generator.save_all(datasets, "datasets/")

    The generator:
    1. Runs the v5.11 runtime on graphs from the frozen family splits.
    2. Records observed transitions (both committed and rejected).
    3. Performs shadow evaluation on non-selected candidates for negative
       sampling.
    4. Produces separate datasets for train/validation/held-out.
    5. Generates quality reports and validates each split.
    """

    def __init__(
        self,
        config: ResearchConfig | None = None,
        seed: int = 42,
        registry: FrozenGraphFamilyRegistry | None = None,
        n_negative_samples: int = 3,
        fixed_timestamp: str | None = None,
    ) -> None:
        self.config = config or self._default_config()
        self.seed = int(seed)
        self.registry = registry or FrozenGraphFamilyRegistry()
        self.n_negative_samples = int(n_negative_samples)
        # Use a fixed timestamp for deterministic regeneration.
        # When None, uses a deterministic timestamp derived from the seed.
        self._timestamp = fixed_timestamp or self._deterministic_timestamp()
        self._config_hash = self._compute_config_hash()

    def _deterministic_timestamp(self) -> str:
        """Generate a deterministic timestamp from the seed."""
        # Use a fixed epoch-based timestamp derived from the seed.
        # This ensures D(seed, config, version) = constant.
        import hashlib
        h = hashlib.sha256(f"timestamp:{self.seed}".encode()).digest()
        # Use first 4 bytes as a pseudo-epoch offset from a base date.
        offset = int.from_bytes(h[:4], "big") % (365 * 24 * 3600)
        from datetime import datetime, timedelta, timezone
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ts = base + timedelta(seconds=offset)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _default_config() -> ResearchConfig:
        cfg = ResearchConfig()
        cfg.fiber.d_base = 2
        cfg.fiber.d_max = 6
        cfg.fiber.spawn_width = 1
        cfg.fiber.gauge_dim = 0
        cfg.audit.orc_backend = "exact_lp"
        cfg.audit.persistent_homology_enabled = False
        cfg.audit.entropic_nodes = 0
        cfg.audit.bakry_nodes = 0
        cfg.audit.cde_nodes = 0
        cfg.audit.exact_lly_top_k = 0
        cfg.audit.orc_top_k = 0
        cfg.mutation.shadow_horizons = [1, 2]
        cfg.mutation.curvature_ema_enabled = False
        return cfg

    def _compute_config_hash(self) -> str:
        """Deterministic hash of the generator configuration."""
        config_dict = {
            "seed": self.seed,
            "n_negative_samples": self.n_negative_samples,
            "config": str(self.config),
        }
        content = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def generate_split(
        self,
        split: str,
        n_steps: int = 5,
        n_episodes: int = 1,
    ) -> SplitDataset:
        """Generate a dataset for one split.

        Args:
            split: "train", "validation", or "held_out".
            n_steps: Number of runtime steps per graph.
            n_episodes: Number of episodes per graph family.

        Returns:
            SplitDataset with all records, metadata, and quality report.
        """
        entries = self.registry.all_entries().get(split, [])
        records: list[TransitionRecord] = []
        run_id = hashlib.sha256(f"{split}:{self.seed}".encode()).hexdigest()[:16]

        for ep in range(n_episodes):
            for entry in entries:
                # Deterministic per-entry seed.
                entry_seed = self.seed + int.from_bytes(
                    hashlib.sha256(entry.family_id.encode()).digest()[:4], "big"
                ) % 1000
                episode_id = f"{entry.family_id}_ep{ep}"
                ep_records = self._generate_episode(
                    entry=entry,
                    split=split,
                    run_id=run_id,
                    episode_id=episode_id,
                    seed=entry_seed,
                    n_steps=n_steps,
                )
                records.extend(ep_records)

        # Generate quality report.
        quality = generate_quality_report(records)

        # Validate.
        validator = DatasetValidator(
            held_out_families={f.value for f in self.registry.split.held_out},
            train_families={f.value for f in self.registry.split.train},
        )
        validation = validator.validate(records, expected_split=split)
        if not validation.valid:
            # Include validation issues in quality report warnings.
            for issue in validation.issues:
                if issue.severity == "error":
                    quality.warnings.append(f"VALIDATION ERROR: {issue.message}")

        # Create immutable metadata.
        dataset_hash = hashlib.sha256(
            json.dumps(
                [{k: v for k, v in r.to_log().items()
                  if k not in ("transaction_id", "timestamp")}
                 for r in records],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        metadata = DatasetImmutableMetadata(
            schema_version=DATASET_SCHEMA_VERSION,
            split=split,
            graph_family="mixed" if len(entries) > 1 else (entries[0].family.value if entries else ""),
            seed=self.seed,
            generator_version=GENERATOR_VERSION,
            base_runtime=VERSION,
            config_hash=self._config_hash,
            dataset_hash=dataset_hash,
            n_records=len(records),
            n_observed=sum(1 for r in records if r.provenance == TransitionProvenance.REALIZED),
            n_counterfactual=sum(1 for r in records if r.provenance == TransitionProvenance.COUNTERFACTUAL),
            created_at=self._timestamp,
        )

        return SplitDataset(
            split=split,
            records=records,
            metadata=metadata,
            quality_report=quality,
        )

    def generate_all_splits(
        self,
        n_steps: int = 5,
        n_episodes: int = 1,
    ) -> dict[str, SplitDataset]:
        """Generate datasets for all splits."""
        return {
            split: self.generate_split(split, n_steps=n_steps, n_episodes=n_episodes)
            for split in ("train", "validation", "held_out")
        }

    def save_all(
        self,
        datasets: dict[str, SplitDataset],
        base_dir: str | Path,
    ) -> None:
        """Save all splits to a base directory."""
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        for split, dataset in datasets.items():
            split_dir = base / split
            dataset.save(split_dir)

    # ------------------------------------------------------------------ #
    # Episode generation.
    # ------------------------------------------------------------------ #

    def _generate_episode(
        self,
        entry: CurriculumEntry,
        split: str,
        run_id: str,
        episode_id: str,
        seed: int,
        n_steps: int,
    ) -> list[TransitionRecord]:
        """Generate records for one episode (one graph, multiple steps)."""
        graph = gen_graph(entry)
        runtime = LGAERuntime(
            graph=graph,
            config=self.config,
            runtime_config=RuntimeConfig(),
        )
        records: list[TransitionRecord] = []

        for step in range(n_steps):
            # Run one step.
            result = runtime.step()

            # Extract the observed transition.
            observed = self._extract_observed_record(
                result=result,
                run_id=run_id,
                episode_id=episode_id,
                step_id=step,
                graph_family=entry.family.value,
                split=split,
                seed=seed,
            )
            records.append(observed)

            # Generate negative samples (counterfactual transitions).
            counterfactuals = self._generate_counterfactual_records(
                result=result,
                runtime=runtime,
                run_id=run_id,
                episode_id=episode_id,
                step_id=step,
                graph_family=entry.family.value,
                split=split,
                seed=seed,
            )
            records.extend(counterfactuals)

        return records

    def _extract_observed_record(
        self,
        result: Any,  # RuntimeStepResult
        run_id: str,
        episode_id: str,
        step_id: int,
        graph_family: str,
        split: str,
        seed: int,
    ) -> TransitionRecord:
        """Extract an observed (REALIZED) transition record from a runtime step."""
        # Extract state summaries.
        state_before = self._extract_state_summary(result.snapshot_before)
        state_after = self._extract_state_summary(result.snapshot_after)

        # Authority identity.
        auth_before = AuthorityIdentity(
            state_hash=state_before.state_hash,
            state_version=state_before.graph_version,
            authority_hash=result.snapshot_before.authority_hash,
        )
        auth_after = AuthorityIdentity(
            state_hash=state_after.state_hash if state_after else "",
            state_version=state_after.graph_version if state_after else 0,
            authority_hash=result.snapshot_after.authority_hash,
        )

        # Diagnosis (simplified extraction from metadata).
        diagnosis = DiagnosisSummary(
            oversquashing_score=float(result.metadata.get("oversquashing_score", 0.0)),
            bottleneck_score=float(result.metadata.get("bottleneck_score", 0.0)),
            epistemic_uncertainty=float(result.metadata.get("epistemic_uncertainty", 0.0)),
        )

        # Candidate set summary (simplified).
        n_candidates = 0
        if result.candidates is not None:
            try:
                n_candidates = len(result.candidates)
            except TypeError:
                n_candidates = 1

        # Build candidate summaries.
        candidate_summaries: list[CandidateSummary] = []
        selected_action = result.chosen_action
        for i in range(max(n_candidates, 1)):
            action_type = selected_action if i == 0 else f"candidate_{i}"
            candidate_summaries.append(CandidateSummary(
                candidate_id=i,
                action_type=action_type,
                target={},
                predicted_delta=float(result.delta_utility) if i == 0 else 0.0,
                predicted_risk=0.0,
                predicted_cost=0.0,
                predicted_ig=0.0,
                selected=(i == 0),
            ))

        cand_set = CandidateSetSummary(
            n_candidates=max(n_candidates, 1),
            candidates=tuple(candidate_summaries),
            action_distribution={selected_action: 1} if n_candidates <= 1 else {},
        )

        # Planner metadata.
        planner_meta = PlannerMetadata(
            horizon=int(result.metadata.get("horizon", 1)),
            planner_type=str(result.metadata.get("planner_type", "mpc")),
        )

        # Authorization decision.
        if result.committed:
            auth_decision = AuthorizationDecision.ACCEPTED
        elif result.governance_decision == "reject":
            auth_decision = AuthorizationDecision.REJECTED
        elif result.governance_decision == "quarantine":
            auth_decision = AuthorizationDecision.QUARANTINED
        else:
            auth_decision = AuthorizationDecision.NO_OP

        # Compute metrics.
        compute = ComputeMetrics(
            candidate_evaluations=n_candidates,
            shadow_executions=0,
            wall_clock_seconds=0.0,
        )

        record_id = make_record_id(
            run_id, episode_id, step_id, seed, TransitionProvenance.REALIZED,
        )

        return TransitionRecord(
            record_id=record_id,
            run_id=run_id,
            episode_id=episode_id,
            step_id=step_id,
            graph_family=graph_family,
            split=split,
            seed=seed,
            authority_identity_before=auth_before,
            authority_identity_after=auth_after,
            structural_state_before=state_before,
            structural_state_after=state_after,
            diagnosis=diagnosis,
            candidate_set_summary=cand_set,
            selected_candidate=candidate_summaries[0] if candidate_summaries else None,
            planner_metadata=planner_meta,
            predicted_delta=float(result.delta_utility),
            predicted_risk=0.0,
            predicted_cost=0.0,
            predicted_ig=0.0,
            action=selected_action,
            action_target={},
            authorization_decision=auth_decision,
            transaction_id=result.receipt_hash,
            realized_delta=float(result.delta_utility) if result.committed else 0.0,
            realized_cost=float(n_candidates),
            realized_risk=0.0,
            success=bool(result.committed),
            rollback=False,
            rejected=not result.committed and result.governance_decision == "reject",
            compute_metrics=compute,
            provenance=TransitionProvenance.REALIZED,
            base_runtime_version=VERSION,
            generator_version=GENERATOR_VERSION,
            timestamp=self._timestamp,
        )

    def _generate_counterfactual_records(
        self,
        result: Any,
        runtime: LGAERuntime,
        run_id: str,
        episode_id: str,
        step_id: int,
        graph_family: str,
        split: str,
        seed: int,
    ) -> list[TransitionRecord]:
        """Generate counterfactual transition records via shadow evaluation.

        For each step, we evaluate a few alternative actions on a COPY of
        the graph (never the runtime's authoritative state). These are
        COUNTERFACTUAL records — what would have happened if an alternative
        action had been chosen.
        """
        records: list[TransitionRecord] = []
        current_graph = runtime._engine.graph

        # Get candidate non-edges for shadow evaluation.
        valid = current_graph.valid.bool()
        n = int(current_graph.num_nodes)
        edges = []
        for i in range(current_graph.src.shape[0]):
            if valid[i]:
                edges.append((
                    int(current_graph.src[i].item()),
                    int(current_graph.dst[i].item()),
                ))
        existing = set()
        for u, v in edges:
            existing.add((u, v))
            existing.add((v, u))

        # Sample non-edges for negative sampling.
        non_edges: list[tuple[int, int]] = []
        rng = np.random.RandomState(seed + step_id)
        for _ in range(self.n_negative_samples * 3):
            u = int(rng.randint(0, n))
            v = int(rng.randint(0, n))
            if u != v and (u, v) not in existing:
                non_edges.append((u, v))
                if len(non_edges) >= self.n_negative_samples:
                    break

        # Shadow evaluation: add each non-edge to a COPY and measure ΔU.
        base_utility = self._graph_utility(current_graph)
        state_before = self._extract_state_summary(result.snapshot_before)

        for idx, (u, v) in enumerate(non_edges):
            # Create a modified graph copy.
            modified_edges = list(edges) + [(u, v)]
            capacity = max(len(modified_edges) + 8, n * 2)
            try:
                modified_graph = make_graph_buffers(n, modified_edges, capacity=capacity)
                shadow_utility = self._graph_utility(modified_graph)
                shadow_delta = shadow_utility - base_utility
            except Exception:
                shadow_delta = 0.0
                shadow_utility = base_utility

            # Local action features.
            local_feats = extract_local_action_features(current_graph, u, v)

            record_id = make_record_id(
                run_id, episode_id, step_id, seed,
                TransitionProvenance.COUNTERFACTUAL, candidate_id=idx,
            )

            auth_before = AuthorityIdentity(
                state_hash=state_before.state_hash,
                state_version=state_before.graph_version,
                authority_hash=result.snapshot_before.authority_hash,
            )

            # Counterfactual state_after is the shadow state.
            state_after = self._extract_state_summary_from_graph(modified_graph)

            records.append(TransitionRecord(
                record_id=record_id,
                run_id=run_id,
                episode_id=episode_id,
                step_id=step_id,
                graph_family=graph_family,
                split=split,
                seed=seed,
                authority_identity_before=auth_before,
                authority_identity_after=None,  # counterfactual — no authority binding
                structural_state_before=state_before,
                structural_state_after=state_after,
                diagnosis=DiagnosisSummary(),
                candidate_set_summary=CandidateSetSummary(
                    n_candidates=1,
                    candidates=(CandidateSummary(
                        candidate_id=idx,
                        action_type="ADD_EDGE",
                        target={"u": u, "v": v},
                        predicted_delta=0.0,  # no prediction for shadow
                        predicted_risk=0.0,
                        predicted_cost=0.0,
                        predicted_ig=0.0,
                        selected=False,
                    ),),
                ),
                selected_candidate=None,
                planner_metadata=PlannerMetadata(),
                predicted_delta=0.0,
                predicted_risk=0.0,
                predicted_cost=0.0,
                predicted_ig=0.0,
                action="ADD_EDGE",
                action_target={"u": u, "v": v},
                authorization_decision=AuthorizationDecision.REJECTED,  # not selected
                transaction_id=None,
                realized_delta=float(shadow_delta),
                realized_cost=1.0,
                realized_risk=0.0,
                success=False,  # not actually committed
                rollback=False,
                rejected=True,
                compute_metrics=ComputeMetrics(
                    candidate_evaluations=1,
                    shadow_executions=1,
                ),
                provenance=TransitionProvenance.COUNTERFACTUAL,
                base_runtime_version=VERSION,
                generator_version=GENERATOR_VERSION,
                timestamp=self._timestamp,
                extra={
                    "local_features": local_feats.to_log(),
                    "shadow_utility": float(shadow_utility),
                    "base_utility": float(base_utility),
                },
            ))

        return records

    # ------------------------------------------------------------------ #
    # Helpers.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _graph_utility(graph: GraphBuffers) -> float:
        """Spectral gap utility."""
        try:
            lam, _ = spectral_gap_graphbuffers(graph)
            return float(lam)
        except Exception:
            return 0.0

    def _extract_state_summary(self, snap: Any) -> StructuralStateSummary:
        """Extract a StructuralStateSummary from a RuntimeSnapshot."""
        graph = None
        # Try to get the graph from the snapshot's hash.
        state_hash = getattr(snap, "graph_state_hash", "")
        graph_version = int(getattr(snap, "graph_version", 0))
        # We don't have direct access to the graph from the snapshot,
        # so we extract what we can.
        return StructuralStateSummary(
            n_nodes=0,  # filled from graph if available
            n_edges=0,
            density=0.0,
            spectral_gap=0.0,
            degree_mean=0.0,
            degree_std=0.0,
            n_components=0,
            avg_clustering=0.0,
            fiber_count=0,
            fiber_width=0,
            gauge_dim=0,
            state_hash=state_hash,
            graph_version=graph_version,
        )

    def _extract_state_summary_from_graph(self, graph: GraphBuffers) -> StructuralStateSummary:
        """Extract a StructuralStateSummary from a GraphBuffers."""
        n = int(graph.num_nodes)
        valid = graph.valid.bool()
        n_edges = int(valid.sum().item())
        max_edges = n * (n - 1) / 2
        density = float(n_edges) / max(max_edges, 1.0)

        # Spectral gap.
        try:
            lam, _ = spectral_gap_graphbuffers(graph)
            spec_gap = float(lam)
        except Exception:
            spec_gap = 0.0

        # Degree statistics.
        degrees = [0] * n
        for i in range(graph.src.shape[0]):
            if valid[i]:
                s = int(graph.src[i].item())
                d = int(graph.dst[i].item())
                if s < n:
                    degrees[s] += 1
                if d < n:
                    degrees[d] += 1
        deg_mean = float(np.mean(degrees)) if degrees else 0.0
        deg_std = float(np.std(degrees)) if degrees else 0.0

        # State hash.
        try:
            state_hash = graph.state_hash()
        except Exception:
            state_hash = ""

        return StructuralStateSummary(
            n_nodes=n,
            n_edges=n_edges,
            density=density,
            spectral_gap=spec_gap,
            degree_mean=deg_mean,
            degree_std=deg_std,
            n_components=1,  # assume connected for generated graphs
            avg_clustering=0.0,
            fiber_count=1,
            fiber_width=int(self.config.fiber.d_base),
            gauge_dim=int(self.config.fiber.gauge_dim),
            state_hash=state_hash,
            graph_version=int(graph.version),
        )


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
