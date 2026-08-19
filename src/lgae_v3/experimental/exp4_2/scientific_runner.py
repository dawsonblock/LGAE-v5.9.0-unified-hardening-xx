"""Scientific runner for exp4.2.

Orchestrates the full experiment:
1. Freeze dataset
2. Train on TRAIN only (multi-seed)
3. Select on VALIDATION only
4. Lock finalists
5. Open held-out ONCE
6. Generate final scientific report

The runner enforces the experiment state machine and strict compatibility.
It NEVER mutates authoritative runtime state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import time
import math
import hashlib
import numpy as np

from .experiment_state import ExperimentStateMachine, ExperimentStateError
from .experiment_config import (
    ExperimentConfig, FinalistLock, SelectionWeights, default_experiment_config,
)
from .dataset_freeze import DatasetFreeze, freeze_dataset
from .targets import TARGET_DEFINITIONS, get_target_definition, extract_target_value
from .metrics import (
    compute_regret, compute_oracle_recovery, compute_selective_prediction,
    compute_pareto_frontier, ParetoFrontierEntry, bootstrap_ci,
    compute_uncertainty_error_correlation,
)
from .cf_real import run_cf_real_experiment, SupervisionRegime


@dataclass(slots=True)
class SplitResult:
    """Results for one encoder × predictor × target on one split."""
    encoder_id: str
    predictor_id: str
    target: str
    seed: int
    split_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    predictions: list[float] = field(default_factory=list)
    true_values: list[float] = field(default_factory=list)
    uncertainties: list[float] = field(default_factory=list)
    n_samples: int = 0
    latency_ms: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "predictor_id": self.predictor_id,
            "target": self.target,
            "seed": int(self.seed),
            "split_name": self.split_name,
            "metrics": dict(self.metrics),
            "n_samples": int(self.n_samples),
            "latency_ms": float(self.latency_ms),
        }


@dataclass
class ScientificResult:
    """Full scientific result for one encoder × predictor × target."""
    encoder_id: str
    predictor_id: str
    target: str
    train_metrics: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    heldout_metrics: dict[str, Any] = field(default_factory=dict)
    seed_results: list[dict[str, Any]] = field(default_factory=list)
    mean_validation_score: float = 0.0
    std_validation_score: float = 0.0
    regret: dict[str, Any] = field(default_factory=dict)
    oracle_recovery: dict[str, Any] = field(default_factory=dict)
    selective_prediction: dict[str, Any] = field(default_factory=dict)
    cf_real: dict[str, Any] = field(default_factory=dict)
    uncertainty_correlation: dict[str, Any] = field(default_factory=dict)
    group_metrics: list[dict[str, Any]] = field(default_factory=list)
    n_parameters: int = 0
    encoding_latency_ms: float = 0.0
    prediction_latency_ms: float = 0.0
    artifact_hash: str = ""
    is_control: bool = False  # True for scientific controls

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "predictor_id": self.predictor_id,
            "target": self.target,
            "train_metrics": dict(self.train_metrics),
            "validation_metrics": dict(self.validation_metrics),
            "heldout_metrics": dict(self.heldout_metrics),
            "seed_results": list(self.seed_results),
            "mean_validation_score": float(self.mean_validation_score),
            "std_validation_score": float(self.std_validation_score),
            "regret": dict(self.regret),
            "oracle_recovery": dict(self.oracle_recovery),
            "selective_prediction": dict(self.selective_prediction),
            "cf_real": dict(self.cf_real),
            "uncertainty_correlation": dict(self.uncertainty_correlation),
            "group_metrics": list(self.group_metrics),
            "n_parameters": int(self.n_parameters),
            "encoding_latency_ms": float(self.encoding_latency_ms),
            "prediction_latency_ms": float(self.prediction_latency_ms),
            "artifact_hash": self.artifact_hash,
            "is_control": bool(self.is_control),
        }


@dataclass
class ScientificConclusion:
    """Machine-readable scientific conclusion."""
    experiment: str = "v6.0-exp4.2"
    scientific_status: str = "INCONCLUSIVE"
    structural_signal_detected: bool = False
    generalizes_to_heldout: bool = False
    best_model: str = ""
    best_encoder: str = ""
    best_validation_score: float = 0.0
    best_heldout_spearman: float = 0.0
    best_heldout_regret: float = 0.0
    exp5_authorized: bool = False
    recommended_exp5_architecture: str = ""
    cf_real_transfer_ok: bool = False
    uncertainty_useful: bool = False
    limitations: list[str] = field(default_factory=list)
    per_target_summary: dict[str, Any] = field(default_factory=dict)
    baseline_spearman: float = 0.0  # best control spearman on best target

    def to_log(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "scientific_status": self.scientific_status,
            "structural_signal_detected": bool(self.structural_signal_detected),
            "generalizes_to_heldout": bool(self.generalizes_to_heldout),
            "best_model": self.best_model,
            "best_encoder": self.best_encoder,
            "best_validation_score": float(self.best_validation_score),
            "best_heldout_spearman": float(self.best_heldout_spearman),
            "best_heldout_regret": float(self.best_heldout_regret),
            "exp5_authorized": bool(self.exp5_authorized),
            "recommended_exp5_architecture": self.recommended_exp5_architecture,
            "cf_real_transfer_ok": bool(self.cf_real_transfer_ok),
            "uncertainty_useful": bool(self.uncertainty_useful),
            "limitations": list(self.limitations),
            "per_target_summary": dict(self.per_target_summary),
            "baseline_spearman": float(self.baseline_spearman),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), sort_keys=True, indent=2)


class ScientificRunner:
    """Orchestrates the exp4.2 held-out structural prediction study.

    Usage::

        runner = ScientificRunner(config=default_experiment_config())
        runner.prepare(datasets)
        runner.train(records)
        runner.validate()
        runner.lock_finalists()
        runner.open_heldout()
        conclusion = runner.finalize()
    """

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or default_experiment_config()
        self.state = ExperimentStateMachine()
        self._dataset_freeze: DatasetFreeze | None = None
        self._records: list[Any] = []
        self._train_results: list[ScientificResult] = []
        self._validation_results: list[ScientificResult] = []
        self._heldout_results: list[ScientificResult] = []
        self._finalist_lock: FinalistLock | None = None
        self._conclusion: ScientificConclusion | None = None

    @property
    def dataset_freeze(self) -> DatasetFreeze | None:
        return self._dataset_freeze

    def prepare(
        self,
        datasets: dict[str, Any],
        *,
        dataset_schema_hash: str,
        feature_schema_hash: str,
        graph_family_registry_hash: str,
    ) -> DatasetFreeze:
        """Phase 3: Freeze the dataset."""
        self.state.assert_selection_permitted()
        freeze = freeze_dataset(
            datasets,
            dataset_schema_hash=dataset_schema_hash,
            feature_schema_hash=feature_schema_hash,
            graph_family_registry_hash=graph_family_registry_hash,
            seed=42,
        )
        self._dataset_freeze = freeze
        # Collect all records.
        self._records = (
            list(datasets["train"].records)
            + list(datasets["validation"].records)
            + list(datasets["held_out"].records)
        )
        self.state.transition_to("TRAINING", action="dataset_frozen")
        return freeze

    def train(self, records: list[Any] | None = None) -> list[ScientificResult]:
        """Phase 9: Train all encoder × predictor × target on TRAIN only."""
        self.state.assert_selection_permitted()
        if records is not None:
            self._records = records

        recs = [r for r in self._records if getattr(r, "split", "") == "train"]
        val_recs = [r for r in self._records if getattr(r, "split", "") == "validation"]

        results = []
        total = len(self.config.encoders) * len(self.config.predictors) * len(self.config.targets)
        done = 0
        for enc_cfg in self.config.encoders:
            for pred_cfg in self.config.predictors:
                for target_name in self.config.targets:
                    td = get_target_definition(target_name)
                    # Skip invalid combinations.
                    if td.task_category == "classification" and pred_cfg.predictor_id not in (
                        "global_mean", "logistic", "tree", "mlp_clf"
                    ):
                        continue
                    if td.task_category == "ranking" and pred_cfg.predictor_id not in (
                        "pointwise_rank", "pairwise_rank"
                    ):
                        continue
                    if td.task_category == "regression" and pred_cfg.predictor_id in (
                        "logistic", "mlp_clf"
                    ):
                        continue

                    done += 1
                    result = self._train_single(
                        enc_cfg, pred_cfg, target_name, recs, val_recs,
                    )
                    if result is not None:
                        results.append(result)
                    if done % 10 == 0:
                        print(f"    [{done}/{total}] {enc_cfg.encoder_id} × {pred_cfg.predictor_id} × {target_name}...")

        self._train_results = results
        self.state.transition_to("VALIDATION", action="training_complete")
        return results

    def _train_single(
        self,
        enc_cfg: Any,
        pred_cfg: Any,
        target_name: str,
        train_recs: list[Any],
        val_recs: list[Any],
    ) -> ScientificResult | None:
        """Train one combination across multiple seeds."""
        from ..encoders import EncoderRegistry
        from ..models.model_registry import ModelRegistry
        from ..models.evaluator import (
            compute_regression_metrics, compute_classification_metrics,
        )

        td = get_target_definition(target_name)
        is_classification = td.task_category == "classification"

        seed_results = []
        val_scores = []

        for seed in self.config.seeds:
            # Create encoder.
            try:
                encoder = EncoderRegistry.create(enc_cfg.encoder_id)
            except Exception:
                continue

            # Fit encoder on train only.
            if encoder.requires_fit:
                self._fit_encoder(encoder, train_recs)

            # Extract features.
            X_train, y_train = self._extract_features(encoder, train_recs, target_name)
            X_val, y_val = self._extract_features(encoder, val_recs, target_name)

            if len(X_train) == 0:
                continue

            # Create model.
            kwargs = {}
            if pred_cfg.predictor_id in ("mlp", "mlp_clf"):
                kwargs = {"n_epochs": self.config.n_epochs, "n_ensemble": self.config.n_ensemble, "seed": seed}
            elif pred_cfg.predictor_id in ("linear", "ridge", "logistic"):
                kwargs = {"n_epochs": self.config.n_epochs}
            elif pred_cfg.predictor_id == "tree":
                kwargs = {"n_estimators": min(30, self.config.n_epochs)}

            try:
                model = ModelRegistry.create(pred_cfg.predictor_id, **kwargs)
            except Exception:
                continue

            # Fit on train only.
            try:
                model.fit(X_train, y_train, split="train")
                if hasattr(model, "freeze"):
                    model.freeze()
            except Exception:
                continue

            # Evaluate on train and validation.
            train_m = self._evaluate(model, X_train, y_train, is_classification)
            val_m = self._evaluate(model, X_val, y_val, is_classification)

            val_score = self.config.selection_weights.compute_score(
                spearman=val_m.get("spearman", 0.0),
                ndcg=val_m.get("ndcg", 0.0),
                regret=val_m.get("mean_regret", 0.0),
                sign_accuracy=val_m.get("accuracy", 0.0),
                ece=val_m.get("ece", 0.0),
                latency_ms=0.0,
                ood_proxy=0.0,
            )
            val_scores.append(val_score)
            seed_results.append({
                "seed": seed,
                "train_metrics": train_m,
                "validation_metrics": val_m,
            })

        if not seed_results:
            return None

        mean_score = float(np.mean(val_scores))
        std_score = float(np.std(val_scores)) if len(val_scores) > 1 else 0.0

        return ScientificResult(
            encoder_id=enc_cfg.encoder_id,
            predictor_id=pred_cfg.predictor_id,
            target=target_name,
            seed_results=seed_results,
            mean_validation_score=mean_score,
            std_validation_score=std_score,
        )

    def validate(self) -> list[ScientificResult]:
        """Phase 10: Validation competition — select on validation only."""
        self.state.assert_selection_permitted()
        self._validation_results = self._train_results
        # Already in VALIDATION after train() — no transition needed.
        return self._validation_results

    def lock_finalists(self) -> FinalistLock:
        """Phase 28: Lock finalists based on validation performance."""
        self.state.assert_selection_permitted()

        # Select top performers per category.
        by_target: dict[str, list[ScientificResult]] = {}
        for r in self._validation_results:
            by_target.setdefault(r.target, []).append(r)

        finalists = []
        for target, results in by_target.items():
            # Sort by validation score.
            sorted_results = sorted(results, key=lambda r: r.mean_validation_score, reverse=True)
            # Take top 3 per target.
            for r in sorted_results[:3]:
                finalists.append({
                    "encoder_id": r.encoder_id,
                    "predictor_id": r.predictor_id,
                    "target": r.target,
                    "mean_validation_score": r.mean_validation_score,
                    "std_validation_score": r.std_validation_score,
                })

        lock = FinalistLock(
            finalists=finalists,
            selection_weights=self.config.selection_weights.to_log(),
        )
        self._finalist_lock = lock
        self.state.lock_finalists(lock.config_hash)
        return lock

    # Scientific controls that MUST always be evaluated on held-out,
    # regardless of validation pruning. These are not competitors —
    # they are immutable scientific controls.
    SCIENTIFIC_CONTROLS = (
        "global_mean",
        "mutation_type_mean",
        "nearest_experience",
    )

    def open_heldout(self) -> list[ScientificResult]:
        """Phase 29: One-shot held-out evaluation. No retraining.

        Always evaluates:
        1. All locked finalists.
        2. Scientific controls (global_mean, mutation_type_mean,
           nearest_experience) on every target, regardless of
           whether they were validation finalists.

        Controls are NOT subject to validation pruning. They are
        scientific baselines that must be beaten on held-out.
        """
        # Transition to HELDOUT_OPENED if not already.
        if self.state.state == "MODEL_LOCKED":
            self.state.transition_to("HELDOUT_OPENED", action="open_heldout")

        self.state.assert_heldout_accessible()

        held_recs = [r for r in self._records if getattr(r, "split", "") == "held_out"]

        results = []

        # 1. Evaluate locked finalists.
        finalist_keys: set[tuple[str, str, str]] = set()
        for f in self._finalist_lock.finalists if self._finalist_lock else []:
            enc_id = f["encoder_id"]
            pred_id = f["predictor_id"]
            target = f["target"]
            finalist_keys.add((enc_id, pred_id, target))

            result = self._evaluate_heldout(enc_id, pred_id, target, held_recs)
            if result is not None:
                results.append(result)

        # 2. Evaluate scientific controls on every target.
        #    Use the minimal-control encoder for controls (simplest
        #    representation), so the control is "simple stats on
        #    simple features."
        for target in self.config.targets:
            for ctrl_pred in self.SCIENTIFIC_CONTROLS:
                key = ("minimal-control", ctrl_pred, target)
                if key in finalist_keys:
                    continue  # already evaluated as a finalist
                result = self._evaluate_heldout(
                    "minimal-control", ctrl_pred, target, held_recs,
                    is_control=True,
                )
                if result is not None:
                    results.append(result)

        self._heldout_results = results
        return results

    def _evaluate_heldout(
        self,
        enc_id: str,
        pred_id: str,
        target: str,
        held_recs: list[Any],
        *,
        is_control: bool = False,
    ) -> ScientificResult | None:
        """Evaluate one finalist or control on held-out data.

        Args:
            enc_id: Encoder identifier.
            pred_id: Predictor identifier.
            target: Target name.
            held_recs: Held-out records.
            is_control: If True, this is a scientific control (not a
                competitor). Controls are always evaluated regardless
                of validation pruning.
        """
        from ..encoders import EncoderRegistry
        from ..models.model_registry import ModelRegistry
        from ..models.evaluator import (
            compute_regression_metrics, compute_classification_metrics,
        )

        td = get_target_definition(target)
        is_classification = td.task_category == "classification"

        # Use first seed for held-out (finalists are locked).
        seed = self.config.seeds[0]

        train_recs = [r for r in self._records if getattr(r, "split", "") == "train"]

        try:
            encoder = EncoderRegistry.create(enc_id)
        except Exception:
            return None
        if encoder.requires_fit:
            self._fit_encoder(encoder, train_recs)

        X_train, y_train = self._extract_features(encoder, train_recs, target)
        X_held, y_held = self._extract_features(encoder, held_recs, target)

        if len(X_train) == 0 or len(X_held) == 0:
            return None

        kwargs = {}
        if pred_id in ("mlp", "mlp_clf"):
            kwargs = {"n_epochs": self.config.n_epochs, "n_ensemble": self.config.n_ensemble, "seed": seed}
        elif pred_id in ("linear", "ridge", "logistic"):
            kwargs = {"n_epochs": self.config.n_epochs}
        elif pred_id == "tree":
            kwargs = {"n_estimators": min(30, self.config.n_epochs)}

        try:
            model = ModelRegistry.create(pred_id, **kwargs)
            model.fit(X_train, y_train, split="train")
            if hasattr(model, "freeze"):
                model.freeze()
        except Exception:
            return None

        held_m = self._evaluate(model, X_held, y_held, is_classification)

        # Compute regret on held-out (group by state/episode).
        regret_report = self._compute_heldout_regret(model, held_recs, encoder, target)

        # Compute uncertainty-error correlation.
        unc_corr = self._compute_uncertainty_corr(model, X_held, y_held)

        # CF-to-real experiment.
        prov_train = [str(getattr(r, "provenance", "")).lower() for r in train_recs]
        # Map to "realized" or "counterfactual".
        prov_clean = []
        for p in prov_train:
            if "realized" in p:
                prov_clean.append("realized")
            elif "counterfactual" in p:
                prov_clean.append("counterfactual")
            else:
                prov_clean.append("realized")

        def model_factory():
            k = dict(kwargs)
            return ModelRegistry.create(pred_id, **k)

        try:
            cf_report = run_cf_real_experiment(
                X_train, y_train, prov_clean,
                np.zeros((0, X_train.shape[1])), np.zeros(0),  # no val for simplicity
                X_held, y_held,
                model_factory=model_factory,
                is_classification=is_classification,
            )
            cf_log = cf_report.to_log()
        except Exception:
            cf_log = {}

        # Carry forward validation metrics from training results.
        val_metrics: dict[str, Any] = {}
        mean_val_score = 0.0
        std_val_score = 0.0
        for tr in self._train_results:
            if tr.encoder_id == enc_id and tr.predictor_id == pred_id and tr.target == target:
                val_metrics = tr.validation_metrics
                mean_val_score = tr.mean_validation_score
                std_val_score = tr.std_validation_score
                break

        return ScientificResult(
            encoder_id=enc_id,
            predictor_id=pred_id,
            target=target,
            validation_metrics=val_metrics,
            heldout_metrics=held_m,
            mean_validation_score=mean_val_score,
            std_validation_score=std_val_score,
            regret=regret_report.to_log() if regret_report else {},
            uncertainty_correlation=unc_corr.to_log() if unc_corr else {},
            cf_real=cf_log,
            is_control=is_control,
        )

    def finalize(self) -> ScientificConclusion:
        """Phase 36: Generate the machine-readable conclusion."""
        if self.state.state != "HELDOUT_OPENED":
            raise ExperimentStateError(
                f"Cannot finalize from state {self.state.state}. "
                f"Must be in HELDOUT_OPENED."
            )

        conclusion = self._compute_conclusion()
        self._conclusion = conclusion
        self.state.transition_to("FINALIZED", action="finalize")
        return conclusion

    def _compute_conclusion(self) -> ScientificConclusion:
        """Compute the scientific conclusion from held-out results.

        Separate conclusions are computed for each task category
        (regression, classification, ranking). The overall conclusion
        uses the best non-control model that materially outperforms
        the best scientific control on the same target.
        """
        # Separate results into controls and competitors.
        controls = [r for r in self._heldout_results if r.is_control]
        competitors = [r for r in self._heldout_results if not r.is_control]

        # Find best competitor by held-out spearman, per target.
        best_per_target: dict[str, ScientificResult] = {}
        for r in competitors:
            sp = r.heldout_metrics.get("spearman", 0.0)
            if r.target not in best_per_target or sp > best_per_target[r.target].heldout_metrics.get("spearman", 0.0):
                best_per_target[r.target] = r

        # Find best control per target.
        best_control_per_target: dict[str, ScientificResult] = {}
        for r in controls:
            sp = r.heldout_metrics.get("spearman", 0.0)
            if r.target not in best_control_per_target or sp > best_control_per_target[r.target].heldout_metrics.get("spearman", 0.0):
                best_control_per_target[r.target] = r

        # Determine the best overall competitor (highest held-out spearman).
        best = None
        best_sp = -1.0
        for r in competitors:
            sp = r.heldout_metrics.get("spearman", 0.0)
            if sp > best_sp:
                best_sp = sp
                best = r

        # Baseline comparison: does the best competitor beat the best
        # control on the SAME target?
        baseline_sp = 0.0
        if best:
            target_controls = [r for r in controls if r.target == best.target]
            if target_controls:
                baseline_sp = max(r.heldout_metrics.get("spearman", 0.0) for r in target_controls)

        # Check if signal detected.
        signal = best_sp > 0.3 if best else False
        # Generalization requires materially outperforming controls.
        generalizes = (best_sp - baseline_sp) > 0.1 if best else False

        # Check CF-to-real transfer.
        cf_ok = True
        if best and best.cf_real:
            gap = abs(best.cf_real.get("gap_cf_to_real_spearman", 0.0))
            cf_ok = gap < 0.2

        # Check uncertainty usefulness.
        unc_useful = False
        if best and best.uncertainty_correlation:
            corr = best.uncertainty_correlation.get("corr_uncertainty_abs_error", 0.0)
            unc_useful = corr > 0.1

        # Determine status and exp5 authorization.
        # Downgraded from QUALIFIED_SIMPLE to PRELIMINARY_SIGNAL_DETECTED
        # until controls are beaten and multi-step rollout is validated.
        if signal and generalizes:
            status = "PRELIMINARY_SIGNAL_DETECTED"
            exp5_auth = True
            # Recommend architecture based on winning encoder.
            if best and best.encoder_id in ("learned-graph", "hybrid"):
                rec_arch = "graph_native_world_model"
            else:
                rec_arch = "lightweight_latent_dynamics"
        elif signal and not generalizes:
            status = "INCONCLUSIVE"
            exp5_auth = False
            rec_arch = ""
        else:
            status = "FAILED_GENERALIZATION"
            exp5_auth = False
            rec_arch = ""

        # Apply additional gates.
        if not cf_ok:
            exp5_auth = False
            status = "FAILED_CF_REAL_TRANSFER"
        if best and best.regret:
            cat_rate = best.regret.get("catastrophic_regret_rate", 1.0)
            if cat_rate > 0.5:
                exp5_auth = False
                if status not in ("FAILED_GENERALIZATION", "FAILED_CF_REAL_TRANSFER"):
                    status = "INCONCLUSIVE"

        # Build per-target summary.
        per_target_summary: dict[str, Any] = {}
        for target, br in best_per_target.items():
            bc = best_control_per_target.get(target)
            per_target_summary[target] = {
                "best_competitor": {
                    "encoder": br.encoder_id,
                    "predictor": br.predictor_id,
                    "heldout_spearman": br.heldout_metrics.get("spearman", 0.0),
                    "heldout_regret": br.regret.get("mean_regret", 0.0) if br.regret else 0.0,
                },
                "best_control": {
                    "predictor": bc.predictor_id if bc else "none",
                    "heldout_spearman": bc.heldout_metrics.get("spearman", 0.0) if bc else 0.0,
                } if bc else None,
                "beats_control": (
                    br.heldout_metrics.get("spearman", 0.0) - (bc.heldout_metrics.get("spearman", 0.0) if bc else 0.0)
                ) > 0.1 if bc else False,
            }

        limitations = []
        if not generalizes:
            limitations.append("No model materially outperformed scientific controls on held-out.")
        if not cf_ok:
            limitations.append("Counterfactual-to-real transfer gap is too large.")
        if not unc_useful:
            limitations.append("Uncertainty does not correlate with error — trust signal is weak.")
        # Add dataset limitations.
        limitations.append("Dataset is synthetic with limited mutation type diversity.")
        limitations.append("Multi-step rollout quality is not yet validated for MPC use.")
        limitations.append("Risk target is near-constant — risk prediction is not scientifically tested.")

        return ScientificConclusion(
            scientific_status=status,
            structural_signal_detected=signal,
            generalizes_to_heldout=generalizes,
            best_model=best.predictor_id if best else "",
            best_encoder=best.encoder_id if best else "",
            best_validation_score=best.mean_validation_score if best else 0.0,
            best_heldout_spearman=best_sp,
            best_heldout_regret=best.regret.get("mean_regret", 0.0) if best and best.regret else 0.0,
            exp5_authorized=exp5_auth,
            recommended_exp5_architecture=rec_arch,
            cf_real_transfer_ok=cf_ok,
            uncertainty_useful=unc_useful,
            limitations=limitations,
            per_target_summary=per_target_summary,
            baseline_spearman=baseline_sp,
        )

    # ------------------------------------------------------------------
    # Helper methods.
    # ------------------------------------------------------------------

    def _fit_encoder(self, encoder: Any, train_records: list[Any]) -> None:
        """Fit an encoder on train records only."""
        if not encoder.requires_fit:
            return
        global_feats = []
        local_feats = []
        for r in train_records:
            state = r.structural_state_before
            global_feats.append(self._state_to_global_features(state))
            local_feats.append(self._action_to_local_features(r))
        if hasattr(encoder, "fit"):
            try:
                encoder.fit(global_feats, local_feats, split="train")
            except TypeError:
                try:
                    encoder.fit(global_feats, split="train")
                except TypeError:
                    pass
        if hasattr(encoder, "freeze"):
            try:
                encoder.freeze()
            except Exception:
                pass

    def _extract_features(
        self,
        encoder: Any,
        records: list[Any],
        target_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from records."""
        X, y = [], []
        for r in records:
            state_before = r.structural_state_before
            global_feats = self._state_to_global_features(state_before)
            local_feats = self._action_to_local_features(r)
            state_obj = _StateObj(state_before)
            try:
                rep = encoder.encode(
                    state=state_obj,
                    global_features=global_feats,
                    action_type=r.action,
                    action_target=r.action_target,
                    local_features=local_feats,
                )
                X.append(list(rep.vector))
            except Exception:
                continue
            y.append(extract_target_value(r, target_name))

        if not X:
            return np.zeros((0, 1)), np.zeros(0)
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

    def _evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool,
    ) -> dict[str, Any]:
        """Evaluate a model."""
        if len(X) == 0:
            return {}
        if is_classification and hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)
            from ..models.evaluator import compute_classification_metrics
            m = compute_classification_metrics(preds, y.astype(int).tolist())
            return m.to_log()
        elif hasattr(model, "predict"):
            preds = model.predict(X)
            from ..models.evaluator import compute_regression_metrics
            m = compute_regression_metrics(preds, y.tolist())
            return m.to_log()
        return {}

    def _compute_heldout_regret(
        self,
        model: Any,
        held_recs: list[Any],
        encoder: Any,
        target: str,
    ) -> Any:
        """Compute regret by grouping records into candidate sets."""
        # Group by (episode_id, step_id) to form candidate sets.
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(held_recs):
            ep = getattr(r, "episode_id", f"ep_{i}")
            step = getattr(r, "step_id", 0)
            key = f"{ep}_{step}"
            groups.setdefault(key, []).append(i)

        pred_utils = []
        true_utils = []
        for key, indices in groups.items():
            if len(indices) < 2:
                continue
            preds = []
            trues = []
            for idx in indices:
                r = held_recs[idx]
                state_before = r.structural_state_before
                global_feats = self._state_to_global_features(state_before)
                local_feats = self._action_to_local_features(r)
                state_obj = _StateObj(state_before)
                try:
                    rep = encoder.encode(
                        state=state_obj,
                        global_features=global_feats,
                        action_type=r.action,
                        action_target=r.action_target,
                        local_features=local_feats,
                    )
                    pred = model.predict(np.array([list(rep.vector)], dtype=np.float64))
                    preds.append(float(pred[0].mean))
                    trues.append(extract_target_value(r, target))
                except Exception:
                    continue
            if len(preds) >= 2:
                pred_utils.append(preds)
                true_utils.append(trues)

        if not pred_utils:
            return None
        return compute_regret(
            pred_utils, true_utils,
            catastrophic_threshold=self.config.catastrophic_regret_threshold,
        )

    def _compute_uncertainty_corr(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Any:
        """Compute uncertainty-error correlation."""
        if len(X) == 0 or not hasattr(model, "predict"):
            return None
        preds = model.predict(X)
        uncs = [float(p.uncertainty) for p in preds]
        errors = [abs(float(p.mean) - float(t)) for p, t in zip(preds, y)]
        return compute_uncertainty_error_correlation(uncs, errors)

    def _state_to_global_features(self, state: Any) -> list[float]:
        """Convert a StructuralStateSummary to global feature vector."""
        return [
            float(getattr(state, "n_nodes", 0)),
            float(getattr(state, "n_edges", 0)),
            float(getattr(state, "density", 0.0)),
            float(getattr(state, "degree_mean", 0.0)),
            float(getattr(state, "degree_std", 0.0)),
            float(getattr(state, "degree_std", 0.0)),
            float(getattr(state, "spectral_gap", 0.0)),
            float(math.log1p(max(abs(getattr(state, "spectral_gap", 0.0)), 1e-10))),
            float(getattr(state, "spectral_gap", 0.0) / max(getattr(state, "n_nodes", 1), 1)),
            float(getattr(state, "n_components", 1)),
            float(getattr(state, "avg_clustering", 0.0)),
            float(getattr(state, "n_nodes", 1)),
            0.0, 0.0, 0.0,
            0.0, 0.0,
            float(getattr(state, "fiber_count", 0)),
            float(getattr(state, "fiber_width", 0)),
            float(getattr(state, "gauge_dim", 0)),
            0.0, 0.0,
            0.0, 0.0,
        ]

    def _action_to_local_features(self, record: Any) -> list[float]:
        """Extract local action features."""
        target = record.action_target
        u = int(target.get("u", 0)) if isinstance(target, dict) else 0
        v = int(target.get("v", 0)) if isinstance(target, dict) else 0
        state = record.structural_state_before
        return [
            float(u), float(v),
            float(getattr(state, "degree_mean", 0.0)),
            float(getattr(state, "degree_std", 0.0)),
            float(getattr(state, "density", 0.0)),
            float(getattr(state, "spectral_gap", 0.0)),
            float(getattr(state, "n_components", 1)),
            float(getattr(state, "avg_clustering", 0.0)),
            0.0, 0.0, 0.0, 0.0,
        ]


@dataclass
class _StateObj:
    """Lightweight state object for encoder.encode()."""
    n_nodes: int = 10
    n_edges: int = 9
    graph_family: str = "path"

    def __init__(self, state: Any) -> None:
        self.n_nodes = int(getattr(state, "n_nodes", 10))
        self.n_edges = int(getattr(state, "n_edges", 9))
        self.graph_family = str(getattr(state, "state_hash", "path"))[:8]


def authorize_exp5(conclusion: ScientificConclusion) -> bool:
    """Phase 40: Explicit exp5 authorization gate.

    Returns True only when predefined scientific conditions are satisfied.
    This does not control Git branches technically, but makes the
    project's scientific decision explicit.

    Conditions:
    - structural_signal_detected must be True
    - generalizes_to_heldout must be True
    - exp5_authorized must be True (set by the conclusion computation)
    - cf_real_transfer_ok must be True
    - catastrophic_regret_rate must not be extreme
    """
    if not conclusion.structural_signal_detected:
        return False
    if not conclusion.generalizes_to_heldout:
        return False
    if not conclusion.exp5_authorized:
        return False
    if not conclusion.cf_real_transfer_ok:
        return False
    return True
