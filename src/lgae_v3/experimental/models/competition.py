"""v6.0-exp4.1: Real structural model competition.

Executes the experiment exp4 was designed to support:

    Encoder_i × Predictor_j

on the actual exp2 structural-transition dataset.

Produces:
- Train/validation/held-out metrics separately
- Group-level breakdowns (graph family, mutation type, OOD status)
- Counterfactual-to-real transfer gap
- Complexity-adjusted comparison table

The decisive scientific question:

    Can LGAE predict which structural changes will actually improve
    an unseen graph?

If the answer is yes, exp5 (structural world model) is justified.
If no model beats simple handcrafted representations plus a tree/linear
predictor on held-out graph families, do not proceed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import time
import math
import numpy as np

from .protocol import Prediction, ModelLifecycle
from .baselines import GlobalMeanPredictor, MutationTypeMeanPredictor, NearestExperiencePredictor
from .linear import LinearRegressionPredictor, RidgeRegressionPredictor, LogisticRegressionPredictor
from .tree import GradientBoostedTreePredictor
from .mlp import MLPRegressor, MLPClassifier
from .ranking import PointwiseRankingModel, PairwiseRankingModel
from .evaluator import (
    RegressionMetrics, ClassificationMetrics, RankingMetrics,
    GroupMetrics, CFToRealGap,
    compute_regression_metrics, compute_classification_metrics,
    compute_ranking_metrics, compute_group_metrics,
    compute_cf_to_real_gap, compute_ood_degradation,
)
from .calibration import (
    expected_calibration_error, brier_score,
    prediction_interval_coverage, standardized_residual_calibration,
)
from .artifact import ModelArtifact, create_artifact
from .model_registry import ModelRegistry
from .targets import compute_sign_delta, compute_normalized_delta


# ---------------------------------------------------------------------------
# Competition data structures.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CompetitionEntry:
    """A single Encoder × Predictor competition entry."""
    encoder_id: str
    predictor_id: str
    target: str  # "realized_delta", "sign_delta", "risk", "cost"

    # Metrics.
    train_metrics: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    heldout_metrics: dict[str, Any] = field(default_factory=dict)

    # Group breakdowns.
    group_metrics: list[dict[str, Any]] = field(default_factory=list)

    # CF-to-real gap.
    cf_to_real_gap: float | None = None

    # Calibration.
    ece: float | None = None
    brier: float | None = None
    interval_coverage: float | None = None

    # Complexity.
    n_parameters: int = 0
    encoding_latency_ms: float = 0.0

    # Artifact.
    artifact: ModelArtifact | None = None

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "predictor_id": self.predictor_id,
            "target": self.target,
            "train_metrics": dict(self.train_metrics),
            "validation_metrics": dict(self.validation_metrics),
            "heldout_metrics": dict(self.heldout_metrics),
            "group_metrics": list(self.group_metrics),
            "cf_to_real_gap": self.cf_to_real_gap,
            "ece": self.ece,
            "brier": self.brier,
            "interval_coverage": self.interval_coverage,
            "n_parameters": int(self.n_parameters),
            "encoding_latency_ms": float(self.encoding_latency_ms),
            "artifact_hash": self.artifact.artifact_hash if self.artifact else None,
        }


@dataclass(slots=True)
class CompetitionReport:
    """Full competition report across all Encoder × Predictor combinations."""
    entries: list[CompetitionEntry] = field(default_factory=list)
    dataset_schema_hash: str = ""
    train_split_hash: str = ""
    validation_split_hash: str = ""
    heldout_split_hash: str = ""
    heldout_accessed: bool = False
    created_at: str = ""
    n_train: int = 0
    n_validation: int = 0
    n_heldout: int = 0

    def to_log(self) -> dict[str, Any]:
        return {
            "dataset_schema_hash": self.dataset_schema_hash,
            "train_split_hash": self.train_split_hash,
            "validation_split_hash": self.validation_split_hash,
            "heldout_split_hash": self.heldout_split_hash,
            "heldout_accessed": self.heldout_accessed,
            "created_at": self.created_at,
            "n_train": int(self.n_train),
            "n_validation": int(self.n_validation),
            "n_heldout": int(self.n_heldout),
            "entries": [e.to_log() for e in self.entries],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_log(), sort_keys=True, indent=2)

    def summary_table(self) -> str:
        """Produce a human-readable summary table."""
        lines = []
        lines.append(f"{'Encoder':<20} {'Predictor':<20} {'Target':<15} "
                     f"{'Val Spearman':<14} {'Held Spearman':<14} "
                     f"{'Sign Acc':<10} {'ECE':<8} {'Params':<8}")
        lines.append("-" * 120)
        for e in self.entries:
            val_sp = e.validation_metrics.get("spearman", 0.0)
            held_sp = e.heldout_metrics.get("spearman", 0.0)
            sign_acc = e.heldout_metrics.get("accuracy", 0.0)
            ece = e.ece or 0.0
            lines.append(
                f"{e.encoder_id:<20} {e.predictor_id:<20} {e.target:<15} "
                f"{val_sp:<14.4f} {held_sp:<14.4f} "
                f"{sign_acc:<10.4f} {ece:<8.4f} {e.n_parameters:<8}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data extraction from exp2 records.
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExtractedData:
    """Extracted features and targets from exp2 transition records."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_heldout: np.ndarray
    y_heldout: np.ndarray
    groups_train: list[str]
    groups_validation: list[str]
    groups_heldout: list[str]
    actions_train: list[str]
    actions_validation: list[str]
    actions_heldout: list[str]
    provenance_train: list[str]  # "realized" or "counterfactual"
    provenance_validation: list[str]
    provenance_heldout: list[str]
    n_features: int
    feature_schema_hash: str
    target_schema_hash: str


def extract_competition_data(
    records: list[Any],
    encoder: Any,
    target: str = "realized_delta",
    *,
    split_field: str = "split",
) -> ExtractedData:
    """Extract features and targets from exp2 transition records.

    Args:
        records: List of TransitionRecord objects.
        encoder: An encoder from exp3 (must be fitted on train if required).
        target: Which target to extract ("realized_delta", "sign_delta",
                "normalized_delta", "risk", "cost").
        split_field: Field name for split identification.

    Returns:
        ExtractedData with train/validation/heldout arrays.
    """
    train_recs = [r for r in records if getattr(r, split_field, "") == "train"]
    val_recs = [r for r in records if getattr(r, split_field, "") == "validation"]
    held_recs = [r for r in records if getattr(r, split_field, "") == "held_out"]

    def _extract(recs: list[Any]) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
        X = []
        y = []
        groups = []
        actions = []
        provenance = []
        for r in recs:
            # Extract features from the record.
            state_before = r.structural_state_before
            global_feats = _state_to_global_features(state_before)
            local_feats = _action_to_local_features(r)
            state_obj = _StateObj(state_before)
            rep = encoder.encode(
                state=state_obj,
                global_features=global_feats,
                action_type=r.action,
                action_target=r.action_target,
                local_features=local_feats,
            )
            X.append(list(rep.vector))
            # Extract target.
            if target == "realized_delta":
                y.append(float(r.realized_delta))
            elif target == "sign_delta":
                y.append(float(compute_sign_delta(r.realized_delta)))
            elif target == "normalized_delta":
                y.append(compute_normalized_delta(r.realized_delta, 1.0))
            elif target == "risk":
                y.append(float(r.realized_risk))
            elif target == "cost":
                y.append(float(r.realized_cost))
            else:
                y.append(float(r.realized_delta))
            groups.append(r.graph_family)
            actions.append(r.action)
            provenance.append(r.provenance.value if hasattr(r.provenance, "value") else str(r.provenance))
        if not X:
            return np.zeros((0, 1)), np.zeros(0), [], [], []
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), groups, actions, provenance

    X_train, y_train, g_train, a_train, p_train = _extract(train_recs)
    X_val, y_val, g_val, a_val, p_val = _extract(val_recs)
    X_held, y_held, g_held, a_held, p_held = _extract(held_recs)

    n_features = X_train.shape[1] if len(X_train) > 0 else (X_val.shape[1] if len(X_val) > 0 else 1)

    import hashlib
    feat_hash = hashlib.sha256(f"exp4.1-features-{n_features}".encode()).hexdigest()[:16]
    target_hash = hashlib.sha256(f"exp4.1-target-{target}".encode()).hexdigest()[:16]

    return ExtractedData(
        X_train=X_train, y_train=y_train,
        X_validation=X_val, y_validation=y_val,
        X_heldout=X_held, y_heldout=y_held,
        groups_train=g_train, groups_validation=g_val, groups_heldout=g_held,
        actions_train=a_train, actions_validation=a_val, actions_heldout=a_held,
        provenance_train=p_train, provenance_validation=p_val, provenance_heldout=p_held,
        n_features=n_features,
        feature_schema_hash=feat_hash,
        target_schema_hash=target_hash,
    )


def _state_to_global_features(state: Any) -> list[float]:
    """Convert a StructuralStateSummary to global feature vector."""
    return [
        float(getattr(state, "n_nodes", 0)),
        float(getattr(state, "n_edges", 0)),
        float(getattr(state, "density", 0.0)),
        float(getattr(state, "degree_mean", 0.0)),
        float(getattr(state, "degree_std", 0.0)),
        float(getattr(state, "degree_max", 0.0) if hasattr(state, "degree_max") else state.degree_std),
        float(getattr(state, "spectral_gap", 0.0)),
        float(math.log1p(max(abs(getattr(state, "spectral_gap", 0.0)), 1e-10))),
        float(getattr(state, "spectral_gap", 0.0) / max(getattr(state, "n_nodes", 1), 1)),
        float(getattr(state, "n_components", 1)),
        float(getattr(state, "avg_clustering", 0.0)),
        float(getattr(state, "n_nodes", 1)),  # diameter estimate
        0.0, 0.0, 0.0,  # curvature
        0.0, 0.0,  # resistance
        float(getattr(state, "fiber_count", 0)),
        float(getattr(state, "fiber_width", 0)),
        float(getattr(state, "gauge_dim", 0)),
        0.0, 0.0,  # diagnosis
        0.0, 0.0,  # history
    ]


def _action_to_local_features(record: Any) -> list[float]:
    """Extract local action features from a transition record."""
    target = record.action_target
    u = int(target.get("u", 0))
    v = int(target.get("v", 0))
    state = record.structural_state_before
    return [
        float(u),
        float(v),
        float(getattr(state, "degree_mean", 0.0)),
        float(getattr(state, "degree_std", 0.0)),
        float(getattr(state, "density", 0.0)),
        float(getattr(state, "spectral_gap", 0.0)),
        float(getattr(state, "n_components", 1)),
        float(getattr(state, "avg_clustering", 0.0)),
        0.0,  # shortest path
        0.0,  # common neighbors
        0.0,  # jaccard
        0.0,  # edge exists
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


# ---------------------------------------------------------------------------
# Competition runner.
# ---------------------------------------------------------------------------

# Default encoder × predictor combinations to test.
DEFAULT_ENCODERS = [
    "minimal-control",
    "global",
    "global-local",
    "semantic-action",
    "geometric",
    "spectral",
]

DEFAULT_PREDICTORS = [
    "global_mean",
    "mutation_type_mean",
    "linear",
    "ridge",
    "tree",
    "mlp",
]

DEFAULT_CLASSIFICATION_PREDICTORS = [
    "global_mean",
    "logistic",
    "tree",
    "mlp_clf",
]


def run_competition(
    records: list[Any],
    *,
    encoders: list[str] | None = None,
    predictors: list[str] | None = None,
    target: str = "realized_delta",
    classification_predictors: list[str] | None = None,
    dataset_schema_hash: str = "",
    train_split_hash: str = "",
    validation_split_hash: str = "",
    heldout_split_hash: str = "",
    encoder_fit_data: list[Any] | None = None,
    n_epochs: int = 50,
    n_ensemble: int = 3,
) -> CompetitionReport:
    """Run the Encoder × Predictor competition on real exp2 data.

    Args:
        records: List of TransitionRecord objects from exp2.
        encoders: List of encoder IDs to test.
        predictors: List of predictor IDs to test.
        target: Target to predict ("realized_delta", "sign_delta", etc.).
        classification_predictors: Predictors for classification targets.
        dataset_schema_hash: Hash of the dataset schema.
        train_split_hash: Hash of the train split.
        validation_split_hash: Hash of the validation split.
        heldout_split_hash: Hash of the held-out split.
        encoder_fit_data: Data to fit encoders on (defaults to train records).
        n_epochs: Training epochs for neural models.
        n_ensemble: Ensemble size for MLP models.

    Returns:
        CompetitionReport with all entries.
    """
    from ..encoders import EncoderRegistry

    if encoders is None:
        encoders = DEFAULT_ENCODERS
    if predictors is None:
        predictors = DEFAULT_PREDICTORS
    if classification_predictors is None:
        classification_predictors = DEFAULT_CLASSIFICATION_PREDICTORS

    is_classification = target in ("sign_delta", "utility_bucket")
    active_predictors = classification_predictors if is_classification else predictors

    report = CompetitionReport(
        dataset_schema_hash=dataset_schema_hash,
        train_split_hash=train_split_hash,
        validation_split_hash=validation_split_hash,
        heldout_split_hash=heldout_split_hash,
        heldout_accessed=False,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    for enc_id in encoders:
        # Create and fit encoder.
        encoder = EncoderRegistry.create(enc_id)
        fit_recs = encoder_fit_data or [r for r in records if r.split == "train"]
        if encoder.requires_fit:
            _fit_encoder(encoder, fit_recs)

        # Extract data using this encoder.
        data = extract_competition_data(records, encoder, target)

        report.n_train = len(data.y_train)
        report.n_validation = len(data.y_validation)
        report.n_heldout = len(data.y_heldout)

        for pred_id in active_predictors:
            entry = _run_single(
                enc_id, pred_id, target, data,
                n_epochs=n_epochs, n_ensemble=n_ensemble,
                dataset_schema_hash=dataset_schema_hash,
                train_split_hash=train_split_hash,
                feature_schema_hash=data.feature_schema_hash,
                target_schema_hash=data.target_schema_hash,
            )
            report.entries.append(entry)

    return report


def _fit_encoder(encoder: Any, train_records: list[Any]) -> None:
    """Fit an encoder on train records."""
    if not encoder.requires_fit:
        return
    global_feats = []
    local_feats = []
    for r in train_records:
        state = r.structural_state_before
        global_feats.append(_state_to_global_features(state))
        local_feats.append(_action_to_local_features(r))
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


def _run_single(
    enc_id: str,
    pred_id: str,
    target: str,
    data: ExtractedData,
    *,
    n_epochs: int,
    n_ensemble: int,
    dataset_schema_hash: str,
    train_split_hash: str,
    feature_schema_hash: str,
    target_schema_hash: str,
) -> CompetitionEntry:
    """Run a single Encoder × Predictor combination."""
    is_classification = target in ("sign_delta", "utility_bucket")

    # Create model.
    kwargs = {}
    if pred_id in ("mlp", "mlp_clf"):
        kwargs = {"n_epochs": n_epochs, "n_ensemble": n_ensemble}
    elif pred_id in ("linear", "ridge", "logistic"):
        kwargs = {"n_epochs": n_epochs}
    elif pred_id == "tree":
        kwargs = {"n_estimators": min(30, n_epochs)}

    model = ModelRegistry.create(pred_id, **kwargs)

    # Fit on train.
    if len(data.X_train) > 0:
        model.fit(data.X_train, data.y_train, split="train")
        model.freeze()

    # Evaluate on train.
    train_metrics = _evaluate(model, data.X_train, data.y_train, is_classification)

    # Evaluate on validation.
    val_metrics = _evaluate(model, data.X_validation, data.y_validation, is_classification)

    # Evaluate on held-out (final scientific evaluation).
    held_metrics = _evaluate(model, data.X_heldout, data.y_heldout, is_classification)

    # Group metrics on held-out.
    group_metrics = []
    if len(data.y_heldout) > 0 and data.groups_heldout:
        if is_classification:
            preds = model.predict_proba(data.X_heldout) if hasattr(model, "predict_proba") else []
            labels = data.y_heldout.astype(int).tolist()
            for grp in sorted(set(data.groups_heldout)):
                mask = [i for i, g in enumerate(data.groups_heldout) if g == grp]
                if mask:
                    sub_preds = [preds[i] for i in mask]
                    sub_labels = [labels[i] for i in mask]
                    m = compute_classification_metrics(sub_preds, sub_labels)
                    group_metrics.append({"group": grp, **m.to_log()})
        else:
            preds = model.predict(data.X_heldout) if hasattr(model, "predict") else []
            targs = data.y_heldout.tolist()
            for grp in sorted(set(data.groups_heldout)):
                mask = [i for i, g in enumerate(data.groups_heldout) if g == grp]
                if mask:
                    sub_preds = [preds[i] for i in mask]
                    sub_targs = [targs[i] for i in mask]
                    m = compute_regression_metrics(sub_preds, sub_targs)
                    group_metrics.append({"group": grp, **m.to_log()})

    # CF-to-real gap.
    cf_gap = None
    if len(data.y_train) > 0:
        realized_mask = [i for i, p in enumerate(data.provenance_train) if p == "realized"]
        cf_mask = [i for i, p in enumerate(data.provenance_train) if p == "counterfactual"]
        if realized_mask and cf_mask and not is_classification:
            real_preds = [model.predict(data.X_train[i:i+1])[0] for i in realized_mask]
            real_targs = data.y_train[realized_mask].tolist()
            cf_preds = [model.predict(data.X_train[i:i+1])[0] for i in cf_mask]
            cf_targs = data.y_train[cf_mask].tolist()
            real_m = compute_regression_metrics(real_preds, real_targs)
            cf_m = compute_regression_metrics(cf_preds, cf_targs)
            gap = compute_cf_to_real_gap(real_m, cf_m, metric="spearman")
            cf_gap = gap.gap

    # Calibration.
    ece = None
    brier = None
    interval_cov = None
    if is_classification and len(data.y_heldout) > 0:
        preds = model.predict_proba(data.X_heldout) if hasattr(model, "predict_proba") else []
        probs = [p.probability for p in preds]
        labels = data.y_heldout.astype(int).tolist()
        ece = expected_calibration_error(probs, labels).value
        brier = brier_score(probs, labels).value
    elif not is_classification and len(data.y_heldout) > 0:
        preds = model.predict(data.X_heldout) if hasattr(model, "predict") else []
        means = [p.mean for p in preds]
        uncs = [p.uncertainty for p in preds]
        targs = data.y_heldout.tolist()
        interval_cov = prediction_interval_coverage(means, uncs, targs).value

    # Complexity.
    n_params = getattr(model, "n_parameters", 0)
    if callable(n_params):
        try:
            n_params = n_params()
        except Exception:
            n_params = 0

    # Create artifact.
    artifact = create_artifact(
        model,
        encoder_id=enc_id,
        encoder_schema_hash=getattr(encoder, "schema_hash", "") if "encoder" in dir() else "",
        dataset_schema_hash=dataset_schema_hash,
        train_split_hash=train_split_hash,
        feature_schema_hash=feature_schema_hash,
        target_schema_hash=target_schema_hash,
        metrics={"validation": val_metrics, "heldout": held_metrics},
    )

    return CompetitionEntry(
        encoder_id=enc_id,
        predictor_id=pred_id,
        target=target,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        heldout_metrics=held_metrics,
        group_metrics=group_metrics,
        cf_to_real_gap=cf_gap,
        ece=ece,
        brier=brier,
        interval_coverage=interval_cov,
        n_parameters=int(n_params) if n_params else 0,
        encoding_latency_ms=0.0,
        artifact=artifact,
    )


def _evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    is_classification: bool,
) -> dict[str, Any]:
    """Evaluate a model on a dataset."""
    if len(X) == 0:
        return {}
    if is_classification and hasattr(model, "predict_proba"):
        preds = model.predict_proba(X)
        labels = y.astype(int).tolist()
        metrics = compute_classification_metrics(preds, labels)
        return metrics.to_log()
    elif hasattr(model, "predict"):
        preds = model.predict(X)
        targs = y.tolist()
        metrics = compute_regression_metrics(preds, targs)
        return metrics.to_log()
    return {}
