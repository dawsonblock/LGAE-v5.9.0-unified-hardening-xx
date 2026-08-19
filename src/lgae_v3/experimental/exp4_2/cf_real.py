"""Counterfactual-to-real transfer experiment.

Runs three supervision regimes:
    R:  realized-only training
    CF: counterfactual-only training
    MIX: realized + counterfactual

All three are evaluated on held-out REALIZED outcomes.

This is one of the most consequential exp4.2 results:
- If CF ≈ Real, cheap shadow data can train useful predictors.
- If CF >> poor real transfer, exp5 needs simulator-gap correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


class SupervisionRegime:
    """Supervision regime identifiers."""
    REALIZED_ONLY = "realized_only"
    COUNTERFACTUAL_ONLY = "counterfactual_only"
    MIXED = "mixed"


@dataclass(slots=True)
class RegimeResult:
    """Result of one supervision regime."""
    regime: str
    train_metrics: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    heldout_metrics: dict[str, Any] = field(default_factory=dict)
    n_train: int = 0

    def to_log(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "train_metrics": dict(self.train_metrics),
            "validation_metrics": dict(self.validation_metrics),
            "heldout_metrics": dict(self.heldout_metrics),
            "n_train": int(self.n_train),
        }


@dataclass(slots=True)
class CFRealTransferReport:
    """Full CF-to-real transfer report."""
    results: list[RegimeResult] = field(default_factory=list)
    gap_cf_to_real_spearman: float = 0.0
    gap_cf_to_real_regret: float = 0.0
    gap_cf_to_real_sign_accuracy: float = 0.0
    gap_cf_to_real_rmse: float = 0.0
    gap_mix_to_real_spearman: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "results": [r.to_log() for r in self.results],
            "gap_cf_to_real_spearman": float(self.gap_cf_to_real_spearman),
            "gap_cf_to_real_regret": float(self.gap_cf_to_real_regret),
            "gap_cf_to_real_sign_accuracy": float(self.gap_cf_to_real_sign_accuracy),
            "gap_cf_to_real_rmse": float(self.gap_cf_to_real_rmse),
            "gap_mix_to_real_spearman": float(self.gap_mix_to_real_spearman),
        }


def run_cf_real_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    provenance_train: list[str],
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_heldout: np.ndarray,
    y_heldout: np.ndarray,
    *,
    model_factory: Any,
    is_classification: bool = False,
) -> CFRealTransferReport:
    """Run the three-regime CF-to-real experiment.

    Args:
        X_train: Training features.
        y_train: Training targets.
        provenance_train: Per-sample provenance ("realized" or "counterfactual").
        X_validation: Validation features.
        y_validation: Validation targets.
        X_heldout: Held-out features.
        y_heldout: Held-out targets.
        model_factory: Callable that creates a fresh model each call.
        is_classification: Whether this is a classification task.

    Returns:
        CFRealTransferReport with all three regimes and gaps.
    """
    from ..models.evaluator import (
        compute_regression_metrics, compute_classification_metrics,
    )

    prov = [str(p).lower() for p in provenance_train]
    realized_mask = np.array([p == "realized" for p in prov])
    cf_mask = np.array([p == "counterfactual" for p in prov])

    regimes = [
        (SupervisionRegime.REALIZED_ONLY, realized_mask),
        (SupervisionRegime.COUNTERFACTUAL_ONLY, cf_mask),
        (SupervisionRegime.MIXED, np.ones(len(prov), dtype=bool)),
    ]

    results = []
    for regime_name, mask in regimes:
        X_sub = X_train[mask]
        y_sub = y_train[mask]

        if len(X_sub) == 0:
            results.append(RegimeResult(regime=regime_name, n_train=0))
            continue

        model = model_factory()
        model.fit(X_sub, y_sub, split="train")
        if hasattr(model, "freeze"):
            model.freeze()

        # Evaluate on all splits.
        def _eval(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
            if len(X) == 0:
                return {}
            if is_classification and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)
                m = compute_classification_metrics(preds, y.astype(int).tolist())
                return m.to_log()
            elif hasattr(model, "predict"):
                preds = model.predict(X)
                m = compute_regression_metrics(preds, y.tolist())
                return m.to_log()
            return {}

        train_m = _eval(X_sub, y_sub)
        val_m = _eval(X_validation, y_validation)
        held_m = _eval(X_heldout, y_heldout)

        results.append(RegimeResult(
            regime=regime_name,
            train_metrics=train_m,
            validation_metrics=val_m,
            heldout_metrics=held_m,
            n_train=len(X_sub),
        ))

    # Compute gaps.
    report = CFRealTransferReport(results=results)
    if len(results) >= 2:
        real_held = results[0].heldout_metrics
        cf_held = results[1].heldout_metrics
        mix_held = results[2].heldout_metrics if len(results) > 2 else {}

        real_sp = real_held.get("spearman", 0.0)
        cf_sp = cf_held.get("spearman", 0.0)
        mix_sp = mix_held.get("spearman", 0.0)

        report.gap_cf_to_real_spearman = cf_sp - real_sp
        report.gap_mix_to_real_spearman = mix_sp - real_sp

        # For regret, we'd need candidate-set structure; use RMSE as proxy.
        real_rmse = real_held.get("rmse", 0.0)
        cf_rmse = cf_held.get("rmse", 0.0)
        report.gap_cf_to_real_rmse = cf_rmse - real_rmse

        real_acc = real_held.get("accuracy", 0.0)
        cf_acc = cf_held.get("accuracy", 0.0)
        report.gap_cf_to_real_sign_accuracy = cf_acc - real_acc

    return report
