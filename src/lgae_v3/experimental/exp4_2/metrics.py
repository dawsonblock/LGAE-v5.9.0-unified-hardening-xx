"""Scientific metrics for exp4.2.

These metrics go beyond standard RMSE/accuracy to measure:
- Top-action regret (operationally meaningful)
- Oracle recovery (how much of available improvement is captured)
- Selective prediction (abstention quality)
- Pareto frontier (quality vs cost)
- Bootstrap confidence intervals
- Uncertainty-error correlation (trust signal quality)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np


@dataclass(slots=True)
class RegretReport:
    """Top-action regret metrics.

    Regret = ΔU(a*) - ΔU(a_model)

    where a* is the best candidate and a_model is the model-selected one.
    """
    mean_regret: float = 0.0
    median_regret: float = 0.0
    p75_regret: float = 0.0
    p90_regret: float = 0.0
    p95_regret: float = 0.0
    max_regret: float = 0.0
    catastrophic_regret_rate: float = 0.0
    n_states: int = 0
    catastrophic_threshold: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "mean_regret": float(self.mean_regret),
            "median_regret": float(self.median_regret),
            "p75_regret": float(self.p75_regret),
            "p90_regret": float(self.p90_regret),
            "p95_regret": float(self.p95_regret),
            "max_regret": float(self.max_regret),
            "catastrophic_regret_rate": float(self.catastrophic_regret_rate),
            "n_states": int(self.n_states),
            "catastrophic_threshold": float(self.catastrophic_threshold),
        }


def compute_regret(
    predicted_utilities: list[list[float]],
    true_utilities: list[list[float]],
    *,
    catastrophic_threshold: float = 0.1,
) -> RegretReport:
    """Compute top-action regret.

    Args:
        predicted_utilities: For each state, predicted utility for each candidate.
        true_utilities: For each state, true utility for each candidate.
        catastrophic_threshold: Regret above this is "catastrophic".
            Must be defined BEFORE held-out evaluation.

    Returns:
        RegretReport with mean, median, percentiles, and catastrophic rate.
    """
    if not predicted_utilities:
        return RegretReport(catastrophic_threshold=catastrophic_threshold)

    regrets = []
    for preds, trues in zip(predicted_utilities, true_utilities):
        if len(preds) == 0 or len(trues) == 0:
            continue
        # Model selects argmax of predictions.
        model_idx = int(np.argmax(preds))
        # Oracle is argmax of true utilities.
        oracle_idx = int(np.argmax(trues))
        regret = float(trues[oracle_idx]) - float(trues[model_idx])
        regrets.append(regret)

    if not regrets:
        return RegretReport(catastrophic_threshold=catastrophic_threshold)

    arr = np.array(regrets)
    n_cat = int(np.sum(arr > catastrophic_threshold))
    return RegretReport(
        mean_regret=float(np.mean(arr)),
        median_regret=float(np.median(arr)),
        p75_regret=float(np.percentile(arr, 75)),
        p90_regret=float(np.percentile(arr, 90)),
        p95_regret=float(np.percentile(arr, 95)),
        max_regret=float(np.max(arr)),
        catastrophic_regret_rate=n_cat / len(regrets),
        n_states=len(regrets),
        catastrophic_threshold=catastrophic_threshold,
    )


@dataclass(slots=True)
class OracleRecoveryReport:
    """Oracle recovery metrics.

    OracleRecovery = (U(a_model) - U(a_baseline)) / (U(a_oracle) - U(a_baseline) + ε)

    Measures how much of the theoretically available improvement the
    learned model recovers relative to a baseline.
    """
    mean_oracle_recovery: float = 0.0
    median_oracle_recovery: float = 0.0
    n_states: int = 0
    epsilon: float = 1e-8

    def to_log(self) -> dict[str, Any]:
        return {
            "mean_oracle_recovery": float(self.mean_oracle_recovery),
            "median_oracle_recovery": float(self.median_oracle_recovery),
            "n_states": int(self.n_states),
            "epsilon": float(self.epsilon),
        }


def compute_oracle_recovery(
    predicted_utilities: list[list[float]],
    true_utilities: list[list[float]],
    baseline_utilities: list[float],
    *,
    epsilon: float = 1e-8,
) -> OracleRecoveryReport:
    """Compute oracle recovery.

    Args:
        predicted_utilities: For each state, predicted utility per candidate.
        true_utilities: For each state, true utility per candidate.
        baseline_utilities: For each state, the baseline-selected utility.
        epsilon: Small constant to avoid division by zero.

    Returns:
        OracleRecoveryReport.
    """
    if not predicted_utilities:
        return OracleRecoveryReport(epsilon=epsilon)

    recoveries = []
    for preds, trues, baseline_u in zip(predicted_utilities, true_utilities, baseline_utilities):
        if len(preds) == 0 or len(trues) == 0:
            continue
        model_idx = int(np.argmax(preds))
        oracle_idx = int(np.argmax(trues))
        model_u = float(trues[model_idx])
        oracle_u = float(trues[oracle_idx])
        baseline = float(baseline_u)
        denom = (oracle_u - baseline) + epsilon
        recovery = (model_u - baseline) / denom
        recoveries.append(recovery)

    if not recoveries:
        return OracleRecoveryReport(epsilon=epsilon)

    arr = np.array(recoveries)
    return OracleRecoveryReport(
        mean_oracle_recovery=float(np.mean(arr)),
        median_oracle_recovery=float(np.median(arr)),
        n_states=len(recoveries),
        epsilon=epsilon,
    )


@dataclass(slots=True)
class SelectivePredictionReport:
    """Selective prediction metrics.

    Evaluates performance when the model is allowed to abstain on
    uncertain cases. Retains only the top X% most-confident predictions.
    """
    coverage_levels: list[float] = field(default_factory=list)
    ranking_spearman: list[float] = field(default_factory=list)
    regret: list[float] = field(default_factory=list)
    error: list[float] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "coverage_levels": list(self.coverage_levels),
            "ranking_spearman": list(self.ranking_spearman),
            "regret": list(self.regret),
            "error": list(self.error),
        }


def compute_selective_prediction(
    predicted_utilities: list[list[float]],
    true_utilities: list[list[float]],
    confidences: list[float],
    *,
    coverage_levels: list[float] | None = None,
    catastrophic_threshold: float = 0.1,
) -> SelectivePredictionReport:
    """Compute selective prediction metrics.

    Args:
        predicted_utilities: Per-state predicted utilities per candidate.
        true_utilities: Per-state true utilities per candidate.
        confidences: Per-state confidence (higher = more confident).
        coverage_levels: Fractions of most-confident states to retain.
        catastrophic_threshold: For regret computation.

    Returns:
        SelectivePredictionReport with metrics at each coverage level.
    """
    if coverage_levels is None:
        coverage_levels = [1.0, 0.9, 0.75, 0.5, 0.25]

    if not predicted_utilities:
        return SelectivePredictionReport(
            coverage_levels=list(coverage_levels),
        )

    # Sort states by confidence (descending).
    n = len(predicted_utilities)
    sorted_indices = np.argsort(confidences)[::-1]  # descending

    report = SelectivePredictionReport(coverage_levels=list(coverage_levels))
    for cov in coverage_levels:
        k = max(1, int(math.ceil(cov * n)))
        selected = sorted_indices[:k]

        sel_preds = [predicted_utilities[i] for i in selected]
        sel_trues = [true_utilities[i] for i in selected]

        # Ranking spearman (macro average over states).
        spearmans = []
        for preds, trues in zip(sel_preds, sel_trues):
            if len(preds) > 1:
                from ..models.evaluator import _spearman
                sp = _spearman(np.array(preds), np.array(trues))
                spearmans.append(sp)
        avg_sp = float(np.mean(spearmans)) if spearmans else 0.0

        # Regret.
        reg = compute_regret(sel_preds, sel_trues, catastrophic_threshold=catastrophic_threshold)

        # Error (mean absolute prediction error).
        errors = []
        for preds, trues in zip(sel_preds, sel_trues):
            for p, t in zip(preds, trues):
                errors.append(abs(float(p) - float(t)))
        avg_err = float(np.mean(errors)) if errors else 0.0

        report.ranking_spearman.append(avg_sp)
        report.regret.append(reg.mean_regret)
        report.error.append(avg_err)

    return report


@dataclass(slots=True)
class ParetoFrontierEntry:
    """One entry on the Pareto frontier."""
    encoder_id: str
    predictor_id: str
    target: str
    quality: float  # primary quality metric (e.g., spearman)
    latency_ms: float
    n_parameters: int
    memory_bytes: float = 0.0
    efficiency: float = 0.0  # quality gain per unit latency
    is_pareto_optimal: bool = False

    def to_log(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "predictor_id": self.predictor_id,
            "target": self.target,
            "quality": float(self.quality),
            "latency_ms": float(self.latency_ms),
            "n_parameters": int(self.n_parameters),
            "memory_bytes": float(self.memory_bytes),
            "efficiency": float(self.efficiency),
            "is_pareto_optimal": bool(self.is_pareto_optimal),
        }


@dataclass
class ParetoFrontier:
    """Pareto frontier over (quality, latency, parameters)."""
    entries: list[ParetoFrontierEntry] = field(default_factory=list)
    baseline_quality: float = 0.0

    def to_log(self) -> dict[str, Any]:
        return {
            "baseline_quality": float(self.baseline_quality),
            "entries": [e.to_log() for e in self.entries],
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_log(), sort_keys=True, indent=2)


def compute_pareto_frontier(
    entries: list[ParetoFrontierEntry],
    *,
    baseline_quality: float = 0.0,
) -> ParetoFrontier:
    """Compute the Pareto frontier.

    An entry is Pareto-optimal if no other entry is better or equal in
    all dimensions (quality, latency, parameters) and strictly better
    in at least one.

    Args:
        entries: All competition entries with quality/latency/params.
        baseline_quality: The baseline quality for efficiency computation.

    Returns:
        ParetoFrontier with is_pareto_optimal flags set.
    """
    # Compute efficiency for each entry.
    for e in entries:
        gain = e.quality - baseline_quality
        e.efficiency = gain / (e.latency_ms + 1e-6)

    # Find Pareto-optimal entries.
    # Maximize quality, minimize latency, minimize parameters.
    for i, ei in enumerate(entries):
        is_dominated = False
        for j, ej in enumerate(entries):
            if i == j:
                continue
            # ej dominates ei if ej is >= in quality and <= in latency
            # and <= in parameters, with at least one strict inequality.
            if (ej.quality >= ei.quality and
                ej.latency_ms <= ei.latency_ms and
                ej.n_parameters <= ei.n_parameters and
                (ej.quality > ei.quality or
                 ej.latency_ms < ei.latency_ms or
                 ej.n_parameters < ei.n_parameters)):
                is_dominated = True
                break
        ei.is_pareto_optimal = not is_dominated

    return ParetoFrontier(entries=entries, baseline_quality=baseline_quality)


def bootstrap_ci(
    values: list[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval.

    Args:
        values: Sample values.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    n = len(arr)
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1.0 - alpha) * 100))
    return lower, upper


@dataclass(slots=True)
class UncertaintyCorrelationReport:
    """Correlation between uncertainty and actual error.

    A useful uncertainty model should produce higher uncertainty when
    prediction error is higher. If this correlation is near zero or
    negative, uncertainty is not useful for trust control.
    """
    corr_uncertainty_abs_error: float = 0.0
    corr_uncertainty_regret: float = 0.0
    n_samples: int = 0

    def to_log(self) -> dict[str, Any]:
        return {
            "corr_uncertainty_abs_error": float(self.corr_uncertainty_abs_error),
            "corr_uncertainty_regret": float(self.corr_uncertainty_regret),
            "n_samples": int(self.n_samples),
        }


def compute_uncertainty_error_correlation(
    uncertainties: list[float],
    absolute_errors: list[float],
    regrets: list[float] | None = None,
) -> UncertaintyCorrelationReport:
    """Compute correlation between uncertainty and error.

    Args:
        uncertainties: Per-prediction uncertainty.
        absolute_errors: Per-prediction absolute error.
        regrets: Optional per-state regret for uncertainty-regret correlation.

    Returns:
        UncertaintyCorrelationReport.
    """
    if not uncertainties or not absolute_errors:
        return UncertaintyCorrelationReport()

    unc = np.array(uncertainties)
    err = np.array(absolute_errors)

    # Pearson correlation.
    if np.std(unc) > 1e-10 and np.std(err) > 1e-10:
        corr_err = float(np.corrcoef(unc, err)[0, 1])
    else:
        corr_err = 0.0

    corr_reg = 0.0
    if regrets and len(regrets) == len(uncertainties):
        reg = np.array(regrets)
        if np.std(reg) > 1e-10:
            corr_reg = float(np.corrcoef(unc, reg)[0, 1])

    return UncertaintyCorrelationReport(
        corr_uncertainty_abs_error=corr_err,
        corr_uncertainty_regret=corr_reg,
        n_samples=len(uncertainties),
    )
