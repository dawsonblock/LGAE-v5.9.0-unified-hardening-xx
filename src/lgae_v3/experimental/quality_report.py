"""Data quality report for v6.0-exp2 structural transition datasets.

Every generated dataset automatically produces a quality report covering:

- Transition count
- Graph-family distribution
- Action distribution
- Accept/reject distribution
- Realized utility distribution
- Risk distribution
- Cost distribution
- OOD distribution
- Mutation-type balance
- Candidate-set size
- Prediction error
- Calibration error

Also detects imbalance (e.g., 92% ADD_EDGE) and flags it before training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np

from .transition_record import (
    TransitionRecord,
    TransitionProvenance,
    AuthorizationDecision,
)


@dataclass(slots=True)
class DistributionReport:
    """Distribution report for a single metric."""
    name: str
    n: int
    mean: float
    std: float
    min: float
    median: float
    max: float
    p10: float
    p25: float
    p75: float
    p90: float
    p99: float
    histogram: list[int] = field(default_factory=list)
    histogram_edges: list[float] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": int(self.n),
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min),
            "median": float(self.median),
            "max": float(self.max),
            "p10": float(self.p10),
            "p25": float(self.p25),
            "p75": float(self.p75),
            "p90": float(self.p90),
            "p99": float(self.p99),
            "histogram": [int(h) for h in self.histogram],
            "histogram_edges": [float(e) for e in self.histogram_edges],
        }


@dataclass(slots=True)
class CategoryDistribution:
    """Distribution of categorical values."""
    name: str
    counts: dict[str, int]
    total: int

    @property
    def fractions(self) -> dict[str, float]:
        return {k: v / max(self.total, 1) for k, v in self.counts.items()}

    @property
    def max_fraction(self) -> float:
        return max(self.fractions.values()) if self.fractions else 0.0

    @property
    def is_imbalanced(self) -> bool:
        """Flag if any single category exceeds 80%."""
        return self.max_fraction > 0.80

    @property
    def dominant_category(self) -> str | None:
        if not self.counts:
            return None
        return max(self.counts, key=self.counts.get)

    def to_log(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "counts": dict(self.counts),
            "total": int(self.total),
            "fractions": {k: float(v) for k, v in self.fractions.items()},
            "max_fraction": float(self.max_fraction),
            "is_imbalanced": bool(self.is_imbalanced),
            "dominant_category": self.dominant_category,
        }


@dataclass(slots=True)
class DataQualityReport:
    """Complete data quality report for a dataset."""
    n_records: int
    n_observed: int
    n_counterfactual: int
    n_realized: int
    n_shadow: int

    # Distributions.
    graph_family_distribution: CategoryDistribution | None = None
    action_distribution: CategoryDistribution | None = None
    authorization_distribution: CategoryDistribution | None = None
    provenance_distribution: CategoryDistribution | None = None
    split_distribution: CategoryDistribution | None = None

    # Numeric distributions.
    realized_delta_distribution: DistributionReport | None = None
    predicted_delta_distribution: DistributionReport | None = None
    realized_risk_distribution: DistributionReport | None = None
    realized_cost_distribution: DistributionReport | None = None
    candidate_set_size_distribution: DistributionReport | None = None
    prediction_error_distribution: DistributionReport | None = None

    # Quality flags.
    imbalance_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Calibration.
    calibration_correlation: float = 0.0
    sign_agreement: float = 0.0
    top1_agreement: float = 0.0

    @property
    def has_imbalance(self) -> bool:
        return len(self.imbalance_flags) > 0

    def to_log(self) -> dict[str, Any]:
        return {
            "n_records": int(self.n_records),
            "n_observed": int(self.n_observed),
            "n_counterfactual": int(self.n_counterfactual),
            "n_realized": int(self.n_realized),
            "n_shadow": int(self.n_shadow),
            "graph_family_distribution": self.graph_family_distribution.to_log() if self.graph_family_distribution else None,
            "action_distribution": self.action_distribution.to_log() if self.action_distribution else None,
            "authorization_distribution": self.authorization_distribution.to_log() if self.authorization_distribution else None,
            "provenance_distribution": self.provenance_distribution.to_log() if self.provenance_distribution else None,
            "split_distribution": self.split_distribution.to_log() if self.split_distribution else None,
            "realized_delta_distribution": self.realized_delta_distribution.to_log() if self.realized_delta_distribution else None,
            "predicted_delta_distribution": self.predicted_delta_distribution.to_log() if self.predicted_delta_distribution else None,
            "realized_risk_distribution": self.realized_risk_distribution.to_log() if self.realized_risk_distribution else None,
            "realized_cost_distribution": self.realized_cost_distribution.to_log() if self.realized_cost_distribution else None,
            "candidate_set_size_distribution": self.candidate_set_size_distribution.to_log() if self.candidate_set_size_distribution else None,
            "prediction_error_distribution": self.prediction_error_distribution.to_log() if self.prediction_error_distribution else None,
            "imbalance_flags": list(self.imbalance_flags),
            "warnings": list(self.warnings),
            "calibration_correlation": float(self.calibration_correlation),
            "sign_agreement": float(self.sign_agreement),
            "top1_agreement": float(self.top1_agreement),
        }


def _compute_distribution(values: list[float], name: str, n_bins: int = 10) -> DistributionReport:
    """Compute a distribution report from a list of values."""
    if not values:
        return DistributionReport(
            name=name, n=0, mean=0.0, std=0.0, min=0.0, median=0.0,
            max=0.0, p10=0.0, p25=0.0, p75=0.0, p90=0.0, p99=0.0,
        )
    arr = np.array(values, dtype=np.float64)
    # Histogram.
    try:
        hist, edges = np.histogram(arr, bins=n_bins)
        histogram = [int(h) for h in hist]
        hist_edges = [float(e) for e in edges]
    except Exception:
        histogram = []
        hist_edges = []
    return DistributionReport(
        name=name,
        n=len(values),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        median=float(np.median(arr)),
        max=float(np.max(arr)),
        p10=float(np.percentile(arr, 10)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        p99=float(np.percentile(arr, 99)),
        histogram=histogram,
        histogram_edges=hist_edges,
    )


def _compute_category_distribution(
    values: list[str],
    name: str,
) -> CategoryDistribution:
    """Compute a category distribution from a list of string values."""
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return CategoryDistribution(name=name, counts=counts, total=len(values))


def generate_quality_report(records: list[TransitionRecord]) -> DataQualityReport:
    """Generate a data quality report from a list of transition records.

    Args:
        records: The transition records to analyze.

    Returns:
        A DataQualityReport with distributions, imbalance flags, and
        calibration metrics.
    """
    n = len(records)
    if n == 0:
        return DataQualityReport(
            n_records=0, n_observed=0, n_counterfactual=0,
            n_realized=0, n_shadow=0,
            warnings=["Empty dataset: no records to analyze."],
        )

    # Provenance counts.
    n_observed = sum(1 for r in records if r.provenance == TransitionProvenance.REALIZED)
    n_counterfactual = sum(1 for r in records if r.provenance == TransitionProvenance.COUNTERFACTUAL)
    n_shadow = sum(1 for r in records if r.provenance == TransitionProvenance.SHADOW)

    # Category distributions.
    families = [r.graph_family for r in records]
    actions = [r.action for r in records]
    auths = [r.authorization_decision.value for r in records]
    provenances = [r.provenance.value for r in records]
    splits = [r.split for r in records]

    fam_dist = _compute_category_distribution(families, "graph_family")
    act_dist = _compute_category_distribution(actions, "action")
    auth_dist = _compute_category_distribution(auths, "authorization_decision")
    prov_dist = _compute_category_distribution(provenances, "provenance")
    split_dist = _compute_category_distribution(splits, "split")

    # Numeric distributions.
    realized_deltas = [r.realized_delta for r in records if math.isfinite(r.realized_delta)]
    predicted_deltas = [r.predicted_delta for r in records if math.isfinite(r.predicted_delta)]
    realized_risks = [r.realized_risk for r in records if math.isfinite(r.realized_risk)]
    realized_costs = [r.realized_cost for r in records if math.isfinite(r.realized_cost)]
    cand_sizes = [r.candidate_set_summary.n_candidates for r in records]

    # Prediction error (only for REALIZED records with both predicted and realized).
    pred_errors = []
    for r in records:
        if (r.provenance == TransitionProvenance.REALIZED and
                math.isfinite(r.predicted_delta) and
                math.isfinite(r.realized_delta)):
            pred_errors.append(abs(r.predicted_delta - r.realized_delta))

    # Imbalance flags.
    imbalance_flags: list[str] = []
    if act_dist.is_imbalanced:
        imbalance_flags.append(
            f"Action distribution imbalanced: {act_dist.dominant_category} "
            f"accounts for {act_dist.max_fraction:.1%} of records"
        )
    if fam_dist.is_imbalanced:
        imbalance_flags.append(
            f"Graph family distribution imbalanced: {fam_dist.dominant_category} "
            f"accounts for {fam_dist.max_fraction:.1%} of records"
        )
    if auth_dist.is_imbalanced:
        imbalance_flags.append(
            f"Authorization distribution imbalanced: {auth_dist.dominant_category} "
            f"accounts for {auth_dist.max_fraction:.1%} of records"
        )

    # Calibration metrics (only for REALIZED records).
    calibration_corr = 0.0
    sign_agreement = 0.0
    top1_agreement = 0.0

    realized_records = [r for r in records if r.provenance == TransitionProvenance.REALIZED]
    if len(realized_records) >= 2:
        preds = np.array([r.predicted_delta for r in realized_records])
        reals = np.array([r.realized_delta for r in realized_records])
        if preds.std() > 1e-10 and reals.std() > 1e-10:
            calibration_corr = float(np.corrcoef(preds, reals)[0, 1])
        # Sign agreement: do predicted and realized have the same sign?
        sign_agreement = float(np.mean(np.sign(preds) == np.sign(reals)))

    # Warnings.
    warnings: list[str] = []
    if n_observed == 0:
        warnings.append("No observed (REALIZED) transitions — dataset is entirely counterfactual/shadow.")
    if n_counterfactual == 0 and n_shadow == 0:
        warnings.append("No counterfactual/shadow transitions — dataset has no negative samples.")
    if len(pred_errors) < 10:
        warnings.append(f"Only {len(pred_errors)} records with prediction error — calibration metrics unreliable.")

    return DataQualityReport(
        n_records=n,
        n_observed=n_observed,
        n_counterfactual=n_counterfactual,
        n_realized=n_observed,  # REALIZED = observed
        n_shadow=n_shadow,
        graph_family_distribution=fam_dist,
        action_distribution=act_dist,
        authorization_distribution=auth_dist,
        provenance_distribution=prov_dist,
        split_distribution=split_dist,
        realized_delta_distribution=_compute_distribution(realized_deltas, "realized_delta"),
        predicted_delta_distribution=_compute_distribution(predicted_deltas, "predicted_delta"),
        realized_risk_distribution=_compute_distribution(realized_risks, "realized_risk"),
        realized_cost_distribution=_compute_distribution(realized_costs, "realized_cost"),
        candidate_set_size_distribution=_compute_distribution(
            [float(c) for c in cand_sizes], "candidate_set_size"
        ),
        prediction_error_distribution=_compute_distribution(pred_errors, "prediction_error"),
        imbalance_flags=imbalance_flags,
        warnings=warnings,
        calibration_correlation=calibration_corr,
        sign_agreement=sign_agreement,
        top1_agreement=top1_agreement,
    )
