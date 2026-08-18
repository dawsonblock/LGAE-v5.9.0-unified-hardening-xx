"""Evaluator: held-out metrics, group-level, OOD degradation, CF-to-real gap.

Prioritized metrics (by how MPC will use them):
1. Candidate rank correlation (Spearman, Kendall tau)
2. Sign accuracy
3. Top-1/top-k action agreement
4. Calibration
5. OOD robustness
6. Absolute RMSE

Also computes group-level metrics by:
- graph family
- mutation type
- graph size bucket
- OOD status
- diagnosis type
- risk bucket
- candidate-set size
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np

from .protocol import Prediction, ClassificationPrediction


@dataclass(slots=True)
class RegressionMetrics:
    """Standard regression metrics."""
    rmse: float
    mae: float
    r2: float
    spearman: float
    kendall_tau: float
    n_samples: int

    def to_log(self) -> dict[str, Any]:
        return {
            "rmse": float(self.rmse),
            "mae": float(self.mae),
            "r2": float(self.r2),
            "spearman": float(self.spearman),
            "kendall_tau": float(self.kendall_tau),
            "n_samples": int(self.n_samples),
        }


@dataclass(slots=True)
class ClassificationMetrics:
    """Classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    n_samples: int

    def to_log(self) -> dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "auc_roc": float(self.auc_roc),
            "n_samples": int(self.n_samples),
        }


@dataclass(slots=True)
class RankingMetrics:
    """Ranking quality metrics."""
    ndcg_at_k: float
    mrr: float
    top1_agreement: float
    top3_recall: float
    pairwise_accuracy: float
    n_groups: int

    def to_log(self) -> dict[str, Any]:
        return {
            "ndcg_at_k": float(self.ndcg_at_k),
            "mrr": float(self.mrr),
            "top1_agreement": float(self.top1_agreement),
            "top3_recall": float(self.top3_recall),
            "pairwise_accuracy": float(self.pairwise_accuracy),
            "n_groups": int(self.n_groups),
        }


@dataclass(slots=True)
class GroupMetrics:
    """Metrics broken down by a grouping variable."""
    group_name: str
    group_values: list[str]
    metrics: list[dict[str, Any]] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "group_values": list(self.group_values),
            "metrics": list(self.metrics),
        }


@dataclass(slots=True)
class CFToRealGap:
    """Counterfactual-to-realized transfer gap."""
    metric: str
    realized_value: float
    counterfactual_value: float
    gap: float

    def to_log(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "realized_value": float(self.realized_value),
            "counterfactual_value": float(self.counterfactual_value),
            "gap": float(self.gap),
        }


# ---------------------------------------------------------------------------
# Metric computations.
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-10 or ry.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    total = n * (n - 1) / 2
    if total == 0:
        return 0.0
    return float((concordant - discordant) / total)


def compute_regression_metrics(
    predictions: list[Prediction],
    targets: list[float],
) -> RegressionMetrics:
    """Compute regression metrics from predictions."""
    n = len(predictions)
    if n == 0:
        return RegressionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    means = np.array([p.mean for p in predictions])
    targs = np.array(targets)
    residuals = means - targs
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((targs - targs.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10) if ss_tot > 1e-10 else 0.0
    spearman = _spearman(means, targs)
    kendall = _kendall_tau(means, targs)
    return RegressionMetrics(rmse, mae, r2, spearman, kendall, n)


def compute_classification_metrics(
    predictions: list[ClassificationPrediction],
    labels: list[int],
) -> ClassificationMetrics:
    """Compute classification metrics."""
    n = len(predictions)
    if n == 0:
        return ClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    probs = np.array([p.probability for p in predictions])
    preds = np.array([p.predicted_class for p in predictions])
    labs = np.array(labels)
    accuracy = float(np.mean(preds == labs))
    # Precision, recall, F1 for positive class.
    tp = int(np.sum((preds == 1) & (labs == 1)))
    fp = int(np.sum((preds == 1) & (labs == 0)))
    fn = int(np.sum((preds == 0) & (labs == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    # AUC-ROC (simplified).
    auc = 0.5  # default
    if len(set(labs)) > 1:
        # Simple AUC: rank-based.
        order = np.argsort(-probs)
        ranked_labs = labs[order]
        n_pos = int(np.sum(labs == 1))
        n_neg = int(np.sum(labs == 0))
        if n_pos > 0 and n_neg > 0:
            tp_count = 0
            auc_sum = 0.0
            for i in range(n):
                if ranked_labs[i] == 1:
                    tp_count += 1
                else:
                    auc_sum += tp_count
            auc = float(auc_sum) / (n_pos * n_neg)
    return ClassificationMetrics(accuracy, precision, recall, f1, auc, n)


def compute_ranking_metrics(
    predicted_ranks: list[int],
    true_ranks: list[int],
    k: int = 3,
) -> RankingMetrics:
    """Compute ranking quality metrics.

    Args:
        predicted_ranks: Predicted rank of each candidate (0 = best).
        true_ranks: True rank of each candidate (0 = best).
        k: K for NDCG@K and top-K recall.
    """
    n = len(predicted_ranks)
    if n == 0:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    pred = np.array(predicted_ranks)
    true = np.array(true_ranks)
    # Top-1 agreement.
    pred_top1 = int(np.argmin(pred))
    true_top1 = int(np.argmin(true))
    top1 = 1.0 if pred_top1 == true_top1 else 0.0
    # Top-3 recall.
    pred_top_k = set(np.argsort(pred)[:k])
    true_top_k = set(np.argsort(true)[:k])
    top3_recall = len(pred_top_k & true_top_k) / max(len(true_top_k), 1)
    # MRR.
    mrr = 1.0 / (true[pred_top1] + 1) if n > 0 else 0.0
    # NDCG@K.
    dcg = sum(1.0 / (true[i] + 1) for i in np.argsort(pred)[:k])
    idcg = sum(1.0 / (i + 1) for i in range(min(k, n)))
    ndcg = dcg / max(idcg, 1e-10)
    # Pairwise accuracy.
    n_pairs = 0
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            pred_order = pred[i] < pred[j]
            true_order = true[i] < true[j]
            if pred_order == true_order:
                correct += 1
    pairwise_acc = correct / max(n_pairs, 1)
    return RankingMetrics(
        ndcg_at_k=float(ndcg),
        mrr=float(mrr),
        top1_agreement=float(top1),
        top3_recall=float(top3_recall),
        pairwise_accuracy=float(pairwise_acc),
        n_groups=1,
    )


def compute_group_metrics(
    predictions: list[Prediction],
    targets: list[float],
    groups: list[str],
    group_name: str,
) -> GroupMetrics:
    """Compute regression metrics broken down by a grouping variable."""
    group_set = sorted(set(groups))
    group_metrics = GroupMetrics(group_name=group_name, group_values=group_set)
    for g in group_set:
        mask = [i for i, grp in enumerate(groups) if grp == g]
        if not mask:
            continue
        preds = [predictions[i] for i in mask]
        targs = [targets[i] for i in mask]
        metrics = compute_regression_metrics(preds, targs)
        group_metrics.metrics.append({
            "group": g,
            **metrics.to_log(),
        })
    return group_metrics


def compute_cf_to_real_gap(
    realized_metrics: RegressionMetrics,
    counterfactual_metrics: RegressionMetrics,
    metric: str = "spearman",
) -> CFToRealGap:
    """Compute counterfactual-to-realized transfer gap.

    Gap_{CF→Real} = metric_realized - metric_counterfactual

    If mixed supervision wins, that strongly supports the counterfactual
    data pipeline. If counterfactual-only performs poorly on realized
    outcomes, there is a simulator-gap problem.
    """
    val_real = getattr(realized_metrics, metric)
    val_cf = getattr(counterfactual_metrics, metric)
    return CFToRealGap(
        metric=metric,
        realized_value=float(val_real),
        counterfactual_value=float(val_cf),
        gap=float(val_real) - float(val_cf),
    )


def compute_ood_degradation(
    in_distribution_metrics: RegressionMetrics,
    ood_metrics: RegressionMetrics,
    metric: str = "spearman",
) -> float:
    """Compute OOD degradation.

    A large degradation means the model cannot be trusted OOD.
    """
    val_id = getattr(in_distribution_metrics, metric)
    val_ood = getattr(ood_metrics, metric)
    return float(val_id) - float(val_ood)
