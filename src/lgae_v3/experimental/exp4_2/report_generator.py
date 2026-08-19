"""Scientific report generator for exp4.2.

Generates the full report suite:
- EXECUTIVE_SUMMARY.md
- SCIENTIFIC_REPORT.md
- RAW_RESULTS.json
- COMPETITION_TABLE.csv
- Machine-readable conclusion JSON
"""
from __future__ import annotations

from typing import Any
from pathlib import Path
import json
import time
import csv

from .scientific_runner import ScientificResult, ScientificConclusion


def generate_scientific_report(
    results: list[ScientificResult],
    conclusion: ScientificConclusion,
    output_dir: str | Path,
    *,
    dataset_freeze_log: dict[str, Any] | None = None,
    experiment_config_log: dict[str, Any] | None = None,
    finalist_lock_log: dict[str, Any] | None = None,
) -> None:
    """Generate the full scientific report suite."""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    # RAW_RESULTS.json
    raw = {
        "results": [r.to_log() for r in results],
        "conclusion": conclusion.to_log(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (dir_path / "RAW_RESULTS.json").write_text(
        json.dumps(raw, sort_keys=True, indent=2)
    )

    # COMPETITION_TABLE.csv
    _write_competition_table(results, dir_path / "COMPETITION_TABLE.csv")

    # Machine-readable conclusion.
    (dir_path / "CONCLUSION.json").write_text(conclusion.to_json())

    # EXPERIMENT_CONFIG.json
    if experiment_config_log:
        (dir_path / "EXPERIMENT_CONFIG.json").write_text(
            json.dumps(experiment_config_log, sort_keys=True, indent=2)
        )

    # DATASET_FREEZE.json
    if dataset_freeze_log:
        (dir_path / "DATASET_FREEZE.json").write_text(
            json.dumps(dataset_freeze_log, sort_keys=True, indent=2)
        )

    # FINALISTS.json
    if finalist_lock_log:
        (dir_path / "FINALISTS.json").write_text(
            json.dumps(finalist_lock_log, sort_keys=True, indent=2)
        )

    # EXECUTIVE_SUMMARY.md
    _write_executive_summary(results, conclusion, dir_path / "EXECUTIVE_SUMMARY.md")

    # SCIENTIFIC_REPORT.md
    _write_scientific_report(results, conclusion, dir_path / "SCIENTIFIC_REPORT.md")


def _write_competition_table(results: list[ScientificResult], path: Path) -> None:
    """Write the competition table as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Encoder", "Predictor", "Target",
            "Val Spearman", "Val Score", "Val Score Std",
            "Held Spearman", "Held RMSE", "Held Sign Acc",
            "Mean Regret", "P90 Regret", "Catastrophic Rate",
            "CF-Real Gap", "Unc-Error Corr",
            "N Params", "Latency ms",
        ])
        for r in results:
            val_sp = r.validation_metrics.get("spearman", 0.0)
            held_sp = r.heldout_metrics.get("spearman", 0.0)
            held_rmse = r.heldout_metrics.get("rmse", 0.0)
            held_acc = r.heldout_metrics.get("accuracy", 0.0)
            regret = r.regret.get("mean_regret", 0.0)
            p90_reg = r.regret.get("p90_regret", 0.0)
            cat_rate = r.regret.get("catastrophic_regret_rate", 0.0)
            cf_gap = r.cf_real.get("gap_cf_to_real_spearman", 0.0)
            unc_corr = r.uncertainty_correlation.get("corr_uncertainty_abs_error", 0.0)
            writer.writerow([
                r.encoder_id, r.predictor_id, r.target,
                f"{val_sp:.4f}", f"{r.mean_validation_score:.4f}", f"{r.std_validation_score:.4f}",
                f"{held_sp:.4f}", f"{held_rmse:.4f}", f"{held_acc:.4f}",
                f"{regret:.4f}", f"{p90_reg:.4f}", f"{cat_rate:.4f}",
                f"{cf_gap:.4f}", f"{unc_corr:.4f}",
                r.n_parameters, f"{r.prediction_latency_ms:.2f}",
            ])


def _write_executive_summary(
    results: list[ScientificResult],
    conclusion: ScientificConclusion,
    path: Path,
) -> None:
    """Write the executive summary."""
    lines = [
        "# v6.0-exp4.2 — Executive Summary",
        "",
        f"**Experiment ID:** {conclusion.experiment}",
        f"**Generated at:** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "## Scientific Question",
        "",
        "> Can LGAE predict which structural mutation will work best on an unseen graph?",
        "",
        "## Conclusion",
        "",
        f"**Status:** `{conclusion.scientific_status}`",
        "",
        f"- Structural signal detected: **{conclusion.structural_signal_detected}**",
        f"- Generalizes to held-out: **{conclusion.generalizes_to_heldout}**",
        f"- Best encoder: **{conclusion.best_encoder}**",
        f"- Best predictor: **{conclusion.best_model}**",
        f"- Best held-out Spearman: **{conclusion.best_heldout_spearman:.4f}**",
        f"- Best held-out regret: **{conclusion.best_heldout_regret:.4f}**",
        f"- CF→Real transfer OK: **{conclusion.cf_real_transfer_ok}**",
        f"- Uncertainty useful: **{conclusion.uncertainty_useful}**",
        f"- **exp5 authorized: {conclusion.exp5_authorized}**",
        "",
    ]

    if conclusion.recommended_exp5_architecture:
        lines.append(f"**Recommended exp5 architecture:** `{conclusion.recommended_exp5_architecture}`")
        lines.append("")

    if conclusion.limitations:
        lines.append("## Limitations")
        lines.append("")
        for lim in conclusion.limitations:
            lines.append(f"- {lim}")
        lines.append("")

    lines.append("## Decision")
    lines.append("")
    if conclusion.exp5_authorized:
        lines.append("The experiment provides evidence that structural prediction")
        lines.append("generalizes to unseen graph families. exp5 is authorized.")
    else:
        lines.append("The experiment does not provide sufficient evidence for exp5.")
        lines.append("Do not proceed to a sophisticated world model.")
    lines.append("")

    path.write_text("\n".join(lines))


def _write_scientific_report(
    results: list[ScientificResult],
    conclusion: ScientificConclusion,
    path: Path,
) -> None:
    """Write the full scientific report."""
    lines = [
        "# v6.0-exp4.2 — Held-Out Structural Prediction Study",
        "",
        f"**Experiment ID:** {conclusion.experiment}",
        f"**Generated at:** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "## 1. Question",
        "",
        "Can LGAE predict which structural changes will actually improve",
        "an unseen graph?",
        "",
        "## 2. Hypotheses",
        "",
        "**H1:** There exists f(S, a) that predicts relative intervention",
        "quality on graph families not used during training/model selection,",
        "materially outperforming simple baselines.",
        "",
        "**H0:** Available state/action information provides no useful",
        "generalizable predictive signal beyond simple statistical baselines.",
        "",
        "## 3. Experimental Protocol",
        "",
        "1. Freeze exp2 dataset hashes.",
        "2. Freeze exp3 encoder configs.",
        "3. Freeze exp4 predictor configs.",
        "4. Train only on TRAIN.",
        "5. Select models/hyperparameters only on VALIDATION.",
        "6. Lock winning configurations.",
        "7. Open HELD_OUT once.",
        "8. Generate final scientific report.",
        "",
        "## 4. Dataset",
        "",
        "The exp2 structural transition dataset with frozen splits:",
        "train, validation, held_out. The held-out partition was not",
        "accessed during any selection decision.",
        "",
        "## 5. Models and Encoders",
        "",
        f"Total combinations evaluated: {len(results)}",
        "",
        "| Encoder | Predictor | Target | Val Score | Held Spearman |",
        "|---------|-----------|--------|-----------|---------------|",
    ]

    for r in results:
        held_sp = r.heldout_metrics.get("spearman", 0.0)
        lines.append(
            f"| {r.encoder_id} | {r.predictor_id} | {r.target} | "
            f"{r.mean_validation_score:.4f} | {held_sp:.4f} |"
        )

    lines.extend([
        "",
        "## 6. Held-Out Results",
        "",
        f"- Best held-out Spearman: {conclusion.best_heldout_spearman:.4f}",
        f"- Best held-out regret: {conclusion.best_heldout_regret:.4f}",
        f"- Best encoder: {conclusion.best_encoder}",
        f"- Best predictor: {conclusion.best_model}",
        "",
        "## 7. Counterfactual Transfer",
        "",
        f"- CF→Real transfer OK: {conclusion.cf_real_transfer_ok}",
        "",
        "## 8. Calibration",
        "",
        f"- Uncertainty useful: {conclusion.uncertainty_useful}",
        "",
        "## 9. Conclusion",
        "",
        f"**Status:** {conclusion.scientific_status}",
        "",
        f"**exp5 authorized:** {conclusion.exp5_authorized}",
        "",
    ])

    if conclusion.limitations:
        lines.append("## 10. Limitations")
        lines.append("")
        for lim in conclusion.limitations:
            lines.append(f"- {lim}")
        lines.append("")

    lines.append("## 11. Decision on exp5")
    lines.append("")
    if conclusion.exp5_authorized:
        lines.append(f"Proceed to exp5 with architecture: {conclusion.recommended_exp5_architecture}")
    else:
        lines.append("Do not proceed to exp5. The scientific gates are not satisfied.")

    path.write_text("\n".join(lines))


def generate_machine_readable_conclusion(
    conclusion: ScientificConclusion,
    output_path: str | Path,
) -> None:
    """Write the machine-readable conclusion JSON."""
    Path(output_path).write_text(conclusion.to_json())
