"""Experiment runner for v6.0-exp6.1: Real candidate prefilter qualification.

Tests whether 1-5 exact calibration transitions enable a learned filter
to eliminate most candidate evaluations while preserving near-oracle
decisions on unseen topology generators.

Protocol (FROZEN before inspecting TEST-C):
1. Train global prior on training families (graphlet representation)
2. For each test family:
   a. Generate real structural candidates from actual graphs
   b. Evaluate all candidates exactly (oracle)
   c. Calibrate with 1-5 transitions
   d. Score candidates with calibrated model
   e. Sweep pruning ratios (K/N = 50%, 25%, 10%, 5%)
   f. Sweep UCB kappa (0, 0.5, 1, 2)
   g. Compare strategies (random, heuristic, unadapted, adapted, UCB)
3. Report:
   - OracleRecall@K
   - NearOracleRecall@K@ε
   - Regret distribution (mean, median, p90, p95, max, catastrophic rate)
   - Exact evaluations saved
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import time
import numpy as np
import torch

from ..exp5_2.state_encoding import (
    encode_normalized_state, encode_normalized_action,
    NORM_STATE_DIM, NORM_ACTION_DIM,
)
from ..exp5_2.dynamics import DeltaDynamicsModel
from ..exp5_3.representations import (
    REPRESENTATION_LADDER, extract_representation,
    RepresentationConfig,
)
from ..exp5_3.experiment_runner import is_realized, extract_realized_data
from .calibration import fit_calibration, identity_calibration, TopologyCalibration
from .calibration_controller import CalibrationConfig, run_calibration_acquisition
from .candidate_generator import (
    StructuralCandidate, generate_candidates, evaluate_candidates_exact,
    apply_candidate, compute_exact_utility,
)
from .metrics import (
    compute_oracle_recall, compute_near_oracle_recall,
    compute_regret_distribution, compute_pruning_ratio_metrics,
    compare_filtering_strategies, RecallMetrics, RegretDistribution,
)
from .test_c import (
    TestCFamilyConfig, generate_test_c_configs, generate_test_c_graph,
)


@dataclass
class FamilyResult:
    """Result for one family."""
    family: str
    n_candidates: int = 0
    # Calibration.
    calibration_state: str = ""
    sample_efficiency: int = -1
    validation_delta_r2: float = 0.0
    n_calibration: int = 0
    # Pruning ratio sweep.
    pruning_results: dict[str, Any] = field(default_factory=dict)
    # Strategy comparison.
    strategy_comparison: dict[str, Any] = field(default_factory=dict)
    # Regret distributions.
    regret_distributions: dict[str, Any] = field(default_factory=dict)
    # Candidate utility stats.
    oracle_best_utility: float = 0.0
    utility_mean: float = 0.0
    utility_std: float = 0.0
    # Calibration parameters.
    scale: list[float] = field(default_factory=list)
    offset: list[float] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_candidates": int(self.n_candidates),
            "calibration_state": self.calibration_state,
            "sample_efficiency": int(self.sample_efficiency),
            "validation_delta_r2": float(self.validation_delta_r2),
            "n_calibration": int(self.n_calibration),
            "pruning_results": self.pruning_results,
            "strategy_comparison": self.strategy_comparison,
            "regret_distributions": self.regret_distributions,
            "oracle_best_utility": float(self.oracle_best_utility),
            "utility_mean": float(self.utility_mean),
            "utility_std": float(self.utility_std),
            "scale": [float(x) for x in self.scale],
            "offset": [float(x) for x in self.offset],
        }


def run_family_experiment(
    family: str,
    graph_buffers: Any,
    z: torch.Tensor,
    global_model: DeltaDynamicsModel,
    rep_config: RepresentationConfig,
    *,
    n_candidates: int = 50,
    pruning_ratios: list[float] | None = None,
    kappa_values: list[float] | None = None,
    epsilons: list[float] | None = None,
    calibration_config: CalibrationConfig | None = None,
    seed: int = 42,
) -> FamilyResult:
    """Run the full prefilter experiment for one family.

    1. Generate real candidates from the graph
    2. Evaluate all candidates exactly (oracle)
    3. Calibrate with a few transitions
    4. Score candidates with calibrated model
    5. Sweep pruning ratios and kappa values
    6. Compare strategies
    """
    if pruning_ratios is None:
        pruning_ratios = [0.5, 0.25, 0.1, 0.05]
    if kappa_values is None:
        kappa_values = [0.0, 0.5, 1.0, 2.0]
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.5]
    if calibration_config is None:
        calibration_config = CalibrationConfig()

    result = FamilyResult(family=family)

    # --- Step 1: Generate real candidates ---
    candidates = generate_candidates(graph_buffers, n_candidates=n_candidates, seed=seed)
    result.n_candidates = len(candidates)

    if len(candidates) < 5:
        return result

    # --- Step 2: Evaluate all candidates exactly (oracle) ---
    evaluate_candidates_exact(graph_buffers, z, candidates)

    # Extract utility arrays.
    utilities = np.array([c.exact_delta_utility for c in candidates])

    result.oracle_best_utility = float(np.max(utilities))
    result.utility_mean = float(np.mean(utilities))
    result.utility_std = float(np.std(utilities))

    # --- Step 3: Calibrate ---
    # Use the first few candidates as calibration transitions.
    # In a real system, these would be exact exploration transitions.
    n_cal = min(5, len(candidates) // 2)
    cal_candidates = candidates[:n_cal]
    eval_candidates = candidates[n_cal:]

    # Build calibration data from exact evaluations.
    # The "state" is the graphlet encoding of the graph before the action.
    # The "delta" is the exact utility change.
    from ..exp5_2.state_encoding import encode_normalized_state
    from ..dataset_generator import DatasetGenerator

    # For calibration, we need (z_t, a_t, delta_z) triples.
    # Use the graphlet features of the current graph as z_t,
    # and the action encoding as a_t.
    # The "delta" is the change in graphlet features after applying the candidate.
    state_before = _extract_state_from_graph(graph_buffers)
    sv_before = encode_normalized_state(state_before)
    rep_z_before = extract_representation(sv_before.vector[np.newaxis, :], rep_config)[0]

    cal_z = []
    cal_a = []
    cal_deltas = []

    for cand in cal_candidates:
        # Apply candidate to get new graph.
        new_graph = apply_candidate(graph_buffers, cand)
        state_after = _extract_state_from_graph(new_graph)
        sv_after = encode_normalized_state(state_after)
        rep_z_after = extract_representation(sv_after.vector[np.newaxis, :], rep_config)[0]

        # Action encoding (simplified).
        n_nodes = int(graph_buffers.num_nodes)
        degree_mean = float(np.mean([len(_get_neighbors(graph_buffers, i)) for i in range(n_nodes)]))
        av = encode_normalized_action(
            cand.action_type, {"u": cand.u, "v": cand.v},
            n_nodes=n_nodes, degree_mean=degree_mean,
        )

        cal_z.append(rep_z_before)
        cal_a.append(av.vector)
        cal_deltas.append(rep_z_after - rep_z_before)

    cal_z = np.array(cal_z)
    cal_a = np.array(cal_a)
    cal_deltas = np.array(cal_deltas)

    # Get base predictions for calibration.
    base_preds_cal = global_model.predict_raw_batch(cal_z, cal_a)

    # Fit calibration.
    cal = fit_calibration(
        base_preds_cal, cal_deltas,
        topology_signature=family,
        regularization=1.0,
    )

    result.calibration_state = "calibrated" if cal.validation_delta_r2 > 0 else "limited"
    result.validation_delta_r2 = cal.validation_delta_r2
    result.n_calibration = n_cal
    result.scale = list(cal.scale)
    result.offset = list(cal.offset)

    # --- Step 4: Score all candidates with calibrated model ---
    # For each candidate, compute the predicted delta.
    all_z = []
    all_a = []
    for cand in candidates:
        all_z.append(rep_z_before)  # same initial state for all
        n_nodes = int(graph_buffers.num_nodes)
        degree_mean = float(np.mean([len(_get_neighbors(graph_buffers, i)) for i in range(n_nodes)]))
        av = encode_normalized_action(
            cand.action_type, {"u": cand.u, "v": cand.v},
            n_nodes=n_nodes, degree_mean=degree_mean,
        )
        all_a.append(av.vector)

    all_z = np.array(all_z)
    all_a = np.array(all_a)

    # Base predictions.
    base_preds = global_model.predict_raw_batch(all_z, all_a)
    # Apply calibration.
    adapted_preds = cal.apply_batch(base_preds)

    # Predicted utility = mean of predicted delta.
    learned_scores = np.mean(adapted_preds, axis=1)
    unadapted_scores = np.mean(base_preds, axis=1)

    # Uncertainty (from ensemble if available, else use prediction variance).
    uncertainties = np.zeros(len(candidates))
    if hasattr(global_model, 'predict_uncertainty_batch'):
        uncertainties = global_model.predict_uncertainty_batch(all_z, all_a)
    else:
        # Use prediction magnitude as a proxy for uncertainty.
        uncertainties = np.std(adapted_preds, axis=1)

    # --- Step 5: Pruning ratio sweep ---
    result.pruning_results = compute_pruning_ratio_metrics(
        utilities, learned_scores,
        pruning_ratios, epsilons,
    )

    # --- Step 6: Strategy comparison ---
    # Use K = 25% of candidates for strategy comparison.
    k_strategy = max(1, len(candidates) // 4)
    result.strategy_comparison = compare_filtering_strategies(
        utilities,
        learned_scores=learned_scores,
        learned_uncertainties=uncertainties,
        k=k_strategy,
        kappa_values=kappa_values,
        seed=seed,
    )

    # Also compare unadapted vs adapted.
    result.strategy_comparison["unadapted_vs_adapted"] = {
        "unadapted_scores": _evaluate_top_k(utilities, unadapted_scores, k_strategy),
        "adapted_scores": _evaluate_top_k(utilities, learned_scores, k_strategy),
    }

    # --- Regret distributions ---
    # For each strategy, compute regret distribution across multiple decision states.
    # Here we have one decision state per family, so regret is per-candidate.
    oracle_best = float(np.max(utilities))

    for strategy_name, strat_data in result.strategy_comparison.items():
        if isinstance(strat_data, dict) and "regret" in strat_data:
            result.regret_distributions[strategy_name] = {
                "regret": float(strat_data["regret"]),
                "oracle_recall": float(strat_data.get("oracle_recall", 0)),
            }

    return result


def _evaluate_top_k(utilities: np.ndarray, scores: np.ndarray, k: int) -> dict[str, Any]:
    """Evaluate top-K selection."""
    n = len(utilities)
    sorted_idx = np.argsort(-scores)
    top_k = sorted_idx[:k]
    oracle_best = float(np.max(utilities))
    oracle_best_idx = int(np.argmax(utilities))
    best_retained = float(np.max(utilities[top_k])) if len(top_k) > 0 else 0.0
    return {
        "oracle_recall": 1.0 if oracle_best_idx in set(top_k.tolist()) else 0.0,
        "best_retained_utility": best_retained,
        "regret": float(oracle_best - best_retained),
        "k": k,
        "n": n,
    }


def _extract_state_from_graph(graph: Any) -> Any:
    """Extract a StructuralStateSummary from a GraphBuffers."""
    from ..dataset_generator import DatasetGenerator
    from ...config import LGAEConfig
    # Create a temporary generator just for state extraction.
    config = LGAEConfig()
    gen = DatasetGenerator.__new__(DatasetGenerator)
    gen.config = config
    return gen._extract_state_summary_from_graph(graph)


def _get_neighbors(graph: Any, node: int) -> set[int]:
    """Get neighbors of a node in the graph."""
    neighbors = set()
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s == node:
                neighbors.add(d)
            if d == node:
                neighbors.add(s)
    return neighbors
