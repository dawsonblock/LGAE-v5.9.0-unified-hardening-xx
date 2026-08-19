"""Mechanism-agnostic observable features for exp6.5.

These features do NOT include the mechanism label. The model must
infer which delayed-value mechanism is active from observables:

- component structure
- edge redundancy
- hub load
- spectral gap
- community separation
- path bottlenecks
- candidate effects
- distance to structural thresholds

This is the key for cross-mechanism generalization: the model sees
the same features regardless of which mechanism generated the task.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers
from ..exp6_4.structural_features import compute_component_info


OBSERVABLE_FEATURE_DIM = 64  # Fixed dimension for all features


def _compute_spectral_gap(graph: GraphBuffers, n: int) -> float:
    """Compute spectral gap (largest - second largest eigenvalue)."""
    adj = np.zeros((n, n))
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n and d < n:
                w = float(graph.weight[i].item())
                adj[s, d] = w
                adj[d, s] = w
    try:
        eigenvalues = np.linalg.eigvalsh(adj)
        sorted_eg = np.sort(eigenvalues)
        return float(sorted_eg[-1] - sorted_eg[-2]) if n > 1 else 0.0
    except Exception:
        return 0.0


def _compute_degree_stats(graph: GraphBuffers, n: int) -> tuple[np.ndarray, float, float, float, float]:
    """Returns (degrees, mean, std, max, min)."""
    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1
    return (degrees,
            float(np.mean(degrees)),
            float(np.std(degrees)),
            float(np.max(degrees)) if n > 0 else 0.0,
            float(np.min(degrees)) if n > 0 else 0.0)


def _compute_redundancy(graph: GraphBuffers, n: int) -> float:
    """Edge redundancy: fraction of node pairs with multiple paths."""
    # Proxy: average degree / max possible degree.
    degrees, mean_d, _, max_d, min_d = _compute_degree_stats(graph, n)
    if n <= 1:
        return 0.0
    return mean_d / (n - 1)


def extract_observable_features(
    graph: GraphBuffers,
    z: torch.Tensor,
    action: tuple[str, int, int, dict] | None = None,
    *,
    threshold: int = 1,
    horizon: int = 2,
) -> np.ndarray:
    """Extract mechanism-agnostic observable features.

    If action is provided, includes action-specific features.
    Returns a fixed-dim vector (OBSERVABLE_FEATURE_DIM).
    """
    n = int(graph.num_nodes)
    comp_info = compute_component_info(graph, n)

    # --- Component structure ---
    n_comp = comp_info.n_components
    components_remaining = max(0, n_comp - threshold)
    largest_frac = comp_info.largest_component_fraction
    second_frac = comp_info.second_largest_fraction
    smallest_frac = comp_info.smallest_component_size / n if n > 0 else 0.0
    sizes = comp_info.component_sizes
    size_imbalance = (sizes[0] - sizes[-1]) / n if len(sizes) > 1 else 0.0

    # --- Degree statistics ---
    degrees, mean_deg, std_deg, max_deg, min_deg = _compute_degree_stats(graph, n)
    n_isolated = int(np.sum(degrees == 0))
    n_edges = int(graph.valid.bool().sum().item())
    density = n_edges / max(n * (n - 1) / 2, 1)

    # --- Redundancy ---
    redundancy = _compute_redundancy(graph, n)

    # --- Hub load ---
    hub_load = max_deg / max(n - 1, 1)
    n_hubs = int(np.sum(degrees > mean_deg + 2 * std_deg)) if std_deg > 0 else 0

    # --- Spectral gap ---
    spectral_gap = _compute_spectral_gap(graph, n)
    spectral_gap_norm = spectral_gap / max(n, 1)

    # --- Additive utility ---
    with torch.no_grad():
        src = graph.src[graph.valid]
        dst = graph.dst[graph.valid]
        w = graph.weight[graph.valid]
        if src.numel() > 0:
            d = (z[src] - z[dst]).pow(2).sum(-1)
            u_add = float(-(w * d).sum().item())
        else:
            u_add = 0.0

    # --- Component size histogram (padded) ---
    hist = np.zeros(20)
    for i, s in enumerate(sizes[:20]):
        hist[i] = s / n

    # --- State features ---
    state_features = np.array([
        n / 50.0,
        n_comp / 10.0,
        components_remaining / 10.0,
        max(0, n_comp - threshold) / 5.0,  # min steps to threshold
        density,
        mean_deg / 10.0,
        std_deg / 10.0,
        max_deg / 20.0,
        min_deg / 5.0,
        n_isolated / max(n, 1),
        u_add / 100.0,
        largest_frac,
        second_frac,
        smallest_frac,
        size_imbalance,
        redundancy,
        hub_load,
        n_hubs / max(n, 1),
        spectral_gap_norm,
        spectral_gap / 10.0,
    ])

    # Combine state + histogram.
    all_state = np.concatenate([state_features, hist])
    # Pad/truncate to fixed dim.
    if len(all_state) < 40:
        all_state = np.pad(all_state, (0, 40 - len(all_state)))
    else:
        all_state = all_state[:40]

    if action is None:
        # Pad to OBSERVABLE_FEATURE_DIM.
        result = np.pad(all_state, (0, OBSERVABLE_FEATURE_DIM - len(all_state)))
        return result

    # --- Action-specific features ---
    mt, u, v, params = action

    # Component info for action.
    if u < n and v < n:
        src_comp = int(comp_info.component_ids[u])
        dst_comp = int(comp_info.component_ids[v])
        same_component = 1.0 if src_comp == dst_comp else 0.0
    else:
        src_comp = -1
        dst_comp = -1
        same_component = 0.0

    src_comp_size = int(np.sum(comp_info.component_ids == src_comp)) if src_comp >= 0 else 0
    dst_comp_size = int(np.sum(comp_info.component_ids == dst_comp)) if dst_comp >= 0 else 0

    # Does this action merge components?
    if mt in ("add_edge", "bridge", "local_rewire", "hub_connect"):
        merges_components = 1.0 if (not same_component and src_comp >= 0 and dst_comp >= 0) else 0.0
    elif mt == "remove_edge":
        merges_components = -1.0
    else:
        merges_components = 0.0

    # Latent distance.
    if u < n and v < n:
        with torch.no_grad():
            d_sq = float((z[u] - z[v]).pow(2).sum().item())
    else:
        d_sq = 0.0

    weight = params.get("weight", 1.0)

    # Action type one-hot.
    action_type = np.zeros(4)
    if mt in ("add_edge", "bridge", "local_rewire", "hub_connect"):
        action_type[0] = 1.0
    elif mt == "remove_edge":
        action_type[1] = 1.0
    elif mt in ("reweight_up", "reweight_down"):
        action_type[2] = 1.0
    else:
        action_type[3] = 1.0

    # After-action predictions (observable, not exact).
    if merges_components > 0:
        n_comp_after = n_comp - 1
    elif merges_components < 0:
        n_comp_after = n_comp + 1
    else:
        n_comp_after = n_comp

    components_remaining_after = max(0, n_comp_after - threshold)
    reaches_threshold = 1.0 if n_comp_after <= threshold else 0.0
    min_steps_after = max(0, n_comp_after - threshold)

    # Degree after action (observable proxy).
    if mt in ("add_edge", "bridge", "hub_connect") and u < n and v < n:
        deg_u_after = degrees[u] + 1
        deg_v_after = degrees[v] + 1
        max_deg_after = max(max_deg, deg_u_after, deg_v_after)
        min_deg_after = min(min_deg, deg_u_after, deg_v_after)
    elif mt == "remove_edge" and u < n and v < n:
        deg_u_after = max(0, degrees[u] - 1)
        deg_v_after = max(0, degrees[v] - 1)
        max_deg_after = max_deg  # conservative
        min_deg_after = min(min_deg, deg_u_after, deg_v_after)
    else:
        max_deg_after = max_deg
        min_deg_after = min_deg

    hub_load_after = max_deg_after / max(n - 1, 1)
    redundancy_after = (mean_deg + (1.0 if mt in ("add_edge", "bridge") else -1.0) * 2 / n) / max(n - 1, 1)

    action_features = np.array([
        same_component,
        merges_components,
        src_comp_size / n,
        dst_comp_size / n,
        d_sq / 10.0,
        weight,
        n_comp_after / 10.0,
        components_remaining_after / 10.0,
        reaches_threshold,
        min_steps_after / 5.0,
        max_deg_after / 20.0,
        min_deg_after / 5.0,
        hub_load_after,
        redundancy_after,
        deg_u_after / 20.0 if u < n else 0.0,
        deg_v_after / 20.0 if v < n else 0.0,
    ])

    all_action = np.concatenate([action_type, action_features])
    # Pad/truncate action features to fixed dim.
    if len(all_action) < 24:
        all_action = np.pad(all_action, (0, 24 - len(all_action)))
    else:
        all_action = all_action[:24]

    result = np.concatenate([all_state, all_action])
    # Pad to OBSERVABLE_FEATURE_DIM.
    if len(result) < OBSERVABLE_FEATURE_DIM:
        result = np.pad(result, (0, OBSERVABLE_FEATURE_DIM - len(result)))
    else:
        result = result[:OBSERVABLE_FEATURE_DIM]

    return result
