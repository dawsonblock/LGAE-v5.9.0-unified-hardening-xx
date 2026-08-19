"""Structural feature extractor for exp6.4.

These features are legitimate online-observable properties of S_t.
They do NOT include exact future utility, future continuation labels,
or any information that would only be available after exact enumeration.

Features include:
- n_components (current, observable)
- component size histogram
- largest/smallest component fraction
- source/target component ID and size for each candidate
- same-component flag
- number of components a candidate merges
- bridge-distance features
- components remaining to threshold
- minimum theoretical steps to threshold
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers


@dataclass
class ComponentInfo:
    """Structural information about graph components."""
    n_components: int = 0
    component_sizes: list[int] = field(default_factory=list)
    component_ids: np.ndarray = None  # node -> component_id mapping
    largest_component_size: int = 0
    largest_component_fraction: float = 0.0
    second_largest_fraction: float = 0.0
    smallest_component_size: int = 0


def compute_component_info(graph: GraphBuffers, n: int) -> ComponentInfo:
    """Compute component information using union-find."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n and d < n:
                union(s, d)

    # Count components and sizes.
    comp_ids = np.array([find(i) for i in range(n)])
    unique_ids, counts = np.unique(comp_ids, return_counts=True)
    sizes = sorted(counts.tolist(), reverse=True)

    info = ComponentInfo(
        n_components=len(unique_ids),
        component_sizes=sizes,
        component_ids=comp_ids,
        largest_component_size=sizes[0] if sizes else 0,
        largest_component_fraction=sizes[0] / n if sizes else 0.0,
        second_largest_fraction=sizes[1] / n if len(sizes) > 1 else 0.0,
        smallest_component_size=sizes[-1] if sizes else 0,
    )
    return info


def extract_structural_features(
    graph: GraphBuffers,
    z: torch.Tensor,
    action: tuple[str, int, int, dict] | None = None,
    *,
    threshold: int = 1,
    horizon: int = 2,
    n_max_components: int = 20,
) -> np.ndarray:
    """Extract legitimate online-observable structural features.

    If action is provided, includes action-specific features
    (source/target component, merge flag, etc.).
    If action is None, returns state-only features.
    """
    n = int(graph.num_nodes)
    comp_info = compute_component_info(graph, n)

    # --- State features (always available) ---
    n_comp = comp_info.n_components
    components_remaining = max(0, n_comp - threshold)

    # Component size distribution features.
    sizes = comp_info.component_sizes
    largest_frac = comp_info.largest_component_fraction
    second_frac = comp_info.second_largest_fraction
    smallest_frac = comp_info.smallest_component_size / n if n > 0 else 0.0
    size_imbalance = (sizes[0] - sizes[-1]) / n if len(sizes) > 1 else 0.0

    # Degree statistics.
    degrees = np.zeros(n)
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n: degrees[s] += 1
            if d < n: degrees[d] += 1

    n_edges = int(valid.sum().item())
    density = n_edges / max(n * (n - 1) / 2, 1)
    n_isolated = int(np.sum(degrees == 0))

    # Additive utility (exact, cheap).
    with torch.no_grad():
        src = graph.src[graph.valid]
        dst = graph.dst[graph.valid]
        w = graph.weight[graph.valid]
        if src.numel() > 0:
            d = (z[src] - z[dst]).pow(2).sum(-1)
            u_add = float(-(w * d).sum().item())
        else:
            u_add = 0.0

    # Component size histogram (padded to n_max_components).
    hist = np.zeros(n_max_components)
    for i, s in enumerate(sizes[:n_max_components]):
        hist[i] = s / n

    # Minimum theoretical steps to threshold.
    min_steps_to_threshold = max(0, n_comp - threshold)

    state_features = np.array([
        n / 50.0,                          # normalized graph size
        n_comp / 10.0,                     # normalized component count
        components_remaining / 10.0,       # components remaining to threshold
        min_steps_to_threshold / 5.0,      # min steps to threshold
        density,                           # edge density
        float(np.mean(degrees)) / 10.0,    # mean degree
        float(np.std(degrees)) / 10.0,     # degree std
        float(np.max(degrees)) / 20.0,     # max degree
        n_isolated / max(n, 1),            # isolated fraction
        u_add / 100.0,                     # additive utility
        largest_frac,                      # largest component fraction
        second_frac,                       # second largest fraction
        smallest_frac,                     # smallest component fraction
        size_imbalance,                    # component size imbalance
    ])

    # Pad state features to consistent size.
    state_features = np.concatenate([state_features, hist])

    if action is None:
        return state_features

    # --- Action-specific features ---
    mt, u, v, params = action

    # Source and target component IDs.
    if u < n and v < n:
        src_comp = int(comp_info.component_ids[u])
        dst_comp = int(comp_info.component_ids[v])
        same_component = 1.0 if src_comp == dst_comp else 0.0
    else:
        src_comp = -1
        dst_comp = -1
        same_component = 0.0

    # Component sizes for source and target.
    src_comp_size = int(np.sum(comp_info.component_ids == src_comp)) if src_comp >= 0 else 0
    dst_comp_size = int(np.sum(comp_info.component_ids == dst_comp)) if dst_comp >= 0 else 0

    # Does this action merge components?
    if mt in ("add_edge", "bridge", "local_rewire", "hub_connect"):
        merges_components = 1.0 if (not same_component and src_comp >= 0 and dst_comp >= 0) else 0.0
    elif mt == "remove_edge":
        # Removing an edge might split a component.
        merges_components = -1.0  # different semantics
    else:
        merges_components = 0.0

    # Latent distance for this specific action.
    if u < n and v < n:
        with torch.no_grad():
            d_sq = float((z[u] - z[v]).pow(2).sum().item())
    else:
        d_sq = 0.0

    # Weight parameter.
    weight = params.get("weight", 1.0)

    # Action type one-hot.
    action_type_features = np.zeros(4)
    if mt in ("add_edge", "bridge", "local_rewire", "hub_connect"):
        action_type_features[0] = 1.0
    elif mt == "remove_edge":
        action_type_features[1] = 1.0
    elif mt in ("reweight_up", "reweight_down"):
        action_type_features[2] = 1.0
    else:
        action_type_features[3] = 1.0

    # After this action, how many components remain?
    if merges_components > 0:
        n_comp_after = n_comp - 1
    elif merges_components < 0:
        n_comp_after = n_comp + 1  # might split
    else:
        n_comp_after = n_comp

    components_remaining_after = max(0, n_comp_after - threshold)

    # Would this action reach the threshold?
    reaches_threshold = 1.0 if n_comp_after <= threshold else 0.0

    # Steps to threshold after this action.
    min_steps_after = max(0, n_comp_after - threshold)

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
    ])

    return np.concatenate([state_features, action_type_features, action_features])
