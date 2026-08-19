"""Non-greedy delayed-value benchmark tasks for exp6.3.

These tasks use NON-ADDITIVE utility functions where greedy one-step
optimization is provably suboptimal. The key insight from the first
exp6.3 attempt was that the additive utility U = -sum(w*d^2) makes
greedy always optimal because each edge contributes independently.

Non-additive utilities create genuine delayed-value structure:
- Connectivity bonus: bridging components gives a global bonus
- Spectral penalty: global structure affects spectral radius
- Diameter penalty: shortcuts reduce global diameter

Each task guarantees:
    argmax_a ΔU_immediate ≠ argmax_a Q_H*
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import torch

from ...types import GraphBuffers, make_graph_buffers


# ---------------------------------------------------------------------------
# Non-additive utility functions
# ---------------------------------------------------------------------------

def _build_adjacency(graph: GraphBuffers, n: int) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n and d < n:
                adj[s].add(d)
                adj[d].add(s)
    return adj


def _count_components(graph: GraphBuffers, n: int) -> int:
    adj = _build_adjacency(graph, n)
    visited = set()
    n_comp = 0
    for start in range(n):
        if start in visited:
            continue
        n_comp += 1
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return n_comp


def utility_connectivity_bonus(graph: GraphBuffers, z: torch.Tensor,
                               lambda_conn: float = 10.0) -> float:
    """Non-additive: U = -sum(w*d^2) + λ*(n - n_components).

    The connectivity bonus is NON-ADDITIVE: adding a bridge between
    two components gives +λ, while adding within a component gives 0.
    This creates delayed-value structure where bridging (negative
    additive term) can be optimal (large connectivity bonus).
    """
    with torch.no_grad():
        src = graph.src[graph.valid]
        dst = graph.dst[graph.valid]
        w = graph.weight[graph.valid]
        if src.numel() == 0:
            u_add = 0.0
        else:
            d = (z[src] - z[dst]).pow(2).sum(-1)
            u_add = float(-(w * d).sum().item())
    n = int(graph.num_nodes)
    n_comp = _count_components(graph, n)
    return u_add + lambda_conn * (n - n_comp)


def utility_threshold_connectivity(graph: GraphBuffers, z: torch.Tensor,
                                    lambda_conn: float = 15.0,
                                    threshold: int = 1) -> float:
    """Non-additive with THRESHOLD: bonus only when n_components <= threshold.

    U = -sum(w*d^2) + λ * max(0, threshold + 1 - n_components)

    This creates GENUINE delayed value:
    - Starting with 3 components, threshold=1:
      Step 1: bridge 3→2, NO bonus (2 > threshold=1)
      Step 2: bridge 2→1, +λ bonus (1 <= threshold=1)
    - Greedy sees only step 1's delta (negative additive, no bonus)
    - MPC sees that two bridges yield +λ total

    This is the key utility for non-greedy benchmarks.
    """
    with torch.no_grad():
        src = graph.src[graph.valid]
        dst = graph.dst[graph.valid]
        w = graph.weight[graph.valid]
        if src.numel() == 0:
            u_add = 0.0
        else:
            d = (z[src] - z[dst]).pow(2).sum(-1)
            u_add = float(-(w * d).sum().item())
    n = int(graph.num_nodes)
    n_comp = _count_components(graph, n)
    bonus = max(0, threshold + 1 - n_comp)
    return u_add + lambda_conn * bonus


def utility_spectral_penalty(graph: GraphBuffers, z: torch.Tensor,
                             lambda_spec: float = 3.0) -> float:
    """Non-additive: U = -sum(w*d^2) - λ*spectral_radius."""
    with torch.no_grad():
        src = graph.src[graph.valid]
        dst = graph.dst[graph.valid]
        w = graph.weight[graph.valid]
        if src.numel() == 0:
            u_add = 0.0
        else:
            d = (z[src] - z[dst]).pow(2).sum(-1)
            u_add = float(-(w * d).sum().item())
    n = int(graph.num_nodes)
    adj = np.zeros((n, n))
    valid = graph.valid.bool()
    for i in range(graph.src.shape[0]):
        if valid[i]:
            s = int(graph.src[i].item())
            d = int(graph.dst[i].item())
            if s < n and d < n:
                wt = float(graph.weight[i].item())
                adj[s, d] = wt
                adj[d, s] = wt
    try:
        eigenvalues = np.linalg.eigvalsh(adj)
        spec_rad = float(np.max(np.abs(eigenvalues)))
    except Exception:
        spec_rad = 0.0
    return u_add - lambda_spec * spec_rad


UTILITY_FUNCTIONS: dict[str, Callable] = {
    "connectivity_bonus": utility_connectivity_bonus,
    "threshold_connectivity": utility_threshold_connectivity,
    "spectral_penalty": utility_spectral_penalty,
}


# ---------------------------------------------------------------------------
# Delayed-value tasks
# ---------------------------------------------------------------------------

@dataclass
class DelayedValueTask:
    """A multi-step task where greedy is suboptimal."""
    name: str
    description: str
    n_nodes: int
    initial_edges: list[tuple[int, int]]
    utility_name: str
    utility_params: dict[str, Any] = field(default_factory=dict)
    latent_dim: int = 4
    latent_seed: int = 42
    available_actions: list[tuple[str, int, int, dict]] = field(default_factory=list)
    optimal_first_action: tuple[str, int, int] = ("", 0, 0)
    greedy_failure_reason: str = ""
    cluster_assignments: list[int] | None = None
    cluster_spacing: float = 5.0

    @property
    def utility_fn(self) -> Callable:
        fn = UTILITY_FUNCTIONS[self.utility_name]
        params = self.utility_params
        return lambda g, z: fn(g, z, **params)


def make_clustered_latent_state(
    n: int, dim: int, seed: int,
    cluster_assignments: list[int] | None = None,
    cluster_spacing: float = 5.0,
) -> torch.Tensor:
    """Create latent states where clusters are far apart in latent space.

    This ensures cross-cluster edges have large ||z_u - z_v||^2
    (negative additive delta), while within-cluster edges have small
    ||z_u - z_v||^2 (less negative additive delta).

    This is essential for creating delayed-value structure:
    - Greedy picks within-cluster edges (small negative additive)
    - MPC picks cross-cluster bridges (large negative additive but threshold bonus)
    """
    rng = torch.Generator().manual_seed(seed)
    z = torch.randn(n, dim, generator=rng) * 0.3

    if cluster_assignments is not None:
        n_clusters = max(cluster_assignments) + 1
        # Assign each cluster a center far from others.
        centers = torch.randn(n_clusters, dim, generator=rng) * cluster_spacing
        for node_idx, cluster_idx in enumerate(cluster_assignments):
            z[node_idx] += centers[cluster_idx]

    return z


def make_latent_state(n: int, dim: int, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=rng) * 0.5


def make_task_graph(task: DelayedValueTask) -> GraphBuffers:
    return make_graph_buffers(
        num_nodes=task.n_nodes,
        edges=task.initial_edges,
        capacity=max(len(task.initial_edges) * 3, task.n_nodes * 3),
    )


def make_task_latent(task: DelayedValueTask) -> torch.Tensor:
    if task.cluster_assignments:
        return make_clustered_latent_state(
            task.n_nodes, task.latent_dim, task.latent_seed,
            cluster_assignments=task.cluster_assignments,
            cluster_spacing=task.cluster_spacing,
        )
    return make_latent_state(task.n_nodes, task.latent_dim, task.latent_seed)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def task_delayed_bridge() -> DelayedValueTask:
    """Three clusters, threshold=1: first bridge gives no bonus.

    With threshold connectivity (threshold=1), the bonus only kicks in
    when the graph becomes fully connected (n_comp <= 1).

    Starting with 3 components:
    - Step 1: bridge 3→2, NO bonus (2 > 1)
    - Step 2: bridge 2→1, +λ bonus (1 <= 1)

    Greedy picks within-cluster edges (less negative additive delta).
    MPC picks bridges because two bridges yield +λ.
    """
    n = 12
    edges = [
        (0, 1), (1, 2), (2, 3),         # cluster 1 (4 nodes)
        (4, 5), (5, 6), (6, 7),         # cluster 2 (4 nodes)
        (8, 9), (9, 10), (10, 11),      # cluster 3 (4 nodes)
    ]
    return DelayedValueTask(
        name="delayed_bridge",
        description="Three clusters, threshold=1: first bridge gives no bonus, second bridge gives +λ. Greedy avoids bridges.",
        n_nodes=n,
        initial_edges=edges,
        utility_name="threshold_connectivity",
        utility_params={"lambda_conn": 30.0, "threshold": 1},
        latent_dim=4,
        latent_seed=100,
        available_actions=[
            # Cross-cluster bridges (optimal for MPC but negative immediate)
            ("add_edge", 3, 4, {"weight": 1.0}),   # bridge 1-2
            ("add_edge", 7, 8, {"weight": 1.0}),   # bridge 2-3
            ("add_edge", 3, 8, {"weight": 1.0}),   # bridge 1-3
            # Within-cluster (greedy: less negative additive)
            ("add_edge", 0, 2, {"weight": 1.0}),   # cluster 1 shortcut
            ("add_edge", 4, 6, {"weight": 1.0}),   # cluster 2 shortcut
            ("add_edge", 8, 10, {"weight": 1.0}),  # cluster 3 shortcut
        ],
        optimal_first_action=("add_edge", 3, 4),
        greedy_failure_reason="Greedy picks within-cluster shortcuts (less negative additive delta). MPC picks bridges because two bridges yield +λ bonus that outweighs the additive cost.",
        cluster_assignments=[0,0,0,0, 1,1,1,1, 2,2,2,2],
        cluster_spacing=1.0,
    )


def task_staged_community_bridge() -> DelayedValueTask:
    """Three communities, threshold=1: need 2 bridges for bonus.

    Starting with 3 components, threshold=1:
    - Step 1: bridge 3→2, NO bonus (2 > 1)
    - Step 2: bridge 2→1, +λ bonus

    Greedy avoids all bridges (large negative additive).
    MPC picks bridges because two bridges yield +λ.
    """
    n = 15
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # community 1 (5 nodes)
        (5, 6), (6, 7), (7, 8), (8, 9),       # community 2 (5 nodes)
        (10, 11), (11, 12), (12, 13), (13, 14), # community 3 (5 nodes)
    ]
    return DelayedValueTask(
        name="staged_community_bridge",
        description="Three communities, threshold=1: need 2 bridges for bonus. Greedy avoids bridges.",
        n_nodes=n,
        initial_edges=edges,
        utility_name="threshold_connectivity",
        utility_params={"lambda_conn": 30.0, "threshold": 1},
        latent_dim=4,
        latent_seed=200,
        available_actions=[
            ("add_edge", 4, 5, {"weight": 1.0}),    # bridge 1-2
            ("add_edge", 9, 10, {"weight": 1.0}),   # bridge 2-3
            ("add_edge", 4, 10, {"weight": 1.0}),   # bridge 1-3
            ("add_edge", 0, 2, {"weight": 1.0}),    # within-cluster
            ("add_edge", 5, 7, {"weight": 1.0}),    # within-cluster
            ("add_edge", 10, 12, {"weight": 1.0}),  # within-cluster
        ],
        optimal_first_action=("add_edge", 4, 5),
        greedy_failure_reason="Greedy picks within-cluster shortcuts. MPC picks bridges because two bridges yield +λ threshold bonus.",
        cluster_assignments=[0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2],
        cluster_spacing=1.0,
    )


def task_hub_decomposition() -> DelayedValueTask:
    """Four clusters in a chain, threshold=1: need 3 bridges.

    Starting with 4 components arranged in a chain:
    - Steps 1-2: bridge 4→3→2, NO bonus (still > 1)
    - Step 3: bridge 2→1, +λ bonus

    At H=2, MPC can only get to 2 components (no bonus).
    But the bridge that sets up step 3 best is still chosen by MPC
    because it has less negative future value than within-cluster.
    """
    n = 16
    edges = [
        (0, 1), (1, 2), (2, 3),               # cluster 1
        (4, 5), (5, 6), (6, 7),               # cluster 2
        (8, 9), (9, 10), (10, 11),            # cluster 3
        (12, 13), (13, 14), (14, 15),         # cluster 4
    ]
    return DelayedValueTask(
        name="hub_decomposition",
        description="Four clusters, threshold=1: need 3 bridges for bonus. At H=2, MPC picks bridges that set up step 3.",
        n_nodes=n,
        initial_edges=edges,
        utility_name="threshold_connectivity",
        utility_params={"lambda_conn": 30.0, "threshold": 1},
        latent_dim=4,
        latent_seed=300,
        available_actions=[
            ("add_edge", 3, 4, {"weight": 1.0}),    # bridge 1-2
            ("add_edge", 7, 8, {"weight": 1.0}),    # bridge 2-3
            ("add_edge", 11, 12, {"weight": 1.0}),  # bridge 3-4
            ("add_edge", 0, 2, {"weight": 1.0}),    # within-cluster (greedy)
            ("add_edge", 5, 7, {"weight": 1.0}),    # within-cluster
            ("add_edge", 9, 11, {"weight": 1.0}),   # within-cluster
        ],
        optimal_first_action=("add_edge", 3, 4),
        greedy_failure_reason="Greedy picks within-cluster shortcuts. MPC picks bridges that set up future connectivity bonus.",
        cluster_assignments=[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3],
        cluster_spacing=1.0,
    )


def task_bottleneck_repair() -> DelayedValueTask:
    """Three components with isolated pair, threshold=1.

    Starting with 3 components (main path, second path, isolated pair):
    - Step 1: connect isolated pair to main, 3→2, NO bonus
    - Step 2: connect second path, 2→1, +λ bonus

    Greedy picks within-path shortcuts. MPC picks the bridge.
    """
    n = 14
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),   # path 1 (6 nodes)
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), # path 2 (6 nodes)
        (12, 13),  # isolated pair (2 nodes)
    ]
    return DelayedValueTask(
        name="bottleneck_repair",
        description="Three components, threshold=1: first bridge gives no bonus, second gives +λ.",
        n_nodes=n,
        initial_edges=edges,
        utility_name="threshold_connectivity",
        utility_params={"lambda_conn": 30.0, "threshold": 1},
        latent_dim=4,
        latent_seed=400,
        available_actions=[
            ("add_edge", 5, 6, {"weight": 1.0}),   # bridge path1-path2
            ("add_edge", 12, 0, {"weight": 1.0}),  # connect isolated to path1
            ("add_edge", 12, 6, {"weight": 1.0}),  # connect isolated to path2
            ("add_edge", 0, 2, {"weight": 1.0}),   # within-path1 (greedy)
            ("add_edge", 7, 9, {"weight": 1.0}),   # within-path2 (greedy)
            ("add_edge", 12, 13, {"weight": 2.0}), # within-isolated (greedy)
        ],
        optimal_first_action=("add_edge", 5, 6),
        greedy_failure_reason="Greedy picks within-path shortcuts. MPC picks bridges because two bridges yield +λ threshold bonus.",
        cluster_assignments=[0,0,0,0,0,0, 1,1,1,1,1,1, 2,2],
        cluster_spacing=1.0,
    )


def task_redundancy_then_shortcut() -> DelayedValueTask:
    """Three clusters with different sizes, threshold=1.

    Starting with 3 components of different sizes:
    - Step 1: bridge two clusters, 3→2, NO bonus
    - Step 2: bridge remaining, 2→1, +λ bonus

    Greedy picks within-cluster edges. MPC picks bridges.
    """
    n = 12
    edges = [
        (0, 1), (1, 2),               # cluster 1 (3 nodes)
        (3, 4), (4, 5), (5, 6),       # cluster 2 (4 nodes)
        (7, 8), (8, 9), (9, 10), (10, 11),  # cluster 3 (5 nodes)
    ]
    return DelayedValueTask(
        name="redundancy_then_shortcut",
        description="Three unequal clusters, threshold=1: first bridge gives no bonus, second gives +λ.",
        n_nodes=n,
        initial_edges=edges,
        utility_name="threshold_connectivity",
        utility_params={"lambda_conn": 30.0, "threshold": 1},
        latent_dim=4,
        latent_seed=500,
        available_actions=[
            ("add_edge", 2, 3, {"weight": 1.0}),   # bridge 1-2
            ("add_edge", 6, 7, {"weight": 1.0}),   # bridge 2-3
            ("add_edge", 2, 7, {"weight": 1.0}),   # bridge 1-3
            ("add_edge", 0, 1, {"weight": 1.0}),   # within-cluster (greedy)
            ("add_edge", 3, 5, {"weight": 1.0}),   # within-cluster
            ("add_edge", 8, 10, {"weight": 1.0}),  # within-cluster
        ],
        optimal_first_action=("add_edge", 2, 3),
        greedy_failure_reason="Greedy picks within-cluster shortcuts (less negative additive). MPC picks bridges for threshold bonus.",
        cluster_assignments=[0,0,0, 1,1,1,1, 2,2,2,2,2],
        cluster_spacing=1.0,
    )


def get_all_delayed_value_tasks() -> list[DelayedValueTask]:
    return [
        task_delayed_bridge(),
        task_staged_community_bridge(),
        task_hub_decomposition(),
        task_bottleneck_repair(),
        task_redundancy_then_shortcut(),
    ]
