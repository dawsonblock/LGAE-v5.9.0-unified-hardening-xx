"""Delayed-value benchmark suite for exp6.3.

These are synthetic multi-step tasks where greedy immediate utility
is KNOWN to be suboptimal. They provide the structural incentive for
long-horizon planning.

Each task has:
- An initial graph + latent state
- A set of available actions at each step
- A known optimal multi-step plan
- The property that greedy one-step selection picks the WRONG first action

Task types:
1. Bridge now, unlock later: adding a bridge edge now has negative
   immediate utility but enables a high-utility shortcut later.

2. Temporary density increase: adding edges now increases density
   (negative utility) but enables removing a bottleneck later.

3. Remove useful-looking edge: pruning a high-utility edge now
   enables a better reroute later.

4. Multi-step hub decomposition: breaking a hub incrementally
   improves long-term connectivity.

5. Community bridge sequence: connecting communities step by step
   builds toward a globally optimal structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch

from ...types import GraphBuffers, make_graph_buffers
from ...mutations import AddEdge, PruneEdge


@dataclass
class DelayedValueTask:
    """A multi-step task where greedy is suboptimal."""
    name: str
    description: str
    n_nodes: int
    initial_edges: list[tuple[int, int]]
    latent_dim: int = 4
    # The optimal first action (NOT the greedy choice).
    optimal_first_action: tuple[str, int, int] = ("", 0, 0)
    # The greedy first action (what one-step optimization would pick).
    greedy_first_action: tuple[str, int, int] = ("", 0, 0)
    # Available actions at step 0.
    available_actions: list[tuple[str, int, int, dict]] = field(default_factory=list)
    # The optimal plan (sequence of actions).
    optimal_plan: list[tuple[str, int, int, dict]] = field(default_factory=list)
    # Why greedy fails.
    greedy_failure_reason: str = ""
    # Latent state seed.
    latent_seed: int = 42


def make_latent_state(n_nodes: int, dim: int, seed: int) -> torch.Tensor:
    """Create a deterministic latent state."""
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(n_nodes, dim, generator=rng) * 0.5


# ---------------------------------------------------------------------------
# Task 1: Bridge now, unlock later
# ---------------------------------------------------------------------------
# Two clusters connected by a long path. Adding a direct bridge edge
# between distant nodes has negative immediate utility (they're far apart
# in latent space) but enables a shortcut that dramatically improves
# future utility.

def task_bridge_now_unlock_later() -> DelayedValueTask:
    """Bridge edge has negative immediate utility but enables future shortcut."""
    n = 12
    # Two clusters: {0,1,2,3,4} and {5,6,7,8,9}, connected via path 4-10-11-5.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # cluster 1
        (5, 6), (6, 7), (7, 8), (8, 9),  # cluster 2
        (4, 10), (10, 11), (11, 5),       # bridge path
    ]

    # Latent states: cluster 1 nodes are close, cluster 2 nodes are close,
    # but clusters are far apart. Nodes 10, 11 are in between.
    # We'll use a custom latent state.
    latent_seed = 100

    # Available actions:
    # A: add_edge(0, 5) — direct bridge, high latent distance → negative ΔU
    # B: add_edge(3, 6) — shorter bridge, less negative ΔU (greedy picks this)
    # C: add_edge(4, 5) — very short bridge, slightly positive ΔU (greedy would pick this)
    # D: reweight(4, 10, 2.0) — strengthen bridge path (greedy might pick this)

    # The OPTIMAL first action is A (add 0-5) because it creates a direct
    # connection that, combined with a second step (add 1-6), creates
    # a much better structure than the greedy path.

    return DelayedValueTask(
        name="bridge_now_unlock_later",
        description="Adding a direct bridge between distant clusters has negative immediate utility but enables high-utility future connections.",
        n_nodes=n,
        initial_edges=edges,
        latent_dim=4,
        optimal_first_action=("add_edge", 0, 5),
        greedy_first_action=("add_edge", 4, 5),  # shorter, less negative
        available_actions=[
            ("add_edge", 0, 5, {"weight": 1.0}),   # A: long bridge (optimal)
            ("add_edge", 3, 6, {"weight": 1.0}),   # B: medium bridge
            ("add_edge", 4, 5, {"weight": 1.0}),   # C: short bridge (greedy)
            ("add_edge", 4, 10, {"weight": 2.0}),  # D: strengthen path
            ("add_edge", 1, 6, {"weight": 1.0}),   # E: another bridge
            ("add_edge", 2, 7, {"weight": 1.0}),   # F: another bridge
        ],
        optimal_plan=[
            ("add_edge", 0, 5, {"weight": 1.0}),
            ("add_edge", 1, 6, {"weight": 1.0}),
        ],
        greedy_failure_reason="Greedy picks the shortest bridge (4-5) which has the least negative immediate utility, but the long bridge (0-5) combined with a second step creates a much better global structure.",
        latent_seed=latent_seed,
    )


# ---------------------------------------------------------------------------
# Task 2: Remove useful-looking edge to enable better reroute
# ---------------------------------------------------------------------------

def task_remove_useful_reroute() -> DelayedValueTask:
    """Removing a high-utility edge enables a better reroute."""
    n = 10
    # A graph where edge (0,1) has high utility (close latent states)
    # but removing it forces a reroute through (0,2)-(2,3)-(3,1)
    # which has higher total utility.
    edges = [
        (0, 1), (0, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
    ]

    return DelayedValueTask(
        name="remove_useful_reroute",
        description="Removing a high-utility edge (0,1) has negative immediate utility but enables adding a better alternative path.",
        n_nodes=n,
        initial_edges=edges,
        latent_dim=4,
        optimal_first_action=("remove_edge", 0, 1),
        greedy_first_action=("add_edge", 1, 3),  # looks good immediately
        available_actions=[
            ("remove_edge", 0, 1, {}),            # A: remove useful edge (optimal)
            ("add_edge", 1, 3, {"weight": 1.0}),  # B: add shortcut (greedy)
            ("add_edge", 0, 5, {"weight": 1.0}),  # C: another connection
            ("add_edge", 2, 8, {"weight": 1.0}),  # D: cross connection
            ("add_edge", 4, 9, {"weight": 1.0}),  # E: another cross
            ("remove_edge", 4, 5, {}),            # F: remove different edge
        ],
        optimal_plan=[
            ("remove_edge", 0, 1, {}),
            ("add_edge", 0, 3, {"weight": 2.0}),
        ],
        greedy_failure_reason="Greedy adds a shortcut (1-3) for immediate gain, but removing the high-utility edge (0-1) and replacing it with a weighted alternative (0-3) gives higher total utility over 2 steps.",
        latent_seed=200,
    )


# ---------------------------------------------------------------------------
# Task 3: Multi-step hub decomposition
# ---------------------------------------------------------------------------

def task_hub_decomposition() -> DelayedValueTask:
    """Breaking a hub incrementally improves long-term connectivity."""
    n = 15
    # Hub at node 0 connected to many nodes. Other nodes poorly connected.
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (0, 6), (0, 7), (0, 8), (0, 9), (0, 10),
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
        (11, 12), (12, 13), (13, 14),
    ]

    return DelayedValueTask(
        name="hub_decomposition",
        description="Removing hub edges has negative immediate utility but enables better distributed connectivity.",
        n_nodes=n,
        initial_edges=edges,
        latent_dim=4,
        optimal_first_action=("remove_edge", 0, 7),
        greedy_first_action=("add_edge", 11, 0),  # connect orphan to hub
        available_actions=[
            ("remove_edge", 0, 7, {}),             # A: break hub edge (optimal)
            ("add_edge", 11, 0, {"weight": 1.0}),  # B: connect orphan (greedy)
            ("add_edge", 11, 1, {"weight": 1.0}),  # C: connect orphan to leaf
            ("add_edge", 12, 3, {"weight": 1.0}),  # D: connect orphan chain
            ("add_edge", 14, 10, {"weight": 1.0}), # E: connect end
            ("remove_edge", 0, 5, {}),             # F: break different hub edge
        ],
        optimal_plan=[
            ("remove_edge", 0, 7, {}),
            ("add_edge", 7, 8, {"weight": 2.0}),
            ("add_edge", 11, 7, {"weight": 1.0}),
        ],
        greedy_failure_reason="Greedy connects orphans to the hub for immediate gain, but breaking hub edges and creating lateral connections distributes load better over 3 steps.",
        latent_seed=300,
    )


# ---------------------------------------------------------------------------
# Task 4: Community bridge sequence
# ---------------------------------------------------------------------------

def task_community_bridge_sequence() -> DelayedValueTask:
    """Connecting communities step by step builds toward optimal structure."""
    n = 16
    # Three communities: {0-4}, {5-9}, {10-14}, plus node 15.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8), (8, 9),
        (10, 11), (11, 12), (12, 13), (13, 14),
    ]

    return DelayedValueTask(
        name="community_bridge_sequence",
        description="Building community bridges in the right order creates better global structure than greedy shortcuts.",
        n_nodes=n,
        initial_edges=edges,
        latent_dim=4,
        optimal_first_action=("add_edge", 4, 10),
        greedy_first_action=("add_edge", 2, 7),  # within-cluster shortcut
        available_actions=[
            ("add_edge", 4, 10, {"weight": 1.0}),  # A: connect comm 1-3 (optimal)
            ("add_edge", 2, 7, {"weight": 1.0}),   # B: within shortcut (greedy)
            ("add_edge", 4, 5, {"weight": 1.0}),   # C: connect comm 1-2
            ("add_edge", 9, 10, {"weight": 1.0}),  # D: connect comm 2-3
            ("add_edge", 0, 15, {"weight": 1.0}),  # E: connect orphan
            ("add_edge", 14, 15, {"weight": 1.0}), # F: connect other orphan
        ],
        optimal_plan=[
            ("add_edge", 4, 10, {"weight": 1.0}),
            ("add_edge", 5, 10, {"weight": 1.0}),
            ("add_edge", 0, 15, {"weight": 1.0}),
        ],
        greedy_failure_reason="Greedy picks a within-cluster shortcut for immediate gain, but connecting communities in sequence builds a globally optimal structure.",
        latent_seed=400,
    )


# ---------------------------------------------------------------------------
# Task 5: Temporary density for later shortcut
# ---------------------------------------------------------------------------

def task_temporary_density() -> DelayedValueTask:
    """Adding edges now (negative utility) enables better structure later."""
    n = 10
    # Sparse graph where adding temporary density enables better shortcuts.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8), (8, 9),
    ]

    return DelayedValueTask(
        name="temporary_density",
        description="Adding a temporary edge with negative utility enables a better permanent structure.",
        n_nodes=n,
        initial_edges=edges,
        latent_dim=4,
        optimal_first_action=("add_edge", 2, 7),
        greedy_first_action=("add_edge", 3, 6),  # shorter, less negative
        available_actions=[
            ("add_edge", 2, 7, {"weight": 1.0}),  # A: cross bridge (optimal)
            ("add_edge", 3, 6, {"weight": 1.0}),  # B: shorter cross (greedy)
            ("add_edge", 4, 5, {"weight": 1.0}),  # C: end-to-end
            ("add_edge", 1, 8, {"weight": 1.0}),  # D: long cross
            ("add_edge", 0, 9, {"weight": 1.0}),  # E: very long cross
            ("add_edge", 2, 5, {"weight": 1.0}),  # F: medium cross
        ],
        optimal_plan=[
            ("add_edge", 2, 7, {"weight": 1.0}),
            ("add_edge", 1, 6, {"weight": 1.0}),
        ],
        greedy_failure_reason="Greedy picks the shortest cross-cluster edge for less negative utility, but the longer bridge creates a better foundation for the second step.",
        latent_seed=500,
    )


def get_all_delayed_value_tasks() -> list[DelayedValueTask]:
    """Get all delayed-value benchmark tasks."""
    return [
        task_bridge_now_unlock_later(),
        task_remove_useful_reroute(),
        task_hub_decomposition(),
        task_community_bridge_sequence(),
        task_temporary_density(),
    ]


def make_task_graph(task: DelayedValueTask) -> GraphBuffers:
    """Create a GraphBuffers from a task's initial edges."""
    return make_graph_buffers(
        num_nodes=task.n_nodes,
        edges=task.initial_edges,
        capacity=max(len(task.initial_edges) * 3, task.n_nodes * 3),
    )


def make_task_latent(task: DelayedValueTask) -> torch.Tensor:
    """Create the latent state for a task."""
    return make_latent_state(task.n_nodes, task.latent_dim, task.latent_seed)
