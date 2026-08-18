"""v5.0 Synthetic benchmark tasks with known-optimal structural changes.

Each task creates a specific graph + latent state with a known structural
problem. The correct structural intervention is known a priori, allowing
direct evaluation of whether LGAE identifies and executes the right change.

Tasks:
    A: Long-range bottleneck → ADD_EDGE (alternate route)
    B: Local representational complexity → SPAWN_FIBER
    C: Noisy spurious edge → PRUNE_EDGE
    D: Coordinate-frame mismatch → CHANGE_GAUGE
    E: Distribution shift → SPAWN_FIBER + consolidate
    F: Nothing wrong → NO_OP
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import random

import torch
from torch import Tensor

from ..types import GraphBuffers, make_graph_buffers
from ..config import LGAEConfig


class StructuralAction(Enum):
    """Actions the structural executive can propose."""
    NO_OP = "no_op"
    ADD_EDGE = "add_edge"
    PRUNE_EDGE = "prune_edge"
    REWEIGHT_AFFINITY = "reweight_affinity"
    REWEIGHT_LENGTH = "reweight_length"
    SPAWN_FIBER = "spawn_fiber"
    PRUNE_FIBER = "prune_fiber"
    CHANGE_GAUGE = "change_gauge"
    COUPLED_REWEIGHT = "coupled_reweight"


# ---------------------------------------------------------------------------
# Canonical action ordering (v5.3.3)
#
# Never derive semantic order from sets, dicts (unless ordered), filesystem
# traversal, or Python hash order.  Use ACTION_TO_INDEX for deterministic
# ordering when selecting from a set of correct actions.
# ---------------------------------------------------------------------------

ACTION_ORDER: tuple[StructuralAction, ...] = tuple(StructuralAction)

ACTION_TO_INDEX: dict[StructuralAction, int] = {
    action: idx for idx, action in enumerate(ACTION_ORDER)
}


def canonical_action(actions: set[StructuralAction]) -> StructuralAction:
    """Pick the canonical (lowest-index) action from a set.

    This replaces ``next(iter(actions))`` which is nondeterministic under
    PYTHONHASHSEED variation.
    """
    if not actions:
        return StructuralAction.NO_OP
    return min(actions, key=lambda a: ACTION_TO_INDEX[a])


@dataclass
class TaskState:
    """Initial state for a benchmark task."""
    graph: GraphBuffers
    z: Tensor
    config: LGAEConfig
    task_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskOutcome:
    """Outcome of applying a structural action to a task."""
    action: StructuralAction
    utility_before: float
    utility_after: float
    delta_utility: float
    correct: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkTask(ABC):
    """Base class for synthetic benchmark tasks.

    Each task defines:
    - initial_state(): Create the starting graph + latent state
    - correct_actions(): Return the set of actions that are correct for this task
    - utility(state): Measure task utility (higher = better)
    - apply_action(state, action): Apply a structural action and return new state
    - evaluate(action): Apply action, measure utility, return outcome
    """

    name: str
    description: str

    @abstractmethod
    def initial_state(self, seed: int = 42) -> TaskState:
        """Create the initial graph + latent state for this task."""
        ...

    @abstractmethod
    def correct_actions(self) -> set[StructuralAction]:
        """Return the set of actions that are correct for this task."""
        ...

    @abstractmethod
    def utility(self, state: TaskState) -> float:
        """Measure task utility. Higher is better."""
        ...

    @abstractmethod
    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        """Apply a structural action and return the new state."""
        ...

    def evaluate(self, state: TaskState, action: StructuralAction) -> TaskOutcome:
        """Apply action, measure utility change, return outcome."""
        u_before = self.utility(state)
        new_state = self.apply_action(state, action)
        u_after = self.utility(new_state)
        return TaskOutcome(
            action=action,
            utility_before=u_before,
            utility_after=u_after,
            delta_utility=u_after - u_before,
            correct=action in self.correct_actions(),
            metadata={"task": self.name},
        )

    def evaluate_all(
        self, state: TaskState, actions: list[StructuralAction] | None = None,
    ) -> list[TaskOutcome]:
        """Evaluate all candidate actions from the same starting state."""
        if actions is None:
            actions = list(StructuralAction)
        return [self.evaluate(state, a) for a in actions]


# ===========================================================================
# Task A: Long-range bottleneck → correct action = ADD_EDGE
# ===========================================================================

class TaskA_Bottleneck(BenchmarkTask):
    """Long-range bottleneck: two clusters connected by a single bridge edge.

    The correct action is to add an alternate route (ADD_EDGE) to reduce
    the bottleneck. Other actions should not improve utility as much.
    """

    name = "A_bottleneck"
    description = "Long-range bottleneck requiring alternate route"

    def initial_state(self, seed: int = 42) -> TaskState:
        torch.manual_seed(seed)
        # Two clusters of 4 nodes each, connected by a single bridge
        N = 8
        edges = [
            (0, 1), (1, 2), (2, 3),  # cluster 1
            (4, 5), (5, 6), (6, 7),  # cluster 2
            (3, 4),                   # bridge (bottleneck)
        ]
        graph = make_graph_buffers(N, edges, capacity=N + 4)
        # Latent: clusters are far apart in latent space
        z = torch.zeros(N, 4)
        z[:4] = torch.randn(4, 4) * 0.1 + torch.tensor([2.0, 0.0, 0.0, 0.0])
        z[4:] = torch.randn(4, 4) * 0.1 + torch.tensor([-2.0, 0.0, 0.0, 0.0])
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        return TaskState(graph=graph, z=z, config=cfg,
                         task_params={"bridge": (3, 4), "cluster1": [0, 1, 2, 3],
                                      "cluster2": [4, 5, 6, 7]})

    def correct_actions(self) -> set[StructuralAction]:
        return {StructuralAction.ADD_EDGE}

    def utility(self, state: TaskState) -> float:
        """Utility = spectral gap (algebraic connectivity) of the graph.

        This is a physical graph invariant — the Fiedler value λ₂ of the
        normalized Laplacian — not a function that rewards the correct
        action's structural signature. The correct action (add an alternate
        inter-cluster route) happens to maximize λ₂ because of the physics
        of bottlenecks, not because this function was written to reward it.
        The previous version added a `+0.1 * inter_cluster_edge_count` term
        that directly encoded the correct action's signature; that circular
        term has been removed.
        """
        from ..operators import spectral_gap_graphbuffers
        lam, _ = spectral_gap_graphbuffers(state.graph)
        return float(lam)

    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        import copy
        graph = state.graph.clone()
        z = state.z.clone()
        params = dict(state.task_params)

        if action == StructuralAction.ADD_EDGE:
            # Add alternate route: connect node 2 to node 5
            graph = self._add_edge(graph, 2, 5, weight=1.0)
        elif action == StructuralAction.PRUNE_EDGE:
            # Prune the bridge (makes things worse)
            graph = self._prune_edge(graph, 3, 4)
        elif action == StructuralAction.REWEIGHT_AFFINITY:
            # Increase bridge weight (helps a little but not as much as new route)
            graph = self._reweight(graph, 3, 4, factor=3.0)
        elif action == StructuralAction.SPAWN_FIBER:
            # Add fiber capacity (doesn't help the bottleneck)
            pass
        elif action == StructuralAction.NO_OP:
            pass
        else:
            pass

        return TaskState(graph=graph, z=z, config=state.config, task_params=params)

    def _add_edge(self, graph: GraphBuffers, u: int, v: int, weight: float = 1.0) -> GraphBuffers:
        """Add an edge to the graph."""
        from ..mutations import AddEdge
        mut = AddEdge(u, v, weight=weight)
        mut.apply(graph)
        return graph

    def _prune_edge(self, graph: GraphBuffers, u: int, v: int) -> GraphBuffers:
        from ..mutations import PruneEdge
        mut = PruneEdge(u, v)
        mut.apply(graph)
        return graph

    def _reweight(self, graph: GraphBuffers, u: int, v: int, factor: float) -> GraphBuffers:
        from ..mutations import ReweightAffinity
        mut = ReweightAffinity(u, v, factor=factor)
        mut.apply(graph)
        return graph


# ===========================================================================
# Task B: Local representational complexity → correct action = SPAWN_FIBER
# ===========================================================================

class TaskB_RepComplexity(BenchmarkTask):
    """Local representational complexity: a node needs more capacity.

    The correct action is to spawn a fiber (SPAWN_FIBER) to increase
    representational capacity at the bottleneck node.
    """

    name = "B_rep_complexity"
    description = "Local representational complexity requiring fiber growth"

    def initial_state(self, seed: int = 42) -> TaskState:
        torch.manual_seed(seed)
        N = 6
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        graph = make_graph_buffers(N, edges, capacity=N + 4)
        # Node 2 has high local complexity (needs more dimensions)
        z = torch.randn(N, 2)  # Only 2D currently — insufficient for node 2
        z[2] = torch.tensor([5.0, -5.0])  # High magnitude at bottleneck
        cfg = LGAEConfig()
        cfg.fiber.d_base = 2; cfg.fiber.d_max = 8
        return TaskState(graph=graph, z=z, config=cfg,
                         task_params={"bottleneck_node": 2, "needed_dim": 4})

    def correct_actions(self) -> set[StructuralAction]:
        return {StructuralAction.SPAWN_FIBER}

    def utility(self, state: TaskState) -> float:
        """Utility = negative representation bottleneck at the critical node."""
        z = state.z
        node = state.task_params["bottleneck_node"]
        needed = state.task_params["needed_dim"]
        current_dim = z.shape[1]
        # If we have enough dimensions, utility is high
        if current_dim >= needed:
            return 1.0 - 0.01 * z[node].norm().item() / max(current_dim, 1)
        # Otherwise, utility is penalized by the dimension deficit
        deficit = needed - current_dim
        return -float(deficit) - 0.1 * z[node].norm().item() / max(current_dim, 1)

    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        graph = state.graph.clone()
        z = state.z.clone()
        params = dict(state.task_params)

        if action == StructuralAction.SPAWN_FIBER:
            # Increase latent dimensionality (simulate fiber birth)
            needed = params["needed_dim"]
            if z.shape[1] < needed:
                # Pad with zeros to needed dimension
                pad = torch.zeros(z.shape[0], needed - z.shape[1])
                z = torch.cat([z, pad], dim=1)
        elif action == StructuralAction.ADD_EDGE:
            graph = self._add_edge(graph, 1, 3, weight=1.0)
        elif action == StructuralAction.PRUNE_EDGE:
            graph = self._prune_edge(graph, 1, 2)
        elif action == StructuralAction.NO_OP:
            pass
        else:
            pass

        return TaskState(graph=graph, z=z, config=state.config, task_params=params)

    def _add_edge(self, graph, u, v, weight=1.0):
        from ..mutations import AddEdge
        AddEdge(u, v, weight=weight).apply(graph)
        return graph

    def _prune_edge(self, graph, u, v):
        from ..mutations import PruneEdge
        PruneEdge(u, v).apply(graph)
        return graph


# ===========================================================================
# Task C: Noisy spurious edge → correct action = PRUNE_EDGE
# ===========================================================================

class TaskC_SpuriousEdge(BenchmarkTask):
    """Noisy spurious edge: a random edge connects unrelated nodes.

    The correct action is to prune the spurious edge (PRUNE_EDGE).
    """

    name = "C_spurious_edge"
    description = "Noisy spurious edge requiring pruning"

    def initial_state(self, seed: int = 42) -> TaskState:
        torch.manual_seed(seed)
        N = 6
        # Clean path graph + one spurious edge
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]  # 0-5 is spurious
        graph = make_graph_buffers(N, edges, capacity=N + 4)
        z = torch.randn(N, 4)
        # Nodes 0 and 5 are far apart in latent space (spurious connection)
        z[0] = torch.tensor([3.0, 0.0, 0.0, 0.0])
        z[5] = torch.tensor([-3.0, 0.0, 0.0, 0.0])
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        return TaskState(graph=graph, z=z, config=cfg,
                         task_params={"spurious_edge": (0, 5)})

    def correct_actions(self) -> set[StructuralAction]:
        return {StructuralAction.PRUNE_EDGE}

    def utility(self, state: TaskState) -> float:
        """Utility = negative edge-geometry mismatch."""
        src = state.graph.src.tolist()
        dst = state.graph.dst.tolist()
        w = state.graph.weight.tolist()
        z = state.z
        # Penalize edges where endpoints are far in latent space
        penalty = 0.0
        for s, d, weight in zip(src, dst, w):
            if weight > 0:
                dist = (z[s] - z[d]).norm().item()
                penalty += weight * dist
        return -penalty

    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        graph = state.graph.clone()
        z = state.z.clone()
        params = dict(state.task_params)

        if action == StructuralAction.PRUNE_EDGE:
            u, v = params["spurious_edge"]
            from ..mutations import PruneEdge
            PruneEdge(u, v).apply(graph)
        elif action == StructuralAction.ADD_EDGE:
            from ..mutations import AddEdge
            AddEdge(1, 4, weight=1.0).apply(graph)
        elif action == StructuralAction.REWEIGHT_AFFINITY:
            u, v = params["spurious_edge"]
            from ..mutations import ReweightAffinity
            ReweightAffinity(u, v, factor=0.01).apply(graph)
        elif action == StructuralAction.NO_OP:
            pass
        else:
            pass

        return TaskState(graph=graph, z=z, config=state.config, task_params=params)


# ===========================================================================
# Task D: Coordinate-frame mismatch → correct action = CHANGE_GAUGE
# ===========================================================================

class TaskD_GaugeMismatch(BenchmarkTask):
    """Coordinate-frame mismatch: two clusters use different reference frames.

    The correct action is to change the gauge (CHANGE_GAUGE) to align
    the coordinate frames.
    """

    name = "D_gauge_mismatch"
    description = "Coordinate-frame mismatch requiring gauge adaptation"

    def initial_state(self, seed: int = 42) -> TaskState:
        torch.manual_seed(seed)
        N = 6
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        graph = make_graph_buffers(N, edges, capacity=N + 4)
        # Structured latent: cluster 1 has clear principal directions
        z = torch.zeros(N, 4)
        z[:3] = torch.tensor([
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.5, 0.5, 1.0, 0.0],
        ])
        # Cluster 2: same structure but rotated 90° in first 2 dims
        R = torch.eye(4)
        R[0, 0] = 0.0; R[0, 1] = -1.0; R[1, 0] = 1.0; R[1, 1] = 0.0
        z[3:] = z[:3].clone() @ R.T  # Same structure, rotated
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        return TaskState(graph=graph, z=z, config=cfg,
                         task_params={"rotation": R, "bridge": (2, 3)})

    def correct_actions(self) -> set[StructuralAction]:
        return {StructuralAction.CHANGE_GAUGE}

    def utility(self, state: TaskState) -> float:
        """Utility = frame alignment between clusters.

        Measures how well the two clusters' coordinate frames are aligned.
        Uses the Frobenius norm of the difference between the clusters'
        principal direction matrices.
        """
        z = state.z
        c1 = z[:3]
        c2 = z[3:]
        # If frames are aligned, c1 ≈ c2 (same structure)
        # If rotated, c1 ≠ c2
        alignment = -float((c1 - c2).pow(2).sum().item())
        return alignment

    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        graph = state.graph.clone()
        z = state.z.clone()
        params = dict(state.task_params)

        if action == StructuralAction.CHANGE_GAUGE:
            # Align cluster 2 to cluster 1's frame
            R = params["rotation"]
            z[3:] = z[3:] @ R  # Undo the rotation
        elif action == StructuralAction.ADD_EDGE:
            from ..mutations import AddEdge
            AddEdge(1, 4, weight=1.0).apply(graph)
        elif action == StructuralAction.PRUNE_EDGE:
            from ..mutations import PruneEdge
            PruneEdge(2, 3).apply(graph)
        elif action == StructuralAction.NO_OP:
            pass
        else:
            pass

        return TaskState(graph=graph, z=z, config=state.config, task_params=params)


# ===========================================================================
# Task E: Distribution shift → correct action = SPAWN_FIBER + consolidate
# ===========================================================================

class TaskE_DistributionShift(BenchmarkTask):
    """Distribution shift: the task distribution has changed, requiring new capacity.

    The correct action is to spawn a fiber (SPAWN_FIBER) to handle the
    new distribution, then consolidate.
    """

    name = "E_distribution_shift"
    description = "Distribution shift requiring growth then consolidation"

    def initial_state(self, seed: int = 42) -> TaskState:
        torch.manual_seed(seed)
        N = 5
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        graph = make_graph_buffers(N, edges, capacity=N + 4)
        # Original distribution: 2D is sufficient
        z = torch.randn(N, 2) * 0.5
        # But the new distribution requires 4D
        cfg = LGAEConfig()
        cfg.fiber.d_base = 2; cfg.fiber.d_max = 8
        return TaskState(graph=graph, z=z, config=cfg,
                         task_params={"needed_dim": 4, "current_dim": 2})

    def correct_actions(self) -> set[StructuralAction]:
        return {StructuralAction.SPAWN_FIBER}

    def utility(self, state: TaskState) -> float:
        """Utility = capacity to represent the new distribution."""
        current_dim = state.z.shape[1]
        needed = state.task_params["needed_dim"]
        if current_dim >= needed:
            return 1.0
        return -float(needed - current_dim)

    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        graph = state.graph.clone()
        z = state.z.clone()
        params = dict(state.task_params)

        if action == StructuralAction.SPAWN_FIBER:
            needed = params["needed_dim"]
            if z.shape[1] < needed:
                pad = torch.zeros(z.shape[0], needed - z.shape[1])
                z = torch.cat([z, pad], dim=1)
        elif action == StructuralAction.PRUNE_EDGE:
            from ..mutations import PruneEdge
            PruneEdge(1, 2).apply(graph)
        elif action == StructuralAction.ADD_EDGE:
            from ..mutations import AddEdge
            AddEdge(0, 4, weight=1.0).apply(graph)
        elif action == StructuralAction.NO_OP:
            pass
        else:
            pass

        return TaskState(graph=graph, z=z, config=state.config, task_params=params)


# ===========================================================================
# Task F: Nothing wrong → correct action = NO_OP
# ===========================================================================

class TaskF_NoOp(BenchmarkTask):
    """Nothing wrong: the graph is already well-structured.

    The correct action is NO_OP. Any structural change should not
    improve utility (and may decrease it).
    """

    name = "F_noop"
    description = "Well-structured graph where no action is needed"

    def initial_state(self, seed: int = 42) -> TaskState:
        torch.manual_seed(seed)
        N = 6
        # Well-connected graph with no issues
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]
        graph = make_graph_buffers(N, edges, capacity=N + 4)
        z = torch.randn(N, 4) * 0.5
        # Ensure good alignment
        z = z - z.mean(dim=0, keepdim=True)
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        initial_w = graph.weight[graph.valid.bool()].clone()
        return TaskState(graph=graph, z=z, config=cfg,
                         task_params={"optimal_edge_count": 7, "initial_weights": initial_w})

    def correct_actions(self) -> set[StructuralAction]:
        return {StructuralAction.NO_OP}

    def utility(self, state: TaskState) -> float:
        """Utility = fidelity to the optimal graph configuration.

        The graph is already near-optimal. Any structural change (adding,
        pruning, or reweighting edges) moves away from the optimum and
        decreases utility. This is the "nothing wrong" task: the best
        action is NO_OP.
        """
        # Compare current graph to the initial (optimal) configuration
        initial_w = state.task_params.get("initial_weights")
        initial_edges = state.task_params.get("optimal_edge_count", 7)
        # initial_w is always set by initial_state(); if somehow missing,
        # treat the current state as optimal (utility = 0).
        if initial_w is None:
            return 0.0
        # Penalize weight changes
        current_w = state.graph.weight[state.graph.valid.bool()]
        if current_w.numel() == initial_w.numel():
            weight_penalty = (current_w - initial_w).pow(2).sum().item()
        else:
            weight_penalty = 10.0  # Large penalty for edge count change
        # Penalize edge count deviation
        edge_count = int(current_w.numel())
        edge_penalty = 5.0 * (edge_count - initial_edges) ** 2
        return -(weight_penalty + edge_penalty)

    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        graph = state.graph.clone()
        z = state.z.clone()
        params = dict(state.task_params)

        if action == StructuralAction.ADD_EDGE:
            from ..mutations import AddEdge
            AddEdge(0, 3, weight=1.0).apply(graph)
        elif action == StructuralAction.PRUNE_EDGE:
            from ..mutations import PruneEdge
            PruneEdge(0, 5).apply(graph)
        elif action == StructuralAction.REWEIGHT_AFFINITY:
            from ..mutations import ReweightAffinity
            ReweightAffinity(0, 1, factor=2.0).apply(graph)
        elif action == StructuralAction.NO_OP:
            pass
        else:
            pass

        return TaskState(graph=graph, z=z, config=state.config, task_params=params)


# ===========================================================================
# Task G: Information gain from exploration (v5.3.2)
#
# The audit found that the IG head predicts information gain but the
# benchmark doesn't test whether it drives active experimentation.  This
# task creates a graph with hidden structural uncertainty (two possible
# topologies consistent with the observed edges) and measures whether
# exploratory mutations (those with high predicted IG) actually reduce
# uncertainty about the true structure.
# ===========================================================================

class TaskG_InformationGain(BenchmarkTask):
    """Hidden structural uncertainty: exploration should reduce it.

    The graph has an ambiguous region where two interpretations are
    possible.  Adding an edge in the ambiguous region resolves the
    ambiguity (high actual IG), while adding an edge elsewhere doesn't
    (low actual IG).  The task tests whether the executive's IG
    predictions correlate with actual information gain.
    """

    name = "G_info_gain"
    description = "Hidden structural uncertainty requiring exploration"

    def initial_state(self, seed: int = 42) -> TaskState:
        torch.manual_seed(seed)
        # 8 nodes: two clusters with an ambiguous middle node
        N = 8
        edges = [
            (0, 1), (1, 2),        # cluster 1
            (3, 4), (4, 5),        # cluster 2
            (2, 6), (5, 6),        # node 6 connects both clusters
            # Node 7 is isolated from the main structure
            (6, 7),
        ]
        graph = make_graph_buffers(N, edges, capacity=N + 4)
        # Latent states with uncertainty in the middle region
        z = torch.randn(N, 4) * 0.5
        z[6] = torch.zeros(4)  # ambiguous middle node
        z[7] = torch.randn(4) * 0.1  # near-zero latent (uncertain)
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4
        cfg.fiber.d_max = 8
        return TaskState(graph=graph, z=z, config=cfg, task_params={"seed": seed})

    def correct_actions(self) -> set[StructuralAction]:
        # Adding an edge to resolve ambiguity is the correct exploratory action
        return {StructuralAction.ADD_EDGE}

    def utility(self, state: TaskState) -> float:
        """Utility = spectral gap + exploration bonus for resolving ambiguity."""
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(state.graph.num_nodes))
        ids = state.graph.valid.nonzero(as_tuple=True)[0]
        for i in ids.tolist():
            u, v = int(state.graph.src[i]), int(state.graph.dst[i])
            G.add_edge(u, v)
        if G.number_of_edges() == 0:
            return 0.0
        eigenvalues = nx.laplacian_spectrum(G).real
        eigenvalues.sort()
        lam2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        # Exploration bonus: edges connecting node 7 to the main structure
        # reduce uncertainty.  Reward connectivity of node 7.
        exploration_bonus = 0.0
        if 7 in G:
            deg7 = G.degree(7)
            exploration_bonus = 0.1 * deg7
        return lam2 + exploration_bonus

    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        import copy
        graph = copy.deepcopy(state.graph)
        z = state.z.clone()
        params = dict(state.task_params)
        if action == StructuralAction.ADD_EDGE:
            from ..mutations import AddEdge
            # Add edge from node 7 to the main structure (resolves ambiguity)
            AddEdge(7, 2).apply(graph)
        elif action == StructuralAction.PRUNE_EDGE:
            from ..mutations import PruneEdge
            ids = graph.valid.nonzero(as_tuple=True)[0]
            if len(ids) > 0:
                PruneEdge(int(graph.src[ids[0]]), int(graph.dst[ids[0]])).apply(graph)
        elif action == StructuralAction.NO_OP:
            pass
        else:
            pass
        return TaskState(graph=graph, z=z, config=state.config, task_params=params)


# ===========================================================================
# Registry
# ===========================================================================

ALL_TASKS: list[BenchmarkTask] = [
    TaskA_Bottleneck(),
    TaskB_RepComplexity(),
    TaskC_SpuriousEdge(),
    TaskD_GaugeMismatch(),
    TaskE_DistributionShift(),
    TaskF_NoOp(),
    TaskG_InformationGain(),
]


# ===========================================================================
# Held-out parametric variants (v5.3.1)
#
# The original ALL_TASKS produce *identical* graph structure for every seed
# (only latent noise differs), so "held-out seeds 101-105" were not actually
# held-out.  These parametric factories generate *structurally different*
# graphs (different sizes, cluster splits, spurious-edge positions) so that
# held-out evaluation measures something beyond seed noise.
# ===========================================================================


class HeldOutBottleneck(BenchmarkTask):
    """Bottleneck with variable cluster size and bridge position.

    Unlike TaskA (fixed 4+4 clusters, bridge (3,4)), this generates two
    clusters of ``cluster`` nodes each joined by a single bridge at a
    caller-chosen position.  Correct action remains ADD_EDGE (alternate
    route); utility is the spectral gap.
    """

    name = "heldout_bottleneck"
    description = "Parametric bottleneck with variable cluster size"

    def __init__(self, cluster: int = 5, bridge_offset: int = 0) -> None:
        self.cluster = max(3, int(cluster))
        self.bridge_offset = int(bridge_offset)

    def initial_state(self, seed: int = 42) -> TaskState:
        torch.manual_seed(seed)
        c = self.cluster
        N = 2 * c
        edges: list[tuple[int, int]] = []
        for i in range(c - 1):
            edges.append((i, i + 1))            # cluster 1 path
            edges.append((c + i, c + i + 1))    # cluster 2 path
        b1 = min(c - 1, max(0, c - 1 + self.bridge_offset))
        b2 = c
        edges.append((b1, b2))                  # the bridge (bottleneck)
        graph = make_graph_buffers(N, edges, capacity=N + 4)
        z = torch.zeros(N, 4)
        z[:c] = torch.randn(c, 4) * 0.1 + torch.tensor([2.0, 0.0, 0.0, 0.0])
        z[c:] = torch.randn(c, 4) * 0.1 + torch.tensor([-2.0, 0.0, 0.0, 0.0])
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        return TaskState(graph=graph, z=z, config=cfg,
                         task_params={"bridge": (b1, b2),
                                      "cluster1": list(range(c)),
                                      "cluster2": list(range(c, N))})

    def correct_actions(self) -> set[StructuralAction]:
        return {StructuralAction.ADD_EDGE}

    def utility(self, state: TaskState) -> float:
        from ..operators import spectral_gap_graphbuffers
        lam, _ = spectral_gap_graphbuffers(state.graph)
        return float(lam)

    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        graph = state.graph.clone()
        z = state.z.clone()
        params = dict(state.task_params)
        c = self.cluster
        if action == StructuralAction.ADD_EDGE:
            # Alternate route: connect an interior node of cluster 1 to one of cluster 2.
            from ..mutations import AddEdge
            AddEdge(max(0, c - 2), c + 1, weight=1.0).apply(graph)
        elif action == StructuralAction.PRUNE_EDGE:
            from ..mutations import PruneEdge
            b1, b2 = params["bridge"]
            PruneEdge(b1, b2).apply(graph)
        elif action == StructuralAction.REWEIGHT_AFFINITY:
            from ..mutations import ReweightAffinity
            b1, b2 = params["bridge"]
            ReweightAffinity(b1, b2, factor=3.0).apply(graph)
        elif action == StructuralAction.NO_OP:
            pass
        return TaskState(graph=graph, z=z, config=state.config, task_params=params)


class HeldOutSpuriousEdge(BenchmarkTask):
    """Spurious-edge task with variable size and spurious-edge position."""

    name = "heldout_spurious_edge"
    description = "Parametric spurious edge with variable size"

    def __init__(self, n: int = 8) -> None:
        self.n = max(5, int(n))

    def initial_state(self, seed: int = 42) -> TaskState:
        torch.manual_seed(seed)
        N = self.n
        edges = [(i, i + 1) for i in range(N - 1)]
        # Spurious long-range edge between the two ends.
        edges.append((0, N - 1))
        graph = make_graph_buffers(N, edges, capacity=N + 4)
        z = torch.randn(N, 4)
        z[0] = torch.tensor([3.0, 0.0, 0.0, 0.0])
        z[N - 1] = torch.tensor([-3.0, 0.0, 0.0, 0.0])
        cfg = LGAEConfig()
        cfg.fiber.d_base = 4; cfg.fiber.d_max = 8
        return TaskState(graph=graph, z=z, config=cfg,
                         task_params={"spurious_edge": (0, N - 1)})

    def correct_actions(self) -> set[StructuralAction]:
        return {StructuralAction.PRUNE_EDGE}

    def utility(self, state: TaskState) -> float:
        src = state.graph.src.tolist()
        dst = state.graph.dst.tolist()
        w = state.graph.weight.tolist()
        z = state.z
        penalty = 0.0
        for s, d, weight in zip(src, dst, w):
            if weight > 0:
                dist = (z[s] - z[d]).norm().item()
                penalty += weight * dist
        return -penalty

    def apply_action(self, state: TaskState, action: StructuralAction) -> TaskState:
        graph = state.graph.clone()
        z = state.z.clone()
        params = dict(state.task_params)
        if action == StructuralAction.PRUNE_EDGE:
            u, v = params["spurious_edge"]
            from ..mutations import PruneEdge
            PruneEdge(u, v).apply(graph)
        elif action == StructuralAction.ADD_EDGE:
            from ..mutations import AddEdge
            AddEdge(1, state.graph.num_nodes - 2, weight=1.0).apply(graph)
        elif action == StructuralAction.REWEIGHT_AFFINITY:
            u, v = params["spurious_edge"]
            from ..mutations import ReweightAffinity
            ReweightAffinity(u, v, factor=0.01).apply(graph)
        elif action == StructuralAction.NO_OP:
            pass
        return TaskState(graph=graph, z=z, config=state.config, task_params=params)


def heldout_tasks(seed: int = 0) -> list[BenchmarkTask]:
    """Return a set of structurally-distinct held-out tasks.

    These are intentionally *not* in ALL_TASKS (the training set).  They vary
    graph size, cluster split, and spurious-edge position so that a model
    trained on ALL_TASKS cannot succeed here by memorizing a single graph.
    The ``seed`` argument only varies which parametric variants are returned,
    not the latent noise of a fixed structure.
    """
    rng = random.Random(seed)
    cluster = rng.choice([5, 6, 7])
    bridge_offset = rng.choice([-1, 0, 1])
    n_spur = rng.choice([8, 9, 10])
    return [
        HeldOutBottleneck(cluster=cluster, bridge_offset=bridge_offset),
        HeldOutSpuriousEdge(n=n_spur),
    ]

