"""v5.3.2 Model-predictive structural control (MPC).

The audit recommended moving toward model-predictive structural control:
planning over a horizon of structural mutations by simulating their
effects, rather than greedily selecting the single best mutation.

This module implements a simple MPC controller that:
  1. Enumerates candidate mutation sequences up to a horizon H.
  2. Simulates each sequence using the engine's shadow evaluation.
  3. Selects the sequence with the highest cumulative utility.
  4. Executes only the first mutation (receding horizon).

The controller is bounded: the branching factor and horizon are
configurable, and the total number of simulated sequences is capped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy
import math

import torch
from torch import Tensor

from .types import GraphBuffers
from .config import LGAEConfig
from .mutations import (
    AddEdge, PruneEdge, ReweightAffinity, ReweightLength,
    MutationAuthorityLevel, mutation_authority_level,
)
from .benchmark.tasks import StructuralAction


@dataclass
class MPCPlanResult:
    """Result of MPC planning."""
    best_sequence: list[Any]  # list of mutations
    predicted_utility: float
    candidates_evaluated: int
    horizon: int
    first_mutation_authority: MutationAuthorityLevel


class StructuralMPC:
    """Model-predictive controller for structural mutations.

    Plans over a horizon by simulating mutation sequences using a
    utility function.  Uses receding horizon control: only the first
    mutation of the best sequence is executed.

    The controller is bounded:
      - max_branching: how many candidate mutations to consider at each step
      - horizon: how many steps to look ahead
      - max_sequences: cap on total simulated sequences (branching^horizon)
    """

    def __init__(
        self,
        utility_fn,
        *,
        horizon: int = 2,
        max_branching: int = 8,
        max_sequences: int = 64,
    ):
        if int(horizon) < 1:
            raise ValueError("horizon must be positive")
        if int(max_branching) < 1:
            raise ValueError("max_branching must be positive")
        if int(max_sequences) < 1:
            raise ValueError("max_sequences must be positive")
        self.utility_fn = utility_fn
        self.horizon = int(horizon)
        self.max_branching = int(max_branching)
        self.max_sequences = int(max_sequences)

    def _generate_candidates(
        self, graph: GraphBuffers, z: Tensor, *, seed: int = 0,
    ) -> list[Any]:
        """Generate bounded candidate mutations for the current state."""
        ids = graph.valid.nonzero(as_tuple=True)[0]
        n = graph.num_nodes
        candidates: list[Any] = []

        # ADD_EDGE candidates: top-k non-edges by latent distance
        if len(ids) > 0 and n > 2:
            existing = set()
            for i in ids.tolist():
                u, v = int(graph.src[i]), int(graph.dst[i])
                existing.add(tuple(sorted((u, v))))

            # Find non-edges sorted by latent distance
            with torch.no_grad():
                dists = torch.cdist(z, z)
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if tuple(sorted((i, j))) not in existing:
                        non_edges.append((float(dists[i, j].item()), i, j))
            non_edges.sort()
            for _, u, v in non_edges[:self.max_branching // 2]:
                candidates.append(AddEdge(u, v))

        # PRUNE_EDGE candidates: edges with lowest weight
        if len(ids) > 1:
            weights = []
            for i in ids.tolist():
                weights.append((float(graph.weight[i].item()), i))
            weights.sort()
            for _, i in weights[:self.max_branching // 4]:
                candidates.append(PruneEdge(int(graph.src[i]), int(graph.dst[i])))

        # REWEIGHT candidates: a few reweighting mutations
        if len(ids) > 0:
            for i in ids.tolist()[:self.max_branching // 4]:
                candidates.append(ReweightAffinity(
                    int(graph.src[i]), int(graph.dst[i]), factor=1.5,
                ))

        # NO_OP is always a candidate
        # (represented as None, which means "do nothing this step")

        return candidates[:self.max_branching]

    def _apply_mutation(self, graph: GraphBuffers, mutation: Any) -> GraphBuffers:
        """Apply a mutation to a copy of the graph (shadow execution)."""
        if mutation is None:
            return copy.deepcopy(graph)
        g2 = copy.deepcopy(graph)
        mutation.apply(g2)
        return g2

    def plan(
        self, graph: GraphBuffers, z: Tensor, *, seed: int = 0,
    ) -> MPCPlanResult:
        """Plan the best mutation sequence over the horizon.

        Returns the best sequence found.  Only the first mutation should
        be executed (receding horizon).
        """
        current_utility = self.utility_fn(graph, z)
        best_sequence: list[Any] = []
        best_utility = current_utility
        candidates_evaluated = 0

        # BFS over the horizon
        # State: (graph, z, sequence, cumulative_utility)
        frontier = [(graph, z, [], current_utility)]

        for step in range(self.horizon):
            next_frontier = []
            for g, zz, seq, util in frontier:
                if len(next_frontier) >= self.max_sequences:
                    break
                cands = self._generate_candidates(g, zz, seed=seed + step)
                # Add NO_OP (None) as a candidate
                cands = cands + [None]
                for mut in cands:
                    if len(next_frontier) >= self.max_sequences:
                        break
                    g2 = self._apply_mutation(g, mut)
                    u2 = self.utility_fn(g2, zz)
                    new_seq = seq + [mut]
                    candidates_evaluated += 1
                    if u2 > best_utility or len(new_seq) == 1:
                        if u2 > best_utility:
                            best_utility = u2
                            best_sequence = new_seq
                    next_frontier.append((g2, zz, new_seq, u2))

            # Keep only top-k by utility for the next step (pruning)
            next_frontier.sort(key=lambda x: x[3], reverse=True)
            frontier = next_frontier[:self.max_branching]

        # Determine authority level of first mutation
        if best_sequence and best_sequence[0] is not None:
            authority = mutation_authority_level(best_sequence[0])
        else:
            authority = MutationAuthorityLevel.REVERSIBLE  # NO_OP

        return MPCPlanResult(
            best_sequence=best_sequence,
            predicted_utility=best_utility,
            candidates_evaluated=candidates_evaluated,
            horizon=self.horizon,
            first_mutation_authority=authority,
        )
