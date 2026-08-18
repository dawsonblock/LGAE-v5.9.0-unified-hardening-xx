"""v5.3.2 Permutation-equivariant executive architecture.

The audit found that the executive uses engineered global statistics and
node/edge MLPs, not a graph-equivariant architecture.  A GNN or graph
transformer would be more appropriate because the executive's decisions
should be invariant to node relabeling.

This module implements a simple message-passing GNN that produces
permutation-equivariant node embeddings.  The embeddings can be pooled
into a graph-level representation for action selection.

The architecture is intentionally simple (2-layer message passing with
mean aggregation) to avoid overcomplicating the codebase.  It can be
replaced with a more sophisticated architecture (GAT, graph transformer)
without changing the interface.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingLayer(nn.Module):
    """Single message-passing layer with mean aggregation.

    h_i' = W_self * h_i + mean_{j in N(i)} W_neigh * h_j + b

    This is permutation-equivariant: relabeling nodes produces the same
    embeddings (relabeled).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """Apply one round of message passing.

        Args:
            h: Node features [N, in_dim]
            edge_index: Edge indices [2, E] (src and dst)

        Returns:
            Updated node features [N, out_dim]
        """
        src, dst = edge_index  # [E], [E]
        # Aggregate neighbor messages (mean aggregation)
        messages = self.w_neigh(h[src])  # [E, out_dim]
        # Mean aggregation: for each dst node, average messages from its src neighbors
        N = h.shape[0]
        aggregated = torch.zeros(N, messages.shape[1], device=h.device, dtype=h.dtype)
        degree = torch.zeros(N, device=h.device, dtype=h.dtype)
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        # Avoid division by zero
        aggregated = aggregated / degree.unsqueeze(1).clamp_min(1.0)
        # Self transform + neighbor aggregation
        return self.w_self(h) + aggregated


class EquivariantExecutiveNetwork(nn.Module):
    """Permutation-equivariant executive network.

    Uses message passing to produce node embeddings, then pools them
    into a graph-level representation for action selection.

    The network is permutation-equivariant in its node embeddings and
    permutation-invariant in its graph-level output (after pooling).

    Architecture:
      1. Input: node features [N, in_dim] + edge_index [2, E]
      2. Message passing layers (default 2)
      3. Mean pooling → graph-level representation [hidden_dim]
      4. Action heads: delta_u, ig, cost, risk, uncertainty, policy

    The graph-level output is permutation-invariant because mean pooling
    is permutation-invariant.
    """

    def __init__(
        self,
        node_feat_dim: int = 6,
        hidden_dim: int = 64,
        num_actions: int = 9,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        # Action heads (same as ExecutiveNetwork)
        self.delta_u_head = nn.Linear(hidden_dim, num_actions)
        self.ig_head = nn.Linear(hidden_dim, num_actions)
        self.cost_head = nn.Linear(hidden_dim, num_actions)
        self.risk_head = nn.Linear(hidden_dim, num_actions)
        self.uncertainty_head = nn.Linear(hidden_dim, num_actions)
        self.policy_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, node_feats: Tensor, edge_index: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            node_feats: Node features [N, node_feat_dim]
            edge_index: Edge indices [2, E]

        Returns:
            Dict of action head outputs [num_actions]
        """
        h = F.relu(self.input_proj(node_feats))
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
        # Mean pooling → graph-level representation
        h_graph = h.mean(dim=0)  # [hidden_dim]
        return {
            "delta_u": self.delta_u_head(h_graph),
            "ig": F.softplus(self.ig_head(h_graph)),
            "cost": F.softplus(self.cost_head(h_graph)),
            "risk": F.softplus(self.risk_head(h_graph)),
            "uncertainty": F.softplus(self.uncertainty_head(h_graph)),
            "policy_logits": self.policy_head(h_graph),
        }

    def node_embeddings(self, node_feats: Tensor, edge_index: Tensor) -> Tensor:
        """Return permutation-equivariant node embeddings [N, hidden_dim]."""
        h = F.relu(self.input_proj(node_feats))
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
        return h


def graphbuffers_to_edge_index(graph) -> Tensor:
    """Convert GraphBuffers to edge_index [2, E] format used by GNNs."""
    src, dst, _ = graph.active()
    # Undirected: add both directions
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return edge_index


def permutation_invariance_test(
    net: EquivariantExecutiveNetwork,
    node_feats: Tensor,
    edge_index: Tensor,
    perm: Tensor,
) -> dict[str, float]:
    """Test that the network's graph-level output is permutation-invariant.

    Args:
        net: The equivariant network
        node_feats: Node features [N, D]
        edge_index: Edge indices [2, E]
        perm: Permutation indices [N]

    Returns:
        Dict of max absolute differences between original and permuted outputs.
    """
    out_orig = net(node_feats, edge_index)
    # Permute nodes
    perm_feats = node_feats[perm]
    # Permute edge indices
    perm_map = torch.zeros_like(perm)
    perm_map[perm] = torch.arange(len(perm))
    perm_edge_index = perm_map[edge_index]
    out_perm = net(perm_feats, perm_edge_index)
    diffs = {}
    for key in out_orig:
        diffs[key] = float((out_orig[key] - out_perm[key]).abs().max().item())
    return diffs
