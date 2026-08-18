"""Encoder 7: SmallLearnedGraphEncoder — 2-3 layer message passing.

A deliberately small learned encoder. Not exotic architecture — just
simple message passing:

    h_v^{(l+1)} = φ(h_v^{(l)}, AGG_{u∈N(v)} ψ(h_u^{(l)}, e_{uv}))

Then pool the affected local subgraph.

Candidate architecture:
- 2-3 graph layers
- small hidden width
- mean + max pooling
- action embedding concatenated afterward
- output: 64 dimensions

This encoder is trained only through an explicit pretraining/objective
experiment — not entangled with full v6 planning yet.
"""
from __future__ import annotations

from typing import Any, Sequence
import hashlib
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import (
    EncodedState, EncodedAction, StateActionRepresentation,
    ActionEncodingSchema, DEFAULT_ACTION_SCHEMA,
    feature_hash, ensure_finite,
)
from .normalization import NormalizationStatistics


class SimpleMessagePassingLayer(nn.Module):
    """A single message passing layer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        node_feats: torch.Tensor,  # (N, in_dim)
        adj: torch.Tensor,         # (N, N) — binary adjacency
    ) -> torch.Tensor:
        # Self transform.
        h_self = self.self_linear(node_feats)
        # Neighbor aggregation (mean).
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        agg = adj @ node_feats / deg  # (N, in_dim)
        h_neigh = self.neighbor_linear(agg)
        return F.relu(h_self + h_neigh)


class SmallLearnedGraphEncoder:
    """Encoder 7: Small learned graph encoder (2-3 layer message passing).

    Uses a simple message passing network with mean pooling.
    Output dimension: 64 (configurable).

    This encoder has a UNFIT → FITTED_TRAIN → FROZEN lifecycle.
    Once frozen, the neural parameters are immutable.
    """

    name = "learned-graph"
    version = "v1"
    deterministic = False  # Neural network (but deterministic with fixed seed)
    requires_fit = True

    def __init__(
        self,
        hidden_dim: int = 32,
        output_dim: int = 64,
        n_layers: int = 2,
        node_feat_dim: int = 8,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.n_layers = int(n_layers)
        self.node_feat_dim = int(node_feat_dim)
        self.seed = int(seed)
        self._schema = DEFAULT_ACTION_SCHEMA
        self._global_norm = NormalizationStatistics()
        self._lifecycle = "unfit"

        # Build the network.
        torch.manual_seed(self.seed)
        self._net = self._build_network()
        self._action_dim = self._schema.n_types

    def _build_network(self) -> nn.Module:
        layers = []
        in_dim = self.node_feat_dim
        for _ in range(self.n_layers):
            layers.append(SimpleMessagePassingLayer(in_dim, self.hidden_dim))
            in_dim = self.hidden_dim
        # Final projection to output_dim.
        self._final = nn.Linear(in_dim, self.output_dim)
        self._layers = nn.ModuleList(layers)
        return nn.ModuleDict({
            "layers": self._layers,
            "final": self._final,
        })

    @property
    def dimension(self) -> int:
        return self.output_dim + self._action_dim

    @property
    def schema_hash(self) -> str:
        content = f"{self.name}:{self.version}:{self.output_dim}:{self.n_layers}:{self.hidden_dim}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._net.parameters())

    def fit(
        self,
        representations: Sequence[Sequence[float]],
        targets: Sequence[float],
        *,
        split: str = "train",
        n_epochs: int = 50,
        lr: float = 0.01,
    ) -> dict[str, Any]:
        """Fit the encoder on training data (simple regression pretraining).

        This is a lightweight pretraining objective that learns to predict
        realized_delta from the representation. It is NOT the final outcome
        model — it just provides a useful initialization.
        """
        if split != "train":
            from .normalization import HeldOutFittingError
            raise HeldOutFittingError(
                f"Cannot fit learned encoder on '{split}' split. "
                "Fitting must use train data only."
            )
        if self._lifecycle == "frozen":
            from .normalization import FrozenNormalizationError
            raise FrozenNormalizationError("Cannot fit frozen encoder.")

        X = torch.tensor(np.array(representations), dtype=torch.float32)
        y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)

        # Simple linear probe for pretraining.
        probe = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(list(self._net.parameters()) + list(probe.parameters()), lr=lr)

        for epoch in range(n_epochs):
            opt.zero_grad()
            pred = probe(X)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()

        self._lifecycle = "fitted_train"
        return {"final_loss": float(loss.item()), "n_epochs": n_epochs}

    def freeze(self) -> None:
        """Freeze the encoder. After this, parameters are immutable."""
        if self._lifecycle != "fitted_train":
            raise RuntimeError("Cannot freeze unfitted encoder.")
        for p in self._net.parameters():
            p.requires_grad = False
        self._lifecycle = "frozen"

    def encode_graph(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Encode a graph using message passing + pooling.

        Args:
            node_feats: (N, node_feat_dim) node feature matrix.
            adj: (N, N) binary adjacency matrix.

        Returns:
            (output_dim,) pooled graph representation.
        """
        h = node_feats
        layers = self._net["layers"]
        for layer in layers:
            h = layer(h, adj)
        # Mean + max pooling.
        mean_pool = h.mean(dim=0)
        max_pool = h.max(dim=0).values
        pooled = torch.cat([mean_pool, max_pool])
        # Project to output_dim.
        out = self._net["final"](pooled)
        return out

    def encode_state(self, state: Any, global_features: Sequence[float]) -> EncodedState:
        # For the state encoder, we use global features as a fallback
        # when no graph is provided.
        normed, mask = self._global_norm.transform(global_features)
        normed = ensure_finite(normed)
        return EncodedState(
            vector=normed, dimension=len(normed),
            encoder_id=self.name, schema_hash=self.schema_hash,
            missing_mask=mask,
        )

    def encode_action(
        self, action_type: str, action_target: dict[str, Any],
        local_features: Sequence[float],
    ) -> EncodedAction:
        type_vec = [0.0] * self._schema.n_types
        idx = self._schema.type_index(action_type)
        if idx >= 0:
            type_vec[idx] = 1.0
        vec = ensure_finite(type_vec)
        return EncodedAction(
            vector=vec, dimension=len(vec),
            encoder_id=self.name, schema_hash=self.schema_hash,
            action_type=action_type,
        )

    def encode(
        self, state: Any, global_features: Sequence[float],
        action_type: str, action_target: dict[str, Any],
        local_features: Sequence[float],
    ) -> StateActionRepresentation:
        es = self.encode_state(state, global_features)
        ea = self.encode_action(action_type, action_target, local_features)
        combined = es.vector + ea.vector
        return StateActionRepresentation(
            encoder_id=self.name, encoder_version=self.version,
            schema_hash=self.schema_hash, vector=combined,
            dimension=len(combined),
            state_feature_hash=feature_hash(es.vector),
            action_feature_hash=feature_hash(ea.vector),
            normalization_hash=self._global_norm.normalization_hash,
            metadata={"n_parameters": self.n_parameters},
        )
