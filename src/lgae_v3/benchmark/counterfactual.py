"""v5.3.1 Counterfactual structural dataset generator and Q-learning trainer.

This module addresses the central limitation identified in the audit:
the learned executive classifies actions from labels rather than learning
Q(S,a) from counterfactual outcomes.  It also scales the training data
from 864 hand-authored samples to tens of thousands of randomized
interventions across varied topology families.

The generator produces (state, action, Δutility) triples by:
  1. Sampling a random graph from a topology family (path, cycle, grid,
     star, barabasi-albert, watts-strogatz, random, complete-bipartite).
  2. Enumerating bounded candidate interventions (ADD_EDGE, PRUNE_EDGE,
     NO_OP, REWEIGHT).
  3. Executing each in a shadow graph.
  4. Measuring Δutility (spectral gap change).
  5. Recording the exact ranking.

The Q-trainer learns Q(S,a) = E[ΔU(S,a)] via regression on these
counterfactual outcomes, then derives the policy as π(S) = argmax_a Q(S,a).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random
import math
import hashlib

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx


def _stable_hash_u64(value: str | float) -> int:
    """Deterministic 64-bit hash.  Not dependent on PYTHONHASHSEED."""
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")

from ..types import GraphBuffers, make_graph_buffers
from ..config import LGAEConfig
from .tasks import StructuralAction

# ACTION_TO_IDX and NUM_ACTIONS are defined in executive.py but imported
# here locally to avoid a circular import (executive → benchmark.tasks →
# benchmark.__init__ → benchmark.counterfactual → executive).
_ACTION_LIST = list(StructuralAction)
ACTION_TO_IDX = {a: i for i, a in enumerate(_ACTION_LIST)}
NUM_ACTIONS = len(_ACTION_LIST)


# ===========================================================================
# Topology families
# ===========================================================================

TOPOLOGY_FAMILIES = [
    "path",
    "cycle",
    "grid",
    "star",
    "barabasi_albert",
    "watts_strogatz",
    "random",
    "complete_bipartite",
]

# Held-out families never seen during training
HELD_OUT_FAMILIES = ["wheel", "lollipop", "caveman"]


def generate_graph(family: str, n: int, seed: int) -> nx.Graph:
    """Generate a random graph from a topology family."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    n = max(4, min(n, 50))  # bounded for speed

    if family == "path":
        G = nx.path_graph(n)
    elif family == "cycle":
        G = nx.cycle_graph(n)
    elif family == "grid":
        side = max(2, int(math.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    elif family == "star":
        G = nx.star_graph(n - 1)
    elif family == "barabasi_albert":
        m = max(1, min(3, n // 4))
        G = nx.barabasi_albert_graph(n, m, seed=np_rng)
    elif family == "watts_strogatz":
        k = max(2, min(4, n // 3))
        G = nx.watts_strogatz_graph(n, k, 0.3, seed=np_rng)
    elif family == "random":
        p = max(0.1, min(0.5, 2.0 / max(n, 4)))
        G = nx.erdos_renyi_graph(n, p, seed=np_rng)
    elif family == "complete_bipartite":
        n1 = n // 2
        n2 = n - n1
        G = nx.complete_bipartite_graph(max(2, n1), max(2, n2))
    elif family == "wheel":
        G = nx.wheel_graph(n)
    elif family == "lollipop":
        G = nx.lollipop_graph(max(3, n // 2), max(2, n // 2))
    elif family == "caveman":
        k = max(2, n // 5)
        G = nx.connected_caveman_graph(max(2, k), max(3, n // max(k, 2)))
    else:
        G = nx.path_graph(n)

    # Ensure connected; if not, take largest component
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


def graph_to_buffers(G: nx.Graph, capacity: int = 256) -> GraphBuffers:
    """Convert a NetworkX graph to GraphBuffers."""
    edges = list(G.edges())
    if not edges:
        edges = [(0, 1)]
    return make_graph_buffers(G.number_of_nodes(), edges, capacity=capacity)


def spectral_gap(G: nx.Graph) -> float:
    """Compute the spectral gap (second-smallest eigenvalue of Laplacian)."""
    if G.number_of_nodes() <= 1:
        return 0.0
    try:
        eigenvalues = nx.laplacian_spectrum(G).real
        eigenvalues.sort()
        if len(eigenvalues) >= 2:
            return float(eigenvalues[1])
        return 0.0
    except Exception:
        return 0.0


# ===========================================================================
# Counterfactual dataset
# ===========================================================================

@dataclass
class CounterfactualSample:
    """A single (state, action, Δutility) triple."""
    observation: Tensor       # StructuralObservation vector
    action_idx: int           # ACTION_TO_IDX[action]
    delta_utility: float      # measured ΔU from shadow execution
    topology: str             # topology family name
    graph_size: int           # number of nodes
    correct: bool             # whether this action has the highest ΔU


def compute_observation(G: nx.Graph, z: Tensor | None = None) -> Tensor:
    """Compute a StructuralObservation-like feature vector from a graph.

    This is a simplified observation that captures the same structural
    statistics the executive uses: spectral gap, degree statistics,
    curvature proxy, latent variance, etc.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    std_deg = float(np.std(degrees)) if degrees else 0.0
    lam2 = spectral_gap(G)
    density = float(e) / max(n * (n - 1) / 2, 1)
    avg_clustering = float(nx.average_clustering(G)) if n > 2 else 0.0
    n_components = nx.number_connected_components(G)

    if z is not None:
        latent_var = float(z.var().item())
        latent_norm = float(z.norm().item())
    else:
        latent_var = 0.0
        latent_norm = 0.0

    # 16-dimensional observation vector
    return torch.tensor([
        math.log(max(n, 1)),
        math.log(max(e, 1)),
        lam2,
        mean_deg,
        std_deg,
        density,
        avg_clustering,
        float(n_components),
        latent_var,
        math.log1p(max(latent_var, 0.0)),
        latent_norm,
        math.log1p(max(latent_norm, 0.0)),
        float(n) / max(e, 1),
        float(e) / max(n, 1),
        float(degrees[0]) if degrees else 0.0,
        1.0,
    ], dtype=torch.float32)


def apply_action_to_graph(G: nx.Graph, action: StructuralAction, seed: int = 0) -> nx.Graph:
    """Apply a structural action to a NetworkX graph (shadow execution)."""
    G2 = G.copy()
    rng = random.Random(seed)
    nodes = list(G2.nodes())

    if action == StructuralAction.NO_OP:
        pass
    elif action == StructuralAction.ADD_EDGE:
        non_edges = list(nx.non_edges(G2))
        if non_edges:
            u, v = rng.choice(non_edges)
            G2.add_edge(u, v)
    elif action == StructuralAction.PRUNE_EDGE:
        edges = list(G2.edges())
        if edges:
            # Prune the edge with lowest betweenness (least important)
            u, v = rng.choice(edges)
            G2.remove_edge(u, v)
            if not nx.is_connected(G2):
                G2.add_edge(u, v)  # revert if disconnects
    elif action == StructuralAction.REWEIGHT_AFFINITY:
        # Reweighting doesn't change topology; approximate as no-op for spectral gap
        pass
    elif action == StructuralAction.SPAWN_FIBER:
        # Fiber spawn doesn't change graph topology
        pass
    elif action == StructuralAction.CHANGE_GAUGE:
        # Gauge change doesn't change graph topology
        pass
    else:
        pass

    return G2


def generate_counterfactual_dataset(
    num_samples: int = 10000,
    families: list[str] | None = None,
    seed: int = 0,
    min_size: int = 8,
    max_size: int = 30,
) -> list[CounterfactualSample]:
    """Generate a counterfactual dataset of (state, action, ΔU) triples.

    For each sample:
      1. Sample a random graph from a topology family.
      2. Compute baseline utility (spectral gap).
      3. For each candidate action, apply it in a shadow graph and measure ΔU.
      4. Record all (state, action, ΔU) triples.
    """
    if families is None:
        families = TOPOLOGY_FAMILIES

    rng = random.Random(seed)
    dataset: list[CounterfactualSample] = []

    samples_per_family = num_samples // len(families)
    for family in families:
        for i in range(samples_per_family):
            s = seed + i * 1000 + (_stable_hash_u64(family) % 1000)
            n = rng.randint(min_size, max_size)
            G = generate_graph(family, n, s)

            # Compute baseline observation and utility
            obs = compute_observation(G)
            u_base = spectral_gap(G)

            # Evaluate all candidate actions
            actions = [
                StructuralAction.NO_OP,
                StructuralAction.ADD_EDGE,
                StructuralAction.PRUNE_EDGE,
                StructuralAction.REWEIGHT_AFFINITY,
                StructuralAction.SPAWN_FIBER,
                StructuralAction.CHANGE_GAUGE,
            ]
            deltas = []
            for action in actions:
                G2 = apply_action_to_graph(G, action, seed=s)
                u_after = spectral_gap(G2)
                delta = u_after - u_base
                deltas.append((action, delta))

            # Find the best action
            best_delta = max(d for _, d in deltas)

            for action, delta in deltas:
                dataset.append(CounterfactualSample(
                    observation=obs.clone(),
                    action_idx=ACTION_TO_IDX[action],
                    delta_utility=float(delta),
                    topology=family,
                    graph_size=n,
                    correct=(delta == best_delta and delta > 0),
                ))

    return dataset


# ===========================================================================
# Q-learning trainer
# ===========================================================================

class QNetwork(nn.Module):
    """Simple Q-network: observation → Q-values for each action.

    Q(S,a) = E[ΔU(S,a)]

    The policy is derived as π(S) = argmax_a Q(S,a).
    """

    def __init__(self, obs_dim: int = 16, hidden_dim: int = 128, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class QTrainingResult:
    """Result of training a Q-network on counterfactual data."""
    q_network: QNetwork
    losses: list[float] = field(default_factory=list)
    train_accuracy: float = 0.0
    train_samples: int = 0


def train_q_network(
    dataset: list[CounterfactualSample],
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> QTrainingResult:
    """Train a Q-network on counterfactual outcomes.

    Learns Q(S,a) = E[ΔU(S,a)] via MSE regression on the counterfactual
    dataset.  The policy π(S) = argmax_a Q(S,a) should recover the
    best-ΔU action for each state.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    if not dataset:
        return QTrainingResult(q_network=QNetwork())

    obs_dim = dataset[0].observation.shape[0]
    q_net = QNetwork(obs_dim=obs_dim, hidden_dim=128)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    observations = torch.stack([s.observation for s in dataset])
    actions = torch.tensor([s.action_idx for s in dataset], dtype=torch.long)
    deltas = torch.tensor([s.delta_utility for s in dataset], dtype=torch.float32)

    n = len(dataset)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            obs_batch = observations[idx]
            act_batch = actions[idx]
            delta_batch = deltas[idx]

            q_values = q_net(obs_batch)  # [B, num_actions]
            predicted_deltas = q_values[torch.arange(len(idx)), act_batch]

            loss = F.mse_loss(predicted_deltas, delta_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        losses.append(epoch_loss / max(num_batches, 1))

    # Compute training accuracy: how often does argmax Q match the best action?
    with torch.no_grad():
        all_q = q_net(observations)
        predicted_actions = all_q.argmax(dim=-1)
        # For each state, find the best action from the dataset
        # Group by observation (approximate: same observation vector = same state)
        correct = 0
        total = 0
        seen_states: dict[int, int] = {}  # hash of obs → best action idx
        for i, s in enumerate(dataset):
            h = _stable_hash_u64(s.observation.sum().item())
            if h not in seen_states:
                # Find best action for this state
                state_samples = [j for j, s2 in enumerate(dataset) if _stable_hash_u64(s2.observation.sum().item()) == h]
                best_j = max(state_samples, key=lambda j: dataset[j].delta_utility)
                seen_states[h] = dataset[best_j].action_idx
            if predicted_actions[i].item() == seen_states[h]:
                correct += 1
            total += 1

    return QTrainingResult(
        q_network=q_net,
        losses=losses,
        train_accuracy=correct / max(total, 1),
        train_samples=n,
    )


# ===========================================================================
# Evaluation
# ===========================================================================

@dataclass
class EvaluationResult:
    """Result of evaluating a Q-network on a set of topology families."""
    family: str
    num_states: int
    accuracy: float
    mean_delta_utility: float
    mean_regret: float


def evaluate_q_network(
    q_net: QNetwork,
    families: list[str],
    num_states_per_family: int = 100,
    seed: int = 999,
) -> list[EvaluationResult]:
    """Evaluate a trained Q-network on topology families.

    For each family, generates fresh states, computes the Q-network's
    chosen action, and compares it to the oracle best action.
    """
    results = []
    for family in families:
        dataset = generate_counterfactual_dataset(
            num_samples=num_states_per_family * 6,  # 6 actions per state
            families=[family],
            seed=seed,
        )
        if not dataset:
            continue

        observations = torch.stack([s.observation for s in dataset])
        with torch.no_grad():
            q_values = q_net(observations)
            predicted_actions = q_values.argmax(dim=-1)

        # Group by state and compute accuracy
        seen_states: dict[int, list[int]] = {}  # obs hash → sample indices
        for i, s in enumerate(dataset):
            h = _stable_hash_u64(s.observation.sum().item())
            if h not in seen_states:
                seen_states[h] = []
            seen_states[h].append(i)

        correct = 0
        total = 0
        deltas = []
        regrets = []
        for h, indices in seen_states.items():
            # Find oracle best action
            best_idx = max(indices, key=lambda j: dataset[j].delta_utility)
            best_action = dataset[best_idx].action_idx
            best_delta = dataset[best_idx].delta_utility

            # Find Q-network's chosen action for this state
            first_idx = indices[0]
            q_chosen = predicted_actions[first_idx].item()

            # Find the delta for the Q-chosen action
            q_delta = 0.0
            for j in indices:
                if dataset[j].action_idx == q_chosen:
                    q_delta = dataset[j].delta_utility
                    break

            if q_chosen == best_action:
                correct += 1
            total += 1
            deltas.append(q_delta)
            regrets.append(best_delta - q_delta)

        results.append(EvaluationResult(
            family=family,
            num_states=total,
            accuracy=correct / max(total, 1),
            mean_delta_utility=float(np.mean(deltas)) if deltas else 0.0,
            mean_regret=float(np.mean(regrets)) if regrets else 0.0,
        ))

    return results
