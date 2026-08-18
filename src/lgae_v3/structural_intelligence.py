"""LGAE v5.8 structural-intelligence qualification primitives.

This module deliberately focuses on *decision quality* rather than adding more
geometry operators.  It provides:

* state-grouped counterfactual replay (multiple actions from one state remain
  together for ranking/listwise training),
* a shared graph encoder with an ensemble of candidate-value heads so epistemic
  disagreement is measurable,
* effective-resistance candidate retrieval as a global bottleneck channel,
* procedural graph families with explicit held-out-family qualification, and
* regret/calibration metrics against exact candidate evaluation.

The learned layer is still advisory.  Exact LGAE shadow evaluation and the
GeometryGovernor remain authoritative for mutation certification and commit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any, Callable, Iterable, Sequence
import math
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import GraphBuffers, make_graph_buffers
from .reasoning import (
    CandidateQNetwork,
    CandidateValue,
    ConcreteAction,
    GraphStateEncoder,
)
from .benchmark.tasks import StructuralAction


@dataclass(slots=True)
class ReplayItem:
    graph: GraphBuffers
    z: Tensor
    candidate: ConcreteAction
    delta_utility: float
    risk: float = 0.0
    information_gain: float = 0.0


class StateGroupedReplayBuffer:
    """Replay buffer whose sampling unit is a graph state, not a lone action.

    Ranking objectives require competing actions from the *same* state.  The
    v5.4-v5.7 item-wise buffer could randomly separate those actions into
    different minibatches, making ranking supervision sparse as the replay grew.
    This buffer keeps each state and its counterfactual action set together.
    """

    def __init__(self, capacity_states: int = 20_000):
        self.capacity_states = int(capacity_states)
        self._order: deque[str] = deque()
        self._groups: dict[str, list[ReplayItem]] = {}

    def __len__(self) -> int:
        return len(self._groups)

    @property
    def action_count(self) -> int:
        return sum(len(v) for v in self._groups.values())

    def add_group(
        self,
        graph: GraphBuffers,
        z: Tensor,
        records: Sequence[tuple[ConcreteAction, float, float, float]],
    ) -> str:
        key = graph.state_hash(include_version=False)
        items = [
            ReplayItem(graph.clone(), z.detach().cpu().clone(), c, float(du), float(risk), float(ig))
            for c, du, risk, ig in records
        ]
        if key not in self._groups:
            while len(self._groups) >= self.capacity_states and self._order:
                old = self._order.popleft()
                self._groups.pop(old, None)
            self._order.append(key)
        self._groups[key] = items
        return key

    def sample_groups(self, batch_states: int, rng: random.Random | None = None) -> list[list[ReplayItem]]:
        rng = rng or random
        keys = list(self._groups)
        if not keys:
            return []
        chosen = rng.sample(keys, min(int(batch_states), len(keys)))
        return [self._groups[k] for k in chosen]


class EnsembleStructuralQ(nn.Module):
    """Shared graph encoder + independent Q heads for epistemic uncertainty."""

    def __init__(
        self,
        d_max: int = 64,
        hidden_dim: int = 128,
        message_passing_layers: int = 3,
        members: int = 5,
        seed: int = 0,
    ):
        super().__init__()
        if members < 2:
            raise ValueError("members must be >= 2 for epistemic uncertainty")
        torch.manual_seed(seed)
        self.encoder = GraphStateEncoder(d_max=d_max, hidden_dim=hidden_dim, layers=message_passing_layers)
        self.heads = nn.ModuleList()
        # Ensure independently initialized heads without changing the global API.
        for i in range(int(members)):
            torch.manual_seed(seed + 1009 * (i + 1))
            self.heads.append(CandidateQNetwork(hidden_dim=hidden_dim))
        self.members = int(members)

    def forward(self, graph: GraphBuffers, z: Tensor, candidates: Sequence[ConcreteAction]) -> dict[str, Tensor]:
        node_h, global_h = self.encoder(graph, z)
        outs = [h(candidates, node_h, global_h, graph, z) for h in self.heads]
        means = torch.stack([o["mean"] for o in outs], dim=0)
        aleatoric_vars = torch.stack([torch.exp(o["log_var"]) for o in outs], dim=0)
        risks = torch.stack([o["risk"] for o in outs], dim=0)
        igs = torch.stack([o["ig"] for o in outs], dim=0)
        mean = means.mean(dim=0)
        epistemic_var = means.var(dim=0, unbiased=False)
        aleatoric_var = aleatoric_vars.mean(dim=0)
        total_var = epistemic_var + aleatoric_var
        return {
            "member_means": means,
            "mean": mean,
            "epistemic_std": epistemic_var.clamp_min(0).sqrt(),
            "aleatoric_std": aleatoric_var.clamp_min(1e-12).sqrt(),
            "total_std": total_var.clamp_min(1e-12).sqrt(),
            "risk": risks.mean(dim=0),
            "ig": igs.mean(dim=0),
        }


class StructuralIntelligenceExecutive(nn.Module):
    """Ensemble candidate valuation with state-grouped ranking training."""

    def __init__(
        self,
        d_max: int = 64,
        hidden_dim: int = 128,
        members: int = 5,
        beta_epistemic: float = 1.0,
        beta_aleatoric: float = 0.5,
        risk_weight: float = 0.25,
        ig_weight: float = 0.10,
        lr: float = 3e-4,
        seed: int = 0,
    ):
        super().__init__()
        self.model = EnsembleStructuralQ(d_max=d_max, hidden_dim=hidden_dim, members=members, seed=seed)
        self.beta_epistemic = float(beta_epistemic)
        self.beta_aleatoric = float(beta_aleatoric)
        self.risk_weight = float(risk_weight)
        self.ig_weight = float(ig_weight)
        self.replay = StateGroupedReplayBuffer()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.rng = random.Random(seed)

    @torch.no_grad()
    def predict(self, graph: GraphBuffers, z: Tensor, candidates: Sequence[ConcreteAction]) -> list[CandidateValue]:
        self.eval()
        p = self.model(graph, z, candidates)
        score = (
            p["mean"]
            - self.beta_epistemic * p["epistemic_std"]
            - self.beta_aleatoric * p["aleatoric_std"]
            - self.risk_weight * p["risk"]
            + self.ig_weight * p["ig"]
        )
        out: list[CandidateValue] = []
        for i, c in enumerate(candidates):
            # CandidateValue has one std field; expose total predictive spread
            # there and keep epistemic/aleatoric decomposition in metadata.
            c.metadata = dict(c.metadata)
            c.metadata.update({
                "epistemic_std": float(p["epistemic_std"][i]),
                "aleatoric_std": float(p["aleatoric_std"][i]),
            })
            out.append(CandidateValue(
                c,
                float(p["mean"][i]),
                float(p["total_std"][i]),
                float(p["risk"][i]),
                float(p["ig"][i]),
                float(score[i]),
            ))
        return sorted(out, key=lambda x: x.score, reverse=True)

    def add_counterfactual_group(
        self,
        graph: GraphBuffers,
        z: Tensor,
        outcomes: Sequence[Any],
    ) -> str:
        records = []
        for o in outcomes:
            records.append((
                o.candidate,
                float(o.delta_utility),
                0.0 if bool(o.accepted_by_governor) else 1.0,
                float(getattr(o, "metadata", {}).get("information_gain", 0.0)),
            ))
        return self.replay.add_group(graph, z, records)

    def train_step(self, batch_states: int = 8, ranking_weight: float = 0.5) -> dict[str, float]:
        groups = self.replay.sample_groups(batch_states, self.rng)
        if not groups:
            return {"loss": 0.0, "samples": 0, "states": 0, "ranking_loss": 0.0}
        self.train(); self.optimizer.zero_grad()
        device = next(self.parameters()).device
        value_losses: list[Tensor] = []
        rank_losses: list[Tensor] = []
        sample_count = 0
        for group in groups:
            if not group:
                continue
            graph = GraphBuffers.from_state_dict(group[0].graph.to_state_dict(), device=device)
            z = group[0].z.to(device)
            candidates = [x.candidate for x in group]
            targets = torch.tensor([x.delta_utility for x in group], device=device, dtype=z.dtype)
            risk_targets = torch.tensor([x.risk for x in group], device=device, dtype=z.dtype)
            ig_targets = torch.tensor([x.information_gain for x in group], device=device, dtype=z.dtype)
            node_h, global_h = self.model.encoder(graph, z)
            head_means: list[Tensor] = []
            for h in self.model.heads:
                p = h(candidates, node_h, global_h, graph, z)
                var = torch.exp(p["log_var"])
                nll = 0.5 * ((p["mean"] - targets).square() / var.clamp_min(1e-8) + p["log_var"])
                risk_loss = F.binary_cross_entropy(p["risk"], risk_targets, reduction="none")
                ig_loss = (p["ig"] - ig_targets).square()
                value_losses.append((nll + 0.2 * risk_loss + 0.05 * ig_loss).mean())
                head_means.append(p["mean"])
            ensemble_mean = torch.stack(head_means).mean(0)
            # All pairwise preference constraints in this state, not merely
            # whichever two actions collided in a random item-wise minibatch.
            diffs: list[Tensor] = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if targets[i] == targets[j]:
                        continue
                    if targets[i] > targets[j]:
                        diffs.append(F.softplus(-(ensemble_mean[i] - ensemble_mean[j])))
                    else:
                        diffs.append(F.softplus(-(ensemble_mean[j] - ensemble_mean[i])))
            if diffs:
                rank_losses.append(torch.stack(diffs).mean())
            sample_count += len(group)
        if not value_losses:
            return {"loss": 0.0, "samples": 0, "states": 0, "ranking_loss": 0.0}
        value_loss = torch.stack(value_losses).mean()
        rank_loss = torch.stack(rank_losses).mean() if rank_losses else value_loss.new_zeros(())
        total = value_loss + float(ranking_weight) * rank_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        return {
            "loss": float(total.detach()),
            "value_loss": float(value_loss.detach()),
            "ranking_loss": float(rank_loss.detach()),
            "samples": sample_count,
            "states": len(groups),
        }


def effective_resistance_matrix(graph: GraphBuffers) -> Tensor:
    """Exact effective-resistance matrix for small/qualification graphs."""
    n = graph.num_nodes
    dtype = graph.weight.dtype
    device = graph.weight.device
    A = torch.zeros((n, n), dtype=dtype, device=device)
    ids = torch.where(graph.valid)[0]
    if ids.numel():
        u = graph.src[ids]; v = graph.dst[ids]; w = graph.weight[ids]
        A[u, v] += w; A[v, u] += w
    L = torch.diag(A.sum(-1)) - A
    Lp = torch.linalg.pinv(L)
    diag = torch.diag(Lp)
    R = diag[:, None] + diag[None, :] - 2.0 * Lp
    return R.clamp_min(0)


def effective_resistance_candidates(graph: GraphBuffers, z: Tensor, top_k: int = 8) -> list[ConcreteAction]:
    """Propose nonedges with highest global effective resistance."""
    if graph.num_nodes < 2:
        return []
    R = effective_resistance_matrix(graph)
    existing = {
        tuple(sorted((int(graph.src[i]), int(graph.dst[i]))))
        for i in torch.where(graph.valid)[0].tolist()
    }
    ranked: list[tuple[float, int, int]] = []
    for u in range(graph.num_nodes):
        for v in range(u + 1, graph.num_nodes):
            if (u, v) in existing:
                continue
            ranked.append((float(R[u, v]), u, v))
    ranked.sort(reverse=True)
    out = []
    for score, u, v in ranked[: int(top_k)]:
        dist = float(torch.linalg.vector_norm(z[u] - z[v])) if z.numel() else 1.0
        out.append(ConcreteAction(
            StructuralAction.ADD_EDGE,
            {"u": u, "v": v, "weight": 1.0, "length": max(dist, 1e-3)},
            channel="effective_resistance",
            prior_score=score,
        ))
    return out


@dataclass(slots=True)
class ProceduralCase:
    family: str
    graph: GraphBuffers
    z: Tensor
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ProceduralGraphFactory:
    """Generate topology families for train/held-out structural evaluation."""

    families = ("erdos_renyi", "block", "tree", "small_world", "scale_free", "geometric")

    def __init__(self, latent_dim: int = 8):
        self.latent_dim = int(latent_dim)

    @staticmethod
    def _connected_tree_edges(n: int, rng: random.Random) -> set[tuple[int, int]]:
        return {tuple(sorted((i, rng.randrange(i)))) for i in range(1, n)}

    def make(self, family: str, n: int = 16, seed: int = 0) -> ProceduralCase:
        if family not in self.families:
            raise ValueError(f"unknown family: {family}")
        rng = random.Random(seed)
        torch.manual_seed(seed)
        edges: set[tuple[int, int]] = set()
        if family == "tree":
            edges = self._connected_tree_edges(n, rng)
        elif family == "erdos_renyi":
            edges = self._connected_tree_edges(n, rng)
            p = min(0.35, max(2.5 / max(n - 1, 1), 0.08))
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        edges.add((u, v))
        elif family == "block":
            split = n // 2
            for lo, hi in ((0, split), (split, n)):
                for u in range(lo, hi):
                    for v in range(u + 1, hi):
                        if rng.random() < 0.45:
                            edges.add((u, v))
            edges.add((max(0, split - 1), min(n - 1, split)))
            edges |= self._connected_tree_edges(n, rng)
        elif family == "small_world":
            k = min(4, max(2, n - 1))
            for u in range(n):
                for d in range(1, k // 2 + 1):
                    v = (u + d) % n
                    edges.add(tuple(sorted((u, v))))
            # Sparse deterministic-ish shortcuts.
            for _ in range(max(1, n // 6)):
                u, v = rng.sample(range(n), 2)
                edges.add(tuple(sorted((u, v))))
        elif family == "scale_free":
            # Lightweight preferential-attachment generator, m=2.
            if n >= 2:
                edges.add((0, 1))
            degree = [1, 1] + [0] * max(0, n - 2)
            for v in range(2, n):
                targets: set[int] = set()
                while len(targets) < min(2, v):
                    weights = [degree[i] + 1 for i in range(v)]
                    total = sum(weights); r = rng.uniform(0, total); acc = 0.0
                    for i, w in enumerate(weights):
                        acc += w
                        if r <= acc:
                            targets.add(i); break
                for u in targets:
                    edges.add((u, v)); degree[u] += 1; degree[v] += 1
        elif family == "geometric":
            xy = torch.rand(n, 2)
            # Ensure connected with nearest-neighbor backbone then radius edges.
            for i in range(1, n):
                d = torch.linalg.vector_norm(xy[:i] - xy[i], dim=-1)
                u = int(torch.argmin(d)); edges.add((u, i))
            for u in range(n):
                for v in range(u + 1, n):
                    if float(torch.linalg.vector_norm(xy[u] - xy[v])) < 0.35:
                        edges.add((u, v))
        weighted = [(u, v, 1.0) for u, v in sorted(edges)]
        g = make_graph_buffers(n, weighted, capacity=max(len(weighted) + n, n * 2))
        z = torch.randn(n, self.latent_dim)
        if family == "block":
            z[: n // 2, 0] += 2.0; z[n // 2 :, 0] -= 2.0
        return ProceduralCase(family, g, z, seed, {"n": n, "edges": len(edges)})

    def sample(self, families: Sequence[str], count: int, n_range: tuple[int, int] = (10, 24), seed: int = 0) -> list[ProceduralCase]:
        rng = random.Random(seed)
        out = []
        for i in range(int(count)):
            fam = families[i % len(families)] if families else self.families[i % len(self.families)]
            n = rng.randint(int(n_range[0]), int(n_range[1]))
            out.append(self.make(fam, n=n, seed=seed + 7919 * i))
        return out


def spectral_utility(graph: GraphBuffers, z: Tensor | None = None, edge_cost: float = 0.002) -> float:
    """Reference utility used by the v5.8 structural-regret qualification."""
    from .operators import spectral_gap_graphbuffers
    gap, _ = spectral_gap_graphbuffers(graph)
    return float(gap) - float(edge_cost) * float(graph.edge_count)


def exact_candidate_deltas(
    graph: GraphBuffers,
    z: Tensor,
    candidates: Sequence[ConcreteAction],
    utility_fn: Callable[[GraphBuffers, Tensor], float] = spectral_utility,
) -> list[float]:
    """Exact candidate evaluation for safe, small qualification graphs.

    This intentionally bypasses the learned model.  It is an oracle for the
    *benchmark candidate set*, not authority for production commits.
    """
    from .action_bridge import action_to_mutation
    before = float(utility_fn(graph, z))
    vals: list[float] = []
    for c in candidates:
        if c.action == StructuralAction.NO_OP:
            vals.append(0.0); continue
        m = action_to_mutation(c.action, graph, z, **c.target)
        if m is None:
            vals.append(float("-inf")); continue
        shadow = graph.clone()
        try:
            m.apply(shadow)
            vals.append(float(utility_fn(shadow, z)) - before)
        except Exception:
            vals.append(float("-inf"))
    return vals


@dataclass(slots=True)
class RegretResult:
    regret: float
    oracle_delta: float
    chosen_delta: float
    oracle_index: int
    chosen_index: int


def candidate_regret(predicted_scores: Sequence[float], exact_deltas: Sequence[float]) -> RegretResult:
    if len(predicted_scores) != len(exact_deltas) or not predicted_scores:
        raise ValueError("score/delta lengths must match and be non-empty")
    oi = max(range(len(exact_deltas)), key=lambda i: exact_deltas[i])
    ci = max(range(len(predicted_scores)), key=lambda i: predicted_scores[i])
    oracle = float(exact_deltas[oi]); chosen = float(exact_deltas[ci])
    return RegretResult(max(0.0, oracle - chosen), oracle, chosen, oi, ci)


def uncertainty_calibration(errors: Sequence[float], predicted_std: Sequence[float]) -> dict[str, float]:
    """Small distribution-free calibration summary for qualification reports."""
    if len(errors) != len(predicted_std) or not errors:
        raise ValueError("errors/std lengths must match and be non-empty")
    e = torch.tensor([abs(float(x)) for x in errors])
    s = torch.tensor([max(float(x), 1e-8) for x in predicted_std])
    return {
        "mean_abs_error": float(e.mean()),
        "mean_predicted_std": float(s.mean()),
        "coverage_1sigma": float((e <= s).float().mean()),
        "coverage_2sigma": float((e <= 2 * s).float().mean()),
        "normalized_error": float((e / s).mean()),
    }

# ---------------------------------------------------------------------------
# v5.8.1 structural-intelligence hardening
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StructuralRegime:
    lambda2: float
    degree_cv: float
    clustering: float
    resistance_q75: float
    density: float


def structural_regime_features(graph: GraphBuffers) -> StructuralRegime:
    """Cheap structural descriptors used for replay stratification.

    Deliberately avoids exact treewidth/girth in the hot path.  The descriptors
    are deterministic and inexpensive on the small/medium graphs used by the
    structural qualification harness.
    """
    from .operators import spectral_gap_graphbuffers
    import networkx as nx
    n = graph.num_nodes
    ids = torch.where(graph.valid)[0]
    deg = torch.zeros(n, dtype=torch.float64)
    G = nx.Graph(); G.add_nodes_from(range(n))
    for i in ids.tolist():
        u, v = int(graph.src[i]), int(graph.dst[i])
        deg[u] += 1; deg[v] += 1; G.add_edge(u, v)
    mean_deg = float(deg.mean()) if n else 0.0
    degree_cv = float(deg.std(unbiased=False) / max(mean_deg, 1e-8)) if n else 0.0
    try:
        lam2, _ = spectral_gap_graphbuffers(graph); lam2 = float(lam2)
    except Exception:
        lam2 = 0.0
    clustering = float(nx.average_clustering(G)) if n > 1 and G.number_of_edges() else 0.0
    density = float(nx.density(G)) if n > 1 else 0.0
    try:
        R = effective_resistance_matrix(graph).detach().cpu()
        tri = R[torch.triu(torch.ones_like(R, dtype=torch.bool), diagonal=1)]
        resistance_q75 = float(torch.quantile(tri, 0.75)) if tri.numel() else 0.0
    except Exception:
        resistance_q75 = 0.0
    return StructuralRegime(lam2, degree_cv, clustering, resistance_q75, density)


class SpectralStratifiedReplayBuffer(StateGroupedReplayBuffer):
    """State-grouped replay sampled across structural regimes.

    Bins are deliberately coarse.  We want diversity, not a brittle taxonomy.
    """
    def __init__(self, capacity_states: int = 20_000, capacity_per_bin: int = 2_000):
        super().__init__(capacity_states=capacity_states)
        self.capacity_per_bin = int(capacity_per_bin)
        self._bin_of: dict[str, tuple[int, ...]] = {}
        self._bins: dict[tuple[int, ...], deque[str]] = defaultdict(deque)

    @staticmethod
    def _bucket(r: StructuralRegime) -> tuple[int, ...]:
        def q(x: float, cuts: tuple[float, ...]) -> int:
            return sum(float(x) > c for c in cuts)
        return (
            q(r.lambda2, (0.03, 0.10, 0.25, 0.6)),
            q(r.degree_cv, (0.25, 0.5, 1.0)),
            q(r.clustering, (0.05, 0.2, 0.5)),
            q(r.resistance_q75, (1.0, 2.5, 5.0, 10.0)),
            q(r.density, (0.08, 0.16, 0.3)),
        )

    def add_group(self, graph: GraphBuffers, z: Tensor, records: Sequence[tuple[ConcreteAction, float, float, float]]) -> str:
        key = super().add_group(graph, z, records)
        b = self._bucket(structural_regime_features(graph))
        old = self._bin_of.get(key)
        if old is not None and old != b:
            try: self._bins[old].remove(key)
            except ValueError: pass
        self._bin_of[key] = b
        dq = self._bins[b]
        if key not in dq: dq.append(key)
        while len(dq) > self.capacity_per_bin:
            stale = dq.popleft()
            if stale in self._groups:
                self._groups.pop(stale, None)
                self._bin_of.pop(stale, None)
                try: self._order.remove(stale)
                except ValueError: pass
        return key

    def sample_groups(self, batch_states: int, rng: random.Random | None = None) -> list[list[ReplayItem]]:
        rng = rng or random
        active = [list(v) for v in self._bins.values() if v]
        if not active: return []
        chosen: list[str] = []
        # Round-robin shuffled bins gives structural diversity before repetition.
        rng.shuffle(active)
        while len(chosen) < int(batch_states):
            progressed = False
            for keys in active:
                avail = [k for k in keys if k not in chosen and k in self._groups]
                if avail:
                    chosen.append(rng.choice(avail)); progressed = True
                    if len(chosen) >= int(batch_states): break
            if not progressed: break
        return [self._groups[k] for k in chosen]


class RandomizedPriorEnsembleQ(nn.Module):
    """Q ensemble with frozen randomized prior functions.

    Each member has an independently initialized trainable head and frozen prior
    head.  The graph encoder is shared for efficiency; persistent prior diversity
    prevents all member values collapsing solely because the shared representation
    is similar on an OOD topology.
    """
    def __init__(self, d_max: int = 64, hidden_dim: int = 128, message_passing_layers: int = 3,
                 members: int = 5, prior_scale: float = 0.5, seed: int = 0):
        super().__init__()
        if members < 2: raise ValueError("members must be >= 2")
        torch.manual_seed(seed)
        self.encoder = GraphStateEncoder(d_max=d_max, hidden_dim=hidden_dim, layers=message_passing_layers)
        self.heads = nn.ModuleList(); self.priors = nn.ModuleList(); self.members = int(members)
        self.prior_scale = float(prior_scale)
        for i in range(self.members):
            torch.manual_seed(seed + 1009 * (i + 1)); self.heads.append(CandidateQNetwork(hidden_dim=hidden_dim))
            torch.manual_seed(seed + 65537 * (i + 1)); p = CandidateQNetwork(hidden_dim=hidden_dim)
            for x in p.parameters(): x.requires_grad_(False)
            self.priors.append(p)

    def member_output(self, i: int, candidates: Sequence[ConcreteAction], node_h: Tensor, global_h: Tensor,
                      graph: GraphBuffers, z: Tensor) -> dict[str, Tensor]:
        o = self.heads[i](candidates, node_h, global_h, graph, z)
        p = self.priors[i](candidates, node_h, global_h, graph, z)
        # Priors perturb value only; risk/IG/aleatoric heads remain trainable measurements.
        return {**o, "mean": o["mean"] + self.prior_scale * p["mean"].detach()}

    def forward(self, graph: GraphBuffers, z: Tensor, candidates: Sequence[ConcreteAction]) -> dict[str, Tensor]:
        node_h, global_h = self.encoder(graph, z)
        outs = [self.member_output(i, candidates, node_h, global_h, graph, z) for i in range(self.members)]
        means = torch.stack([o['mean'] for o in outs]); av = torch.stack([torch.exp(o['log_var']) for o in outs])
        risks = torch.stack([o['risk'] for o in outs]); igs = torch.stack([o['ig'] for o in outs])
        epi = means.var(0, unbiased=False); alea = av.mean(0)
        return {'member_means':means,'mean':means.mean(0),'epistemic_std':epi.clamp_min(0).sqrt(),
                'aleatoric_std':alea.clamp_min(1e-12).sqrt(),'total_std':(epi+alea).clamp_min(1e-12).sqrt(),
                'risk':risks.mean(0),'ig':igs.mean(0)}


class ContrastiveCandidateRetriever(nn.Module):
    """Dual-encoder nonedge retriever trained from exact/oracle advantages."""
    def __init__(self, hidden_dim: int = 128, embed_dim: int = 64, temperature: float = 0.10):
        super().__init__(); self.temperature=float(temperature)
        self.left = nn.Linear(hidden_dim, embed_dim); self.right = nn.Linear(hidden_dim, embed_dim)

    def logits(self, node_h: Tensor) -> Tensor:
        a=F.normalize(self.left(node_h),dim=-1); b=F.normalize(self.right(node_h),dim=-1)
        return a @ b.T / max(self.temperature,1e-6)

    def loss(self, node_h: Tensor, oracle_advantage: Tensor, valid_nonedges: Tensor) -> Tensor:
        logits=self.logits(node_h)
        mask=valid_nonedges.bool()
        # Symmetric target from positive exact advantage.  Fall back to top oracle
        # nonedge when no candidate has positive advantage.
        target=torch.where(mask, oracle_advantage.clamp_min(0), torch.zeros_like(oracle_advantage))
        if float(target.sum()) <= 0 and mask.any():
            masked=torch.where(mask, oracle_advantage, torch.full_like(oracle_advantage,-torch.inf))
            idx=torch.argmax(masked.flatten()); target.flatten()[idx]=1.0
        target=(target+target.T)/2
        target=target/target.sum().clamp_min(1e-8)
        flat_mask=mask.flatten(); logp=F.log_softmax(torch.where(mask,logits,torch.full_like(logits,-1e9)).flatten(),dim=0)
        return -(target.flatten()[flat_mask]*logp[flat_mask]).sum()

    @torch.no_grad()
    def candidates(self, graph: GraphBuffers, z: Tensor, node_h: Tensor, top_k: int = 16) -> list[ConcreteAction]:
        scores=self.logits(node_h); existing={tuple(sorted((int(graph.src[i]),int(graph.dst[i])))) for i in torch.where(graph.valid)[0].tolist()}
        rows=[]
        for u in range(graph.num_nodes):
            for v in range(u+1,graph.num_nodes):
                if (u,v) in existing: continue
                rows.append((float((scores[u,v]+scores[v,u])/2),u,v))
        rows.sort(reverse=True); out=[]
        for s,u,v in rows[:int(top_k)]:
            d=float(torch.linalg.vector_norm(z[u]-z[v]))
            out.append(ConcreteAction(StructuralAction.ADD_EDGE,{'u':u,'v':v,'weight':1.0,'length':max(d,1e-3)},channel='learned_retrieval',prior_score=s))
        return out


def fosr_candidates(graph: GraphBuffers, z: Tensor, top_k: int = 8) -> list[ConcreteAction]:
    """Qualification-grade FoSR-style spectral proposals.

    For bounded benchmark graphs this evaluates the exact lambda2 gain of each
    nonedge.  It is intentionally an expensive reference proposal mechanism,
    not the production hot-path implementation.
    """
    from .operators import spectral_gap_graphbuffers
    base=float(spectral_gap_graphbuffers(graph)[0]); existing={tuple(sorted((int(graph.src[i]),int(graph.dst[i])))) for i in torch.where(graph.valid)[0].tolist()}
    rows=[]
    for u in range(graph.num_nodes):
        for v in range(u+1,graph.num_nodes):
            if (u,v) in existing: continue
            g=graph.clone()
            d=float(torch.linalg.vector_norm(z[u]-z[v]))
            from .mutations import AddEdge
            try:
                AddEdge(u,v,1.0,max(d,1e-3)).apply(g)
            except RuntimeError:
                continue
            try: gain=float(spectral_gap_graphbuffers(g)[0])-base
            except Exception: continue
            rows.append((gain,u,v,d))
    rows.sort(reverse=True)
    return [ConcreteAction(StructuralAction.ADD_EDGE,{'u':u,'v':v,'weight':1.0,'length':max(d,1e-3)},channel='fosr_reference',prior_score=s) for s,u,v,d in rows[:int(top_k)]]


def forman_flow_candidates(graph: GraphBuffers, z: Tensor, top_k: int = 8) -> list[ConcreteAction]:
    """BORF-style curvature-flow proposal baseline.

    This is explicitly a BORF-*style* reference, not a claim of paper-exact BORF.
    It links endpoints adjacent to the most negatively curved AF3 edges and ranks
    nonedges by the curvature deficit of their endpoint neighborhoods.
    """
    import networkx as nx
    from .curvature.forman import af3_edge
    G=nx.Graph(); G.add_nodes_from(range(graph.num_nodes)); existing=set()
    for i in torch.where(graph.valid)[0].tolist():
        u,v=int(graph.src[i]),int(graph.dst[i]); existing.add(tuple(sorted((u,v)))); G.add_edge(u,v)
    deficit=torch.zeros(graph.num_nodes)
    for u,v in G.edges():
        k=af3_edge(G,u,v); d=max(0.0,-float(k)); deficit[u]+=d; deficit[v]+=d
    rows=[]
    for u in range(graph.num_nodes):
        for v in range(u+1,graph.num_nodes):
            if (u,v) in existing: continue
            score=float(deficit[u]+deficit[v]); dist=float(torch.linalg.vector_norm(z[u]-z[v])); rows.append((score,u,v,dist))
    rows.sort(reverse=True)
    return [ConcreteAction(StructuralAction.ADD_EDGE,{'u':u,'v':v,'weight':1.0,'length':max(d,1e-3)},channel='forman_flow_reference',prior_score=s) for s,u,v,d in rows[:int(top_k)]]


def merge_candidate_channels(*channels: Sequence[ConcreteAction], max_candidates: int = 64) -> list[ConcreteAction]:
    out=[]; seen=set()
    # NOOP is always admissible and first.
    noop=ConcreteAction(StructuralAction.NO_OP,channel='baseline'); out.append(noop); seen.add(noop.key())
    for ch in channels:
        for c in ch:
            if c.key() in seen: continue
            seen.add(c.key()); out.append(c)
            if len(out)>=int(max_candidates): return out
    return out


class ConservativeStructuralExecutive(StructuralIntelligenceExecutive):
    """v5.8.1 executive with anchored priors, stratified replay and LCB arbitration."""
    def __init__(self, d_max: int=64, hidden_dim: int=128, members: int=5, prior_scale: float=.5,
                 beta_lcb: float=1.96, lr: float=3e-4, seed: int=0):
        nn.Module.__init__(self)
        self.model=RandomizedPriorEnsembleQ(d_max=d_max,hidden_dim=hidden_dim,members=members,prior_scale=prior_scale,seed=seed)
        self.beta_epistemic=float(beta_lcb); self.beta_aleatoric=0.25; self.risk_weight=.25; self.ig_weight=.10
        self.replay=SpectralStratifiedReplayBuffer(); self.optimizer=torch.optim.Adam([p for p in self.parameters() if p.requires_grad],lr=lr); self.rng=random.Random(seed)

    @torch.no_grad()
    def predict(self, graph: GraphBuffers, z: Tensor, candidates: Sequence[ConcreteAction]) -> list[CandidateValue]:
        self.eval(); p=self.model(graph,z,candidates)
        score=p['mean']-self.beta_epistemic*p['epistemic_std']-self.beta_aleatoric*p['aleatoric_std']-self.risk_weight*p['risk']+self.ig_weight*p['ig']
        vals=[]
        for i,c in enumerate(candidates):
            c.metadata=dict(c.metadata); c.metadata.update({'epistemic_std':float(p['epistemic_std'][i]),'aleatoric_std':float(p['aleatoric_std'][i]),'lcb_beta':self.beta_epistemic})
            vals.append(CandidateValue(c,float(p['mean'][i]),float(p['total_std'][i]),float(p['risk'][i]),float(p['ig'][i]),float(score[i])))
        return sorted(vals,key=lambda x:x.score,reverse=True)

    def train_step(self, batch_states: int=8, ranking_weight: float=.5) -> dict[str,float]:
        groups=self.replay.sample_groups(batch_states,self.rng)
        if not groups: return {'loss':0.0,'samples':0,'states':0,'ranking_loss':0.0}
        self.train(); self.optimizer.zero_grad(); device=next(self.parameters()).device; value_losses=[]; rank_losses=[]; sample_count=0
        for group in groups:
            graph=GraphBuffers.from_state_dict(group[0].graph.to_state_dict(),device=device); z=group[0].z.to(device); cs=[x.candidate for x in group]
            targets=torch.tensor([x.delta_utility for x in group],device=device,dtype=z.dtype); rt=torch.tensor([x.risk for x in group],device=device,dtype=z.dtype); it=torch.tensor([x.information_gain for x in group],device=device,dtype=z.dtype)
            nh,gh=self.model.encoder(graph,z); means=[]
            for i in range(self.model.members):
                p=self.model.member_output(i,cs,nh,gh,graph,z); var=torch.exp(p['log_var']); nll=.5*((p['mean']-targets).square()/var.clamp_min(1e-8)+p['log_var'])
                value_losses.append((nll+.2*F.binary_cross_entropy(p['risk'],rt,reduction='none')+.05*(p['ig']-it).square()).mean()); means.append(p['mean'])
            em=torch.stack(means).mean(0); diffs=[]
            for i in range(len(group)):
                for j in range(i+1,len(group)):
                    if targets[i]==targets[j]: continue
                    diffs.append(F.softplus(-((em[i]-em[j]) if targets[i]>targets[j] else (em[j]-em[i]))))
            if diffs: rank_losses.append(torch.stack(diffs).mean())
            sample_count+=len(group)
        vl=torch.stack(value_losses).mean(); rl=torch.stack(rank_losses).mean() if rank_losses else vl.new_zeros(()); total=vl+float(ranking_weight)*rl; total.backward(); torch.nn.utils.clip_grad_norm_(self.parameters(),5.0); self.optimizer.step()
        return {'loss':float(total.detach()),'value_loss':float(vl.detach()),'ranking_loss':float(rl.detach()),'samples':sample_count,'states':len(groups)}

# ---------------------------------------------------------------------------
# v5.8.2 scalable structural intelligence
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CalibrationResult:
    scale: float
    exponent: float
    nll_before: float
    nll_after: float


class EpistemicScaleCalibrator(nn.Module):
    """Monotone online calibration for ensemble epistemic spread.

    Maps raw sigma to ``scale * sigma**exponent`` and fits the two positive
    parameters against observed absolute Q errors using Gaussian NLL.
    """
    def __init__(self, init_scale: float = 1.0, init_exponent: float = 1.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(init_scale, 1e-6)), dtype=torch.float32))
        self.log_exponent = nn.Parameter(torch.tensor(math.log(max(init_exponent, 1e-6)), dtype=torch.float32))

    def forward(self, raw_std: Tensor) -> Tensor:
        scale = self.log_scale.exp()
        exponent = self.log_exponent.exp().clamp(0.25, 4.0)
        return scale * raw_std.clamp_min(1e-8).pow(exponent)

    def fit(self, raw_std: Tensor, errors: Tensor, steps: int = 100, lr: float = 0.03) -> CalibrationResult:
        raw_std = raw_std.detach().float().clamp_min(1e-8)
        errors = errors.detach().float().abs()
        def nll(std: Tensor) -> Tensor:
            var = std.square().clamp_min(1e-10)
            return .5 * (torch.log(var) + errors.square() / var).mean()
        before = float(nll(raw_std))
        opt = torch.optim.Adam(self.parameters(), lr=float(lr))
        for _ in range(int(steps)):
            opt.zero_grad(); loss = nll(self(raw_std)); loss.backward(); opt.step()
        after = float(nll(self(raw_std)).detach())
        return CalibrationResult(float(self.log_scale.exp().detach()), float(self.log_exponent.exp().detach()), before, after)


def wl_graph_hash(graph: GraphBuffers, iterations: int = 2) -> str:
    """Cheap 1-WL redundancy hash. Used as a replay gate, never as proof of isomorphism."""
    import hashlib
    nbrs=[[] for _ in range(graph.num_nodes)]
    for i in torch.where(graph.valid)[0].tolist():
        u,v=int(graph.src[i]),int(graph.dst[i]); nbrs[u].append(v); nbrs[v].append(u)
    labels=[str(len(n)) for n in nbrs]
    for _ in range(int(iterations)):
        nxt=[]
        for u in range(graph.num_nodes):
            token=labels[u]+'|'+','.join(sorted(labels[v] for v in nbrs[u]))
            nxt.append(hashlib.sha256(token.encode()).hexdigest()[:16])
        labels=nxt
    return hashlib.sha256('|'.join(sorted(labels)).encode()).hexdigest()


class WLDeduplicatedSpectralReplayBuffer(SpectralStratifiedReplayBuffer):
    """Spectral replay with bounded repetitions of WL-equivalent structural states."""
    def __init__(self, *args, max_per_wl_hash: int = 3, **kwargs):
        super().__init__(*args, **kwargs); self.max_per_wl_hash=int(max_per_wl_hash); self._wl_keys=defaultdict(deque); self._wl_of={}

    def add_group(self, graph: GraphBuffers, z: Tensor, records: Sequence[tuple[ConcreteAction,float,float,float]]) -> str:
        key=graph.state_hash(include_version=False); wh=wl_graph_hash(graph)
        if key not in self._groups and len(self._wl_keys[wh]) >= self.max_per_wl_hash:
            stale=self._wl_keys[wh].popleft(); self._groups.pop(stale,None); self._bin_of.pop(stale,None); self._wl_of.pop(stale,None)
            try: self._order.remove(stale)
            except ValueError: pass
            for dq in self._bins.values():
                try: dq.remove(stale)
                except ValueError: pass
        key=super().add_group(graph,z,records); self._wl_of[key]=wh
        if key not in self._wl_keys[wh]: self._wl_keys[wh].append(key)
        return key


class ANNCandidateRetriever(ContrastiveCandidateRetriever):
    """Subquadratic inference path for learned structural candidate retrieval.

    Training may still use dense contrastive logits on bounded minibatch graphs;
    production inference projects node embeddings then queries ANNNeighborIndex.
    """
    def __init__(self, hidden_dim: int=128, embed_dim: int=64, temperature: float=.10,
                 ann_backend: str='auto', ann_candidates: int=96):
        super().__init__(hidden_dim,embed_dim,temperature); self.ann_backend=ann_backend; self.ann_candidates=int(ann_candidates)

    @torch.no_grad()
    def candidates(self, graph: GraphBuffers, z: Tensor, node_h: Tensor, top_k: int=16, neighbors_per_node: int=8) -> list[ConcreteAction]:
        from .ann_index import ANNNeighborIndex
        left=F.normalize(self.left(node_h),dim=-1); right=F.normalize(self.right(node_h),dim=-1)
        # Symmetric retrieval cloud keeps pair scoring compatible with the dual encoder.
        emb=F.normalize((left+right)*.5,dim=-1)
        idx=ANNNeighborIndex(dim=int(emb.shape[-1]),n_candidates=max(self.ann_candidates,neighbors_per_node*4),n_final=neighbors_per_node,backend=self.ann_backend)
        _, inds=idx.search(emb,neighbors_per_node)
        existing={tuple(sorted((int(graph.src[i]),int(graph.dst[i])))) for i in torch.where(graph.valid)[0].tolist()}
        rows={}
        for u in range(graph.num_nodes):
            for vv in inds[u].tolist():
                v=int(vv)
                if v<0 or v==u: continue
                a,b=sorted((u,v))
                if (a,b) in existing: continue
                s=float((left[a]@right[b]+left[b]@right[a])/(2*max(self.temperature,1e-6)))
                rows[(a,b)]=max(rows.get((a,b),-float('inf')),s)
        ranked=sorted(((s,u,v) for (u,v),s in rows.items()),reverse=True)[:int(top_k)]
        out=[]
        for s,u,v in ranked:
            d=float(torch.linalg.vector_norm(z[u]-z[v])); out.append(ConcreteAction(StructuralAction.ADD_EDGE,{'u':u,'v':v,'weight':1.0,'length':max(d,1e-3)},channel='learned_ann',prior_score=s))
        return out


def approximate_fosr_candidates(graph: GraphBuffers, z: Tensor, top_k: int=16) -> list[ConcreteAction]:
    """Production FoSR channel using first-order Fiedler perturbation.

    For unit edge addition, delta lambda2 is approximated by (f_u-f_v)^2.
    This avoids cloning/re-eigendecomposing the graph for every nonedge.
    """
    from .operators import symmetric_normalized_laplacian_sparse
    L=symmetric_normalized_laplacian_sparse(graph).to_dense()
    vals, vecs=torch.linalg.eigh(L)
    f=vecs[:,1] if graph.num_nodes > 1 else torch.zeros(graph.num_nodes, dtype=z.dtype, device=z.device)
    existing={tuple(sorted((int(graph.src[i]),int(graph.dst[i])))) for i in torch.where(graph.valid)[0].tolist()}
    # top-k endpoint separation can be found from extremes of the Fiedler vector;
    # bounded oversampling avoids materializing all N^2 scores.
    m=min(graph.num_nodes,max(8,int(math.ceil(math.sqrt(max(top_k,1)*8)))))
    lo=torch.argsort(f)[:m].tolist(); hi=torch.argsort(f,descending=True)[:m].tolist(); rows={}
    for u in lo:
        for v in hi:
            if u==v: continue
            a,b=sorted((int(u),int(v)))
            if (a,b) in existing: continue
            rows[(a,b)]=float((f[a]-f[b]).square())
    ranked=sorted(((s,u,v) for (u,v),s in rows.items()),reverse=True)[:int(top_k)]
    out=[]
    for s,u,v in ranked:
        d=float(torch.linalg.vector_norm(z[u]-z[v])); out.append(ConcreteAction(StructuralAction.ADD_EDGE,{'u':u,'v':v,'weight':1.0,'length':max(d,1e-3)},channel='fosr_approx',prior_score=s))
    return out


def contextual_lcb_beta(base_beta: float, *, epistemic_std: float, risk: float, reversibility: float=1.0,
                        governor_margin: float=1.0, ood_score: float=0.0) -> float:
    """Action-conditional conservatism: more risk/OOD, less reversibility => larger beta."""
    rev=max(0.0,min(1.0,float(reversibility))); margin=max(0.0,min(1.0,float(governor_margin)))
    factor=(1.0 + .75*max(0.0,float(risk)) + .75*max(0.0,float(ood_score)) + .25*max(0.0,float(epistemic_std)))
    factor*= (1.25-.5*rev) * (1.25-.5*margin)
    return max(.10,float(base_beta)*factor)


class ScalableStructuralExecutive(ConservativeStructuralExecutive):
    """v5.8.2 executive: calibrated epistemic spread + contextual LCB + WL replay."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.replay=WLDeduplicatedSpectralReplayBuffer(); self.epistemic_calibrator=EpistemicScaleCalibrator()

    @torch.no_grad()
    def predict(self, graph: GraphBuffers, z: Tensor, candidates: Sequence[ConcreteAction]) -> list[CandidateValue]:
        self.eval(); p=self.model(graph,z,candidates); cal=self.epistemic_calibrator(p['epistemic_std'])
        vals=[]
        for i,c in enumerate(candidates):
            meta=dict(c.metadata); risk=float(p['risk'][i]); epi=float(cal[i]); rev=float(meta.get('reversibility',1.0 if c.action!=StructuralAction.PRUNE_EDGE else .25)); margin=float(meta.get('governor_margin',1.0)); ood=float(meta.get('ood_score',0.0))
            beta=contextual_lcb_beta(self.beta_epistemic,epistemic_std=epi,risk=risk,reversibility=rev,governor_margin=margin,ood_score=ood)
            score=float(p['mean'][i])-beta*epi-self.beta_aleatoric*float(p['aleatoric_std'][i])-self.risk_weight*risk+self.ig_weight*float(p['ig'][i])
            c.metadata=meta; c.metadata.update({'epistemic_std_raw':float(p['epistemic_std'][i]),'epistemic_std':epi,'lcb_beta':beta})
            vals.append(CandidateValue(c,float(p['mean'][i]),float((cal[i].square()+p['aleatoric_std'][i].square()).sqrt()),risk,float(p['ig'][i]),score))
        return sorted(vals,key=lambda x:x.score,reverse=True)
