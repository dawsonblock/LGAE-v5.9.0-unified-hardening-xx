"""LGAE v5.4 counterfactual structural reasoning executive.

This module changes the structural-learning problem from action-class
classification to concrete candidate valuation::

    (state, concrete structural intervention) -> outcome distribution

The learned model is a proposal/ranking mechanism only.  The existing
GeometryGovernor/LGAEEngine remain the sole authority for certification and
commit.  Candidate generation is deliberately multi-channel so a learned
retriever cannot silently make a useful intervention unreachable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Callable, Iterable, Sequence
import math
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .action_bridge import action_to_mutation
from .benchmark.tasks import StructuralAction
from .types import GraphBuffers, MutationDecision


EDGE_ACTIONS = (
    StructuralAction.ADD_EDGE,
    StructuralAction.PRUNE_EDGE,
    StructuralAction.REWEIGHT_AFFINITY,
    StructuralAction.REWEIGHT_LENGTH,
    StructuralAction.COUPLED_REWEIGHT,
)
REASONING_ACTIONS = (StructuralAction.NO_OP,) + EDGE_ACTIONS
_REASON_ACTION_TO_IDX = {a: i for i, a in enumerate(REASONING_ACTIONS)}


@dataclass(slots=True)
class ConcreteAction:
    """A fully specified structural intervention candidate."""

    action: StructuralAction
    target: dict[str, Any] = field(default_factory=dict)
    channel: str = "unknown"
    prior_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[Any, ...]:
        if self.action == StructuralAction.NO_OP:
            return (self.action.value,)
        u = self.target.get("u")
        v = self.target.get("v")
        if u is not None and v is not None:
            u, v = sorted((int(u), int(v)))
        factor = self.target.get("factor")
        weight = self.target.get("weight")
        length = self.target.get("length")
        # Quantize continuous proposal values only for deduplication.
        q = lambda x: None if x is None else round(float(x), 6)
        return (self.action.value, u, v, q(factor), q(weight), q(length))


@dataclass(slots=True)
class CandidateValue:
    candidate: ConcreteAction
    mean_delta_utility: float
    std_delta_utility: float
    risk: float
    information_gain: float
    score: float
    memory_prior: float = 0.0

    @property
    def lcb(self) -> float:
        return self.mean_delta_utility - self.std_delta_utility


@dataclass(slots=True)
class CounterfactualOutcome:
    candidate: ConcreteAction
    delta_utility: float
    utility_before: float
    utility_after: float
    accepted_by_governor: bool
    decision: str
    graph_hash_before: str
    graph_hash_after: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReasoningPlan:
    ranked: list[CandidateValue]
    selected: CandidateValue
    candidates_considered: int
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphStateEncoder(nn.Module):
    """Dependency-free permutation-equivariant graph encoder.

    It intentionally uses only PyTorch scatter/index_add operations rather than
    requiring PyG/DGL.  Latent state is padded/truncated to ``d_max`` so the
    network shape is independent of the active fiber width.
    """

    def __init__(self, d_max: int, hidden_dim: int = 128, layers: int = 3):
        super().__init__()
        self.d_max = int(d_max)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        # latent + degree + weighted degree + norm + local edge-length mean
        self.input = nn.Linear(self.d_max + 4, hidden_dim)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.neigh_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 8, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _node_features(self, graph: GraphBuffers, z: Tensor) -> Tensor:
        device, dtype = z.device, z.dtype
        n = graph.num_nodes
        zp = torch.zeros((n, self.d_max), device=device, dtype=dtype)
        width = min(self.d_max, z.shape[1] if z.ndim == 2 else 0)
        if width:
            zp[:, :width] = z[:, :width]
        deg = torch.zeros(n, device=device, dtype=dtype)
        wdeg = torch.zeros(n, device=device, dtype=dtype)
        lsum = torch.zeros(n, device=device, dtype=dtype)
        ids = torch.where(graph.valid)[0]
        if ids.numel():
            src = graph.src[ids].to(device)
            dst = graph.dst[ids].to(device)
            w = graph.weight[ids].to(device=device, dtype=dtype)
            ell = graph.length[ids].to(device=device, dtype=dtype) if graph.length is not None else w.reciprocal()
            ones = torch.ones_like(w)
            deg.index_add_(0, src, ones); deg.index_add_(0, dst, ones)
            wdeg.index_add_(0, src, w); wdeg.index_add_(0, dst, w)
            lsum.index_add_(0, src, ell); lsum.index_add_(0, dst, ell)
        lmean = lsum / deg.clamp_min(1.0)
        znorm = torch.linalg.vector_norm(zp, dim=-1)
        scale = max(float(n), 1.0)
        return torch.cat([
            zp,
            (deg / scale).unsqueeze(-1),
            (wdeg / scale).unsqueeze(-1),
            znorm.unsqueeze(-1),
            lmean.unsqueeze(-1),
        ], dim=-1)

    def forward(self, graph: GraphBuffers, z: Tensor) -> tuple[Tensor, Tensor]:
        h = F.relu(self.input(self._node_features(graph, z)))
        ids = torch.where(graph.valid)[0]
        src = graph.src[ids].to(z.device) if ids.numel() else torch.empty(0, dtype=torch.long, device=z.device)
        dst = graph.dst[ids].to(z.device) if ids.numel() else torch.empty(0, dtype=torch.long, device=z.device)
        weight = graph.weight[ids].to(z.device, dtype=h.dtype) if ids.numel() else torch.empty(0, device=z.device, dtype=h.dtype)
        for self_lin, neigh_lin, norm in zip(self.self_layers, self.neigh_layers, self.norms):
            agg = torch.zeros_like(h)
            mass = torch.zeros((graph.num_nodes, 1), device=h.device, dtype=h.dtype)
            if ids.numel():
                agg.index_add_(0, src, h[dst] * weight[:, None])
                agg.index_add_(0, dst, h[src] * weight[:, None])
                mass.index_add_(0, src, weight[:, None])
                mass.index_add_(0, dst, weight[:, None])
            agg = agg / mass.clamp_min(1.0)
            h = norm(h + F.relu(self_lin(h) + neigh_lin(agg)))
        mean_h = h.mean(dim=0)
        max_h = h.max(dim=0).values
        edge_count = float(graph.edge_count)
        valid = graph.valid
        if edge_count:
            w = graph.weight[valid].to(z.device, dtype=h.dtype)
            ell = graph.length[valid].to(z.device, dtype=h.dtype) if graph.length is not None else w.reciprocal()
            stats = torch.stack([
                torch.tensor(float(graph.num_nodes), device=z.device, dtype=h.dtype),
                torch.tensor(edge_count, device=z.device, dtype=h.dtype),
                w.mean(), w.std(unbiased=False), ell.mean(), ell.std(unbiased=False),
                torch.linalg.vector_norm(z, dim=-1).mean(),
                torch.linalg.vector_norm(z, dim=-1).std(unbiased=False),
            ])
        else:
            stats = torch.tensor([
                float(graph.num_nodes), 0.0, 0.0, 0.0, 0.0, 0.0,
                float(torch.linalg.vector_norm(z, dim=-1).mean().item()) if z.numel() else 0.0,
                float(torch.linalg.vector_norm(z, dim=-1).std(unbiased=False).item()) if z.numel() else 0.0,
            ], device=z.device, dtype=h.dtype)
        # Normalize the two size statistics to keep scale manageable.
        stats = stats.clone()
        stats[0] = torch.log1p(stats[0]); stats[1] = torch.log1p(stats[1])
        g = self.global_proj(torch.cat([mean_h, max_h, stats], dim=-1))
        return h, g


class CandidateQNetwork(nn.Module):
    """Scores concrete interventions conditioned on graph/node embeddings."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        # global + endpoint pair (hu,hv,|hu-hv|,hu*hv) + action onehot + 6 scalars
        dim = hidden_dim + hidden_dim * 4 + len(REASONING_ACTIONS) + 6
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_var = nn.Linear(hidden_dim, 1)
        self.risk = nn.Linear(hidden_dim, 1)
        self.ig = nn.Linear(hidden_dim, 1)

    def _features(self, candidate: ConcreteAction, node_h: Tensor, global_h: Tensor, graph: GraphBuffers, z: Tensor) -> Tensor:
        zero = torch.zeros(self.hidden_dim, device=node_h.device, dtype=node_h.dtype)
        u = candidate.target.get("u")
        v = candidate.target.get("v")
        hu = node_h[int(u)] if u is not None else zero
        hv = node_h[int(v)] if v is not None else zero
        onehot = torch.zeros(len(REASONING_ACTIONS), device=node_h.device, dtype=node_h.dtype)
        onehot[_REASON_ACTION_TO_IDX[candidate.action]] = 1.0
        if u is not None and v is not None:
            dist = torch.linalg.vector_norm(z[int(u)] - z[int(v)]).to(node_h.dtype)
        else:
            dist = torch.zeros((), device=node_h.device, dtype=node_h.dtype)
        scalars = torch.tensor([
            float(candidate.target.get("weight", 1.0)),
            float(candidate.target.get("length", 1.0)),
            float(candidate.target.get("factor", 1.0)),
            float(candidate.prior_score),
            float(graph.edge_count) / max(float(graph.num_nodes), 1.0),
            0.0,
        ], device=node_h.device, dtype=node_h.dtype)
        scalars[-1] = dist
        return torch.cat([global_h, hu, hv, (hu-hv).abs(), hu*hv, onehot, scalars], dim=-1)

    def forward(self, candidates: Sequence[ConcreteAction], node_h: Tensor, global_h: Tensor, graph: GraphBuffers, z: Tensor) -> dict[str, Tensor]:
        if not candidates:
            raise ValueError("at least one candidate is required")
        x = torch.stack([self._features(c, node_h, global_h, graph, z) for c in candidates])
        h = self.trunk(x)
        return {
            "mean": self.mean(h).squeeze(-1),
            "log_var": self.log_var(h).squeeze(-1).clamp(-8.0, 5.0),
            "risk": torch.sigmoid(self.risk(h).squeeze(-1)),
            "ig": F.softplus(self.ig(h).squeeze(-1)),
        }


class CandidateGenerator:
    """Bounded multi-channel concrete action retrieval."""

    def __init__(self, max_candidates: int = 64, learned_fraction: float = 0.25, seed: int = 0):
        if max_candidates < 2:
            raise ValueError("max_candidates must be >= 2")
        self.max_candidates = int(max_candidates)
        self.learned_fraction = float(learned_fraction)
        self.rng = random.Random(seed)

    @staticmethod
    def _edge_set(graph: GraphBuffers) -> set[tuple[int, int]]:
        return {tuple(sorted((int(graph.src[i]), int(graph.dst[i])))) for i in torch.where(graph.valid)[0].tolist()}

    @staticmethod
    def _degrees(graph: GraphBuffers, device: torch.device) -> Tensor:
        deg = torch.zeros(graph.num_nodes, device=device)
        ids = torch.where(graph.valid)[0]
        if ids.numel():
            src, dst = graph.src[ids].to(device), graph.dst[ids].to(device)
            deg.index_add_(0, src, torch.ones_like(src, dtype=deg.dtype))
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=deg.dtype))
        return deg

    def generate(self, graph: GraphBuffers, z: Tensor, node_h: Tensor | None = None) -> list[ConcreteAction]:
        existing = self._edge_set(graph)
        deg = self._degrees(graph, z.device)
        n = graph.num_nodes
        proposals: list[ConcreteAction] = [ConcreteAction(StructuralAction.NO_OP, channel="baseline")]
        budget = self.max_candidates - 1

        # Candidate node pool.  Use learned norm when embeddings are supplied,
        # otherwise latent norm; merge with low-degree nodes to preserve recall.
        if node_h is not None:
            importance = torch.linalg.vector_norm(node_h.detach(), dim=-1)
        else:
            importance = torch.linalg.vector_norm(z.detach(), dim=-1)
        k_nodes = min(n, max(8, int(math.sqrt(max(self.max_candidates, 4))) * 3))
        top = torch.topk(importance, k=k_nodes).indices.tolist()
        low_deg = torch.topk(-deg, k=min(k_nodes, n)).indices.tolist()
        pool = list(dict.fromkeys(top + low_deg))[: min(n, k_nodes * 2)]

        nonedges: list[tuple[float, int, int, float]] = []
        for i, u in enumerate(pool):
            for v in pool[i+1:]:
                if (min(u,v), max(u,v)) in existing:
                    continue
                latent_dist = float(torch.linalg.vector_norm(z[u] - z[v]).item())
                bridge_score = float((1.0/(1.0+deg[u])).item() + (1.0/(1.0+deg[v])).item())
                nonedges.append((latent_dist + bridge_score, u, v, latent_dist))
        nonedges.sort(reverse=True)

        add_budget = max(1, budget // 3)
        for score, u, v, dist in nonedges[:add_budget]:
            proposals.append(ConcreteAction(
                StructuralAction.ADD_EDGE,
                {"u": u, "v": v, "weight": 1.0, "length": max(dist, 1e-3)},
                channel="latent_bridge", prior_score=score,
            ))

        ids = torch.where(graph.valid)[0]
        existing_rank: list[tuple[float, int, int, float, float]] = []
        for slot in ids.tolist():
            u, v = int(graph.src[slot]), int(graph.dst[slot])
            w = float(graph.weight[slot].item())
            ell = float(graph.length[slot].item()) if graph.length is not None else 1.0 / max(w, 1e-12)
            d = float(torch.linalg.vector_norm(z[u] - z[v]).item())
            # High latent mismatch on weak edges are useful prune candidates.
            existing_rank.append((d / max(w, 1e-6), u, v, w, ell))
        existing_rank.sort(reverse=True)
        edge_budget = max(1, budget // 6)
        for score, u, v, w, ell in existing_rank[:edge_budget]:
            proposals.append(ConcreteAction(StructuralAction.PRUNE_EDGE, {"u":u,"v":v}, "mismatch_prune", score))
            proposals.append(ConcreteAction(StructuralAction.REWEIGHT_AFFINITY, {"u":u,"v":v,"factor":1.25}, "affinity_up", score*0.5))
        for score, u, v, w, ell in list(reversed(existing_rank))[:edge_budget]:
            proposals.append(ConcreteAction(StructuralAction.REWEIGHT_AFFINITY, {"u":u,"v":v,"factor":0.8}, "affinity_down", -score))
            proposals.append(ConcreteAction(StructuralAction.REWEIGHT_LENGTH, {"u":u,"v":v,"factor":0.8}, "length_down", -score))

        # Random exploration protects against candidate-distribution collapse.
        # Sample nonedges directly instead of materializing O(N^2) pairs.
        exploration_budget = max(1, budget // 10)
        sampled: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = max(32, exploration_budget * 32)
        while len(sampled) < exploration_budget and attempts < max_attempts and n > 1:
            attempts += 1
            u = self.rng.randrange(n); v = self.rng.randrange(n - 1)
            if v >= u:
                v += 1
            pair = (min(u, v), max(u, v))
            if pair in existing or pair in sampled:
                continue
            sampled.add(pair)
        for u, v in sorted(sampled):
            dist = float(torch.linalg.vector_norm(z[u] - z[v]).item())
            proposals.append(ConcreteAction(StructuralAction.ADD_EDGE, {"u":u,"v":v,"weight":1.0,"length":max(dist,1e-3)}, "exploration", 0.0))

        dedup: dict[tuple[Any,...], ConcreteAction] = {}
        for p in proposals:
            key = p.key()
            if key not in dedup or p.prior_score > dedup[key].prior_score:
                dedup[key] = p
        out = list(dedup.values())
        # NO_OP always first; remainder prioritize heuristic prior before Q ranking.
        noop = next(p for p in out if p.action == StructuralAction.NO_OP)
        rest = [p for p in out if p.action != StructuralAction.NO_OP]
        rest.sort(key=lambda p: p.prior_score, reverse=True)
        return [noop] + rest[: self.max_candidates - 1]


class CounterfactualReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self._items: deque[tuple[GraphBuffers, Tensor, ConcreteAction, float, float, float]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, graph: GraphBuffers, z: Tensor, candidate: ConcreteAction, delta_u: float, risk: float = 0.0, ig: float = 0.0) -> None:
        self._items.append((graph.clone(), z.detach().cpu().clone(), candidate, float(delta_u), float(risk), float(ig)))

    def sample(self, batch_size: int, rng: random.Random | None = None):
        rng = rng or random
        return rng.sample(list(self._items), min(int(batch_size), len(self._items)))


class StructuralReasoningExecutive(nn.Module):
    """Concrete candidate generator + graph encoder + probabilistic Q model."""

    def __init__(
        self,
        d_max: int = 64,
        hidden_dim: int = 128,
        message_passing_layers: int = 3,
        max_candidates: int = 64,
        beta_uncertainty: float = 1.0,
        risk_weight: float = 0.25,
        ig_weight: float = 0.10,
        memory_weight: float = 0.15,
        lr: float = 3e-4,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = GraphStateEncoder(d_max=d_max, hidden_dim=hidden_dim, layers=message_passing_layers)
        self.q = CandidateQNetwork(hidden_dim=hidden_dim)
        self.generator = CandidateGenerator(max_candidates=max_candidates, seed=seed)
        self.beta_uncertainty = float(beta_uncertainty)
        self.risk_weight = float(risk_weight)
        self.ig_weight = float(ig_weight)
        self.memory_weight = float(memory_weight)
        self.memory: Any | None = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.replay = CounterfactualReplayBuffer()
        self.rng = random.Random(seed)

    def attach_memory(self, memory: Any | None) -> None:
        """Attach a derived experience-memory provider.

        The provider may influence ranking but never certification/commit.
        It must expose ``action_prior(graph, z, candidate)``.
        """
        self.memory = memory

    @torch.no_grad()
    def generate_candidates(self, graph: GraphBuffers, z: Tensor) -> list[ConcreteAction]:
        self.eval()
        node_h, _ = self.encoder(graph, z)
        return self.generator.generate(graph, z, node_h=node_h)

    def predict(self, graph: GraphBuffers, z: Tensor, candidates: Sequence[ConcreteAction]) -> list[CandidateValue]:
        self.eval()
        with torch.no_grad():
            node_h, global_h = self.encoder(graph, z)
            pred = self.q(candidates, node_h, global_h, graph, z)
            std = torch.exp(0.5 * pred["log_var"])
            base_score = pred["mean"] - self.beta_uncertainty * std - self.risk_weight * pred["risk"] + self.ig_weight * pred["ig"]
        memory_priors: list[float] = []
        for c in candidates:
            prior = 0.0
            if self.memory is not None:
                try:
                    prior, _ = self.memory.action_prior(graph, z, c)
                except Exception:
                    prior = 0.0
            memory_priors.append(float(prior))
        values = [CandidateValue(
            candidate=c,
            mean_delta_utility=float(pred["mean"][i].item()),
            std_delta_utility=float(std[i].item()),
            risk=float(pred["risk"][i].item()),
            information_gain=float(pred["ig"][i].item()),
            score=float(base_score[i].item()) + self.memory_weight * memory_priors[i],
            memory_prior=memory_priors[i],
        ) for i,c in enumerate(candidates)]
        values.sort(key=lambda x: x.score, reverse=True)
        return values

    def plan(self, graph: GraphBuffers, z: Tensor) -> ReasoningPlan:
        candidates = self.generate_candidates(graph, z)
        ranked = self.predict(graph, z, candidates)
        selected = ranked[0]
        # A mutation must beat NO_OP under the same learned score.
        noop = next((v for v in ranked if v.candidate.action == StructuralAction.NO_OP), None)
        if noop is not None and selected.candidate.action != StructuralAction.NO_OP and selected.score <= noop.score:
            selected = noop
        return ReasoningPlan(ranked, selected, len(candidates), {"no_op_guard": True})

    def record(self, graph: GraphBuffers, z: Tensor, outcome: CounterfactualOutcome) -> None:
        self.replay.add(
            graph, z, outcome.candidate, outcome.delta_utility,
            risk=0.0 if outcome.accepted_by_governor else 1.0,
            ig=float(outcome.metadata.get("information_gain", 0.0)),
        )

    def train_step(self, batch_size: int = 32, ranking_weight: float = 0.25) -> dict[str,float]:
        if len(self.replay) < batch_size:
            return {"loss": 0.0, "samples": 0, "ranking_loss": 0.0}
        self.train(); self.optimizer.zero_grad()
        batch = self.replay.sample(batch_size, self.rng)
        losses: list[Tensor] = []
        group_scores: dict[str, list[tuple[Tensor,float]]] = {}
        for graph_cpu, z_cpu, cand, target, risk_target, ig_target in batch:
            device = next(self.parameters()).device
            graph = GraphBuffers.from_state_dict(graph_cpu.to_state_dict(), device=device)
            z = z_cpu.to(device)
            node_h, global_h = self.encoder(graph, z)
            pred = self.q([cand], node_h, global_h, graph, z)
            mean, log_var = pred["mean"][0], pred["log_var"][0]
            t = torch.tensor(target, device=device, dtype=mean.dtype)
            # Gaussian heteroscedastic NLL gives the uncertainty head a proper target.
            nll = 0.5 * (torch.exp(-log_var) * (mean-t).square() + log_var)
            risk_loss = F.binary_cross_entropy(pred["risk"][0], torch.tensor(risk_target, device=device, dtype=mean.dtype))
            ig_loss = (pred["ig"][0] - torch.tensor(ig_target, device=device, dtype=mean.dtype)).square()
            losses.append(nll + 0.2*risk_loss + 0.05*ig_loss)
            key = graph.state_hash(include_version=False)
            group_scores.setdefault(key, []).append((mean, target))
        ranking_losses: list[Tensor] = []
        for pairs in group_scores.values():
            if len(pairs) < 2:
                continue
            best = max(pairs, key=lambda x:x[1]); worst = min(pairs, key=lambda x:x[1])
            if best[1] > worst[1]:
                ranking_losses.append(F.softplus(-(best[0]-worst[0])))
        value_loss = torch.stack(losses).mean()
        rank_loss = torch.stack(ranking_losses).mean() if ranking_losses else value_loss.new_zeros(())
        total = value_loss + float(ranking_weight)*rank_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        return {"loss": float(total.detach()), "value_loss": float(value_loss.detach()), "ranking_loss": float(rank_loss.detach()), "samples": len(batch)}


class CounterfactualFactory:
    """Generate exact, governor-grounded supervision for concrete candidates.

    The default path is edge-mutation only.  Fiber/gauge reasoning can be added
    later with an explicit evaluator without weakening the governor boundary.
    """

    def __init__(self, horizon: int = 1, rejected_utility_penalty: float = -1.0):
        self.horizon = int(horizon)
        self.rejected_utility_penalty = float(rejected_utility_penalty)

    def evaluate(
        self,
        graph: GraphBuffers,
        z: Tensor,
        candidates: Sequence[ConcreteAction],
        *,
        governor: Any,
        utility_fn: Callable[[GraphBuffers, Tensor], float],
        rollout_fn: Callable[[GraphBuffers, Tensor, int], tuple[GraphBuffers, Tensor]] | None = None,
        seed: int = 0,
    ) -> list[CounterfactualOutcome]:
        before = float(utility_fn(graph, z))
        before_hash = graph.state_hash()
        outcomes: list[CounterfactualOutcome] = []
        for idx, c in enumerate(candidates):
            if c.action == StructuralAction.NO_OP:
                outcomes.append(CounterfactualOutcome(c, 0.0, before, before, True, "accept", before_hash, before_hash, {"baseline": True}))
                continue
            mutation = action_to_mutation(c.action, graph, z, **c.target)
            if mutation is None:
                outcomes.append(CounterfactualOutcome(c, self.rejected_utility_penalty, before, before+self.rejected_utility_penalty, False, "reject", before_hash, before_hash, {"reason":"unmappable"}))
                continue
            result, shadow = governor.evaluate_mutation(graph, z, mutation, seed=seed+idx)
            accepted = result.decision == MutationDecision.ACCEPT
            if not accepted or shadow is None:
                after = before + self.rejected_utility_penalty
                outcomes.append(CounterfactualOutcome(c, after-before, before, after, False, result.decision.value, before_hash, before_hash, {"reasons":list(result.reasons)}))
                continue
            shadow_z = z.detach().clone()
            if rollout_fn is not None and self.horizon > 0:
                shadow, shadow_z = rollout_fn(shadow, shadow_z, self.horizon)
            after = float(utility_fn(shadow, shadow_z))
            outcomes.append(CounterfactualOutcome(c, after-before, before, after, True, result.decision.value, before_hash, shadow.state_hash(), {"reasons":list(result.reasons)}))
        return outcomes


def certify_ranked_candidates(
    ranked: Sequence[CandidateValue],
    *,
    graph: GraphBuffers,
    z: Tensor,
    governor: Any,
    top_k: int = 3,
    seed: int = 0,
) -> tuple[CandidateValue | None, Any | None]:
    """Return the first governor-accepted learned candidate without committing."""
    for i, value in enumerate(ranked[:max(1,int(top_k))]):
        c = value.candidate
        if c.action == StructuralAction.NO_OP:
            return value, None
        mutation = action_to_mutation(c.action, graph, z, **c.target)
        if mutation is None:
            continue
        result, _ = governor.evaluate_mutation(graph, z, mutation, seed=seed+i)
        if result.decision == MutationDecision.ACCEPT:
            return value, result
    return None, None
