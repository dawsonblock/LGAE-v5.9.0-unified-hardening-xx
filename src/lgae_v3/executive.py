"""Learned structural executive for LGAE v5.1.1.

The executive is a proposal model only. It predicts action value/information
/cost/risk and also scores concrete node/edge targets. The engine/governor
remains the sole authority that may commit a structural change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .benchmark.tasks import StructuralAction
from .types import GraphBuffers
from .config import LGAEConfig

ACTION_LIST: list[StructuralAction] = [
    StructuralAction.NO_OP,
    StructuralAction.ADD_EDGE,
    StructuralAction.PRUNE_EDGE,
    StructuralAction.REWEIGHT_AFFINITY,
    StructuralAction.REWEIGHT_LENGTH,
    StructuralAction.SPAWN_FIBER,
    StructuralAction.PRUNE_FIBER,
    StructuralAction.CHANGE_GAUGE,
    StructuralAction.COUPLED_REWEIGHT,
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
NUM_ACTIONS = len(ACTION_LIST)


@dataclass
class StructuralObservation:
    spectral_gap: float = 0.0
    mean_affinity: float = 0.0
    std_affinity: float = 0.0
    mean_length: float = 1.0
    std_length: float = 0.0
    num_edges: float = 0.0
    num_nodes: float = 0.0
    mean_gamma: float = 0.0
    max_gamma: float = 0.0
    min_lly: float = 0.0
    mean_lly: float = 0.0
    operator_discrepancy: float = 0.0
    lambda2: float = 0.0
    # v5.1.1: real active capacity statistics rather than z.shape[1].
    fiber_capacity: float = 0.0       # mean active width
    fiber_utilization: float = 0.0    # active fraction of D_max
    fiber_max_active: float = 0.0
    fiber_dormant_fraction: float = 0.0
    # Compact latent-state diagnostics used by the structural policy.
    latent_mean_norm: float = 0.0
    latent_std_norm: float = 0.0
    latent_max_norm: float = 0.0
    edge_latent_mismatch_mean: float = 0.0
    edge_latent_mismatch_max: float = 0.0
    task_loss: float = 0.0
    task_loss_delta: float = 0.0
    epistemic_uncertainty: float = 0.0
    recent_mutations: list[float] = field(default_factory=lambda: [0.0] * NUM_ACTIONS)

    def to_vector(self) -> Tensor:
        base = [
            self.spectral_gap, self.mean_affinity, self.std_affinity,
            self.mean_length, self.std_length, self.num_edges, self.num_nodes,
            self.mean_gamma, self.max_gamma, self.min_lly, self.mean_lly,
            self.operator_discrepancy, self.lambda2,
            self.fiber_capacity, self.fiber_utilization,
            self.fiber_max_active, self.fiber_dormant_fraction,
            self.latent_mean_norm, self.latent_std_norm, self.latent_max_norm,
            self.edge_latent_mismatch_mean, self.edge_latent_mismatch_max,
            self.task_loss, self.task_loss_delta, self.epistemic_uncertainty,
        ]
        return torch.tensor(base + self.recent_mutations, dtype=torch.float32)


@dataclass
class ActionProposal:
    action: StructuralAction
    expected_delta_utility: float
    information_gain: float
    cost: float
    risk: float
    score: float
    uncertainty: float = 0.0
    lcb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutiveNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.num_actions = num_actions
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.delta_u_head = nn.Linear(hidden_dim, num_actions)
        self.ig_head = nn.Linear(hidden_dim, num_actions)
        self.cost_head = nn.Linear(hidden_dim, num_actions)
        self.risk_head = nn.Linear(hidden_dim, num_actions)
        self.uncertainty_head = nn.Linear(hidden_dim, num_actions)
        self.policy_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs: Tensor) -> dict[str, Tensor]:
        h = self.encoder(obs)
        return {
            "delta_u": self.delta_u_head(h),
            "ig": F.softplus(self.ig_head(h)),
            "cost": F.softplus(self.cost_head(h)),
            "risk": F.softplus(self.risk_head(h)),
            "uncertainty": F.softplus(self.uncertainty_head(h)),
            "policy_logits": self.policy_head(h),
        }


class StructuralExecutive:
    def __init__(
        self,
        config: LGAEConfig | None = None,
        hidden_dim: int = 64,
        nu: float = 0.1,
        lam: float = 0.05,
        mu: float = 0.1,
        beta: float = 1.0,
        lcb_threshold: float = 0.0,
        quarantine_uncertainty: float = 0.5,
        lr: float = 1e-3,
        policy_prior_weight: float = 0.35,
        candidate_top_k: int = 64,
        candidate_max_pairs: int = 512,
        candidate_knn_per_node: int = 4,
    ):
        self.config = config or LGAEConfig()
        self.nu, self.lam, self.mu = float(nu), float(lam), float(mu)
        self.beta = float(beta)
        self.lcb_threshold = float(lcb_threshold)
        self.quarantine_uncertainty = float(quarantine_uncertainty)
        self.policy_prior_weight = float(policy_prior_weight)
        # v5.3.1: Hierarchical candidate retrieval parameters.
        # Previous default was top-24 nodes / 256 pairs, which made correct
        # endpoints outside top-24 impossible to find.  New defaults are
        # top-64 / 512 pairs / 4 KNN-per-node, giving much higher recall.
        self.candidate_top_k = int(candidate_top_k)
        self.candidate_max_pairs = int(candidate_max_pairs)
        self.candidate_knn_per_node = int(candidate_knn_per_node)
        self._obs_dim = self._compute_obs_dim()
        self.network = ExecutiveNetwork(self._obs_dim, hidden_dim=hidden_dim)
        # Target scorers use fixed local feature dimensions and therefore work
        # across graph sizes without changing network shapes.
        self.node_target_scorer = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
        self.edge_target_scorer = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 1))
        # v5.2: bounded mutation-magnitude heads.  The executive now proposes
        # not only *where* to mutate but also a conservative magnitude/width.
        self.node_magnitude_scorer = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
        self.edge_magnitude_scorer = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 1))
        params = (
            list(self.network.parameters())
            + list(self.node_target_scorer.parameters())
            + list(self.edge_target_scorer.parameters())
            + list(self.node_magnitude_scorer.parameters())
            + list(self.edge_magnitude_scorer.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self._mutation_history = torch.zeros(NUM_ACTIONS)
        self._experience: list[dict] = []
        self._policy_experience: list[dict] = []

    def _compute_obs_dim(self) -> int:
        return StructuralObservation().to_vector().shape[0]

    def observe(
        self,
        graph: GraphBuffers,
        z: Tensor,
        audit_snapshot: Any | None = None,
        task_loss: float = 0.0,
        task_loss_delta: float = 0.0,
        epistemic_uncertainty: float = 0.0,
        fiber_state: Any | None = None,
    ) -> StructuralObservation:
        obs = StructuralObservation()
        obs.num_nodes = float(graph.num_nodes)
        valid = graph.valid.bool()
        obs.num_edges = float(valid.sum().item())
        if obs.num_edges:
            w = graph.weight[valid]
            obs.mean_affinity = float(w.mean().item())
            obs.std_affinity = float(w.std().item()) if w.numel() > 1 else 0.0
            if graph.length is not None:
                ell = graph.length[valid]
                obs.mean_length = float(ell.mean().item())
                obs.std_length = float(ell.std().item()) if ell.numel() > 1 else 0.0
        if audit_snapshot is not None:
            obs.lambda2 = float(audit_snapshot.lambda2)
            obs.spectral_gap = obs.lambda2
            obs.operator_discrepancy = float(getattr(audit_snapshot, "operator_discrepancy", 0.0))
            gamma = audit_snapshot.details.get("gamma") if hasattr(audit_snapshot, "details") else None
            if isinstance(gamma, Tensor) and gamma.numel():
                obs.mean_gamma = float(gamma.mean().item())
                obs.max_gamma = float(gamma.max().item())
            elif isinstance(gamma, dict) and gamma:
                vals = list(gamma.values()); obs.mean_gamma = float(sum(vals)/len(vals)); obs.max_gamma = float(max(vals))
            lly = audit_snapshot.details.get("lly") if hasattr(audit_snapshot, "details") else None
            if isinstance(lly, dict) and lly:
                vals = list(lly.values()); obs.min_lly = float(min(vals)); obs.mean_lly = float(sum(vals)/len(vals))

        dmax = float(self.config.fiber.d_max)
        if fiber_state is not None and hasattr(fiber_state, "active_mask"):
            capacity = fiber_state.active_mask.sum(dim=-1).to(torch.float32)
            obs.fiber_capacity = float(capacity.mean().item())
            obs.fiber_max_active = float(capacity.max().item())
            obs.fiber_utilization = float(capacity.mean().item() / max(dmax, 1.0))
            obs.fiber_dormant_fraction = 1.0 - obs.fiber_utilization
        else:
            # For external callers with arbitrary z, do not assume every padded
            # channel is active. Clamp the observed width to the configured max.
            width = float(min(z.shape[1], self.config.fiber.d_max)) if z.ndim == 2 else 0.0
            obs.fiber_capacity = width
            obs.fiber_max_active = width
            obs.fiber_utilization = width / max(dmax, 1.0)
            obs.fiber_dormant_fraction = max(0.0, 1.0 - obs.fiber_utilization)
        if z.ndim == 2 and z.shape[0] == graph.num_nodes and z.numel():
            zn = torch.linalg.vector_norm(z.detach(), dim=-1).to(torch.float32)
            obs.latent_mean_norm = float(zn.mean().item())
            obs.latent_std_norm = float(zn.std().item()) if zn.numel() > 1 else 0.0
            obs.latent_max_norm = float(zn.max().item())
            ids = torch.where(graph.valid)[0]
            if ids.numel():
                src = graph.src[ids].to(z.device)
                dst = graph.dst[ids].to(z.device)
                mismatch = torch.linalg.vector_norm(z[src] - z[dst], dim=-1).to(torch.float32)
                obs.edge_latent_mismatch_mean = float(mismatch.mean().item())
                obs.edge_latent_mismatch_max = float(mismatch.max().item())

        obs.task_loss = float(task_loss)
        obs.task_loss_delta = float(task_loss_delta)
        obs.epistemic_uncertainty = float(epistemic_uncertainty)
        obs.recent_mutations = self._mutation_history.tolist()
        return obs

    def propose(self, observation: StructuralObservation) -> list[ActionProposal]:
        self.network.eval()
        with torch.no_grad():
            preds = self.network(observation.to_vector())
        out: list[ActionProposal] = []
        logp = F.log_softmax(preds["policy_logits"], dim=-1)
        centered_logp = logp - logp.mean()
        for i, action in enumerate(ACTION_LIST):
            du = float(preds["delta_u"][i].item())
            ig = float(preds["ig"][i].item())
            cost = float(preds["cost"][i].item())
            risk = float(preds["risk"][i].item())
            sigma = float(preds["uncertainty"][i].item())
            policy_bonus = self.policy_prior_weight * float(centered_logp[i].item())
            score = du + self.nu * ig - self.lam * cost - self.mu * risk + policy_bonus
            out.append(ActionProposal(
                action, du, ig, cost, risk, score, sigma, du - self.beta*sigma,
                {"action_idx": i, "policy_logp": float(logp[i].item()), "policy_bonus": policy_bonus},
            ))
        out.sort(key=lambda p: p.score, reverse=True)
        return out

    def best_proposal(self, observation: StructuralObservation) -> ActionProposal:
        return self.propose(observation)[0]

    def should_quarantine(self, proposal: ActionProposal) -> bool:
        return proposal.uncertainty > self.quarantine_uncertainty and proposal.lcb < self.lcb_threshold

    def _degrees(self, graph: GraphBuffers, device: torch.device) -> Tensor:
        deg = torch.zeros(graph.num_nodes, dtype=torch.float32, device=device)
        ids = torch.where(graph.valid)[0]
        if ids.numel():
            deg.index_add_(0, graph.src[ids].to(device), torch.ones(ids.numel(), device=device))
            deg.index_add_(0, graph.dst[ids].to(device), torch.ones(ids.numel(), device=device))
        return deg

    def select_target(self, action: StructuralAction, graph: GraphBuffers, z: Tensor, fiber_state: Any | None = None) -> dict[str, Any]:
        """Score concrete mutation targets with learned node/edge scorers.

        This removes the old hard-coded "first disconnected pair" / "weakest
        edge" authority. The scorer remains a proposal mechanism; the engine
        still validates the selected target transactionally.
        """
        if action == StructuralAction.NO_OP:
            return {}
        device = z.device
        deg = self._degrees(graph, device)
        action_norm = float(ACTION_TO_IDX[action]) / max(NUM_ACTIONS - 1, 1)
        active_width = None
        if fiber_state is not None and hasattr(fiber_state, "active_mask"):
            active_width = fiber_state.active_mask.sum(-1).to(z.device, dtype=z.dtype)
        else:
            active_width = torch.full((graph.num_nodes,), float(z.shape[1]), device=device, dtype=z.dtype)
        norm = torch.linalg.vector_norm(z, dim=-1)
        node_feat = torch.stack([
            norm.to(torch.float32),
            deg,
            active_width.to(torch.float32) / max(float(self.config.fiber.d_max), 1.0),
            torch.full_like(deg, action_norm),
            torch.full_like(deg, float(graph.edge_count) / max(float(graph.num_nodes), 1.0)),
            torch.ones_like(deg),
        ], dim=-1)

        if action in (StructuralAction.SPAWN_FIBER, StructuralAction.PRUNE_FIBER):
            with torch.no_grad(): scores = self.node_target_scorer(node_feat).squeeze(-1)
            if action == StructuralAction.SPAWN_FIBER:
                allowed = active_width < self.config.fiber.d_max
            else:
                allowed = active_width > self.config.fiber.d_base
            scores = torch.where(allowed, scores, torch.full_like(scores, -torch.inf))
            if not bool(torch.isfinite(scores).any().item()):
                return {}
            j = int(torch.argmax(scores).item())
            with torch.no_grad():
                raw_mag = float(self.node_magnitude_scorer(node_feat[j]).squeeze().item())
            max_width = max(1, int(self.config.fiber.spawn_width))
            # Map to an integer width in [1, spawn_width].
            width = 1 + min(max_width - 1, int(torch.sigmoid(torch.tensor(raw_mag)).item() * max_width))
            return {
                "node": j,
                "width": int(width),
                "target_score": float(scores[j].item()),
                "magnitude_raw": raw_mag,
            }

        ids = torch.where(graph.valid)[0]
        if action == StructuralAction.ADD_EDGE:
            # v5.3.1: Hierarchical candidate retrieval.
            #
            # The previous implementation took top-24 nodes by learned score
            # and enumerated all nonedges within that set (capped at 256).
            # This creates a severe recall bottleneck: if the correct endpoint
            # is outside the top-24, the correct mutation is *impossible*.
            #
            # The new approach has two stages:
            #   1. Score-based retrieval: take a larger top-K (default 64)
            #      by learned node score — this is the "high-confidence" pool.
            #   2. Latent-distance retrieval: for each top-K node, find its
            #      k nearest non-adjacent neighbors in latent space.  This
            #      catches structurally useful endpoints that the node scorer
            #      missed but that are geometrically close to high-score nodes.
            #
            # The two pools are merged, deduplicated, and capped at max_pairs
            # (default 512).  This dramatically improves recall while keeping
            # the candidate set bounded.
            with torch.no_grad(): nscore = self.node_target_scorer(node_feat).squeeze(-1)
            top_k = min(getattr(self, 'candidate_top_k', 64), graph.num_nodes)
            max_pairs = getattr(self, 'candidate_max_pairs', 512)
            knn_per_node = getattr(self, 'candidate_knn_per_node', 4)
            top = torch.topk(nscore, k=top_k).indices.tolist()
            existing = {tuple(sorted((int(graph.src[i]), int(graph.dst[i])))) for i in ids.tolist()}

            # Stage 1: score-based pairs within top-K
            pairs_set: set[tuple[int, int]] = set()
            for ii, u in enumerate(top):
                for v in top[ii+1:]:
                    key = tuple(sorted((u, v)))
                    if key not in existing:
                        pairs_set.add(key)

            # Stage 2: latent-distance KNN retrieval from top-K nodes
            # For each top-K node, find its nearest non-adjacent neighbors.
            if z.shape[0] > top_k and knn_per_node > 0:
                top_nodes = torch.tensor(top, device=device)
                z_top = z[top_nodes]  # [top_k, dim]
                # Compute distances from top-K nodes to ALL nodes
                # [top_k, N]
                dists = torch.cdist(z_top, z)
                # Mask out self and existing neighbors
                for idx_in_top, u in enumerate(top):
                    dists[idx_in_top, u] = float('inf')
                    for i in ids.tolist():
                        s, d = int(graph.src[i]), int(graph.dst[i])
                        if s == u: dists[idx_in_top, d] = float('inf')
                        if d == u: dists[idx_in_top, s] = float('inf')
                # Take KNN for each top node
                knn_indices = torch.topk(dists, k=min(knn_per_node, dists.shape[1]), largest=False, dim=-1).indices
                for idx_in_top, u in enumerate(top):
                    for v in knn_indices[idx_in_top].tolist():
                        key = tuple(sorted((u, int(v))))
                        if key not in existing:
                            pairs_set.add(key)

            pairs = sorted(pairs_set)
            if not pairs:
                return {}
            pairs = pairs[:max_pairs]
            src = torch.tensor([p[0] for p in pairs], device=device)
            dst = torch.tensor([p[1] for p in pairs], device=device)
            aff = torch.ones(len(pairs), device=device)
            ell = torch.linalg.vector_norm(z[src] - z[dst], dim=-1).clamp_min(1e-6)
        else:
            if ids.numel() == 0:
                return {}
            src, dst = graph.src[ids].to(device), graph.dst[ids].to(device)
            aff = graph.weight[ids].to(device)
            ell = graph.length[ids].to(device) if graph.length is not None else aff.reciprocal()

        dist = torch.linalg.vector_norm(z[src] - z[dst], dim=-1)
        efeat = torch.stack([
            aff.to(torch.float32), ell.to(torch.float32), dist.to(torch.float32),
            norm[src].to(torch.float32), norm[dst].to(torch.float32),
            deg[src], deg[dst],
            torch.full_like(deg[src], action_norm), torch.ones_like(deg[src]),
        ], dim=-1)
        with torch.no_grad():
            escore = self.edge_target_scorer(efeat).squeeze(-1)
        j = int(torch.argmax(escore).item())
        with torch.no_grad():
            raw_mag = float(self.edge_magnitude_scorer(efeat[j]).squeeze().item())
        # Conservative bounded factors prevent a learned proposal from bypassing
        # governor safety with an arbitrarily large one-shot mutation.
        factor = float(torch.exp(torch.tensor(0.6931471805599453) * torch.tanh(torch.tensor(raw_mag))).item())
        out = {
            "u": int(src[j]),
            "v": int(dst[j]),
            "target_score": float(escore[j].item()),
            "magnitude_raw": raw_mag,
        }
        if action == StructuralAction.ADD_EDGE:
            out["weight"] = factor
            out["length"] = float(max(1e-6, ell[j].item()))
        elif action in (StructuralAction.REWEIGHT_AFFINITY, StructuralAction.REWEIGHT_LENGTH, StructuralAction.COUPLED_REWEIGHT):
            out["factor"] = factor
        elif action == StructuralAction.CHANGE_GAUGE:
            # Gauge perturbations remain deliberately small: [0.0025, 0.1].
            out["magnitude"] = float(0.0025 + 0.0975 * torch.sigmoid(torch.tensor(raw_mag)).item())
        return out

    def record_policy_label(
        self,
        observation: StructuralObservation,
        optimal_action: StructuralAction,
        *,
        sample_weight: float = 1.0,
    ) -> None:
        """Add a supervised structural-action label for policy qualification.

        Live LGAE does not assume such labels exist; they are supplied by
        synthetic/controlled counterfactual benchmarks where the optimum is known.
        """
        self._policy_experience.append({
            "observation": observation.to_vector().detach().clone(),
            "action_idx": ACTION_TO_IDX[optimal_action],
            "sample_weight": float(sample_weight),
        })

    def record_mutation(self, action: StructuralAction) -> None:
        idx = ACTION_TO_IDX.get(action)
        if idx is not None:
            self._mutation_history *= 0.9
            self._mutation_history[idx] += 1.0

    def record_outcome(
        self,
        observation: StructuralObservation,
        action: StructuralAction,
        actual_delta_utility: float,
        *,
        cost_target: float | None = None,
        risk_target: float | None = None,
        ig_target: float | None = None,
        uncertainty_target: float | None = None,
        target_features: Tensor | None = None,
        target_kind: str | None = None,
        supervise_delta_u: bool = True,
        sample_weight: float = 1.0,
    ) -> None:
        self._experience.append({
            "observation": observation.to_vector().detach().clone(),
            "action_idx": ACTION_TO_IDX[action],
            "actual_delta_u": float(actual_delta_utility),
            "cost_target": cost_target,
            "risk_target": risk_target,
            "ig_target": ig_target,
            "uncertainty_target": uncertainty_target,
            "target_features": None if target_features is None else target_features.detach().clone(),
            "target_kind": target_kind,
            "supervise_delta_u": bool(supervise_delta_u),
            "sample_weight": float(sample_weight),
        })

    def record_long_term_outcome(self, observation: StructuralObservation, action: StructuralAction, actual_return: float) -> None:
        self.record_outcome(
            observation, action, actual_return,
            risk_target=0.0 if actual_return >= 0 else 1.0,
            sample_weight=2.0,
        )

    def record_governance_outcome(
        self,
        observation: StructuralObservation,
        action: StructuralAction,
        decision: str,
        *,
        cost_target: float | None = None,
        uncertainty_target: float | None = None,
    ) -> None:
        """Teach risk/cost heads from non-committed governance outcomes.

        REJECT and QUARANTINE must inform the proposal model even though there
        is no task-utility target because the mutation was not committed.
        """
        severity = {"accept": 0.0, "quarantine": 0.5, "reject": 1.0}.get(str(decision), 1.0)
        self.record_outcome(
            observation,
            action,
            0.0,
            cost_target=cost_target,
            risk_target=severity,
            uncertainty_target=uncertainty_target,
            supervise_delta_u=False,
            sample_weight=1.5 if severity > 0 else 1.0,
        )

    def train_step(self, batch_size: int = 32) -> dict[str, float]:
        """Train structural value/policy heads with a vectorized minibatch.

        Earlier releases executed one network forward per experience sample.  That
        was mathematically valid but made policy qualification unnecessarily slow
        and amplified Python overhead.  This implementation preserves the same
        per-sample masking/weighting semantics while evaluating each minibatch in
        one tensor program.
        """
        if len(self._experience) < batch_size:
            return {"loss": 0.0, "samples": 0}
        self.network.train(); self.node_target_scorer.train(); self.edge_target_scorer.train()
        self.node_magnitude_scorer.train(); self.edge_magnitude_scorer.train()
        batch = random.sample(self._experience, min(batch_size, len(self._experience)))
        self.optimizer.zero_grad()

        observations = torch.stack([exp["observation"] for exp in batch], dim=0)
        preds = self.network(observations)
        device = preds["delta_u"].device
        dtype = preds["delta_u"].dtype
        action_idx = torch.tensor([int(exp["action_idx"]) for exp in batch], dtype=torch.long, device=device)
        rows = torch.arange(len(batch), device=device)
        loss_vec = torch.zeros(len(batch), dtype=dtype, device=device)

        actual_du = torch.tensor([float(exp["actual_delta_u"]) for exp in batch], dtype=dtype, device=device)
        supervise_du = torch.tensor([bool(exp.get("supervise_delta_u", True)) for exp in batch], dtype=torch.bool, device=device)
        pred_du = preds["delta_u"][rows, action_idx]
        loss_vec = loss_vec + torch.where(supervise_du, (pred_du - actual_du).square(), torch.zeros_like(loss_vec))

        def add_optional_target(head: str, key: str, scale: float) -> None:
            nonlocal loss_vec
            present = torch.tensor([exp.get(key) is not None for exp in batch], dtype=torch.bool, device=device)
            target = torch.tensor([
                0.0 if exp.get(key) is None else float(exp[key]) for exp in batch
            ], dtype=dtype, device=device)
            pred = preds[head][rows, action_idx]
            loss_vec = loss_vec + float(scale) * torch.where(present, (pred - target).square(), torch.zeros_like(loss_vec))

        add_optional_target("cost", "cost_target", 0.25)
        add_optional_target("risk", "risk_target", 0.25)
        add_optional_target("ig", "ig_target", 0.25)
        add_optional_target("uncertainty", "uncertainty_target", 0.10)

        weights = torch.tensor([float(exp.get("sample_weight", 1.0)) for exp in batch], dtype=dtype, device=device)
        loss_vec = loss_vec * weights
        value_loss = loss_vec.mean()

        policy_loss_value = 0.0
        if self._policy_experience:
            pbatch = random.sample(self._policy_experience, min(batch_size, len(self._policy_experience)))
            pobs = torch.stack([exp["observation"] for exp in pbatch], dim=0).to(device=device)
            logits = self.network(pobs)["policy_logits"]
            target = torch.tensor([int(exp["action_idx"]) for exp in pbatch], dtype=torch.long, device=device)
            pweights = torch.tensor([float(exp.get("sample_weight", 1.0)) for exp in pbatch], dtype=dtype, device=device)
            policy_loss = (F.cross_entropy(logits, target, reduction="none") * pweights).mean()
            policy_loss_value = float(policy_loss.detach().item())
            # Preserve the historical weighting: the policy loss occupied one
            # slot beside B individual value losses in torch.stack(losses).mean().
            total_loss = (loss_vec.sum() + policy_loss) / float(len(batch) + 1)
        else:
            total_loss = value_loss

        total_loss.backward()
        self.optimizer.step()
        return {
            "loss": float(value_loss.detach().item()),
            "policy_loss": policy_loss_value,
            "samples": len(batch),
        }

    def save_state(self, path: str) -> None:
        torch.save({
            "network_state": self.network.state_dict(),
            "node_target_state": self.node_target_scorer.state_dict(),
            "edge_target_state": self.edge_target_scorer.state_dict(),
            "node_magnitude_state": self.node_magnitude_scorer.state_dict(),
            "edge_magnitude_state": self.edge_magnitude_scorer.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "mutation_history": self._mutation_history,
            "experience_count": len(self._experience),
            "policy_experience_count": len(self._policy_experience),
            "policy_prior_weight": self.policy_prior_weight,
        }, path)

    def load_state(self, path: str) -> None:
        state = torch.load(path, weights_only=False)
        self.network.load_state_dict(state["network_state"])
        if "node_target_state" in state: self.node_target_scorer.load_state_dict(state["node_target_state"])
        if "edge_target_state" in state: self.edge_target_scorer.load_state_dict(state["edge_target_state"])
        if "node_magnitude_state" in state: self.node_magnitude_scorer.load_state_dict(state["node_magnitude_state"])
        if "edge_magnitude_state" in state: self.edge_magnitude_scorer.load_state_dict(state["edge_magnitude_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._mutation_history = state["mutation_history"]
        self.policy_prior_weight = float(state.get("policy_prior_weight", self.policy_prior_weight))
