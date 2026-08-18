#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgae_v3.ann_index import ANNNeighborIndex
from lgae_v3.credit import MutationCreditTracker
from lgae_v3.curvature.bakry_emery import analytic_markov_generator, bakry_emery_curvature
from lgae_v3.executive import StructuralAction
from lgae_v3.neighbor_index import ExactChunkedKNN
from lgae_v3.production_dynamics import CurvatureHysteresisController, LatentEquilibriumBarrier
from lgae_v3.sheaf_diffusion import gauge_orthogonality_penalty, sheaf_laplacian_diffusion
from lgae_v3.transactions import graph_transaction
from lgae_v3.types import GraphBuffers
from lgae_v3.version import VERSION, QUALIFICATION_SCHEMA


def graph() -> GraphBuffers:
    src = torch.tensor([0, 1, 0, 0], dtype=torch.long)
    dst = torch.tensor([1, 2, 0, 0], dtype=torch.long)
    weight = torch.tensor([1.0, 1.0, 0.0, 0.0])
    valid = torch.tensor([True, True, False, False])
    return GraphBuffers(3, src, dst, weight, valid)


def main() -> int:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    z = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    U_bad = (2.0 * torch.eye(2)).unsqueeze(0)
    out = sheaf_laplacian_diffusion(
        z, torch.tensor([0]), torch.tensor([1]), U_bad, torch.ones(1),
        eta=1.0, transport_norm_ratio=1.0,
    )
    transported_norm = float(torch.linalg.vector_norm(out[1]))
    checks['sheaf_nonexpansive_external_map'] = transported_norm <= 1.0 + 1e-6
    checks['gauge_penalty_detects_drift'] = float(gauge_orthogonality_penalty(U_bad)) > 1.0
    details['sheaf_transported_norm'] = transported_norm

    ema = CurvatureHysteresisController(alpha=0.5, variance_alpha=0.5, min_samples=3, sigma_guard=1.0)
    for _ in range(3):
        ema.update({(0, 1): -1.0})
    allowed, meta = ema.allows('add', 0, 2, add_threshold=-0.2, prune_threshold=0.2)
    checks['curvature_ema_hysteresis_persistent_signal'] = bool(allowed)
    details['curvature_hysteresis'] = meta

    P = torch.tensor([[0.10,0.90,0.00],[0.00,0.10,0.90],[0.80,0.20,0.00]], dtype=torch.float64)
    Q, pi, mode = analytic_markov_generator(P, directed_policy='symmetrize')
    checks['directed_gamma2_stationary_symmetrization'] = (
        'symmetrized' in mode
        and torch.allclose(Q.sum(-1), torch.zeros(3, dtype=Q.dtype), atol=1e-10)
        and bool(torch.all(pi > 0))
        and math.isfinite(bakry_emery_curvature(Q, 0))
    )
    details['gamma2_mode'] = mode
    details['stationary_measure'] = [float(x) for x in pi]

    g = graph()
    index = ExactChunkedKNN()
    index.build(torch.randn(3, 4, generator=torch.Generator().manual_seed(7)))
    before = g.state_hash()
    with graph_transaction(g, index):
        g.valid[0] = False
        g.weight[0] = 0.0
        g.length[0] = 0.0
        g.bump_version()
    checks['transaction_atomic_rollback'] = g.state_hash() == before
    checks['ann_cache_invalidated_on_rollback'] = bool(index.cache_dirty)
    details['ann_cache'] = index.cache_metadata()

    barrier = LatentEquilibriumBarrier(delta_tol=1e-3, required_consecutive=2)
    x = torch.ones(4, 3)
    barrier.observe(x)
    barrier.observe(x + 1e-5)
    barrier.observe(x + 1e-5)
    checks['latent_equilibrium_barrier'] = bool(barrier.is_equilibrated)
    details['equilibrium'] = barrier.summary()

    tracker = MutationCreditTracker(gamma=1.0, horizons=[1])
    tracker.record_mutation(
        StructuralAction.ADD_EDGE, step=0, predicted_delta_u=0.5,
        predicted_uncertainty=0.1, governance_decision='accept',
        governance_reasons=[], graph_hash_before='g0', graph_hash_after='g1',
        config_governance_hash='cfg', counterfactual_baseline=0.4,
    )
    tracker.record_utility(0, 0.0)
    tracker.record_utility(1, 1.0)
    outcome = tracker.get_outcomes()[0]
    checks['counterfactual_advantage_credit'] = abs(float(outcome.advantage) - 0.6) < 1e-9
    details['credit'] = {
        'return': outcome.discounted_return,
        'baseline': outcome.baseline_return,
        'advantage': outcome.advantage,
    }

    payload = {
        'version': VERSION,
        'schema': QUALIFICATION_SCHEMA,
        'suite': 'production_dynamics',
        'checks': checks,
        'details': details,
        'passed': all(checks.values()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
