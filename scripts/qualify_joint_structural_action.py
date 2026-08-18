#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import torch

from lgae_v3 import (
    JointStructuralGaugePolicy, localized_dirichlet_credit,
    connection_dirichlet_energy, commit_joint_connection,
)
from lgae_v3.reasoning import ConcreteAction
from lgae_v3.benchmark.tasks import StructuralAction
from lgae_v3.types import make_graph_buffers
from lgae_v3.fibers import SOConnectionBank
from lgae_v3.mutations import AddEdge


torch.manual_seed(583)
g = make_graph_buffers(5, [(0,1,1.0),(1,2,1.0),(2,3,1.0),(3,4,1.0)], capacity=8)
h = torch.randn(5, 32)
c = ConcreteAction(StructuralAction.ADD_EDGE, {"u":0,"v":4,"weight":1.0,"length":1.0})
policy = JointStructuralGaugePolicy(hidden_dim=32, gauge_dim=4, lie_rank=3)
joint = policy(h, [c])[0]
I = torch.eye(4)
orth_err = float(torch.linalg.matrix_norm(joint.connection.T @ joint.connection - I, ord='fro'))
det = float(torch.det(joint.connection))

bank = SOConnectionBank(g.src.numel(), 4, parameterization='cayley')
md = AddEdge(0,4).apply(g)
commit_joint_connection(bank, md['slot'], joint, graph=g)
commit_err = float(torch.linalg.matrix_norm(bank.matrices()[md['slot']] - joint.connection, ord='fro'))

zb = torch.randn(5,4); za = zb.clone(); za[0] = joint.connection @ zb[0]
credit = localized_dirichlet_credit(global_advantage=0.4, graph=g, z_before=zb, z_after=za,
                                    u=0, v=4, W_before=I, W_after=joint.connection,
                                    global_mix=0.5, distance_tau=1.5)
checks = {
    'joint_connection_orthogonal': orth_err < 1e-5,
    'joint_connection_special': abs(det-1.0) < 1e-5,
    'post_commit_connection_matches_shadow': commit_err < 3e-5,
    'credit_weights_normalized': abs(float(credit.node_weights.sum())-1.0) < 1e-6,
    'credit_conserves_blended_advantage': abs(float(credit.node_credits.sum())-credit.blended_advantage) < 1e-6,
    'localized_credit_bounded': -1.0 <= credit.normalized_dirichlet_improvement <= 1.0,
}
metrics = {
    'orthogonality_error': orth_err,
    'determinant': det,
    'commit_connection_error': commit_err,
    'dirichlet_before': credit.dirichlet_before,
    'dirichlet_after': credit.dirichlet_after,
    'normalized_dirichlet_improvement': credit.normalized_dirichlet_improvement,
}
payload = {
    'schema':'LGAE_JOINT_STRUCTURAL_ACTION_QUALIFICATION_V5_8_3',
    'status':'PASS' if all(checks.values()) else 'FAIL',
    'checks':checks,
    'metrics':metrics,
    'scientific_generalization_status':'NOT_YET_QUALIFIED',
    'claim_boundary':'PASS certifies SO(d) joint-action construction, shadow/commit connection consistency, and conservative localized credit accounting. It does not certify learned joint-action superiority on unseen topologies.'
}
Path('joint_structural_action_qualification_report.json').write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if payload['status']=='PASS' else 1)
