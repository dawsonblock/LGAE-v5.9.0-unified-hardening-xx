#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
import torch
from lgae_v3 import (
    cayley_retraction, paired_restriction_maps, assemble_paired_connection_laplacian,
    two_sided_connection_dirichlet_energy, localized_dirichlet_credit,
)
from lgae_v3.types import make_graph_buffers
from lgae_v3.version import VERSION

torch.manual_seed(584)
B,d=128,8
raw=torch.randn(B,d,d)
A=0.08*(raw-raw.transpose(-1,-2))
# Warmup
_ = cayley_retraction(A); _ = torch.matrix_exp(A)

def bench(fn, reps=5):
    vals=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); vals.append(time.perf_counter()-t)
    return min(vals)

cayley_s=bench(lambda: cayley_retraction(A))
exp_s=bench(lambda: torch.matrix_exp(A))
W=cayley_retraction(A)
I=torch.eye(d).expand_as(W)
orth=float(torch.linalg.matrix_norm(W.transpose(-1,-2)@W-I,ord='fro',dim=(-2,-1)).max())
det_err=float((torch.linalg.det(W)-1).abs().max())
Au=A[0]; Wu,Wv=paired_restriction_maps(Au)
L=assemble_paired_connection_laplacian(4,0,3,Wu,Wv,weight=1.25)
sym=float(torch.linalg.matrix_norm(L-L.T,ord='fro'))
min_eval=float(torch.linalg.eigvalsh(L).min())

g=make_graph_buffers(4,[(0,1,1.),(1,2,1.),(2,3,1.)],capacity=6)
zb=torch.randn(4,d); za=zb.clone(); za[0]=0.7*za[0]
credit=localized_dirichlet_credit(global_advantage=.3,graph=g,z_before=zb,z_after=za,u=0,v=1,
    W_before=Wu,W_after=Wu,Wv_before=Wv,Wv_after=Wv,global_mix=.55)
credit_err=abs(float(credit.node_credits.sum())-credit.blended_advantage)
checks={
 'cayley_special_orthogonal': orth < 2e-5 and det_err < 2e-5,
 'paired_laplacian_self_adjoint': sym < 1e-6,
 'paired_laplacian_psd': min_eval > -1e-6,
 'two_sided_credit_conservative': credit_err < 1e-6,
 'runtime_measurement_finite': cayley_s > 0 and exp_s > 0,
}
payload={
 'schema':'LGAE_JOINT_ACTION_RUNTIME_QUALIFICATION_V5_8_4',
 'version':VERSION,
 'status':'PASS' if all(checks.values()) else 'FAIL',
 'checks':checks,
 'metrics':{
   'batch':B,'gauge_dim':d,'cayley_seconds_best_of_5':cayley_s,
   'matrix_exp_seconds_best_of_5':exp_s,
   'matrix_exp_over_cayley_ratio':exp_s/max(cayley_s,1e-12),
   'max_orthogonality_error':orth,'max_det_error':det_err,
   'laplacian_symmetry_error':sym,'laplacian_min_eigenvalue':min_eval,
   'credit_conservation_error':credit_err,
 },
 'claim_boundary':'PASS certifies Cayley SO(d) generation, paired self-adjoint connection-Laplacian assembly, two-sided Dirichlet credit conservation, and reproducible runtime measurement. It does not claim hardware-independent Cayley speedup or learned policy superiority.',
 'scientific_generalization_status':'NOT_YET_QUALIFIED',
}
Path('joint_action_runtime_qualification_report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
raise SystemExit(0 if payload['status']=='PASS' else 1)
