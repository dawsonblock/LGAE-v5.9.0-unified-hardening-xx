#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import torch
from lgae_v3.adaptive_geometry import DependencyRegistry, monitor_orthogonality, AdaptiveCurvatureCascade, CurvatureStage

r=DependencyRegistry()
I=torch.eye(3); W=I.clone(); W[0,0]=1.1
health=monitor_orthogonality(W, warn_threshold=1e-6, repair_threshold=1e-4)
def ev(v,a): return lambda:(v,a)
c=AdaptiveCurvatureCascade({CurvatureStage.FORM:ev(.1,.8),CurvatureStage.LLY:ev(.15,.5),CurvatureStage.SINKHORN:ev(.17,.1),CurvatureStage.EXACT:ev(.18,0)})
a=c.evaluate(risk=.2); e=c.evaluate(risk=.99)
checks={
 'forman_radius_1': r.cache_dependency('forman').radius==1,
 'ollivier_radius_2': r.cache_dependency('ollivier_sinkhorn').radius==2,
 'spectral_global': r.cache_dependency('spectral_gap').radius is None,
 'threshold_repair': health.action=='repaired' and health.repaired is not None,
 'ambiguous_escalates': a.selected.stage is CurvatureStage.SINKHORN,
 'high_risk_exact': e.selected.stage is CurvatureStage.EXACT,
}
payload={'schema':'LGAE_ADAPTIVE_GEOMETRY_QUALIFICATION_V5_7_0','status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}
Path('adaptive_geometry_qualification_report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
raise SystemExit(0 if payload['status']=='PASS' else 1)
