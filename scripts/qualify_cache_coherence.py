#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from lgae_v3.cache_coherence import ChangeKind, CommitEventBus, GenerationStampedCache
from lgae_v3.transactions import journaled_graph_transaction
from lgae_v3.types import make_graph_buffers
from lgae_v3.version import VERSION


def main() -> int:
    checks={}; details={}
    g=make_graph_buffers(4,[(0,1,1.0),(1,2,1.0)],capacity=6)
    bus=CommitEventBus()
    topo=GenerationStampedCache(dependencies=ChangeKind.TOPOLOGY)
    weights=GenerationStampedCache(dependencies=ChangeKind.WEIGHTS)
    topo.bind(g.version); weights.bind(g.version); bus.register(topo); bus.register(weights)
    before=g.state_hash()
    with journaled_graph_transaction(g,event_bus=bus) as tx:
        tx.set_slot(2,src=2,dst=3,weight=1.0,length=1.0,valid=True)
    checks['rollback_no_commit_event']=bus.last_generation is None
    checks['rollback_state_restored']=g.state_hash()==before
    checks['rollback_cache_unchanged']=not topo.dirty and not weights.dirty
    with journaled_graph_transaction(g,event_bus=bus) as tx:
        tx.set_slot(0,weight=2.0,bump_generation=False); tx.commit()
    checks['commit_generation_monotonic']=g.version==1 and bus.last_generation==1
    checks['selective_weight_invalidation']=weights.dirty and not topo.dirty
    checks['unaffected_cache_generation_advanced']=topo.bound_generation==g.version
    details['generation']=g.version
    details['topology_cache']={'dirty':topo.dirty,'bound_generation':topo.bound_generation}
    details['weight_cache']={'dirty':weights.dirty,'bound_generation':weights.bound_generation}
    payload={'schema':'LGAE_CACHE_COHERENCE_QUALIFICATION_V5_6_3','version':VERSION,'checks':checks,'details':details,'passed':all(checks.values())}
    (ROOT/'cache_coherence_qualification_report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0 if payload['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
