#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from lgae_v3.cache_coherence import ChangeKind, GraphCommitEvent, LocalizedGenerationCache, SpatialCacheDependency
from lgae_v3.version import VERSION

def neigh(seeds, radius):
    out=set(seeds); frontier=set(seeds)
    for _ in range(radius):
        nxt=set()
        for n in frontier:
            if n>0: nxt.add(n-1)
            if n<9: nxt.add(n+1)
        out |= nxt; frontier=nxt
    return out

def main():
    c=LocalizedGenerationCache[int,int](dependency=SpatialCacheDependency(ChangeKind.TOPOLOGY, radius=2), neighborhood_resolver=neigh)
    c.bind(0)
    for n in range(10): c.put(n,n,generation=0)
    c.on_graph_commit(GraphCommitEvent(1, ChangeKind.TOPOLOGY, changed_nodes=(5,)))
    checks={
      'radius2_exact_dirty_region': set(c.last_invalidated)=={3,4,5,6,7},
      'untouched_partition_preserved': set(c.keys())=={0,1,2,8,9},
      'authority_generation_advanced': c.bound_generation==1,
    }
    g=LocalizedGenerationCache[int,int](dependency=SpatialCacheDependency(ChangeKind.TOPOLOGY, radius=None)); g.bind(0); g.put(1,1,generation=0)
    g.on_graph_commit(GraphCommitEvent(1, ChangeKind.TOPOLOGY, changed_nodes=(1,)))
    checks['global_dependency_flushes']=g.keys()==()
    f=LocalizedGenerationCache[int,int](dependency=SpatialCacheDependency(ChangeKind.TOPOLOGY, radius=1), neighborhood_resolver=neigh); f.bind(0); f.put(1,1,generation=0)
    f.on_graph_commit(GraphCommitEvent(1, ChangeKind.TOPOLOGY))
    checks['missing_locality_fails_closed']=f.keys()==()
    payload={'schema':'LGAE_LOCALIZED_CACHE_QUALIFICATION_V5_6_3','version':VERSION,'checks':checks,'passed':all(checks.values())}
    (ROOT/'localized_cache_qualification_report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(payload,indent=2,sort_keys=True)); return 0 if payload['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
