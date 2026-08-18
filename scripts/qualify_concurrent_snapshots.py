#!/usr/bin/env python3
from __future__ import annotations
import json, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from lgae_v3.cache_coherence import GraphReadCoordinator, GraphReadView, StaleReadError, run_consistent_read
from lgae_v3.transactions import journaled_graph_transaction
from lgae_v3.types import make_graph_buffers
from lgae_v3.version import VERSION


def main() -> int:
    checks={}; details={}
    g=make_graph_buffers(5,[(0,1,1.0),(1,2,1.0),(2,3,1.0)],capacity=8)
    rc=GraphReadCoordinator()
    token=rc.begin_read(g.version)
    with journaled_graph_transaction(g,read_coordinator=rc) as tx:
        tx.set_slot(0,weight=2.0,bump_generation=False); tx.commit()
    try:
        rc.validate(token,g.version); stale=False
    except StaleReadError:
        stale=True
    checks['overlapping_read_rejected']=stale
    checks['commit_generation_advanced']=g.version==1
    checks['writer_epoch_closed']=not rc.writer_active and rc.mutation_epoch==2

    before=g.state_hash(); token2=rc.begin_read(g.version)
    with journaled_graph_transaction(g,read_coordinator=rc) as tx:
        tx.set_slot(0,weight=9.0,bump_generation=False)
    checks['rollback_restores_state']=g.state_hash()==before
    try:
        rc.validate(token2,g.version); rollback_stale=False
    except StaleReadError:
        rollback_stale=True
    checks['rollback_overlap_rejected']=rollback_stale

    rc.begin_write()
    def release():
        time.sleep(0.02); rc.end_write()
    t=threading.Thread(target=release); t.start()
    value=run_consistent_read(rc,lambda:g.version,lambda:float(g.weight[0]),max_retries=20,retry_delay_s=0.003)
    t.join()
    checks['retry_after_writer']=abs(value-2.0)<1e-7
    with GraphReadView(rc,lambda:g.version) as view:
        details['stable_generation']=view.generation
    details['mutation_epoch']=rc.mutation_epoch
    payload={'schema':'LGAE_CONCURRENT_SNAPSHOT_QUALIFICATION_V5_6_3','version':VERSION,'checks':checks,'details':details,'passed':all(checks.values())}
    (ROOT/'concurrent_snapshot_qualification_report.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0 if payload['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
