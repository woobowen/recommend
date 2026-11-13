"""Reconcile task charges from the accepted runner's terminal cleanup receipts.

Pipeline finally values are provisional. A killed pipeline need not execute
finally: only a terminal, verified-empty runner permits closing its ledger row.
No process is signaled here, and an unresolved prior attempt prevents restart.
"""
import datetime as dt
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen_preflight_20261110/scripts"))
from common import write


def reconcile(path):
    path=Path(path)
    ledger=json.loads(path.read_text()) if path.exists() else {'attempts':[]}
    for row in ledger['attempts']:
        job=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')/row['job_id']
        status=json.loads((job/'status.json').read_text())
        assert status['job_id']==row['job_id']
        assert status['status'] not in ['ACCEPTED','RUNNING','CONTROL_LOST']
        assert status.get('cleanup',{}).get('verified') and status.get('ended_at'), 'prior cleanup unresolved'
        start=dt.datetime.fromisoformat(status['started_at'])
        end=dt.datetime.fromisoformat(status['ended_at'])
        elapsed=(end-start).total_seconds()
        assert elapsed>=0
        if 'wall_seconds' in row and 'pipeline_wall_seconds' not in row:row['pipeline_wall_seconds']=row['wall_seconds']
        row.update(ended_utc=status['ended_at'],wall_seconds=max(elapsed,row.get('wall_seconds',0)),
                   status=status['status'],runner_cleanup_verified=True,runner_receipt=str(job/'status.json'))
        row['gpu_device_seconds']=2*row['wall_seconds']
    if ledger['attempts']:write(path,ledger)
    return ledger
