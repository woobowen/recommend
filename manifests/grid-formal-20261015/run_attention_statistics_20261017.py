"""Execute the unchanged, previously reviewed CPU statistics on eight existing jobs."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

R=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
S=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
P=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
OUT=S/'attention-statistics-20261017'
SCRIPT=R/'manifests/grid-formal-20261015/attention_statistics.py'

def identity(p):
    b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def write_new(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    assert subprocess.check_output(['git','--no-optional-locks','branch','--show-current'],cwd=R,text=True).strip()=='wbw'
    static=json.loads((R/'manifests/grid-formal-20261015/attention-results-review/attention-statistics-static-review.json').read_text())
    assert identity(SCRIPT)['sha256']==static['script']['sha256']
    names=['a-normal','a-shuffle3','a-shuffle4','b-normal','b-shuffle3','b-shuffle4','toys-mlp','sports-mlp']
    jobs={'gf-20261015-'+n+'-attention-v1':str(P/('gf-20261015-'+n+'-attention-v1')/'retrieved') for n in names}
    originals=[]
    for job,directory in jobs.items():
        root=Path(directory);status=json.loads((root/'status.json').read_text())
        assert status['status']=='SUCCEEDED' and status['exit_code']==0
        originals.extend(identity(root/n) for n in ['request.json','status.json','cleanup.json','export-manifest.json','output/RESULT.json'])
        for i in range(4):
            for rel in ['bundle/record.json','bundle/arrays.npz','figures/plot_receipt.json','off_on_checks.json']:
                originals.append(identity(root/'output'/f'batch-{i:02d}'/rel))
    OUT.mkdir(exist_ok=False)
    plan=OUT/'plan.json';result=OUT/'attention-descriptive-statistics.json'
    write_new(plan,{'jobs':jobs,'scope':'eight newly captured formal fixed32 conditions; inherited Beauty MLP excluded because original RESULT.status=PASS does not meet unchanged COMPLETE contract','expected_pair_path_records':3072})
    argv=[sys.executable,'-B',str(SCRIPT),'--plan',str(plan),'--output',str(result)]
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''
    invocation={'started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'argv':argv,'cwd':str(R),'interpreter':sys.executable,'source_commit':subprocess.check_output(['git','--no-optional-locks','rev-parse','HEAD'],cwd=R,text=True).strip(),'script':identity(SCRIPT),'plan':identity(plan),'raw_inputs_before':originals,'GPU_budget':0,'model_calls':0,'new_packages':[]}
    write_new(OUT/'invocation.json',invocation)
    with (OUT/'stdout.log').open('x') as stdout,(OUT/'stderr.log').open('x') as stderr:
        q=subprocess.run(argv,cwd=R,env=env,stdout=stdout,stderr=stderr,timeout=120)
    receipt={'ended_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'exit_code':q.returncode,'script':identity(SCRIPT),'plan':identity(plan),'stdout':identity(OUT/'stdout.log'),'stderr':identity(OUT/'stderr.log')}
    if result.exists():receipt['result']=identity(result)
    if q.returncode==0:
        report=json.loads(result.read_text());assert len(report['jobs'])==8
        assert sum(len(j['per_sample_query_pairs']) for j in report['jobs'])==3072
        assert all(identity(Path(x['path']))==x for x in originals)
        receipt.update(status='CPU_STATISTICS_COMPLETE_PENDING_NONAUTHOR_REVIEW',jobs=8,pair_path_records=3072,raw_inputs_unchanged=True,research_correction_rounds_used=0)
    write_new(OUT/'execution-receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    if q.returncode:raise SystemExit(q.returncode)

if __name__=='__main__':main()
