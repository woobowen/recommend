"""Execute unchanged reviewed CPU attention statistics on the six new conditions."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT=Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
JOBS=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
SCRIPT=REPO/'manifests/grid-formal-20261015/attention_statistics.py'


def identity(path):
    value=path.read_bytes()
    return {'path':str(path),'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}


def write(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    output=SUPPORT/'attention-statistics';output.mkdir(exist_ok=False)
    jobs={};originals=[]
    for domain in ('toys','sports'):
        for condition in ('normal','shuffle3','shuffle4'):
            job=f'gs-20261017-{domain}-b-{condition}-attention-v1'
            root=JOBS/job/'retrieved';status=json.loads((root/'status.json').read_text())
            assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
            jobs[job]=str(root)
            originals.extend(identity(root/name) for name in ('request.json','status.json','cleanup.json','export-manifest.json','output/RESULT.json'))
            for i in range(4):
                originals.extend(identity(root/'output'/f'batch-{i:02d}'/name) for name in
                    ('bundle/record.json','bundle/arrays.npz','figures/plot_receipt.json','off_on_checks.json'))
    plan=output/'plan.json';result=output/'attention-descriptive-statistics.json'
    write(plan,{'jobs':jobs,'expected_pair_path_records':2304,'scope':'Toys/Sports B three fixed32 conditions; actual Top1 ancestry and local first path separate'})
    argv=[sys.executable,'-B',str(SCRIPT),'--plan',str(plan),'--output',str(result)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    invocation={'started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'argv':argv,
        'cwd':str(REPO),'script':identity(SCRIPT),'plan':identity(plan),'raw_inputs_before':originals,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'model_calls':0,'GPU_calls':0,'statistics_formula_changed':False}
    write(output/'invocation.json',invocation)
    with (output/'stdout.log').open('x') as stdout,(output/'stderr.log').open('x') as stderr:
        process=subprocess.run(argv,cwd=REPO,env=env,stdout=stdout,stderr=stderr,timeout=120)
    receipt={'exit_code':process.returncode,'ended_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'script':identity(SCRIPT),'plan':identity(plan),'stdout':identity(output/'stdout.log'),'stderr':identity(output/'stderr.log')}
    if result.exists():receipt['result']=identity(result)
    if process.returncode==0:
        report=json.loads(result.read_text())
        assert len(report['jobs'])==6 and sum(len(j['per_sample_query_pairs']) for j in report['jobs'])==2304
        assert all(identity(Path(r['path']))==r for r in originals)
        receipt.update(status='CPU_STATISTICS_COMPLETE_PENDING_NONAUTHOR_REVIEW',conditions=6,pair_path_records=2304,raw_inputs_unchanged=True)
    write(output/'execution-receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))
    if process.returncode:raise SystemExit(process.returncode)


if __name__=='__main__':main()
