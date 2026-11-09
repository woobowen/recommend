#!/usr/bin/env python3
"""One independent <=90 minute resource chain; optimizer first, CPU param only on GPU OOM.
No repair/retry of a failed candidate. All processes are descendants of accepted runner.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job-id',required=True)
    p.add_argument('--run-commit',required=True)
    a=p.parse_args()
    assert os.environ.get('GRID_JOB_ID')==a.job_id
    assert (a.release/'RUN_VERSION.json').exists()
    version=json.loads((a.release/'RUN_VERSION.json').read_text())
    assert version['RUN_COMMIT']==a.run_commit
    base=a.release/'lc_rec/offload'
    started=time.monotonic()
    result={'schema':1,'job_id':a.job_id,'RUN_COMMIT':a.run_commit,
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'combined_candidate_limit_seconds':5370,'candidate_runs':[],
            'repair_retries':0,'formal_training':False,'status':'RUNNING'}
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    write(out/'chain-status.json',result)
    model=Path('/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549')
    identity=model/'download-validation-stable-20261109.json'
    manifest=json.loads(identity.read_text())
    assert manifest['status']=='PASS' and manifest['revision']=='4782ad278652c7c71b72204d462d6d01eaaf7549'
    assert all((model/Path(r['path']).name).stat().st_size==r['bytes'] for r in manifest['files'])
    write(out/'model-validation-inherited.json',manifest)
    for candidate in ['optimizer-only','optimizer-parameter']:
        elapsed=time.monotonic()-started
        remaining=int(5370-elapsed-15)
        if remaining<=30:
            result['status']='BUDGET_EXHAUSTED';break
        cid=a.job_id+'-'+candidate
        childout=out/candidate
        childout.mkdir()
        name=candidate.replace('-','_')
        cfg=json.loads((base/('config/probe_'+name+'.json')).read_text())
        cfg['single_gpu_seconds']=min(cfg['single_gpu_seconds'],remaining)
        cfg['new_disk_roots'].append(str(out))
        write(childout/'effective-guard.json',cfg)
        argv=[sys.executable,str(base/'scripts/resource_guard.py'),
              '--config',str(childout/'effective-guard.json'),'--output',str(childout),
              '--job-id',cid,'--',sys.executable,'-m','torch.distributed.run',
              '--standalone','--nnodes=1','--nproc_per_node=2',
              str(base/'scripts/offload_probe.py'),'--source',str(a.release/'lc_rec/vendor/original'),
              '--data-root','/data/bwwu/lc-rec/assets/data','--model',str(model),
              '--model-identity',str(identity),'--model-revision',manifest['revision'],
              '--deepspeed-config',str(base/('config/ds_'+name+'.json')),
              '--output',str(childout/'probe'),'--run-commit',a.run_commit,
              '--job-id',cid,'--seed','42','--optimizer-updates','2']
        event={'candidate':candidate,'job_id':cid,'argv':argv,'start_elapsed_seconds':elapsed}
        result['candidate_runs'].append(event);write(out/'chain-status.json',result)
        env=os.environ.copy();env['GRID_JOB_ID']=cid
        with (childout/'console.log').open('wb',buffering=0) as log:
            completed=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        event['exit_code']=completed.returncode
        event['end_elapsed_seconds']=time.monotonic()-started
        event['rank_reports']=[]
        for q in sorted((childout/'probe').glob('probe_rank*.json')):
            data=json.loads(q.read_text())
            event['rank_reports'].append({'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),
                 'rank':data['rank'],'status':data['status'],'phase':data['phase'],
                 'errors':data.get('errors',[])})
        text=(childout/'console.log').read_text(errors='replace')
        gpu_oom=('CUDA out of memory' in text or 'torch.cuda.OutOfMemoryError' in text)
        event['actual_gpu_oom_detected']=gpu_oom
        if completed.returncode==0:
            assert len(event['rank_reports'])==2 and all(r['status']=='TECHNICAL_TWO_UPDATE_PASS' for r in event['rank_reports'])
            result['status']='TECHNICAL_TWO_UPDATE_PASS'
            result['parameter_offload']='NOT_REQUIRED' if candidate=='optimizer-only' else 'USED_AFTER_OPTIMIZER_ONLY_OOM'
            break
        if candidate=='optimizer-only' and gpu_oom:
            result['fallback_reason']='Actual optimizer-only GPU OOM, authorized CPU parameter offload'
            write(out/'chain-status.json',result)
            continue
        result['status']='BLOCKED'
        result['parameter_offload']='INELIGIBLE_NON_RESOURCE_FAILURE' if candidate=='optimizer-only' else 'FAILED'
        break
    result['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['elapsed_seconds']=time.monotonic()-started
    write(out/'chain-status.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='TECHNICAL_TWO_UPDATE_PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
