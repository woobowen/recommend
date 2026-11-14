"""Prepare and submit immutable server-owned serial requests after publication."""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import subprocess
import sys
from control import REPO,SUPPORT,REMOTE,DATA,connection
import transport
import gridctl

SERIES_JOB='letter-shuffle-instruments-seed42-20261114-series-r0'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('mode',choices=['prepare','submit']);p.add_argument('--commit',required=True)
    a=p.parse_args();commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=REPO,text=True).strip()
    t=connection();dep=json.loads((SUPPORT/'receipts'/('deployment-'+commit+'.json')).read_text())
    release=dep['release'];assert dep['RUN_COMMIT']==commit
    publication=json.loads((SUPPORT/'receipts/github_prelaunch_readback.json').read_text())
    assert publication['status']=='PASS' and publication['RUN_COMMIT']==commit
    if a.mode=='submit':
        exists=json.loads(t.python('from pathlib import Path\nimport json\nprint(json.dumps(Path('+repr(t.profile['job_root']+'/'+SERIES_JOB+'/request.json')+').exists()))'))
        result=gridctl.query(t,SERIES_JOB) if exists else gridctl.submit(t,SUPPORT/'requests'/(SERIES_JOB+'.json'))
        transport.atomic(SUPPORT/'receipts'/('submit-'+SERIES_JOB+'.json'),result)
        print(json.dumps(result));return
    # Bind original author code (outside the new export), not merely its name.
    cfg=json.loads((REPO/'lc_rec/shuffle/config.json').read_text())
    external={}
    vendor=REPO/'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'
    for b in cfg['baselines'].values():
        for name in ['data.py','collator.py','evaluate.py','test_ddp.py','prompt.py','utils.py']:
            external[b['source-root']+'/'+name]=sha(vendor/name)
    model_config_shas={'a0':'e4c3dea29aefc16c0e63825f9ae097dbf5cb5d7c320843a84257e11be3b452c8',
                       'a1':'504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59'}
    for key,b in cfg['baselines'].items():external[b['model']+'/config.json']=model_config_shas[key]
    external[cfg['baselines']['a1']['model']+'/ASSET_MANIFEST.json']='c0e6ea66647148035385feae5b5bdb9c89a7f3629b6bc92f154e57387aab896e'
    meta=json.loads(t.python('from pathlib import Path\nimport json,hashlib\nexternal='+repr(external)+'\nfor p,h in external.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p\nroot=Path('+repr(DATA+'/cpu-bound-r0')+')\nreceipt=Path('+repr(DATA+'/preflight_final_receipt.json')+')\nr={"cpu_audit_sha256":{k:hashlib.sha256((root/k/"audit.json").read_bytes()).hexdigest() for k in ["a0","a1"]},"shared_permutation_manifest_sha256":hashlib.sha256((root/"shared_permutations.jsonl").read_bytes()).hexdigest(),"preflight_receipt":{"path":str(receipt),"sha256":hashlib.sha256(receipt.read_bytes()).hexdigest()}}\nassert json.loads(receipt.read_text())["status"]=="PASS"\nprint(json.dumps(r))'))
    gate={'status':'PASS','SHUFFLE_RUN_COMMIT':commit,'A1_NORMAL_REVIEW':'PASS',
          'A1_FINAL_AUDIT_DELIVERY_COMMIT':commit,'preflight_job_id':'letter-shuffle-20261114-preflight-r1',
          **meta,'core_source_sha256':{posixpath.relpath(r['path'],'lc_rec/shuffle'):r['sha256'] for r in dep['files'] if Path(r['path']).suffix in ['.py','.json']},
          'external_source_sha256':external,'publication_readback':publication}
    gate_path=SUPPORT/'requests'/('shuffle-'+commit+'.prelaunch.json');transport.atomic(gate_path,gate)
    remote_gate=t.profile['remote_support']+'/requests/'+gate_path.name;t.put(gate_path,remote_gate)
    old=json.loads(json.loads((SUPPORT/'receipts/initial_remote.json').read_text())['jobs']['letter-a1-q06-20261113-normal-seed42-r0']['request.json'])
    template={k:old[k] for k in ['schema','host','user','uid','namespace','deployment_sha256']}
    environment=dict(old['env'],CUDA_VISIBLE_DEVICES='0,1',PYTHONOPTIMIZE='0',PYTHONUNBUFFERED='1');environment.pop('DS_ACCELERATOR',None)
    inputs=[{'path':release+'/'+r['path'],'sha256':r['sha256']} for r in dep['files']]
    inputs += [{'path':p,'sha256':h} for p,h in external.items()]
    inputs += [{'path':remote_gate,'sha256':sha(gate_path)},meta['preflight_receipt']]
    jobs=[];previous=None
    for key,n in cfg['conditions']:
        # These prefixes are the accepted runner's explicitly supported marked
        # elastic-worker namespaces. The later date is this experiment's date.
        prefix='letter-a0-20261112-' if key=='a0' else 'letter-a1-q06-20261113-'
        job=f'{prefix}shuffle{n}-instruments-seed42-20261114-r0'
        argv=[cfg['baselines']['a1']['python'],release+'/lc_rec/shuffle/condition.py','--model-key',key,
              '--shuffle-n',str(n),'--output',DATA+'/'+job,'--cpu-root',DATA+'/cpu-bound-r0',
              '--run-commit',commit,'--job-id',job,'--prelaunch-gate',remote_gate]
        if previous:argv+=['--previous-job',previous]
        request=dict(template,job_id=job,capture_version=commit,cwd=release,argv=argv,env=environment,
                     inputs=inputs,timeout_seconds=(14400 if key=='a0' else 10800)+1800)
        path=SUPPORT/'requests'/(job+'.json');transport.atomic(path,request)
        t.put(path,t.profile['remote_support']+'/requests/'+path.name);jobs.append(request);previous=job
    plan={'order':cfg['conditions'],'prelaunch_status':'PASS','RUN_COMMIT':commit,'jobs':jobs,
          'runner':t.profile['runner'],'runner_sha256':t.profile['runner_sha256'],'job_root':t.profile['job_root'],
          'requests_root':t.profile['remote_support']+'/requests'}
    plan_path=SUPPORT/'requests'/('shuffle-'+commit+'.series-plan.json');transport.atomic(plan_path,plan)
    remote_plan=t.profile['remote_support']+'/requests/'+plan_path.name;t.put(plan_path,remote_plan)
    env=dict(environment,CUDA_VISIBLE_DEVICES='')
    series=dict(template,job_id=SERIES_JOB,capture_version=commit,cwd=release,
                argv=[cfg['baselines']['a1']['python'],release+'/lc_rec/shuffle/series.py','--plan',remote_plan,'--output',DATA+'/formal-series-r0'],
                env=env,inputs=inputs+[{'path':remote_plan,'sha256':sha(plan_path)}],timeout_seconds=64800)
    transport.atomic(SUPPORT/'requests'/(SERIES_JOB+'.json'),series)
    print(json.dumps({'status':'PREPARED_NOT_SUBMITTED','RUN_COMMIT':commit,'job_ids':[j['job_id'] for j in jobs],
                      'series_job':SERIES_JOB,'prelaunch_gate':remote_gate,'series_plan':remote_plan}))


if __name__=='__main__':main()
