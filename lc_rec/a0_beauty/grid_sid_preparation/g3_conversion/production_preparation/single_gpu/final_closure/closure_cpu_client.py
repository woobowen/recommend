#!/usr/bin/env python3
"""Deploy and retrieve the scoped final-plan CPU check over the existing runner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

SUPPORT = Path('/home/addaswsw/recommend-support/sid-extensions-p0')
sys.path.insert(0,str(SUPPORT/'scripts'))
from transport import Transport, atomic, sha
import gridctl

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p/'.git').exists())
PRIOR = '6db8ab4ec98f5692014eeac02b03f3f55015d1eb'
RUN = '21d495b0be5b606853ec7cec68a3eaa1b58acbd7'
CLOSURE = '/home/bwwu/grid-audits/sid-extensions-p0/g3-production-20261118/formal-closure-20261118'
DRAFT = '/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b/preparation-plan.json'


def git_file(commit, path):
    return subprocess.check_output(['git','show',commit+':'+str(path.relative_to(REPO))],cwd=REPO)


def observation(t, plan, require_new_plan):
    names = [r['job_id'] for r in json.loads(git_file(PRIOR,HERE.parent/'final_job_ledger.json'))['jobs']]
    logs = ['g3-lcrec-single-train-20261118-r2','g3-lcrec-single-train-20261118-r3',
            'g3-lcrec-single-generation-20261118-r3','g3-lcrec-single-evaluation-20261118-r3']
    expected = {n:sha(HERE.parent/'jobs'/n/'work.log') for n in logs}
    payload = {'runner':t.profile['runner'],'historical_job_root':t.profile['job_root'],
               'names':names,'expected':expected,'formal_root':plan['root'],
               'formal_job_root':plan['job_root'],'task_id':plan['task_id'],'draft':DRAFT,
               'require_new_plan':require_new_plan}
    code = """
from pathlib import Path
import datetime, hashlib, json, os, runpy, socket
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old_root=Path(data['historical_job_root'])
run=runpy.run_path(data['runner'],run_name='closure_readonly_observation')
rows=[]
for name in data['names']:
    r=run['query'](old_root/name)
    assert r['status'] in ('SUCCEEDED','FAILED') and not r['worker_alive'] and not r['workgroup_live'] and r['cleanup_verified']
    rows.append({k:r[k] for k in ('job_id','status','exit_code','request_sha256','worker_alive','workgroup_live','cleanup_verified')})
log_identities={}
for n,expected in data['expected'].items():
    p=old_root/n/'work.log'
    assert sha(p)==expected,n
    log_identities[str(p)]=expected
root=Path(data['formal_root'])
children=sorted(x.name for x in root.iterdir())
assert children==(['plan.json','preparation-plan.json'] if data['require_new_plan'] else ['preparation-plan.json'])
formal_ids=sorted(x.name for x in Path(data['formal_job_root']).glob('letter-a0-beauty-formal-'+data['task_id']+'-*'))
assert not formal_ids
draft=Path(data['draft'])
assert draft.stat().st_size==214758 and sha(draft)=='44e6cc1a03f232f1d62a81ae0bde8fa1162ffd18830497bea369390c285efcbe'
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),uid=os.getuid(),
    archive_jobs=rows,historical_logs_verified=log_identities,formal_root_children=children,formal_job_ids=formal_ids,
    readonly_observation=True,old_draft_sha256=sha(draft))))
"""
    return json.loads(t.python('data='+repr(payload)+'\n'+code,timeout=60))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('submit-cpu-check','collect-cpu-check'))
    p.add_argument('--commit',required=True)
    p.add_argument('--job-id',required=True)
    p.add_argument('--prior-cpu-job-id',help='Retry the check on an existing immutable plan')
    a = p.parse_args()
    assert subprocess.check_output(['git','rev-parse',a.commit+'^{commit}'],cwd=REPO,text=True).strip()==a.commit
    t = Transport('new-server')
    old = json.loads(git_file(PRIOR,HERE.parent/'formal_preparation_plan.json'))
    destination = HERE/'jobs'/a.job_id
    destination.mkdir(parents=True,exist_ok=True)
    if a.mode=='submit-cpu-check':
        if a.prior_cpu_job_id:
            prior = gridctl.query(t,a.prior_cpu_job_id)
            assert prior['status']=='FAILED' and prior['cleanup_verified'] and not prior['workgroup_live']
        before = observation(t,old,bool(a.prior_cpu_job_id))
        atomic(destination/'before.json',before)
        rows = []
        sources = {'approved_budget.json':HERE/'approved_budget.json',
                   'budget_approval.json':HERE/'budget_approval.json',
                   'web_technical_review_source.md':HERE/'web_technical_review_source.md',
                   'external_technical_review.json':HERE/'external_technical_review.json',
                   'prior_acceptance.json':HERE.parent/'acceptance.json',
                   'final_plan_check.py':HERE/'final_plan_check.py'}
        with tempfile.TemporaryDirectory(prefix='g3-final-closure-') as tmp:
            for name, source in sources.items():
                fixed = git_file(PRIOR if name=='prior_acceptance.json' else a.commit,source)
                if source.read_bytes()!=fixed:
                    raise ValueError('worktree bytes differ from fixed source '+str(source))
                path = Path(tmp)/name
                path.write_bytes(fixed)
                folder = 'checker/'+a.commit if name.endswith('.py') else 'materials'
                row = t.put(path,CLOSURE+'/'+folder+'/'+name)
                rows.append({**row,'git_commit':PRIOR if name=='prior_acceptance.json' else a.commit,
                             'git_path':str(source.relative_to(REPO))})
        atomic(destination/'material_transfer.json',rows)
        release = old['release']
        target = t.profile
        job_path = target['job_root']+'/'+a.job_id
        argv = [old['python'],CLOSURE+'/checker/'+a.commit+'/final_plan_check.py',
                '--release',release,'--old-draft',DRAFT,'--materials',CLOSURE+'/materials',
                '--output',job_path+'/output']
        req = dict(schema=1,host=target['hostname'],user=target['user'],uid=target['uid'],
                   namespace=target['namespace'],deployment_sha256=target['deployment_sha256'],
                   capture_version=a.commit,model_run_commit=RUN,job_id=a.job_id,cwd=release,argv=argv,
                   timeout_seconds=1200,env={'CUDA_VISIBLE_DEVICES':'','DS_ACCELERATOR':'cpu',
                       'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
                       'PYTHONUNBUFFERED':'1','PYTHONOPTIMIZE':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                   inputs=[{'path':r['path'],'sha256':r['sha256']} for r in rows]+
                          [dict(path=DRAFT,sha256=sha(HERE.parent/'formal_preparation_plan.json')),
                           {'path':old['export_manifest']['path'],'sha256':old['export_manifest']['sha256']}],
                   task_scope='G3_FINAL_FORMAL_PLAN_CPU_DRY_RUN_ONLY')
        if a.prior_cpu_job_id:
            req['prior_cpu_job_id'] = a.prior_cpu_job_id
        atomic(destination/'request.json',req)
        accepted = gridctl.submit(t,destination/'request.json')
        atomic(destination/'submission.json',accepted)
        print(json.dumps({k:accepted.get(k) for k in ('job_id','status','submission','request_sha256')}))
    else:
        terminal = gridctl.query(t,a.job_id)
        atomic(destination/'terminal_query.json',terminal)
        if not terminal.get('cleanup_verified') or terminal.get('worker_alive') or terminal.get('workgroup_live'):
            raise RuntimeError('collection requires a clean terminal job')
        job = t.profile['job_root']+'/'+a.job_id
        files = ['accepted.json','status.json','cleanup.json','work.log','output/dry_run.json',
                 'output/preparation-deployment.json']
        if terminal['status']=='SUCCEEDED':
            files += ['output/SYNTHETIC_CPU_INTEGRATION_ONLY/'+n for n in
                      ('old-experiment-invalid.json','wrong-review-kind.json','wrong-evidence-SHA.json',
                       'counter-test-state.json','train-denial/failure.rank0.json')]
        paths = {name:job+'/'+name for name in files}
        if terminal['status']=='SUCCEEDED':
            paths['final_plan.json'] = old['root']+'/plan.json'
        code = """
import hashlib,json
from pathlib import Path
rows={}
for name,value in paths.items():
    p=Path(value)
    if p.is_file():
        assert not p.is_symlink() and p.stat().st_size<2*1024**2
        rows[name]={'path':value,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(rows))
"""
        expected = json.loads(t.python('paths='+repr(paths)+'\n'+code))
        receipts = {}
        for name,row in expected.items():
            target_path = HERE/name if name=='final_plan.json' else destination/name
            receipts[name]=t.get_verified(row['path'],target_path,row)
        atomic(destination/'collection.json',receipts)
        if terminal['status']=='SUCCEEDED':
            if set(paths)!=set(expected):
                raise ValueError('missing required CPU evidence: '+str(set(paths)-set(expected)))
            atomic(destination/'after.json',observation(t,old,True))
        print(json.dumps({k:terminal.get(k) for k in ('job_id','status','exit_code','cleanup_verified')}))


if __name__=='__main__':
    main()
