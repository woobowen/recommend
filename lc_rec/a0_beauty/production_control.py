"""Beauty server-side control: immutable plans, accepted Runner jobs, bounded recovery.

The CLI accepts formal plans only. Authorization is a separately supplied user
receipt AND its digest taken from the subsequent explicit user authorization;
no approval is created here and a plan cannot authorize itself.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from formal_control import durable_write, locked, read, digest, Runner
from production_identity import need, sha, check_file, scope_guard, ASSETS, SYNTHETIC
from g3_identity import EXPERIMENT_ID, verify_assets, verify_base_model, verify_launch_authorization

STAGES=('train','verify_best_adapter','normal','normal_eval','s3','s3_eval','s4','s4_eval','paired','collect')
ACTIVE={'ACCEPTED','RUNNING','UNCONFIRMED','CONTROL_LOST'}
# This is a server-owned Beauty registry, not a path supplied by the plan or
# Runner sidecar. Moving a job root must never create another budget ledger.
EXPERIMENT_REGISTRY=Path('/data/bwwu/lc-rec/a0-beauty-20261115/formal-experiment-registry')
INITIAL_SERIES=EXPERIMENT_ID

def group(stage):
    return 'train_normal' if stage in ('train','normal') else stage if stage in ('s3','s4') else 'cpu'


def approval_documents(plan, require_pass=False):
    """Approval bytes are part of the plan, never a mutable path-only input."""
    documents={}
    for key in ('prelaunch_review','precision_approval'):
        if key=='prelaunch_review' and not plan.get(key):
            need(not require_pass,'G3 web second review pending')
            continue
        row=plan['approval_files'][key]
        need(row['path']==plan[key], 'approval path binding: '+key)
        check_file(row)
        raw=Path(row['path']).read_bytes()
        import hashlib
        need(hashlib.sha256(raw).hexdigest()==row['sha256'], 'approval changed during read')
        documents[key]=json.loads(raw)
    need(documents['precision_approval'].get('formal_a_only_bf16_authorized') is True,
         'existing user precision approval required')
    if require_pass:
        review=documents['prelaunch_review']
        need(review['status']=='PASS' and review['RUN_COMMIT']==plan['run_commit'] and
             review['config_sha256']==plan['config_sha256'] and
             review['formal_train_sha256']==sha(Path(plan['release'])/'lc_rec/a0_beauty/formal_train.py'),
             'prelaunch review identity')
        need(review.get('review_kind') in ('INDEPENDENT_NONAUTHOR','EXTERNAL_GPT_REVIEW') and
             review.get('reviewer') and review.get('evidence_files'), 'independent review evidence required')
        for row in review['evidence_files']:check_file(row)
    return documents


def experiment_binding(plan, seal=False):
    """A formal seed cannot acquire another ledger by changing root/job ID.

    Preparation only checks existing facts. The first authorized submission
    seals the stable experiment identity in a plan-independent registry.
    """
    if plan['evidence_scope']!='FORMAL':return
    series=plan.get('experiment_series',INITIAL_SERIES)
    need(re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',series), 'invalid experiment series')
    key=digest({'dataset':'Beauty','assets':plan['asset_sha256'],
                'config':plan['config_sha256'],'model_revision':Path(plan['model']).name,
                'seed':42,'protocol':'GRID-LETTER-SHUFFLE-V1','conditions':['normal','s3','s4'],
                'series':series})
    directory=EXPERIMENT_REGISTRY
    record={'experiment_sha256':key,'root':str(Path(plan['root']).resolve()),
            'task_id':plan['task_id'],'plan_sha256':digest(plan)}
    if not directory.exists() and not seal:return
    with locked(directory/(key+'.lock')):
        path=directory/(key+'.json')
        if path.exists():
            need(read(path)==record,'formal experiment already bound to original ledger/plan')
            need((Path(record['root'])/'state.json').is_file(),
                 'sealed experiment state missing; do not reset the budget ledger')
        elif seal:durable_write(path,record)


def validate_plan(p):
    need(p['scope'] in ('FORMAL_PREPARE','FORMAL_SUBMIT','SYNTHETIC_TEST'),'unknown plan scope')
    need(p['evidence_scope']==(SYNTHETIC if p['scope']=='SYNTHETIC_TEST' else 'FORMAL'),'scope mismatch')
    scope_guard(p['root'],p['evidence_scope'])
    need(re.fullmatch('[0-9a-f]{40}',p['run_commit']) and re.fullmatch('[0-9a-f]{40}',p['prediction_run_commit']),'full source commits required')
    need(p['prediction_run_commit']==p['run_commit'],'new generation must use pinned source commit')
    need(sha(p['config'])==p['config_sha256'],'configuration SHA')
    cfg=read(p['config'])
    need(cfg['dataset']=='Beauty' and cfg['epochs']==4 and cfg['storage_save_total_limit']==2 and
         (cfg['seed'],cfg['per_device_batch_size'],cfg['gradient_accumulation_steps'],cfg['world_size'])==(42,8,4,2),'frozen training configuration')
    need(p['asset_sha256']==ASSETS,'frozen data identities')
    need(p.get('experiment_id')==EXPERIMENT_ID and p.get('experiment_series')==EXPERIMENT_ID,'G3 experiment identity')
    for key in ('root','release','python','data_root','source_root','model','config','runner','job_root'):
        need(Path(p[key]).is_absolute() and '..' not in Path(p[key]).parts,'absolute canonical path: '+key)
    need(cfg==read(Path(__file__).parent/'config/formal_g3_beauty_seed42.json'),'frozen G3 effective recipe')
    verify_assets(p['data_root'])
    for k,v in ASSETS.items():need(sha(Path(p['data_root'])/'Beauty'/f'Beauty.{k}.json')==v,'data asset hash')
    need(set(p['stage_seconds'])==set(STAGES) and all(type(x) is int and 0<x<=345600 for x in p['stage_seconds'].values()),'stage wall limits')
    need(set(p['budget'])=={'train_normal','s3','s4','cpu','total','cleanup_reserve'},'budget schema')
    need(all(type(x) is int and x>0 for x in p['budget'].values()),'positive budgets')
    need(p['budget']['cleanup_reserve']>=10 and p['budget']['total']>=sum(p['budget'][k] for k in ('train_normal','s3','s4','cpu')),'total budget must cover all groups')
    need(re.fullmatch(r'[A-Za-z0-9_.-]{1,35}',p['task_id']),'stable task identity')
    need(sha(p['runner'])==p['runner_sha256'],'runner byte identity')
    side=read(p['runner']+'.deployment.json')
    need(side['job_root']==p['job_root'] and side['namespace']==p['namespace']['namespace'],'isolated runner sidecar')
    need(sha(p['runner']+'.deployment.json')==p['namespace']['deployment_sha256'],'sidecar hash')
    for row in p['inputs']:check_file(row)
    version=read(Path(p['release'])/'RUN_VERSION.json')
    need(version['run_commit']==p['run_commit'],'export is not the pinned code commit')
    required=('g3_identity.py','production_control.py','production_pipeline.py','production_identity.py','formal_train.py','formal_evaluate.py','paired_beauty.py','offline_metrics.py','checkpoint_receipt.py','shuffle_entry.py','shuffle_runtime.py','shuffle_cpu_inputs.py')
    supplied={x['path']:x['sha256'] for x in p['inputs']}
    version_rows={x['path']:x['sha256'] for x in version['files']}
    for name in required:
        target=str(Path(p['release'])/'lc_rec/a0_beauty'/name)
        need(target in supplied and supplied[target]==version_rows[target],'production source not bound: '+name)
    need(p['inputs'] and any(Path(r['path'])==Path(p['config']) for r in p['inputs']),'config must be pinned input')
    for relative in ('lc_rec/shuffle/core.py','lc_rec/letter/config/author_instruments.json',
                     'lc_rec/letter/scripts/compat.py','lc_rec/letter/scripts/formal_attention.py','lc_rec/letter/scripts/formal_bnb_reload.py'):
        target=str(Path(p['release'])/relative)
        need(target in supplied and supplied[target]==version_rows[target],'missing production dependency: '+relative)
    from production_pipeline import actual_source_root
    actual_source_root(p)
    if p['evidence_scope']=='FORMAL':
        check_file(p['export_manifest'])
        need(p['export_manifest']['path']==str(Path(p['release'])/'RUN_VERSION.json'),'export manifest path')
        for row in version['files']:
            need(supplied.get(row['path'])==row['sha256'],'incomplete fixed export inputs')
        approval_documents(p)
        verify_base_model(p['model'])
        need(Path(p['model']).name=='4782ad278652c7c71b72204d462d6d01eaaf7549','model revision')
        need(sha(Path(p['model'])/'config.json')=='e4c3dea29aefc16c0e63825f9ae097dbf5cb5d7c320843a84257e11be3b452c8','base model config')
        need(p['env'].get('CUDA_VISIBLE_DEVICES') and p['env'].get('PYTHONOPTIMIZE','0')=='0','formal environment')
        need(len(p['gpu_uuids'])==2 and len(set(p['gpu_uuids']))==2 and
             all(re.fullmatch('GPU-[0-9a-f-]{36}',x) for x in p['gpu_uuids']) and
             p['env']['CUDA_VISIBLE_DEVICES']=='0,1','two fixed physical GPU mappings required')
        need('cpu-only' not in p['namespace']['namespace'],'CPU deployment is not formal deployment')
    else:need(p['env'].get('CUDA_VISIBLE_DEVICES')=='','CPU test visibility')
    return p


def authorize(plan, receipt, expected_digest):
    need(plan['scope']=='FORMAL_SUBMIT','FORMAL_PREPARE never submits')
    if not receipt or not expected_digest:raise PermissionError('separate user launch authorization and user-pinned digest required')
    need(re.fullmatch('[0-9a-f]{64}',expected_digest) and sha(receipt)==expected_digest,'user authorization digest')
    r=verify_launch_authorization(receipt,expected_digest,plan['run_commit'],plan['config_sha256'])
    scope_guard(receipt,'FORMAL')
    need(r.get('scope')=='FORMAL' and r.get('authorization_kind')=='USER_LAUNCH_AUTHORIZATION',
         'formal user launch receipt required; test receipts are never accepted')
    need(r.get('authorized_by')=='user' and r.get('user_message_reference') and r.get('authorization_text'),'authorization provenance')
    need(r.get('action')=='FORMAL_BEAUTY_TRAIN_NORMAL_S3_S4' and r.get('plan_sha256')==digest(plan),'authorization scope/immutable plan')
    need(r.get('run_commit')==plan['run_commit'] and r.get('config_sha256')==plan['config_sha256'] and
         r.get('budget')==plan['budget'] and r.get('task_id')==plan['task_id'],'authorization version/config/budget/job')
    series=plan.get('experiment_series',INITIAL_SERIES)
    need(r.get('experiment_series')==series,'authorization experiment series')
    if series==EXPERIMENT_ID:
        need(r.get('new_independent_experiment_authorized') is True and
             r.get('prior_experiment_series') and r['prior_experiment_series']!=series,
             'new series requires separate explicit user authorization and prior-series reference')
    need(dt.datetime.fromisoformat(r['expires_utc'])>dt.datetime.now(dt.timezone.utc),'authorization expired')
    # An independent prelaunch technical review is distinct from user launch approval.
    approval_documents(plan,require_pass=True)
    return True


def resume_point(root,commit,config_sha):
    from formal_train import verify_checkpoint
    cps=sorted((x for x in Path(root).glob('checkpoint-*') if (x/'complete.json').is_file()),key=lambda x:int(x.name.split('-')[-1]))
    need(cps,'no complete recovery checkpoint')
    verify_checkpoint(cps[-1],commit,config_sha)
    return cps[-1]


def request(plan,stage,attempt,timeout,plan_path,authorization=None,authorization_sha256=None,resume=None):
    prefix='letter-a0-beauty-cpu-runner-check-' if plan['evidence_scope']==SYNTHETIC else 'letter-a0-beauty-formal-'
    job=prefix+plan['task_id']+'-'+stage.replace('_','-')+'-a'+str(attempt)
    need(len(job)<=91,'job ID length')
    output=str(Path(plan['root'])/'attempts'/stage/f'a{attempt}')
    script=Path(plan['release'])/'lc_rec/a0_beauty/production_pipeline.py'
    argv=[plan['python'],str(script),'--plan',str(plan_path),'--stage',stage,'--output',output]
    if authorization:argv+=['--authorization',authorization,'--authorization-sha256',authorization_sha256]
    if resume:argv+=['--resume',resume]
    ns=plan['namespace']
    env=dict(plan['env']);env['CUDA_VISIBLE_DEVICES']=env.get('CUDA_VISIBLE_DEVICES','') if stage in ('train','normal','s3','s4') else ''
    inputs=[{'path':x['path'],'sha256':x['sha256']} for x in plan['inputs']]+[{'path':str(plan_path),'sha256':sha(plan_path)}]
    if plan['evidence_scope']=='FORMAL':
        approval_documents(plan,require_pass=True)
        bindings=[*plan['approval_files'].values(),plan['export_manifest'],
                  {'path':plan['runner'],'sha256':plan['runner_sha256']},
                  {'path':plan['runner']+'.deployment.json','sha256':plan['namespace']['deployment_sha256']}]
        bindings+=approval_documents(plan,require_pass=True)['prelaunch_review']['evidence_files']
        for row in bindings:inputs.append({'path':row['path'],'sha256':row['sha256']})
    if authorization:inputs.append({'path':authorization,'sha256':authorization_sha256})
    return dict(schema=1,host=ns['host'],user=ns['user'],uid=ns['uid'],namespace=ns['namespace'],
                deployment_sha256=ns['deployment_sha256'],capture_version=plan['run_commit'],
                cwd=plan['release'],argv=argv,env=env,timeout_seconds=timeout,inputs=inputs,job_id=job,
                beauty_stage=stage,beauty_config_sha256=plan['config_sha256'],beauty_input_identity=plan['asset_sha256']),output


class Control:
    def __init__(self,plan_path,*,request_factory=request,validator=None,runner=None):
        self.path=Path(plan_path).resolve();self.plan=validate_plan(read(self.path))
        need(self.plan['scope']!='FORMAL_PREPARE','draft plans must not be sealed by Control')
        experiment_binding(self.plan)
        self.root=Path(self.plan['root']);self.root.mkdir(parents=True,exist_ok=True)
        self.factory=request_factory;self.validator=validator
        self.runner=runner or Runner(self.plan['runner'],self.plan['runner_sha256'],self.plan['job_root'])
        self.state_path=self.root/'state.json'
        with locked(self.root/'.identity.lock'):
            identity_path=self.root/'plan.identity.json'
            sealed={'sha256':digest(self.plan),'path':str(self.path),'task_id':self.plan['task_id']}
            if identity_path.exists():need(read(identity_path)==sealed,'task plan changed; budget/job identity cannot reset')
            else:durable_write(identity_path,sealed)
            if not self.state_path.exists():durable_write(self.state_path,{'stages':{},'attempts':[],'status':'PREPARED_NOT_SUBMITTED'})

    def charge(self,state):
        for a in state['attempts']:
            status=self.runner.query(a['job_id'])
            if status is None:
                need(a.get('status')=='SUBMITTING','accepted job missing; cannot reset ledger')
                continue
            need(status['request_sha256']==a['request_sha256'],'queried request identity')
            start=status.get('started_at') or status.get('received_at')
            end=status.get('ended_at')
            if start:
                now=dt.datetime.now(dt.timezone.utc)
                elapsed=((dt.datetime.fromisoformat(end) if end else now)-dt.datetime.fromisoformat(start)).total_seconds()
                need(elapsed>=0,'clock moved backwards; budget is unresolved')
                a['charged_seconds']=max(a.get('charged_seconds',0),elapsed)
            a.update(status=status['status'],runner=status)
        durable_write(self.state_path,state)

    def step(self,authorization=None,authorization_sha256=None,retry=None,resume=None):
        p=self.plan
        if p['evidence_scope']=='FORMAL':authorize(p,authorization,authorization_sha256)
        with locked(self.root/'.control.lock'):
            if p['evidence_scope']=='FORMAL':experiment_binding(p,seal=True)
            state=read(self.state_path);self.charge(state)
            disk=os.statvfs(self.root)
            need(disk.f_bavail*disk.f_frsize>=max(100*1024**3,disk.f_blocks*disk.f_frsize//10),'disk safety reserve')
            memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
            need(memory['MemAvailable']>=40*1024**3,'host memory reserve')
            if p['evidence_scope']==SYNTHETIC:
                temp=Path(p.get('temporary_root',p['root']))
                need(sum(x.stat().st_size for x in temp.rglob('*') if x.is_file())<=8*1024**3,'8GiB CPU temporary space limit')
            stage=next((s for s in STAGES if state['stages'].get(s,{}).get('status')!='COMPLETE'),None)
            if stage is None:
                state['status']='COMPLETE';durable_write(self.state_path,state);return state
            old=state['stages'].get(stage)
            if old:
                a=state['attempts'][old['attempt_index']];status=a.get('runner')
                if status is None:raise RuntimeError('submit outcome unknown; reconcile original job before restart')
                need(status['status'] not in ('UNCONFIRMED','CONTROL_LOST'),'unresolved supervisor; query original identity, no automatic restart')
                if status['status'] in ACTIVE or status.get('worker_alive') or status.get('workgroup_live'):
                    state['status']='WAITING_OR_UNKNOWN';durable_write(self.state_path,state);return state
                need(status.get('cleanup_verified') is True,'cleanup unconfirmed; dependency stopped')
                if status['status']=='SUCCEEDED' and status.get('exit_code')==0:
                    from production_pipeline import validate_stage
                    (self.validator or validate_stage)(p,stage,old['output'])
                    old['status']='COMPLETE';state['status']='COMPLETE' if stage==STAGES[-1] else 'ADVANCING';durable_write(self.state_path,state);return state
                need(retry==stage,'failed stage requires explicit retry; downstream stopped')
                if stage=='train':
                    from production_identity import verify_training, check_file
                    from formal_train import verify_checkpoint
                    root=Path(p['root'])/'training'
                    need(not (root/'best_selection.json').exists(),'completed training must not restart')
                    cp=resume_point(root,p['run_commit'],p['config_sha256'])
                    need(resume and Path(resume)==cp,'only latest complete checkpoint may resume')
                else:need(not resume,'resume checkpoint belongs only to training')
            used={g:sum(a.get('charged_seconds',0) for a in state['attempts'] if group(a['stage'])==g) for g in ('train_normal','s3','s4','cpu')}
            stage_used=sum(a.get('charged_seconds',0) for a in state['attempts'] if a['stage']==stage)
            remaining=min(p['stage_seconds'][stage]-stage_used,p['budget'][group(stage)]-used[group(stage)],p['budget']['total']-sum(used.values()))-p['budget']['cleanup_reserve']
            need(remaining>=1,'cumulative budget exhausted')
            if p['evidence_scope']=='FORMAL' and stage in ('train','normal','s3','s4'):
                gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
                need(not gpu.strip(),'runtime GPU resource check: existing compute process')
                actual=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
                need(actual==p['gpu_uuids'],'runtime physical GPU identities changed')
            n=sum(a['stage']==stage for a in state['attempts'])
            req,out=self.factory(p,stage,n,min(345600,int(remaining)),self.path,authorization,authorization_sha256,resume)
            path=self.root/'requests'/f'{stage}-a{n}.json';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():need(read(path)==req,'request ID conflict')
            else:durable_write(path,req)
            accepted=self.runner.query(req['job_id'])
            attempt={'stage':stage,'attempt':n,'job_id':req['job_id'],'request_sha256':digest(req),
                     'request_path':str(path),'status':'SUBMITTING','charged_seconds':0}
            state['stages'][stage]={'status':'SUBMITTING','output':out,'attempt_index':len(state['attempts'])}
            state['attempts'].append(attempt);durable_write(self.state_path,state)
            # The durable SUBMITTING record precedes the external call. A lost
            # response is resolved by querying this same ID, never by resubmit.
            reply=accepted if accepted is not None else self.runner.submit(path)
            need(reply['request_sha256']==digest(req),'runner accepted another request')
            attempt.update(status=reply['status'],runner=reply);state['status']='RUNNING'
            durable_write(self.state_path,state);return state

    def run(self,authorization=None,authorization_sha256=None,retry=None,resume=None):
        # Only one live controller loop; per-step lock still protects queries.
        with (self.root/'.daemon.lock').open('a') as f:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                state=self.step(authorization,authorization_sha256,retry,resume)
                if all(state['stages'].get(s,{}).get('status')=='COMPLETE' for s in STAGES):return state
                retry=None;resume=None;time.sleep(1 if self.plan['evidence_scope']==SYNTHETIC else 15)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('mode',choices=['prepare','query','step','run','start'])
    p.add_argument('--plan',required=True);p.add_argument('--authorization');p.add_argument('--authorization-sha256')
    p.add_argument('--retry',choices=STAGES);p.add_argument('--resume')
    a=p.parse_args();plan=read(a.plan)
    need(plan['evidence_scope']=='FORMAL','formal CLI does not expose test boundaries')
    if a.mode=='prepare' or (a.mode=='query' and plan['scope']=='FORMAL_PREPARE'):
        validate_plan(plan)
        print(json.dumps({'status':'PREPARED_NOT_SUBMITTED','plan_sha256':digest(plan)}));return
    c=Control(a.plan)
    if a.mode=='query':
        with locked(c.root/'.control.lock'):
            state=read(c.state_path);c.charge(state)
        print(json.dumps(state));return
    authorize(plan,a.authorization,a.authorization_sha256)
    if a.mode=='start':
        args=[plan['python'],str(Path(__file__).resolve()),'run','--plan',str(c.path),
              '--authorization',a.authorization,'--authorization-sha256',a.authorization_sha256]
        if a.retry:args+=['--retry',a.retry]
        if a.resume:args+=['--resume',a.resume]
        env={k:v for k,v in os.environ.items() if k not in ('GRID_JOB_ID','GRID_JOB_DIR')}
        with (c.root/'controller.log').open('ab') as log:
            proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
        print(json.dumps({'controller_pid':proc.pid,'plan':str(c.path),'state':str(c.state_path)}));return
    state=(c.step if a.mode=='step' else c.run)(a.authorization,a.authorization_sha256,a.retry,a.resume)
    print(json.dumps(state))

if __name__=='__main__':main()
