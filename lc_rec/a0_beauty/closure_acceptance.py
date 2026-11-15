"""Server CPU acceptance of production control, real runner and full data pipeline."""
import argparse
import concurrent.futures
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from formal_control import durable_write, read, digest, Runner
from production_control import Control, request, authorize
from production_identity import need, identity, SYNTHETIC


def factory(p,stage,n,timeout,path,*args):
    req,out=request(p,stage,n,timeout,path)
    req['argv']=[p['python'],str(Path(__file__).with_name('closure_fixture.py')),'--plan',str(path),'--stage',stage,'--output',out]
    return req,out


def wait(runner,job,seconds=90):
    end=time.monotonic()+seconds
    while True:
        result=runner.query(job)
        if result and result['status'] not in ('ACCEPTED','RUNNING'):return result
        need(time.monotonic()<end,'test wait timeout '+job);time.sleep(.25)


def rejection(fn):
    try:fn()
    except Exception as e:return {'rejected':True,'exception':repr(e)}
    raise AssertionError('negative accepted')


def negative_tests(p,path,runner):
    from production_identity import verify_training,verify_predictions
    from production_pipeline import best_evidence,locations
    from unittest.mock import patch
    import shutil
    root=Path(p['root']);results={}
    results['formal_authorization_missing']=rejection(lambda:authorize(dict(p,scope='FORMAL_SUBMIT'),None,None))
    results['plan_budget_reset']=rejection(lambda:Control(path,request_factory=factory,runner=runner) if False else _changed_plan(path,p))
    best=best_evidence(p);training=root/'training'
    cp=training/f"checkpoint-{best['best_step']}";manifest=cp/'complete.json';raw=manifest.read_bytes()
    try:
        manifest.write_text('{}');results['corrupt_checkpoint']=rejection(lambda:best_evidence(p))
    finally:manifest.write_bytes(raw)
    selected=training/'best_selection.json';raw=selected.read_bytes();v=read(selected)
    try:
        durable_write(selected,dict(v,checkpoint=str(training/'checkpoint-8214')))
        results['wrong_best']=rejection(lambda:best_evidence(p))
    finally:selected.write_bytes(raw)
    receipt=next((training/'checkpoint_receipts').glob('*.json'));raw=receipt.read_bytes();v=read(receipt)
    try:
        durable_write(receipt,dict(v,run_commit='0'*40));results['receipt_run_mismatch']=rejection(lambda:best_evidence(p))
    finally:receipt.write_bytes(raw)
    test=locations(p,'normal')/'test';rpath=test/'rank0.json';raw=rpath.read_bytes();v=read(rpath)
    try:
        durable_write(rpath,dict(v,adapter_files={}));results['wrong_generation_adapter']=rejection(lambda:verify_predictions(test,p['run_commit'],best,'normal',SYNTHETIC))
    finally:rpath.write_bytes(raw)
    results['synthetic_into_formal']=rejection(lambda:verify_training(training,p['run_commit'],p['config_sha256'],'FORMAL'))
    # Fault boundaries are injected into the actual durable controller; no model stage rerun.
    control=Control(path,request_factory=factory,runner=runner)
    state=read(control.state_path);last=state['stages']['collect'];last_copy=dict(last)
    try:
        last['status']='RUNNING';durable_write(control.state_path,state)
        with patch.object(control,'validator',side_effect=ValueError('injected CPU evaluation failure')):
            results['evaluation_failure']=rejection(control.step)
        need(read(control.state_path)['stages']['normal']['status']=='COMPLETE','evaluation failure reran generation')
        actual=runner.query(state['attempts'][last['attempt_index']]['job_id'])
        query0=runner.query
        target=actual['job_id']
        with patch.object(runner,'query',side_effect=lambda job:dict(actual,cleanup_verified=False) if job==target else query0(job)):
            results['cleanup_unconfirmed']=rejection(control.step)
        with patch.object(runner,'query',side_effect=ConnectionError('injected unreachable transport')):
            results['runner_unreachable']=rejection(control.step)
    finally:
        state['stages']['collect']=last_copy;durable_write(control.state_path,state)
    # Same actual receipt helper; failures must leave retained checkpoint entities untouched.
    from checkpoint_receipt import write_receipt
    r=read(next((training/'checkpoint_receipts').glob('*4107*.json')))
    kwargs=dict(run_commit=p['run_commit'],config_sha256=p['config_sha256'],data_identity=r['data_identity'],
        epoch=r['epoch'],step=r['global_step'],eval_loss=r['eval_loss'],best_checkpoint=r['best_checkpoint'],
        best_metric=r['best_metric'],attempt=r['attempt_job_id'],required_files={x['path'] for x in r['manifest']['files']},
        expected_manifest=r['manifest'],scope=SYNTHETIC)
    results['receipt_conflict']=rejection(lambda:write_receipt(cp,training,**dict(kwargs,eval_loss=123.)))
    with patch('checkpoint_receipt.os.link',side_effect=OSError('INJECTED_RECEIPT_WRITE_FAILURE')):
        results['receipt_write_failure']=rejection(lambda:write_receipt(cp,training,**dict(kwargs,attempt='write-failure-fixture')))
    need(len(list(training.glob('checkpoint-*')))==2,'receipt failure changed retained checkpoints')
    # Durable-controller fault injection retains the original state separately.
    import copy
    baseline=read(control.state_path);faultroot=root/'fault_injection';faultroot.mkdir(exist_ok=True)
    durable_write(faultroot/'original_state.json',baseline)
    for mode in ('worker_live','cleanup_unknown','budget_exhausted','unknown_supervisor','cpu_retry'):
        fake=copy.deepcopy(baseline);fake['stages']['collect']['status']='RUNNING'
        idx=fake['stages']['collect']['attempt_index'];target=fake['attempts'][idx]['job_id']
        actual=query0(target);submit0=runner.submit
        def q(job):
            value=query0(job)
            if job!=target:return value
            if mode=='worker_live':return dict(value,status='RUNNING',worker_alive=True,workgroup_live=True)
            if mode=='unknown_supervisor':return dict(value,status='CONTROL_LOST')
            if mode=='cleanup_unknown':return dict(value,cleanup_verified=False)
            return dict(value,status='FAILED',exit_code=1)
        if mode=='budget_exhausted':fake['attempts'][idx]['charged_seconds']=p['budget']['cpu']
        durable_write(control.state_path,fake)
        try:
            with patch.object(runner,'query',side_effect=q):
                if mode=='worker_live':
                    with patch.object(runner,'submit',side_effect=AssertionError('duplicate while alive')):
                        observation=control.step();need(observation['stages']['collect']['status']!='COMPLETE','live worker marked complete')
                    results[mode]={'no_resubmit':True}
                elif mode=='cpu_retry':
                    with patch.object(runner,'submit',wraps=submit0) as submitted:
                        control.step(retry='collect')
                        need(submitted.call_count==1,'CPU retry not submitted once')
                        req=read(Path(read(control.state_path)['attempts'][-1]['request_path']))
                        need(req['beauty_stage']=='collect','CPU retry reran GPU generation')
                        need(req['timeout_seconds']<=p['stage_seconds']['collect']-fake['attempts'][idx].get('charged_seconds',0)-10,'retry reset stage budget')
                        retry_job=req['job_id']
                    result=wait(runner,retry_job);need(result['status']=='SUCCEEDED' and result['cleanup_verified'],'real CPU-only retry failed')
                    results[mode]={'job_id':retry_job,'timeout_seconds':req['timeout_seconds'],'completed_generation_preserved':True}
                else:results[mode]=rejection(lambda:control.step(retry='collect'))
            durable_write(faultroot/(mode+'.json'),read(control.state_path))
        finally:
            current=read(control.state_path)
            known_ids={x['job_id'] for x in baseline['attempts']}
            for row in current['attempts']:
                if row['job_id'] not in known_ids:baseline['attempts'].append(row)
            durable_write(control.state_path,baseline)
    # Different clients concurrently query a completed stage: no new accepted request.
    with patch.object(runner,'submit',side_effect=AssertionError('duplicate completed pipeline')):
        with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(lambda _:Control(path,request_factory=factory,runner=runner).step(),range(2)))
    results['two_clients_completed_pipeline_no_resubmit']=True
    # Paired CLI processing rejects newly covered column and user-axis drift before bootstrap.
    from paired_beauty import process
    from production_pipeline import locations
    from types import SimpleNamespace
    import numpy as np
    for kind in ('columns','user_axis'):
        derivative=faultroot/kind;shutil.copytree(locations(p,'normal_eval')/'primary',derivative)
        f=derivative/'per_user_metrics.npz'
        with np.load(f,allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
        if kind=='columns':payload['keys']=payload['keys'][::-1]
        else:payload['row_idx']=payload['row_idx'][::-1]
        np.savez_compressed(f,**payload)
        m=read(derivative/'metrics.json')
        m['per_user_files']=[identity(derivative/Path(x['path']).name) for x in m['per_user_files']]
        durable_write(derivative/'metrics.json',m)
        args=SimpleNamespace(normal_primary=str(derivative),shuffle_primary=str(locations(p,'s3_eval')/'primary'),
             normal_test=str(locations(p,'normal')/'test'),shuffle_test=str(locations(p,'s3')/'test'),output=str(faultroot/(kind+'-rejected')))
        results['paired_'+kind]=rejection(lambda:process(args,source_context={'scope':SYNTHETIC,'best':best_evidence(p)}))
    return results


def _changed_plan(path,p):
    raw=path.read_bytes()
    try:
        durable_write(path,dict(p,task_id=p['task_id']+'x'))
        Control(path,request_factory=factory)
    finally:path.write_bytes(raw)


def runner_tests(p,runner,out):
    from unittest.mock import patch
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    prototype,_=factory(p,'normal',90,60,Path(p['root'])/'plan.json')
    script=Path(__file__).with_name('closure_torch_worker.py')
    results={};requests=[]
    import signal
    unmarked_env={k:v for k,v in os.environ.items() if k not in ('GRID_JOB_ID','GRID_JOB_DIR')}
    control=subprocess.Popen([p['python'],'-c','import time;time.sleep(180)'],env=unmarked_env,start_new_session=True,stdin=subprocess.DEVNULL)
    pidfd=os.pidfd_open(control.pid)
    try:
        for mode,limit in [('success',45),('failure',45),('timeout',8)]:
            req=dict(prototype,job_id='letter-a0-beauty-cpu-runner-check-'+p['task_id']+'-'+mode,
                     argv=[p['python'],'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max-restarts=0',str(script),'--mode',mode,'--output',str(root/mode)],timeout_seconds=limit)
            path=root/(mode+'.request.json');durable_write(path,req);requests.append((mode,req,path))
            accepted=runner.submit(path)
            # Separate subprocess clients on every Runner operation; caller/session can exit.
            live=runner.query(req['job_id']);durable_write(root/(mode+'.initial.json'),live)
            status=wait(runner,req['job_id'],75);durable_write(root/(mode+'.terminal.json'),status)
            need(status['cleanup_verified'] and not status['worker_alive'] and not status['workgroup_live'],'torchrun process cleanup')
            need(status['status']==('SUCCEEDED' if mode=='success' else 'FAILED' if mode=='failure' else 'TIMED_OUT'),'real worker exit propagation')
            workers=[read(root/mode/f'rank{rank}.identity.json') for rank in (0,1)]
            need(len({x['pid'] for x in workers})==2 and all(x['job_id']==req['job_id'] and x['job_dir']==str(Path(p['job_root'])/req['job_id']) for x in workers),'actual CPU worker identities/markers')
            need(all(x['pid']!=status['worker']['pid'] for x in workers),'worker is torchrun agent')
            duplicate=runner.submit(path);need(duplicate['submission']=='EXISTING_NO_RESTART','duplicate restart')
            results[mode]={'job_id':req['job_id'],'status':status['status'],'exit_code':status['exit_code'],'cleanup':status['cleanup_verified'],'agent':status['worker'],'workers':workers}
            conflict=dict(req,timeout_seconds=limit+1);cp=root/(mode+'.conflict.json');durable_write(cp,conflict)
            results[mode]['identity_conflict']=rejection(lambda:runner.submit(cp))
            need(control.poll() is None,'unrelated unmarked control was terminated')
    finally:
        if control.poll() is None:signal.pidfd_send_signal(pidfd,signal.SIGTERM)
        control.wait(timeout=10);os.close(pidfd)
    results['unrelated_control_protected']={'pid':control.pid,'checked_after_all_cases':True,'cleanup':'own pidfd'}

    spec=importlib.util.spec_from_file_location('fixed_beauty_runner',p['runner']);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    sample=dict(requests[0][1],timeout_seconds=345600);r.verify(sample)
    results['deadline_345600']=True;results['deadline_345601']=rejection(lambda:r.verify(dict(sample,timeout_seconds=345601)))
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args();p=read(a.plan)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only acceptance')
    started=time.monotonic();root=Path(p['root'])
    c=Control(a.plan,request_factory=factory)
    runner=c.runner
    # Accepted reply loss: the real submission happens, client loses its reply.
    from unittest.mock import patch
    original=runner.submit
    def lost(path):original(path);raise TimeoutError('injected lost submit response AFTER acceptance')
    with patch.object(runner,'submit',side_effect=lost):lost_result=rejection(c.step)
    c=Control(a.plan,request_factory=factory) # reconstruct client from durable facts
    c.run()
    state=read(c.state_path);need(len(state['attempts'])==10 and all(x['status']=='COMPLETE' for x in state['stages'].values()),'full ten-stage pipeline')
    tests=negative_tests(p,a.plan,runner)
    tests['lost_submit_response']=lost_result
    process_tests=runner_tests(p,runner,root/'torchrun_acceptance')
    final={'scope':SYNTHETIC,'status':'PASS','elapsed_seconds':time.monotonic()-started,
           'pipeline_state':identity(c.state_path),'stages':list(state['stages']),
           'negative_tests':tests,'runner_tests':process_tests,'gpu_used':False,
           'external_boundaries':['train GPU computation replaced by labeled checkpoint fixture',
            'generation replaced by hash-checked byte copies of existing candidates'],
           'not_replaced':['Runner submission/query','locks/persistent state/budget','training/Adapter source verification',
            'original LETTER evaluator','offline_metrics 16 metrics/CCE/5000 bootstrap','full paired CSV/NPZ/change/5000 bootstrap processing']}
    durable_write(root/'acceptance.json',final);print(json.dumps(final),flush=True)

if __name__=='__main__':main()
