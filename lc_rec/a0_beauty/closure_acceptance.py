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
        with patch.object(runner,'query',return_value=dict(actual,cleanup_verified=False)):
            results['cleanup_unconfirmed']=rejection(control.step)
        with patch.object(runner,'query',side_effect=ConnectionError('injected unreachable transport')):
            results['runner_unreachable']=rejection(control.step)
    finally:
        state['stages']['collect']=last_copy;durable_write(control.state_path,state)
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
        duplicate=runner.submit(path);need(duplicate['submission']=='EXISTING_NO_RESTART','duplicate restart')
        results[mode]={'job_id':req['job_id'],'status':status['status'],'exit_code':status['exit_code'],'cleanup':status['cleanup_verified']}
        conflict=dict(req,timeout_seconds=limit+1);cp=root/(mode+'.conflict.json');durable_write(cp,conflict)
        results[mode]['identity_conflict']=rejection(lambda:runner.submit(cp))
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
