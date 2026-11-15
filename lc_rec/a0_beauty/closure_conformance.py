"""Additional bounded real-client lock/output tests; no model or evaluator run."""
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from formal_control import durable_write, read, Runner
from production_control import Control,request
from production_identity import need,identity,SYNTHETIC


def factory(p,stage,n,timeout,path,*unused):
    req,out=request(p,stage,n,timeout,path)
    req['argv']=[p['python'],'-c','import time;time.sleep(5)']
    return req,out


def main():
    plan=read(sys.argv[1]);need(plan['evidence_scope']==SYNTHETIC,'test-only entry')
    root=Path(plan['root'])/'concurrent_clients';root.mkdir(exist_ok=False)
    release=Path(__file__).resolve().parents[2]
    version=read(release/'RUN_VERSION.json')
    p=dict(plan,root=str(root),task_id=version['run_commit'][:12]+'-lock',
           release=str(release),run_commit=version['run_commit'],prediction_run_commit=version['run_commit'],
           config=str(release/'lc_rec/a0_beauty/config/formal_beauty_seed42.json'),inputs=version['files'])
    path=root/'plan.json';durable_write(path,p)
    clients=[subprocess.Popen([sys.executable,__file__,'--client',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    observations=[]
    for client in clients:
        stdout,stderr=client.communicate(timeout=30)
        need(client.returncode==0,'client failure: '+stderr)
        observations.append(json.loads(stdout))
    state=read(root/'state.json');need(len(state['attempts'])==1,'concurrent clients duplicated accepted job')
    job=state['attempts'][0]['job_id'];runner=Runner(p['runner'],p['runner_sha256'],p['job_root'])
    live_after_clients_exit=runner.query(job)
    need(live_after_clients_exit['worker_alive'],'test clients did not exit while worker alive')
    third=subprocess.run([sys.executable,__file__,'--client',str(path)],capture_output=True,text=True,timeout=30)
    need(third.returncode==0 and len(read(root/'state.json')['attempts'])==1,'reopened client duplicated job')
    deadline=time.monotonic()+20
    while True:
        s=runner.query(job)
        if s['status'] not in ('ACCEPTED','RUNNING'):break
        need(time.monotonic()<deadline,'bounded concurrent test');time.sleep(.1)
    need(s['status']=='SUCCEEDED' and s['cleanup_verified'],'real CPU request cleanup')
    try:Control(path,request_factory=factory).step()
    except FileNotFoundError as e:missing={'rejected':True,'error':repr(e)}
    else:raise AssertionError('exit0 without artifacts accepted')
    need(read(root/'state.json')['stages']['train']['status']!='COMPLETE','missing output advanced chain')
    from production_control import resume_point
    training=Path(plan['root'])/'training'
    selected=resume_point(training,plan['run_commit'],plan['config_sha256'])
    need(selected.name=='checkpoint-8214','resume selected best rather than latest')
    incomplete=training/'checkpoint-9000';incomplete.mkdir(exist_ok=False)
    try:need(resume_point(training,plan['run_commit'],plan['config_sha256'])==selected,'incomplete checkpoint was selected')
    finally:incomplete.rmdir()
    # Exercise genuine FORMAL preparation against the existing Beauty/base
    # assets. These are explicitly unapproved prospective limits, not a grant.
    formalroot=Path(plan['root']).parent/('FORMAL_PREPARED_NOT_SUBMITTED_'+version['run_commit'][:12])
    formal=dict(p,scope='FORMAL_SUBMIT',evidence_scope='FORMAL',root=str(formalroot),
        task_id=version['run_commit'][:12]+'-prepared',model='/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549',
        env=dict(p['env'],CUDA_VISIBLE_DEVICES='0,1'),
        prelaunch_review=str(formalroot/'NOT_GRANTED.prelaunch.json'),precision_approval=str(formalroot/'NOT_SUPPLIED.precision.json'),
        gpu_uuids=[],budget={'train_normal':345600,'s3':16200,'s4':16200,'cpu':3600,'total':381600,'cleanup_reserve':20})
    formalpath=formalroot/'plan.json';durable_write(formalpath,formal)
    cli=str(Path(__file__).with_name('formal_control.py'))
    prepared=subprocess.run([sys.executable,cli,'prepare','--plan',str(formalpath)],capture_output=True,text=True,timeout=60)
    need(prepared.returncode==0,'genuine formal preparation failed: '+prepared.stderr)
    denied=subprocess.run([sys.executable,cli,'step','--plan',str(formalpath)],capture_output=True,text=True,timeout=60)
    need(denied.returncode!=0 and 'separate user launch authorization' in denied.stderr,'formal CLI submitted without authorization')
    need(not read(formalroot/'state.json')['attempts'],'unauthorized formal request was accepted')
    formal_check={'prepare_exit':prepared.returncode,'prepare_stdout':prepared.stdout,
                  'submit_without_authorization_exit':denied.returncode,'submit_stderr':denied.stderr,
                  'prospective_budgets_are_authorized':False,'plan':identity(formalpath)}
    result={'formal_prepare_and_rejection':formal_check,'scope':SYNTHETIC,'latest_complete_checkpoint':str(selected),'status':'PASS','two_clients_one_accepted_job':job,'runner_terminal':s,
            'exit0_missing_output_rejected':missing,'observations':observations,'worker_alive_after_clients_exit':live_after_clients_exit,'third_client_exit':third.returncode}
    durable_write(root/'conformance.json',result);print(json.dumps(result))

if __name__=='__main__':
    if sys.argv[1]=='--client':
        print(json.dumps(Control(sys.argv[2],request_factory=factory).step()))
    else:main()
