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
    p=dict(plan,root=str(root),task_id=plan['task_id']+'-lock')
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
    result={'scope':SYNTHETIC,'status':'PASS','two_clients_one_accepted_job':job,'runner_terminal':s,
            'exit0_missing_output_rejected':missing,'observations':observations,'worker_alive_after_clients_exit':live_after_clients_exit,'third_client_exit':third.returncode}
    durable_write(root/'conformance.json',result);print(json.dumps(result))

if __name__=='__main__':
    if sys.argv[1]=='--client':
        print(json.dumps(Control(sys.argv[2],request_factory=factory).step()))
    else:main()
