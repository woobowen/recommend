"""Bounded two-GPU acceptance; later formal conditions use their own requests."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

UUIDS=['GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae','GPU-59cac3bd-3645-668b-79c8-839e0d878893']


def gpu_idle():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(rows)==2
    for row,expected in zip(rows,UUIDS):
        index,uuid,used=map(str.strip,row.split(','));assert uuid==expected and int(used)<100
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert not processes.strip(),'GPU is occupied; do not interfere with external work'
    return rows


def main():
    p=argparse.ArgumentParser()
    for n in ['output','cpu-root','run-commit']:p.add_argument('--'+n,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parent
    config=json.loads((root/'config.json').read_text())
    cpu=Path(a.cpu_root);assert json.loads((cpu/'summary.json').read_text())['status']=='PASS_CPU_PREPARATION'
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    ledger_path=out.parent/'preflight_ledger.json'
    lock_path=out.parent/'gpu.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[]}
        used=sum(r['device_seconds'] for r in ledger['attempts'])
        assert used<3600
        attempt={'output':str(out),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'status':'RUNNING','device_seconds':0,'stages':[],'pid':os.getpid()}
        ledger['attempts'].append(attempt)
        started=time.monotonic()
        def save():
            attempt['device_seconds']=2*(time.monotonic()-started)
            tmp=ledger_path.with_suffix('.partial');tmp.write_text(json.dumps(ledger,indent=2)+'\n');tmp.replace(ledger_path)
        try:
            gpu_idle();save()
            for key,b in config['baselines'].items():
                gpu_idle()
                remaining=(3600-used-attempt['device_seconds'])/2-30
                assert remaining>60
                argv=[b['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                      str(root/'entry.py'),'--model-key',key,'--shuffle-n','0','--acceptance',
                      '--cpu-audit',str(cpu/key/'audit.json'),'--source-root',b['source-root'],
                      '--data-root',b['data-root'],'--model',b['model'],'--adapter',b['adapter'],
                      '--output',str(out/key),'--run-commit',a.run_commit,
                      '--preflight-users-per-rank','16','--max-seconds',str(int(remaining))]
                tick=time.monotonic()
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',PYTHONOPTIMIZE='0')
                with (out/(key+'.log')).open('x') as log:
                    proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env)
                    while proc.poll() is None:
                        time.sleep(2);save()
                        if used+attempt['device_seconds']>=3540:
                            # torchrun belongs to this runner workgroup; fail and
                            # let accepted runner own bounded process cleanup.
                            proc.terminate()
                            raise TimeoutError('Preflight cumulative device-time reserve reached')
                    code=proc.returncode
                stage={'model_key':key,'argv':argv,'exit_code':code,'wall_seconds':time.monotonic()-tick}
                attempt['stages'].append(stage);save();assert code==0,key
                for rank in [0,1]:
                    r=json.loads((out/key/('rank%d.json'%rank)).read_text())
                    e=json.loads((out/key/('acceptance.rank%d.json'%rank)).read_text())
                    assert r['status']=='PREFLIGHT_INTERFACE_PASS' and e['status']=='PASS' and e['users']==16
                gpu_idle()
            attempt['status']='PASS'
        except BaseException as error:
            attempt['status']='FAIL';attempt['error']=repr(error);raise
        finally:
            attempt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
            (out/'summary.json').write_text(json.dumps(attempt,indent=2)+'\n')
            assert sum(r['device_seconds'] for r in ledger['attempts'])<=3600
    print(json.dumps(attempt))


if __name__=='__main__':main()
