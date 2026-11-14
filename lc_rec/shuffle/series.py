"""Server-owned serial accepted-job orchestration; fail closed at every stage."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(a.output)
    assert plan['order']==[['a0',3],['a0',4],['a1',3],['a1',4]]
    assert plan['prelaunch_status']=='PASS' and len(plan['jobs'])==4
    out.mkdir(parents=True,exist_ok=False)
    runner=Path(plan['runner']);assert hashlib.sha256(runner.read_bytes()).hexdigest()==plan['runner_sha256']
    environment={k:v for k,v in os.environ.items() if k not in ['GRID_JOB_ID','GRID_JOB_DIR']}
    started=time.monotonic();result={'status':'RUNNING','stages':[],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save():
        result['wall_seconds']=time.monotonic()-started
        tmp=out/'summary.partial';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'summary.json')
    try:
        for index,request in enumerate(plan['jobs']):
            args=request['argv'];model,n=plan['order'][index]
            assert args[args.index('--model-key')+1]==model and int(args[args.index('--shuffle-n')+1])==n
            if index:
                assert args[args.index('--previous-job')+1]==plan['jobs'][index-1]['job_id']
            else:assert '--previous-job' not in args
            assert time.monotonic()-started<64800
            root=Path(plan['job_root']);job=root/request['job_id']
            request_path=Path(plan['requests_root'])/(request['job_id']+'.json')
            assert json.loads(request_path.read_text())==request
            if job.exists():
                assert json.loads((job/'request.json').read_text())==request,'Existing job identity conflict'
            argv=['/usr/bin/python3','-I','-S','-B',str(runner),'submit',str(root),str(request_path)]
            # Removing parent markers prevents the independently supervised
            # child from being owned/cleaned as part of the orchestrator job.
            reply=json.loads(subprocess.check_output(argv,env=environment,text=True,timeout=60))
            stage={'job_id':request['job_id'],'acceptance':reply,'status':reply['status']}
            result['stages'].append(stage);save()
            while True:
                status=json.loads(subprocess.check_output(['/usr/bin/python3','-I','-S','-B',str(runner),
                    'query',str(job)],env=environment,text=True,timeout=45))
                stage['status']=status['status'];save()
                if status['status'] not in ['ACCEPTED','RUNNING']:
                    break
                assert time.monotonic()-started<64800,'Series18h hard wall reserve'
                time.sleep(30)
            assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup_verified'] and not status['worker_alive'] and not status['workgroup_live'],request['job_id']
            output=request['argv'][request['argv'].index('--output')+1]
            summary=json.loads((Path(output)/'summary.json').read_text())
            assert summary['status']=='PASS'
            stage['condition_summary']=summary;stage['cleanup_verified']=True;save()
        result['status']='PASS_ALL_FOUR_CONDITIONS'
    except BaseException as error:
        result['status']='FAIL_STOPPED_DEPENDENCY_CHAIN';result['error']=repr(error);raise
    finally:save()
    print(json.dumps(result))


if __name__=='__main__':main()
