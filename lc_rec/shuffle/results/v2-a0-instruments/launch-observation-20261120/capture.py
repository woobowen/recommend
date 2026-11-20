"""Capture a single live observation of the accepted first-group jobs."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from lc_rec.shuffle_v2 import control

OUTPUT = Path(__file__).resolve().parent
PARENT = control.PREFIX + 'group-series-r0'
CODE = r'''
from datetime import datetime, timezone
import json, os, re, shutil, subprocess
from pathlib import Path

def read(path):
    return json.loads(path.read_text())

def tail(path, howmany=8):
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-65536))
        return stream.read().decode('utf-8', 'replace').splitlines()[-howmany:]

def stamp(path):
    if not path.is_file():
        return None
    s=path.stat()
    return {'path':str(path),'bytes':s.st_size,'mtime_utc':datetime.fromtimestamp(s.st_mtime, timezone.utc).isoformat()}

def nvidia(*argv):
    return subprocess.check_output(['nvidia-smi', *argv],text=True).strip()

mark = datetime.now(timezone.utc).isoformat()
states = {}
for folder in sorted(Path(job_root).glob(prefix+'*')):
    state_file = folder/'status.json'
    if state_file.is_file():
        states[folder.name] = read(state_file)
tracked = {k:v for k,v in states.items() if any(t in k for t in ('group-series','gpu-formal-','cpu-eval-shuffle','cpu-summary'))}
reports = {}
for name,state in tracked.items():
    job_dir = Path(job_root)/name
    accepted = job_dir/'accepted.json'
    query = None
    if state['status'] in ('RUNNING','ACCEPTED') or state.get('ended_at'):
        query = json.loads(subprocess.check_output(['/usr/bin/python3','-I','-S','-B',runner,'query',str(job_dir)],text=True,timeout=30))
    worker = state.get('worker',{})
    record = {
        'status':state['status'],'start':state.get('started_at'),'end':state.get('ended_at'),'exit_code':state.get('exit_code'),
        'worker':worker,'worker_alive':query.get('worker_alive') if query else None,
        'owned_live':query.get('workgroup_observation',{}).get('owned_live',[]) if query else [],
        'cleanup_verified':query.get('cleanup_verified') if query else None,
        'accepted_request':read(accepted) if accepted.is_file() else None,
    }
    if 'gpu-formal-' in name:
        job=name.removeprefix(prefix)
        mode=job.removeprefix('gpu-formal-').rsplit('-',1)[0]
        output=Path(data_root)/job
        rank_file=output/'test/rank0.json'
        report=read(rank_file) if rank_file.is_file() else {}
        path=output/'test/predictions.rank0.jsonl'
        log=output/(mode+'.log')
        req=read(job_dir/'request.json')
        fields=('RUN_COMMIT','status','completed_users','assigned_users','world_size','rank','local_rank','model_load_and_strict_verification_seconds','elapsed_generation_seconds','gpu_peak_reserved_bytes','error')
        record.update(mode=mode, request_env_cuda=req['env']['CUDA_VISIBLE_DEVICES'], timeout_seconds=req['timeout_seconds'],
                      request_run=req['capture_version'], report={k:report.get(k) for k in fields},
                      prediction=stamp(path), unexpected_rank1=(output/'test/predictions.rank1.jsonl').exists(),
                      log=stamp(log), log_tail=tail(log), rank_report=stamp(rank_file))
        record['error_patterns']=[line for line in tail(log,30) if re.search(r'Traceback|CUDA out of memory|OutOfMemoryError|CHILDFAILEDERror|terminated by signal|\bFAIL\b',line,re.I)]
    reports[name]=record

gpus=nvidia('--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits')
compute=nvidia('--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits')
owned={int(m['pid']):name for name, record in reports.items() for m in record['owned_live']}
processes=[]
for line in compute.splitlines():
    uuid,pid,used=map(str.strip,line.split(',')); pid=int(pid)
    process={'uuid':uuid,'pid':pid,'memory_mib':int(used),'task_job':owned.get(pid)}
    if pid in owned:
        base=Path('/proc')/str(pid)
        try:
            fields=dict(pair.split('=',1) for pair in (base/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in pair)
            process['runtime_env']={k:fields.get(k) for k in ('CUDA_VISIBLE_DEVICES','WORLD_SIZE','RANK','LOCAL_RANK','GRID_JOB_ID')}
            process['argv']=(base/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        except FileNotFoundError:
            process['exited_during_capture']=True
    processes.append(process)
print(json.dumps({'observed_utc':mark,'host':os.uname().nodename,'gpus':[list(map(str.strip,r.split(','))) for r in gpus.splitlines()],
                  'compute_processes':processes,'stages':reports,'free_bytes':shutil.disk_usage(data_root).free,
                  'preflight_job':{k:v for k,v in states[prefix+'gpu-preflight-r0'].items() if k in ('status','exit_code','started_at','ended_at','cleanup')},
                  'task_job_ids':list(states)}))
'''


def take(label):
    c=control.connect()
    constants=f'job_root={c.profile["job_root"]!r}\nrunner={c.profile["runner"]!r}\nprefix={control.PREFIX!r}\ndata_root={control.ROOT!r}\n'
    record=json.loads(control.remote_python(c, constants+CODE, timeout=45))
    record['observation_label']=label
    record['observed_eastern']=datetime.fromisoformat(record['observed_utc']).astimezone(ZoneInfo('America/New_York')).isoformat()
    path=OUTPUT/f'{label}.json'
    assert not path.exists(), path
    control.transport.atomic(path,record)
    profiles=[(n,s.get('mode'),s.get('report',{}).get('completed_users'),s['status']) for n,s in record['stages'].items()]
    print(json.dumps({'label':label,'timestamp':record['observed_eastern'],'profiles':profiles,'gpus':record['gpus'],
                      'compute':record['compute_processes'],'free_bytes':record['free_bytes']},ensure_ascii=False),flush=True)
    return record


def main():
    take(sys.argv[1])


if __name__=='__main__':
    main()
