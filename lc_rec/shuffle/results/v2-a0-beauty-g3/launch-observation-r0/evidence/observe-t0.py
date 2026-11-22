"""Collect seven read-only snapshots, and an optional separate delivery check."""
import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from lc_rec.shuffle_v2 import g3_control as gc

EASTERN = ZoneInfo('America/New_York')
EMPTY = {'prediction': {'complete_bytes':0,'complete_users':0,'bytes':0}, 'logs':{}}
REMOTE = r"""
import base64,datetime,hashlib,json,os,re,socket,subprocess,sys,time
from pathlib import Path

job_root=Path(o['job_root'])
gpu_root=Path(o['gpu_output'])
parent_root=Path(o['parent_output'])
ts=time.time_ns()
now=datetime.datetime.fromtimestamp(ts/1e9,datetime.timezone.utc)

def read(path):
 return json.loads(path.read_bytes()) if path.is_file() else None

def query(job):
 if not (job_root/job/'status.json').is_file(): return None
 return json.loads(subprocess.check_output(
  [sys.executable,'-I','-S','-B',o['runner'],'query',str(job_root/job)],text=True,timeout=20))

def file_info(s):
 return {'bytes':s.st_size,'mtime_utc':datetime.datetime.fromtimestamp(s.st_mtime_ns/1e9,datetime.timezone.utc).isoformat(),
         'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev}

def log_part(path,previous):
 if not path.is_file(): return None
 with path.open('rb') as stream:
  s=os.fstat(stream.fileno())
  start=max(previous,s.st_size-32768,0)
  assert start<=s.st_size
  stream.seek(start)
  raw=stream.read(s.st_size-start)
 return {**file_info(s),'path':str(path),'offset_start':start,'offset_end':s.st_size,'delta_bytes':s.st_size-previous,
         'excerpt_sha256':hashlib.sha256(raw).hexdigest(),'excerpt':raw.decode('utf-8','replace')}

def rank_processes(state):
 rows=[]
 if not state: return rows
 wanted={'GRID_JOB_ID','GRID_JOB_DIR','CUDA_VISIBLE_DEVICES','CUDA_HOME','LD_LIBRARY_PATH',
         'DS_ACCELERATOR','WORLD_SIZE','RANK','LOCAL_RANK','TORCHINDUCTOR_COMPILE_THREADS','MASTER_PORT'}
 for member in state['workgroup_observation']['owned_live']:
  proc=Path('/proc')/str(member['pid'])
  try:
   fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
   if fields[19]!=str(member['start_ticks']) or proc.stat().st_uid!=member['uid']: continue
   argv=(proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
   env={}
   for kv in (proc/'environ').read_bytes().split(b'\0'):
    k,ok,v=kv.partition(b'=')
    if ok and k.decode(errors='replace') in wanted: env[k.decode()]=v.decode(errors='replace')
   if env.get('GRID_JOB_ID')==o['child'] and 'RANK' in env and 'lc_rec.shuffle_v2.g3_generate' in argv:
    rows.append({**member,'argv':argv,'rank_env':env,'proc_parent_pid':int(fields[1]),
                 'cwd':str((proc/'cwd').resolve()),'state':fields[0]})
  except (FileNotFoundError,ProcessLookupError): pass
 return rows

def predictions(path,old,rank):
 if not path.is_file():
  assert old['complete_users']==old['complete_bytes']==0
  return {**old,'bytes':0,'delta_users':0,'delta_bytes':0,'buffered_suffix_bytes':0}
 with path.open('rb') as stream:
  s=os.fstat(stream.fileno())
  start=old['complete_bytes']
  assert s.st_size>=start and ('inode' not in old or s.st_ino==old['inode'])
  stream.seek(start)
  raw=stream.read(s.st_size-start)
 assert len(raw)==s.st_size-start
 cut=raw.rfind(b'\n')+1
 body=raw[:cut]
 assert rank and rank['scope']=='FORMAL'
 assigned=rank['assigned_row_indices']
 start_count=old['complete_users']
 last=old.get('last_row_idx')
 for offset,line in enumerate(body.splitlines()):
  row=json.loads(line)
  assert (row['row_idx'],row['rank'],row['scope'],row['condition'])==(assigned[start_count+offset],0,'FORMAL','shuffle2')
  last=row['row_idx']
 count=body.count(b'\n')
 return {**file_info(s),'path':str(path),'complete_bytes':start+cut,'complete_users':start_count+count,
         'delta_users':count,'delta_bytes':s.st_size-old['bytes'],'delta_complete_bytes':cut,'delta_offset_start':start,
         'delta_complete_sha256':hashlib.sha256(body).hexdigest(),'buffered_suffix_bytes':len(raw)-cut,'last_row_idx':last}

def ledger():
 groups={'formal_gpu':{},'preflight_gpu':{},'cpu':{},'parent_wall':{}}
 prefixes=(('gpu-formal-','formal_gpu'),('gpu-preflight-','preflight_gpu'),('cpu-','cpu'),('group-formal-','parent_wall'))
 for item in job_root.glob('g3-beauty-v2-*'):
  record=read(item/'status.json')
  if not record: continue
  group=next((k for prefix,k in prefixes if item.name.startswith('g3-beauty-v2-'+prefix)),None)
  if not group: continue
  start,finish=record.get('started_at'),record.get('ended_at')
  seconds=max(0.0,((datetime.datetime.fromisoformat(finish) if finish else now)-datetime.datetime.fromisoformat(start)).total_seconds()) if start else 0.0
  groups[group][item.name]={'status':record['status'],'started_at':start,'ended_at':finish,'worker_seconds':seconds}
 limits={'formal_gpu':129600,'preflight_gpu':10800,'cpu':28800,'parent_wall':126000}
 return {k:{'used_seconds':sum(v['worker_seconds'] for v in g.values()),'limit_seconds':limits[k],
            'remaining_seconds':limits[k]-sum(v['worker_seconds'] for v in g.values()),'jobs':g}
         for k,g in groups.items()}

parent=query(o['parent'])
child=query(o['child'])
rank=read(gpu_root/'rank0.json')
previous=o['previous']
pred=predictions(gpu_root/'predictions.rank0.jsonl',previous['prediction'],rank)
logs={}
for key,job,filename in [('gpu_worker',o['child'],'work.log'),('gpu_supervisor',o['child'],'supervisor.log'),
                         ('parent_worker',o['parent'],'work.log'),('parent_supervisor',o['parent'],'supervisor.log')]:
 logs[key]=log_part(job_root/job/filename,previous['logs'].get(key,0))
errors_re=re.compile(r'Traceback \(most recent|RuntimeError:|AssertionError:|ValueError:|CUDA out of memory|NCCL (?:error|failure)|ChildFailedError',re.I)
errors=[{'source':k,'line':line} for k,v in logs.items() if v for line in v['excerpt'].splitlines() if errors_re.search(line)]
def smi(fields,keys):
 return [dict(zip(keys,r.split(', '),strict=True)) for r in subprocess.check_output(
  ['nvidia-smi',fields,'--format=csv,noheader,nounits'],text=True).splitlines()]
gpus=smi('--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
         ('physical_index','uuid','name','memory_used_mib','memory_total_mib','utilization_pct'))
gpu,=[row for row in gpus if row['uuid']==o['gpu_uuid']]
apps=smi('--query-compute-apps=gpu_uuid,pid,used_gpu_memory',('uuid','pid','memory_used_mib'))
disk=os.statvfs(gpu_root.parent)
rank_keys=('status','scope','rank','world_size','local_rank','conditions','assigned_users','completed_users',
           'assignment_sha256','model_load_and_strict_verification_seconds','elapsed_generation_seconds','model_identity_sha256',
           'v2_model','RUN_COMMIT','MODEL_TRAINING_COMMIT','allowed_callback_sha256','error')
point=dict(utc=now.isoformat(),unix_ns=ts,host=socket.gethostname(),parent=parent,child=child,controller=read(parent_root/'summary.json'),
           gpu_rank_processes=rank_processes(child),rank={k:rank.get(k) for k in rank_keys} if rank else None,
           prediction=pred,logs=logs,gpu=gpu,all_gpus=gpus,compute_apps=apps,
           output_free_bytes=disk.f_bavail*disk.f_frsize,error_matches=errors,budget=ledger())
if o['need_first_six'] and pred['complete_users']>=6:
 with (gpu_root/'predictions.rank0.jsonl').open('rb') as stream:
  lines=[stream.readline() for _ in range(6)]
  assert all(line.endswith(b'\n') for line in lines)
  bound=stream.tell()
 raw=b''.join(lines)
 point['first_six_bundle']={'prediction_bytes_b64':base64.b64encode(raw).decode(),'prediction_sha256':hashlib.sha256(raw).hexdigest(),
                           'start_offset':0,'end_offset_exclusive':bound,'rank':rank,'binding':read(gpu_root/'v2-binding.rank0.json'),
                           'source_path':str(gpu_root/'predictions.rank0.jsonl'),'captured_utc':now.isoformat()}
print(json.dumps(point,ensure_ascii=False,allow_nan=False))
"""


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def capture(target, plan, previous, want_six):
    parent = plan['parent']
    first = plan['modes']['shuffle2']['gpu']
    p_args, g_args = parent['request']['argv'], first['request']['argv']
    options = {'job_root':plan['job_root'],'runner':plan['runner'],'gpu_uuid':plan['card'],
               'parent':parent['name'],'child':first['name'],'parent_output':p_args[p_args.index('--output')+1],
               'gpu_output':g_args[g_args.index('--output')+1],'previous':previous,'need_first_six':want_six}
    point = json.loads(gc.control.remote_python(target, 'o = '+repr(options)+'\n'+REMOTE, timeout=55))
    point['america_new_york']=dt.datetime.fromisoformat(point['utc']).astimezone(EASTERN).isoformat()
    return point


def stem(point):
    p = point['prediction']
    return {'prediction':{k:p[k] for k in ('complete_users','complete_bytes','bytes','inode','last_row_idx') if k in p},
            'logs':{k:v['bytes'] for k,v in point['logs'].items() if v}}


def live(point):
    return all(s and s['status']=='RUNNING' and s['supervisor_alive'] and s['worker_alive']
               for s in (point['parent'],point['child']))


def check(point):
    assert live(point)
    rank, = point['gpu_rank_processes']
    env=rank['rank_env']
    assert (env['WORLD_SIZE'],env['RANK'],env['LOCAL_RANK'],env['CUDA_VISIBLE_DEVICES'])==('1','0','0',point['gpu']['uuid'])
    assert not point['error_matches'],point['error_matches']
    apps=[a for a in point['compute_apps'] if a['uuid']==point['gpu']['uuid']]
    assert all(int(a['pid'])==rank['pid'] for a in apps)
    if point['prediction']['complete_users']:
        assert len(apps)==1
        report=point['rank']
        assert report['scope']=='FORMAL' and report['assigned_users']==22363
        assert 0 <= report['completed_users'] <= point['prediction']['complete_users'] < 22363
    assert point['budget']['formal_gpu']['remaining_seconds']>0 and point['output_free_bytes']>=20*1024**3


def print_point(label,p):
    print(json.dumps({'label':label,'time':p['america_new_york'],'users':p['prediction']['complete_users'],
                      'delta':p['prediction']['delta_users'],'prediction_bytes':p['prediction']['bytes'],
                      'memory_mib':p['gpu']['memory_used_mib'],'instant_gpu_percent':p['gpu']['utilization_pct'],
                      'rank_pid':p['gpu_rank_processes'][0]['pid']},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    options=parser.parse_args()
    target,_,_=gc.connection()
    plan=json.loads((options.root/'formal-plan.json').read_text())
    out=options.root/'observations'
    if options.final:
        prior=json.loads((out/'T30.json').read_text())
        point=capture(target,plan,stem(prior),False)
        write(options.root/'delivery-check.json',point)
        check(point)
        print_point('delivery-check',point)
        return
    out.mkdir(exist_ok=False)
    previous=EMPTY
    until=time.monotonic()+150
    while True:
        first=capture(target,plan,previous,True)
        if live(first) and len(first['gpu_rank_processes'])==1:
            break
        if any(s and s.get('ended_at') for s in (first['parent'],first['child'])) or time.monotonic()>=until:
            write(options.root/'startup-blocker.json',first)
            raise RuntimeError('Formal parent or S2 rank is absent or terminal. Query the original IDs.')
        time.sleep(2)
    tick=time.monotonic()
    first_identity=[first['gpu_rank_processes'][0][k] for k in ('pid','start_ticks','boot_id')]
    saved_six=False
    for minute in range(0,31,5):
        retries=[]
        if minute==0:
            point=first
        else:
            time.sleep(max(0,tick+minute*60-time.monotonic()))
            for retry in range(3):
                try:
                    point=capture(target,plan,previous,not saved_six)
                    break
                except (OSError,RuntimeError,ValueError) as exc:
                    retries.append({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'error':repr(exc)})
                    if retry==2:
                        write(options.root/'capture-error.json',retries)
                        raise
                    time.sleep(3)
        label=f'T{minute:02}'
        point['label']=label
        point['elapsed_since_t0_seconds']=(point['unix_ns']-first['unix_ns'])/1e9
        point['transport_retries']=retries
        bundle=point.pop('first_six_bundle',None)
        if bundle:
            raw=base64.b64decode(bundle.pop('prediction_bytes_b64'))
            assert raw.count(b'\n')==6 and len(raw)==bundle['end_offset_exclusive']
            assert hashlib.sha256(raw).hexdigest()==bundle['prediction_sha256']
            with (options.root/'first-six.jsonl').open('xb') as stream: stream.write(raw)
            bundle['observation']=label
            write(options.root/'first-six-capture.json',bundle)
            saved_six=True
        write(out/(label+'.json'),point)
        check(point)
        assert [point['gpu_rank_processes'][0][k] for k in ('pid','start_ticks','boot_id')]==first_identity
        if minute>0: assert point['prediction']['complete_users']>previous['prediction']['complete_users']
        print_point(label,point)
        previous=stem(point)
    assert saved_six and point['elapsed_since_t0_seconds']>=1800
    write(options.root/'window-complete.json',{'t0_utc':first['utc'],'t30_utc':point['utc'],
        'span_seconds':point['elapsed_since_t0_seconds'],'seven_observations':True,'scope':'formal S2 startup'})


if __name__=='__main__':
    main()
