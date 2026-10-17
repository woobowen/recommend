"""Read existing job identities and progress; never submit, restart or signal."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport,atomic

parser=argparse.ArgumentParser()
parser.add_argument('--queue-deployment',type=Path,required=True)
parser.add_argument('--label',required=True)
args=parser.parse_args();deployment=json.loads(args.queue_deployment.read_text())
t=Transport('new-server')
code=r'''import json,datetime,pathlib,subprocess,shutil,os
config_path=CONFIG_PATH;config=json.loads(pathlib.Path(config_path).read_text());state_path=pathlib.Path(config['state_dir'])/'queue-status.json'
state=json.loads(state_path.read_text()) if state_path.exists() else None
root=pathlib.Path(config['runner']['job_root'])
def query(job):
 path=root/job
 if not (path/'accepted.json').exists():return {'status':'NOT_SUBMITTED','planned_job_id':job}
 r=config['runner'];cmd=[r['python'],'-I','-S','-B',r['script'],'query',str(path)]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr
 status=json.loads(p.stdout);output=path/'output';details={}
 for name in ('progress.json','test/progress.json','latest_validation.json','stage_result.json','RESULT.json','stage-entry.json','stage-exit.json'):
  f=output/name
  if f.exists():
   details[name]={'mtime_utc':datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat(),'data':json.loads(f.read_text())}
 logs={}
 for name in ('work.log','supervisor.log'):
  f=path/name
  if f.exists():
   with f.open('rb') as stream:
    stream.seek(max(0,f.stat().st_size-5000));text=stream.read().decode('utf-8','replace')
   logs[name]={'mtime_utc':datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat(),'tail':text,'bytes':f.stat().st_size}
 first_update=output/'optimizer_updates.jsonl'
 if first_update.exists():
  with first_update.open() as stream:line=stream.readline()
  if line:details['first_optimizer_update']={'data':json.loads(line)}
 return {'status':status,'details':details,'logs':logs}
jobs={'ga-20261017-queue-v1':query('ga-20261017-queue-v1')}
if state:
 for stage in config['stages']:
  name=stage['request']['job_id']
  if (root/name/'accepted.json').exists():jobs[name]=query(name)
  else:jobs[name]={'status':'NOT_SUBMITTED','planned_job_id':name,'queue_stage_status':state['stages'][stage['id']]['status']}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
owners={}
for line in apps.splitlines():
 pid=line.split(',')[1].strip();owners[pid]=subprocess.check_output(['ps','-p',pid,'-o','pid=,user=,etime=,stat='],text=True).strip()
disk={p:{'total':shutil.disk_usage(p).total,'used':shutil.disk_usage(p).used,'free':shutil.disk_usage(p).free} for p in ('/home/bwwu','/data/bwwu')}
now=datetime.datetime.now(datetime.timezone.utc)
print(json.dumps({'observed_at_utc':now.isoformat(),'observed_at_cst':now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'queue_state':state,'jobs':jobs,'gpu':gpu,'compute_apps':apps,'process_owners':owners,'disk':disk,'loadavg':os.getloadavg(),
 'memory_available':next(l.strip() for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))}))
'''.replace('CONFIG_PATH',repr(deployment['remote_config']))
value=json.loads(t.python(code,timeout=60))
path=args.queue_deployment.parent/('health-'+args.label+'.json');atomic(path,value)
active={j:{'status':v['status']['status'],'progress':v['details'].get('progress.json',v['details'].get('test/progress.json',{})).get('data'),'upstream':v['details'].get('stage_result.json',{}).get('data',{}).get('items')} for j,v in value['jobs'].items() if isinstance(v['status'],dict)}
print(json.dumps({'at':value['observed_at_cst'],'record':str(path),'queue_status':value['queue_state']['status'] if value['queue_state'] else None,'received_jobs':active,'gpu':value['gpu'],'disk_free':{p:d['free'] for p,d in value['disk'].items()}},indent=2))
