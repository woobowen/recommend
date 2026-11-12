"""Lightweight, read-only server observation of one immutable A0 job ID."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import formal_control as fc


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--commit',required=True)
    p.add_argument('--output');a=p.parse_args()
    assert a.job.startswith('letter-a0-20261112-');fc.gridctl.valid(a.job)
    commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=fc.REPO,text=True).strip()
    t=fc.transport.Transport('new-server')
    deployment=json.loads((fc.SUPPORT/'receipts'/('deployment-'+commit+'.json')).read_text())
    t.profile=dict(t.profile,runner=deployment['runner'],runner_sha256=deployment['runner_sha256'])
    state=fc.gridctl.query(t,a.job)
    code="""import json,subprocess,datetime,os
from pathlib import Path
job=Path(%r);root=Path('/data/bwwu/lc-rec/a0-normal-20261111');output=root/%r
request=json.loads((job/'request.json').read_text());kind=request['argv'][request['argv'].index('--kind')+1] if '--kind' in request['argv'] else 'cpu'
training=root/'formal_seed42/training' if kind=='formal' else output/'training'
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':job.name,'kind':kind,'training':str(training),'rank':{}}
for rank in [0,1]:
 result={};report=training/('rank%%d.json'%%rank)
 if report.exists():
  data=json.loads(report.read_text());result['report']={k:data[k] for k in ['status','run_commit','config_sha256','global_step','epoch','best_checkpoint','best_metric','initial_global_step','elapsed_seconds','peak_allocated_bytes','peak_reserved_bytes','formal_attention','error'] if k in data}
 events=training/('events.rank%%d.jsonl'%%rank)
 if events.exists():
  with events.open('rb') as f:
   size=events.stat().st_size;offset=max(0,size-65536);f.seek(offset);lines=f.read().splitlines()
  if offset:lines=lines[1:]
  rows=[json.loads(x) for x in lines if x.strip()]
  result['recent_events']=[v for v in rows if v['event'] not in ['batch','micro_loss']][-8:]
 r['rank'][str(rank)]=result
for key,path in [('pipeline',job/'output/pipeline.json'),('best_selection',training/'best_selection.json'),('metrics',output/'test/metrics.json'),('ledger',root/('formal_seed42_ledger.json' if kind=='formal' else 'formal_det_preflight_20261112_ledger.json'))]:
 if path.exists():r[key]=json.loads(path.read_text())
r['test_rank']={}
for rank in [0,1]:
 path=output/('test' if kind=='formal' else 'test_preflight')/('rank%%d.json'%%rank)
 if path.exists():
  v=json.loads(path.read_text());r['test_rank'][str(rank)]={k:v[k] for k in ['status','completed_users','assigned_users','elapsed_generation_seconds','elapsed_seconds','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','error'] if k in v}
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if len(x.split())>=2};disk=os.statvfs(root)
r['mem_available']=mem['MemAvailable'];r['swap_used']=mem['SwapTotal']-mem['SwapFree'];r['disk_available']=disk.f_bavail*disk.f_frsize
print(json.dumps(r))
"""%(t.profile['job_root']+'/'+a.job,a.job)
    result=json.loads(t.python(code,timeout=45))
    result['runner']={k:state.get(k) for k in ['status','started_at','ended_at','exit_code','request_sha256','worker_alive','supervisor_alive','cleanup_verified','workgroup_observation']}
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target=Path(a.output) if a.output else fc.REPO/'lc_rec/letter/evidence/formal_det_20261112/observations'/a.job/(stamp+'.json')
    fc.transport.atomic(target,result)
    brief={'utc':result['utc'],'job_id':a.job,'runner':result['runner']['status'],
           'phase':result.get('pipeline',{}).get('phase'),'ranks':{},'test':result['test_rank'],
           'gpu':result['gpu'],'mem_available':result['mem_available'],'disk_available':result['disk_available']}
    for rank,v in result['rank'].items():
        brief['ranks'][rank]={'report':v.get('report',{}),'recent_events':v.get('recent_events',[])[-2:]}
    print(json.dumps(brief,indent=2))


if __name__=='__main__':main()
