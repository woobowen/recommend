"""Read-only A1 observation; this local monitor never starts/stops computation."""
import argparse,datetime,importlib.util,json,subprocess,time
from pathlib import Path

REPO=Path('/home/addaswsw/recommend-worktrees/lc-rec')
SUPPORT=Path('/home/addaswsw/recommend-support/a1-qwen3-06b-instruments-20261113')
TASK=REPO/'lc_rec/a1_qwen3_06b'
RUN='0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5'
JOB='letter-a1-q06-20261113-normal-seed42-r0'

def once():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    p=SUPPORT/'monitor';p.mkdir(exist_ok=True)
    r=subprocess.run(['python3',str(TASK/'scripts/control.py'),'observe','--commit',RUN,'--job',JOB],cwd=REPO,capture_output=True,text=True,timeout=120)
    (p/(stamp+'.stdout')).write_text(r.stdout)
    (p/(stamp+'.stderr')).write_text(r.stderr)
    if r.returncode:raise RuntimeError('read-only observation failed; original server job is untouched')
    raw=json.loads(sorted((TASK/'evidence/observations'/JOB).glob('*.json'))[-1].read_text())
    spec=importlib.util.spec_from_file_location('a1_control',TASK/'scripts/control.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    code="""from pathlib import Path
import json
root=Path('/data/bwwu/lc-rec/a1-qwen3-06b-instruments-20261113/formal_seed42/training')
rows=[]
for step in [2060,4120,6180,8240]:
 p=root/('checkpoint-'+str(step))
 if (p/'complete.json').is_file() and (p/'trainer_state.json').is_file():
  v=json.loads((p/'trainer_state.json').read_text())
  rows.append({'checkpoint':str(p),'step':v['global_step'],'best_checkpoint':v['best_model_checkpoint'],'best_metric':v['best_metric'],'validation':[x for x in v['log_history'] if 'eval_loss' in x]})
print(json.dumps(rows))
"""
    checkpoints=json.loads(c.transport.Transport('new-server').python(code,timeout=45))
    raw['sealed_checkpoints_readonly']=checkpoints
    out=TASK/'evidence/observations'/JOB/(stamp+'-monitor.json')
    out.write_text(json.dumps(raw,indent=2,allow_nan=False)+'\n')
    rank={}
    for key,v in raw['rank'].items():
        steps=[x for x in v['events'] if x['event']=='optimizer_step']
        logs=[x for x in v['events'] if x['event']=='log']
        rank[key]={'status':v.get('status'),'step':v.get('global_step',steps[-1]['step'] if steps else None),'epoch':v.get('epoch',steps[-1]['epoch'] if steps else None),'last_step':steps[-1] if steps else None,'last_log':logs[-1] if logs else None}
    summary={'utc':raw['utc'],'job_id':JOB,'runner':raw['runner'],'phase':raw.get('pipeline',{}).get('phase'),'rank':rank,'test_rank':raw['test_rank'],'checkpoints':checkpoints,'gpu':raw['gpu'],'resources':raw['resources'],'raw_receipt':str(out)}
    with (p/'summary.jsonl').open('a') as f:f.write(json.dumps(summary,allow_nan=False)+'\n')
    print(json.dumps(summary,allow_nan=False),flush=True)
    return raw['runner']['status']

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        try:status=once()
        except Exception as e:
            print(json.dumps({'monitor_error':repr(e),'server_computation':'NOT_CANCELLED','recovery':'query original job_id; never resubmit'}),flush=True)
            if not a.watch:raise
            status='OBSERVATION_UNAVAILABLE'
        if not a.watch or status in ['SUCCEEDED','FAILED','TIMED_OUT','CANCELLED']:return
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:time.sleep(min(60,max(0,deadline-time.monotonic())))

if __name__=='__main__':main()
