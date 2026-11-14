"""Read-only terminal audit capture after all four accepted jobs have succeeded.

This delivery helper is outside the immutable formal inference/evaluation RUN.
It cannot submit jobs, load models, or change the frozen server release.
"""
import datetime
import hashlib
import json
from pathlib import Path
from control import connection, SUPPORT, DATA, REMOTE, REPO
from collect_results import RUN


def main():
    t = connection()
    deployment = json.loads((SUPPORT/'receipts'/('deployment-'+RUN+'.json')).read_text())
    plan_path = SUPPORT/'requests'/('shuffle-'+RUN+'.series-plan.json')
    plan = json.loads(plan_path.read_text())
    series = 'letter-shuffle-instruments-seed42-20261114-series-r0'
    script = '''from pathlib import Path
import datetime,hashlib,json,subprocess
root=Path(JOB_ROOT);data=Path(DATA);release=Path(RELEASE)
version=json.loads((release/'RUN_VERSION.json').read_text())
canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert canonical(version)==DEPLOYMENT_SHA and version['RUN_COMMIT']==RUN
PLAN=json.loads(Path(PLAN_PATH).read_text());assert canonical(PLAN)==PLAN_SHA
source=[]
for row in version['files']:
 p=release/row['path'];assert p.is_file() and not p.is_symlink()
 raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 source.append(row)
runner=Path(RUNNER);assert hashlib.sha256(runner.read_bytes()).hexdigest()==RUNNER_SHA
documents={};queries={}
for name in [SERIES]+[j['job_id'] for j in PLAN['jobs']]:
 job=root/name
 state=json.loads(subprocess.check_output(['/usr/bin/python3','-I','-S','-B',str(runner),'query',str(job)],text=True))
 assert state['status']=='SUCCEEDED' and state['exit_code']==0 and state['cleanup_verified']
 assert not state['worker_alive'] and not state['workgroup_live']
 queries[name]=state
 for leaf in ['request.json','status.json','accepted.json','cleanup.json']:
  p=job/leaf;raw=p.read_bytes();assert len(raw)<2*1024**2 and not p.is_symlink()
  documents['jobs/'+name+'/'+leaf]={'source':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'json':json.loads(raw)}
for name in ['formal-series-r0/summary.json','formal_ledger.json']:
 p=data/name;raw=p.read_bytes();assert len(raw)<2*1024**2 and not p.is_symlink()
 documents[name]={'source':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'json':json.loads(raw)}
summary=documents['formal-series-r0/summary.json']['json']
assert summary['status']=='PASS_ALL_FOUR_CONDITIONS' and len(summary['stages'])==4
assert [x['job_id'] for x in summary['stages']]==[x['job_id'] for x in PLAN['jobs']]
assert all(x['status']=='SUCCEEDED' and x['cleanup_verified'] and x['condition_summary']['status']=='PASS' for x in summary['stages'])
attempts=documents['formal_ledger.json']['json']['attempts']
assert len(attempts)==4 and [x['job_id'] for x in attempts]==[x['job_id'] for x in PLAN['jobs']]
assert all(x['status']=='PASS' and x['generation_exit_code']==x['evaluation_exit_code']==0 and x['RUN_COMMIT']==RUN for x in attempts)
assert summary['wall_seconds']<=64800 and sum(x['runner_wall_seconds'] for x in attempts)<=64800
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)
print(json.dumps({'status':'PASS_FINAL_CONTROL_AND_DEPLOYMENT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'SHUFFLE_RUN_COMMIT':RUN,'release':str(release),'not_a_git_checkout':True,'source_files_rehashed':source,
 'runner_sha256':RUNNER_SHA,'queries':queries,'documents':documents,'gpu':gpu,'gpu_processes':apps,
 'cleanup_scope':'All five accepted job workgroups verified empty; GPU process snapshot retained without managing unrelated work.'}))
'''
    prelude = '\n'.join(k+'='+repr(v) for k,v in {
        'JOB_ROOT':t.profile['job_root'], 'DATA':DATA, 'RELEASE':REMOTE+'/releases/'+RUN,
        'DEPLOYMENT_SHA':hashlib.sha256(json.dumps(deployment,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'RUN':RUN, 'RUNNER':t.profile['runner'],
        'RUNNER_SHA':t.profile['runner_sha256'], 'SERIES':series,
        'PLAN_PATH':t.profile['remote_support']+'/requests/'+plan_path.name,
        'PLAN_SHA':hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()}.items())+'\n'
    report = json.loads(t.python(prelude+script, timeout=60))
    target = REPO/'lc_rec/shuffle/evidence/final_control'
    target.mkdir(parents=True, exist_ok=True)
    # This JSON embeds exact parsed server documents and their original-byte identities.
    raw=(json.dumps(report,indent=2)+'\n').encode()
    destination=target/'terminal_and_deployment.json'
    assert not destination.exists(), 'Preserve an existing final capture; inspect it instead of overwriting'
    destination.write_bytes(raw)
    observations=target/'observations';observations.mkdir()
    for path in sorted((SUPPORT/'observations').glob('*.json')):
        value=json.loads(path.read_text())
        assert any(j['job_id']==series for j in value['jobs'])
        assert path.stat().st_size<2*1024**2
        (observations/path.name).write_bytes(path.read_bytes())
    capture={'status':report['status'],'path':str(destination),'bytes':len(raw),
             'sha256':hashlib.sha256(raw).hexdigest(),'monitor_receipts':len(list(observations.iterdir())),
             'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (target/'capture.json').write_text(json.dumps(capture,indent=2)+'\n')
    print(json.dumps(capture))


if __name__=='__main__':main()
