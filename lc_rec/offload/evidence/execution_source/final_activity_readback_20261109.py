import json,os,sys,datetime
from pathlib import Path
import transport
root=Path(__file__).resolve().parents[1];t=transport.Transport('new-server')
code='''from pathlib import Path
import os,json,subprocess,datetime
jobs=['lc-rec-offload-20261108-environment-r'+str(i) for i in range(4)]+['lc-rec-offload-20261109-bounded']
jobroot=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs');out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]}
for job in jobs:
 p=jobroot/job/'status.json';c=jobroot/job/'cleanup.json'
 d=json.loads(p.read_text());out['jobs'].append({'job_id':job,'status':d['status'],'exit_code':d.get('exit_code'),'cleanup_file_exists':c.exists()})
cmd=subprocess.run(['ps','-u',str(os.getuid()),'-o','pid=,etimes=,args='],capture_output=True,text=True)
active=[]
for line in cmd.stdout.splitlines():
 fields=line.strip().split(None,2)
 if len(fields)!=3:continue
 args=fields[2].split()
 matching=[a for a in args if a.startswith(('/home/bwwu/lc-rec/','/home/bwwu/grid-audits/sid-extensions-p0/jobs/lc-rec-offload-')) and (a.endswith('.py') or a.endswith('lc-rec-offload-20261109-bounded'))]
 if matching:active.append({'pid':int(fields[0]),'elapsed_seconds':int(fields[1]),'matched_task_paths':matching})
out['active_owned_task_processes']=active
r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],capture_output=True,text=True,timeout=10);out['gpu_processes']={'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
r=subprocess.run(['free','-b'],capture_output=True,text=True);out['memory']=r.stdout
print(json.dumps(out))'''
s=t.python(code,timeout=40);Path('/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/offload/evidence/preparation/final_activity_20261109.json').write_text(s)
print(s)
# WSL process names/IDs only; never read secret environment or arbitrary argv payloads.
import subprocess
v=subprocess.run(['ps','-u',str(os.getuid()),'-o','pid=,args='],capture_output=True,text=True)
active=[]
for line in v.stdout.splitlines():
 fields=line.strip().split(None,1)
 if len(fields)!=2:continue
 matching=[x for x in fields[1].split() if x.startswith(str(root.parent)+'/') and x.endswith('.py') and any(z in x for z in ['stable_download','relay_weights','verify_server_weights','collect_metadata'])]
 if matching:active.append({'pid':int(fields[0]),'task_script_paths':matching})
p=Path('/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/offload/evidence/preparation/final_local_asset_activity_20261109.json');p.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'active_known_asset_collectors':active,'scope':'Owned processes with exact task script arguments; no environment/secret read'},indent=2)+'\n')
print(json.dumps({'active_known_local_asset_collectors':active}))
