"""Read-only end-of-task resource, job, asset and fixed-source receipts.

Does not import a GPU framework, start a job, remove files or alter environments.
Final publication receipts can live outside Git to avoid self-referential hashes.
"""
import argparse, datetime, hashlib, json, shutil, subprocess, sys
from pathlib import Path
import control


def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=control.REPO,text=True).strip();assert len(commit)==40
 t=control.transport.Transport('new-server')
 basis=json.loads(subprocess.check_output(['git','show',commit+':lc_rec/qwen_preflight_20261110/metadata/method_basis.json'],cwd=control.REPO,text=True))
 # The query code remains small enough for Windows OpenSSH's command-line limit.
 code="""import hashlib,json,os,shutil,subprocess,time;from pathlib import Path
root=Path('/home/bwwu/lc-rec/qwen-preflight-20261110');data=Path('/data/bwwu/lc-rec/qwen-preflight-20261110');jobs=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')
envs=[Path('/data/bwwu/envs/qwen-preflight-20261110-'+family) for family in ['qwen3','qwen35']]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'uid':os.getuid(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True),'memory':Path('/proc/meminfo').read_text(),'disk':{},'allocated_and_logical_bytes':{},'ledgers':{},'jobs':{},'releases':{},'assets':{}}
for base in [Path('/home/bwwu'),Path('/data/bwwu')]:
 d=shutil.disk_usage(base);r['disk'][str(base)]={'total':d.total,'free':d.free,'minimum_reserve':max(30*1024**3,d.total//10)};assert d.free>=max(30*1024**3,d.total//10)
for name in ['gpu-budget.json','update-budget.json']:
 p=root/name;r['ledgers'][name]={'sha256':sha(p),'content':json.loads(p.read_text())} if p.exists() else None
for base in [root,data]+envs+sorted(jobs.glob('qwen-20261110-*')):
 allocated=logical=0
 for parent,dirs,files in os.walk(base,followlinks=False):
  dirs[:]=[x for x in dirs if not (Path(parent)/x).is_symlink()]
  for name in files:
   p=Path(parent)/name
   if p.is_symlink():continue
   try:s=p.stat();allocated+=s.st_blocks*512;logical+=s.st_size
   except FileNotFoundError:pass
 r['allocated_and_logical_bytes'][str(base)]={'allocated':allocated,'logical':logical}
for job in sorted(jobs.glob('qwen-20261110-*')):
 r['jobs'][job.name]={name:json.loads((job/name).read_text()) for name in ['request.json','status.json','cleanup.json','attempt.json','accepted.json'] if (job/name).exists()}
for release in sorted((root/'releases').iterdir()):
 p=release/'RUN_VERSION.json'
 if not p.exists():continue
 version=json.loads(p.read_text());rows=[]
 for row in version['files']:
  if '/scripts/' not in row['path'] and '/config/' not in row['path']:continue
  actual=release/row['path'];value=sha(actual);assert value==row['sha256'];rows.append({'path':row['path'],'bytes':actual.stat().st_size,'sha256':value})
 r['releases'][release.name]={'RUN_VERSION':version,'RUN_VERSION_sha256':sha(p),'actual_script_config_rehash':rows,'not_a_git_checkout':True}
for model in sorted((data/'models').glob('*/*')):
 p=model/'ASSET_MANIFEST.json'
 if not p.exists():continue
 m=json.loads(p.read_text());r['assets'][str(model)]={'manifest':m,'manifest_sha256':sha(p),'actual_files':[{'path':str(q.relative_to(model)),'bytes':q.stat().st_size} for q in sorted(model.rglob('*')) if q.is_file() and not q.is_symlink()]}
available=int(next(line.split()[1] for line in r['memory'].splitlines() if line.startswith('MemAvailable:')))*1024;assert available>=40*1024**3
r['new_task_allocated_total_bytes']=sum(v['allocated'] for v in r['allocated_and_logical_bytes'].values());r['new_task_logical_total_bytes']=sum(v['logical'] for v in r['allocated_and_logical_bytes'].values());assert r['new_task_allocated_total_bytes']<=60*1024**3
"""
 code+="\nr['frozen_letter_sources']=[]\n"
 code+="for row in "+repr(basis['frozen_reference_files'])+":\n"
 code+=" p=Path('/home/bwwu/lc-rec/letter-20261110/releases/8e6399c2a1269dfde58d705eef25a499400dc59b')/row['path'];value=sha(p);assert value==row['sha256'];r['frozen_letter_sources'].append({**row,'server_path':str(p),'actual_sha256':value})\n"
 code+="print(json.dumps(r))\n"
 remote=json.loads(t.python(code,timeout=90))
 queries={}
 for job in remote['jobs']:queries[job]=control.gridctl.query(t,job)
 source_mapping=[]
 for release,record in remote['releases'].items():
  for row in record['actual_script_config_rehash']:
   final=subprocess.run(['git','show',commit+':'+row['path']],cwd=control.REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   if final.returncode==0:
    source_mapping.append({'run_commit':release,'path':row['path'],'run_sha256':row['sha256'],'delivery_comparison_commit':commit,'delivery_sha256':hashlib.sha256(final.stdout).hexdigest(),'identical':hashlib.sha256(final.stdout).hexdigest()==row['sha256']})
 r={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'argv':sys.argv,'interpreter':sys.executable,'comparison_commit':commit,'remote':remote,'live_runner_queries':queries,'fixed_source_mapping':source_mapping,'local':{'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=control.REPO,text=True).strip(),'git_status':subprocess.check_output(['git','status','--short'],cwd=control.REPO,text=True),'disk':{str(path):{'total':shutil.disk_usage(path).total,'free':shutil.disk_usage(path).free} for path in [control.SUPPORT,Path('/mnt/c')]}}}
 a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists(),'Keep previous receipts immutable';a.output.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'output':str(a.output),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'jobs':{k:{x:v[x] for x in ['status','exit_code','cleanup_verified','worker_alive','workgroup_live'] if x in v} for k,v in queries.items()},'new_task_allocated_bytes':remote['new_task_allocated_total_bytes']},indent=2))


if __name__=='__main__':main()
