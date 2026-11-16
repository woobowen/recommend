"""Task-scoped WSL deployment/collection using the existing Windows SSH transport."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import transport

REPO=Path('/home/addaswsw/recommend-worktrees/lc-rec')
LOCAL=Path('/home/addaswsw/recommend-support/beauty-hardening-cpu-6e1b211')
BASE='/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115'
PYTHON='/data/bwwu/envs/letter-20261110/bin/python'
FIXTURE='/data/bwwu/lc-rec/a0-beauty-20261115/SYNTHETIC_CPU_INTEGRATION_ONLY_271df1e33054_r0'
DATA='/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3'
SOURCE='/home/bwwu/grid-audits/sid-extensions-p0/transfers/beauty-infer-63f13def364d/release'

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['deploy','submit','query','collect']);p.add_argument('--commit',required=True);a=p.parse_args()
    a.commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=REPO,text=True).strip()
    LOCAL.mkdir(parents=True,exist_ok=True);t=transport.Transport('new-server')
    t.profile=dict(t.profile,remote_support=BASE) # scoped instance only; no shared profile edit
    release=BASE+'/releases/'+a.commit;root=BASE+'/SYNTHETIC_CPU_INTEGRATION_ONLY_'+a.commit[:12]
    runner=BASE+'/runner-beauty-cc1a1e277faa.py';jobs=BASE+'/cpu-jobs'
    job='letter-a0-beauty-cpu-runner-check-main-'+a.commit[:12]
    def save(n,v):transport.atomic(LOCAL/n,v)
    if a.mode=='deploy':
        t.identity()
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',a.commit,'lc_rec/a0_beauty'],cwd=REPO,text=True).splitlines()
        names=[x for x in names if Path(x).suffix in ('.py','.json') and '/evidence/' not in x]
        names+=['lc_rec/letter/config/a0_evaluation_v1.json','lc_rec/letter/review/final_audit_20261113/external/official_cce.py','lc_rec/letter/review/final_audit_20261113/statistics_verify.py','lc_rec/letter/scripts/formal_runner.py']
        names += ['lc_rec/shuffle/core.py','lc_rec/letter/config/author_instruments.json']
        names += ['lc_rec/letter/scripts/'+x for x in ('compat.py','formal_attention.py','formal_bnb_reload.py')]
        names += ['lc_rec/letter/vendor/LETTER/LETTER-LC-Rec/'+x for x in ('data.py','collator.py','evaluate.py','test_ddp.py','prompt.py','utils.py','modeling_letter.py','config/ds_z2_bf16.json')]
        archive=LOCAL/(a.commit+'.tar.gz')
        subprocess.run(['git','archive','--format=tar.gz','--output='+str(archive),a.commit,*names],cwd=REPO,check=True)
        remote_archive=BASE+'/'+archive.name;t.put(archive,remote_archive)
        code=f'''from pathlib import Path
import tarfile,json,hashlib,os,pwd,socket,datetime
base=Path({BASE!r});release=Path({release!r});archive=Path({remote_archive!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={transport.sha(archive)!r}
with tarfile.open(archive) as tf:
 members=tf.getmembers();assert all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 assert len({{m.name for m in members}})==len(members)
 release.mkdir(parents=True,exist_ok=True)
 for m in members:
  if not m.isfile():continue
  b=tf.extractfile(m).read();p=release/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==b
  else:p.write_bytes(b)
rows=[{{'path':str(release/m.name),'relative':m.name,'bytes':(release/m.name).stat().st_size,'sha256':hashlib.sha256((release/m.name).read_bytes()).hexdigest()}} for m in members if m.isfile()]
import sys
sys.path.insert(0,str(release/'lc_rec/a0_beauty'))
from derive_formal_runner import derive
runner=Path({runner!r});payload=derive(release/'lc_rec/letter/scripts/formal_runner.py')
assert hashlib.sha256(payload).hexdigest()=='cc1a1e277faa9bc713a4d0d1b620248d47c398e8ed17d4ed3d7ab52805cc657e'
if runner.exists():assert runner.read_bytes()==payload
else:runner.write_bytes(payload)
Path({jobs!r}).mkdir(exist_ok=True)
side={{'job_root':{jobs!r},'hostname':socket.gethostname(),'uid':os.getuid(),'user':pwd.getpwuid(os.getuid()).pw_name,'namespace':'beauty-production-closure-cpu-only'}}
sidepath=Path(str(runner)+'.deployment.json');b=(json.dumps(side,sort_keys=True)+'\\n').encode()
if sidepath.exists():assert sidepath.read_bytes()==b
else:sidepath.write_bytes(b)
# Inherit every known Beauty CPU attempt conservatively; nested test job costs
# will also be retained rather than creating a fresh 7200s allowance per name.
prior=[]
for p in list(Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs').glob('letter-beauty*/request.json'))+list(Path({jobs!r}).glob('*/request.json')):
 req=json.loads(p.read_text())
 if req.get('env',{{}}).get('CUDA_VISIBLE_DEVICES')!='':continue
 state=json.loads((p.parent/'status.json').read_text())
 assert state['status'] not in ('ACCEPTED','RUNNING') and state.get('cleanup',{{}}).get('verified'),str(p)
 if state.get('started_at') and state.get('ended_at'):
  seconds=(datetime.datetime.fromisoformat(state['ended_at'])-datetime.datetime.fromisoformat(state['started_at'])).total_seconds()
  prior.append({{'job_id':req['job_id'],'wall_seconds':seconds,'status_sha256':hashlib.sha256((p.parent/'status.json').read_bytes()).hexdigest()}})
version={{'run_commit':{a.commit!r},'release':str(release),'files':rows,'runner_sha256':hashlib.sha256(payload).hexdigest(),'runner':str(runner),'sidecar':side,'sidecar_sha256':hashlib.sha256(b).hexdigest(),'historical_cpu_costs':prior,'historical_cpu_wall_seconds':sum(r['wall_seconds'] for r in prior),'export_not_checkout':True}}
(release/'RUN_VERSION.json').write_text(json.dumps(version,indent=2)+'\\n')
print(json.dumps(version))
'''
        dep=json.loads(t.python(code,timeout=90));save('deployment.json',dep);print(json.dumps({'release':release,'files':len(dep['files']),'inherited_cpu_seconds':dep['historical_cpu_wall_seconds']}));return
    if a.mode=='submit':
        dep=json.loads((LOCAL/'deployment.json').read_text());assert dep['run_commit']==a.commit;remaining=int(7200-dep['historical_cpu_wall_seconds']-60)
        assert remaining>1800
        env={'CUDA_VISIBLE_DEVICES':'','CUDA_HOME':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','USE_TF':'0','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        ns={'host':dep['sidecar']['hostname'],'user':dep['sidecar']['user'],'uid':dep['sidecar']['uid'],'namespace':dep['sidecar']['namespace'],'deployment_sha256':dep['sidecar_sha256']}
        config=release+'/lc_rec/a0_beauty/config/formal_beauty_seed42.json'
        plan={'scope':'SYNTHETIC_TEST','evidence_scope':'SYNTHETIC_CPU_INTEGRATION_ONLY','run_commit':a.commit,'prediction_run_commit':a.commit,
              'root':root,'release':release,'python':PYTHON,'data_root':DATA,'source_root':SOURCE,'model':'SYNTHETIC_NO_MODEL',
              'config':config,'config_sha256':'24b568161f2d71d3c5962372a38683b2d33ee5d6d5064de0612215946a8106fd',
              'asset_sha256':{'index':'be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82','inter':'8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5','item':'74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330'},
              'runner':runner,'runner_sha256':dep['runner_sha256'],'job_root':jobs,'namespace':ns,'env':env,
              'task_id':a.commit[:12],'temporary_root':BASE,'fixture_root':FIXTURE,'stage_seconds':{s:1200 for s in ['train','verify_best_adapter','normal','normal_eval','s3','s3_eval','s4','s4_eval','paired','collect']},
              'budget':{'train_normal':700,'s3':400,'s4':400,'cpu':remaining-1600,'total':remaining-100,'cleanup_reserve':10},
              'inputs':dep['files']}
        # Synthetic model is a non-compute placeholder but still an absolute path.
        plan['model']=root+'/SYNTHETIC_NO_MODEL'
        local_plan=LOCAL/'plan.json';save('plan.json',plan);remote_plan=root+'/plan.json';t.put(local_plan,remote_plan)
        req=dict(schema=1,**ns,capture_version=a.commit,cwd=release,
                 argv=[PYTHON,release+'/lc_rec/a0_beauty/closure_acceptance.py','--plan',remote_plan],
                 env=env,timeout_seconds=remaining,inputs=[{'path':x['path'],'sha256':x['sha256']} for x in dep['files']]+[{'path':remote_plan,'sha256':transport.sha(local_plan)}],job_id=job)
        save('request.json',req);remote_req=BASE+'/'+job+'.json';t.put(LOCAL/'request.json',remote_req)
        reply=json.loads(t.ssh(shlex.join(['/usr/bin/python3','-I','-S','-B',runner,'submit',jobs,remote_req]),timeout=60));save('submit.json',reply);print(json.dumps(reply));return
    result=json.loads(t.ssh(shlex.join(['/usr/bin/python3','-I','-S','-B',runner,'query',jobs+'/'+job]),timeout=45));save('last-query.json',result)
    if a.mode=='query':
        print(json.dumps({k:result.get(k) for k in ['job_id','status','exit_code','cleanup_verified','worker_alive','workgroup_live']}))
        print(t.ssh(shlex.join(['tail','-n','35',jobs+'/'+job+'/work.log'])));return
    code=f'''from pathlib import Path
import json,hashlib,base64
roots=[(Path({jobs!r})/{job!r},'runner'),(Path({root!r}),'pipeline')];files={{}}
for child in Path({jobs!r}).glob('*'+{a.commit[:12]!r}+'*'):
 if child.name!={job!r}:roots.append((child,'child-jobs/'+child.name))
for base,prefix in roots:
 for p in base.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.log') and p.stat().st_size<2*1024**2 and not p.name.startswith('predictions') and 'training/checkpoint-' not in str(p):
   b=p.read_bytes();files[prefix+'/'+str(p.relative_to(base))]={{'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}}
print(json.dumps(files))'''
    got=json.loads(t.python(code,timeout=90));dest=REPO/'lc_rec/a0_beauty/evidence/production-closure/server'/a.commit[:12]
    for n,v in got.items():
        path=dest/n;path.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(v['data']);assert hashlib.sha256(b).hexdigest()==v['sha256'];path.write_bytes(b)
    for name in ('deployment.json','plan.json','request.json','submit.json','last-query.json'):
        (dest/name).write_bytes((LOCAL/name).read_bytes())
    print(json.dumps({'collected_files':len(got),'status':result['status']}))

if __name__=='__main__':main()
