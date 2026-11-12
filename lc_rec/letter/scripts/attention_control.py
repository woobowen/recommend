#!/usr/bin/env python3
"""Task-scoped diagnosis transport, preserving accepted runner and old evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import formal_control as fc
fc.SUPPORT=Path('/home/addaswsw/recommend-support/letter-attention-20261112')
fc.REMOTE='/home/bwwu/lc-rec/letter-attention-20261112'
REPO=fc.REPO
EVIDENCE=REPO/'lc_rec/letter/evidence/attention_20261112'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['snapshot','deploy','submit','query','collect','cpu'])
    p.add_argument('--commit');p.add_argument('--job');p.add_argument('--script');p.add_argument('--args',default='[]')
    p.add_argument('--timeout',type=int,default=1800);p.add_argument('--cpu-only',action='store_true')
    a=p.parse_args();EVIDENCE.mkdir(parents=True,exist_ok=True)
    if a.mode=='deploy':
        sys.argv=[sys.argv[0],'deploy','--commit',a.commit or 'HEAD'];return fc.main()
    t=fc.transport.Transport('new-server')
    def save(name,v):fc.transport.atomic(EVIDENCE/name,v)
    if a.mode=='snapshot':
        code="""from pathlib import Path
import json,hashlib,subprocess,os,datetime
base=Path('/data/bwwu/envs/letter-20261110/lib/python3.10/site-packages')
bases=[base,Path('/data/bwwu/envs/lc-rec-offload-20261108/lib/python3.10/site-packages'),Path('/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages')]
names=['bitsandbytes/nn/modules.py','bitsandbytes/autograd/_functions.py','bitsandbytes/functional.py','deepspeed/runtime/zero/stage_1_and_2.py','deepspeed/runtime/fp16/loss_scaler.py','torch/optim/adamw.py','peft/tuners/lora.py','fastchat/train/llama2_flash_attn_monkey_patch.py','torch/utils/checkpoint.py']
files={}
for name in names:
 p=next(b/name for b in bases if (b/name).is_file());data=p.read_text();files[name]={'source':str(p),'text':data,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
old=Path('/home/bwwu/lc-rec/letter-a0-normal-20261111/releases/87ba6556ce7237f8ef32c690165449d0d6a28294')
mapping=json.loads((old/'RUN_VERSION.json').read_text())
for row in mapping['files']:
 assert hashlib.sha256((old/row['path']).read_bytes()).hexdigest()==row['sha256']
cp=Path('/data/bwwu/lc-rec/a0-normal-20261111/letter-a0-20261111-acceptance-cont-r3/training/checkpoint-2')
rows=[]
for x in sorted(cp.rglob('*')):
 if x.is_file():rows.append({'path':str(x.relative_to(cp)),'bytes':x.stat().st_size,'mtime_ns':x.stat().st_mtime_ns,'sha256':hashlib.sha256(x.read_bytes()).hexdigest() if x.stat().st_size<8*1024**2 else None})
jobs={}
for name in ['letter-a0-20261111-acceptance-cont-r3','letter-a0-20261111-acceptance-resume-r3']:
 root=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')/name
 jobs[name]={f:json.loads((root/f).read_text()) for f in ['request.json','status.json','cleanup.json'] if (root/f).exists()}
commands={'identity':['id'],'gpu':['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],'compute':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],'ram':['free','-b'],'disk':['df','-B1','/data/bwwu','/home/bwwu'],'process':['ps','-u','bwwu','-o','pid,ppid,lstart,args']}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'historical_deployment':mapping,'checkpoint':rows,'jobs':jobs,'resource':{k:subprocess.check_output(v,text=True) for k,v in commands.items()}}))
"""
        v=json.loads(t.python(code,timeout=60));files=v.pop('files')
        for name,r in files.items():
            dest=EVIDENCE/'dependency_sources'/(name.replace('/','_')+'.txt');dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(r['text']);assert fc.transport.sha(dest)==r['sha256'];r.pop('text')
        v['dependency_files']=files;save('initial_snapshot.json',v)
        print(json.dumps({'sources':len(files),'historical_export_files':len(v['historical_deployment']['files']),'checkpoint_files':len(v['checkpoint']),'resources':v['resource']}));return
    commit=subprocess.check_output(['git','rev-parse',a.commit or 'HEAD'],cwd=REPO,text=True).strip()
    release=fc.REMOTE+'/releases/'+commit
    deployment=json.loads((fc.SUPPORT/'receipts'/('deployment-'+commit+'.json')).read_text())
    t.profile=dict(t.profile,runner=deployment['runner'],runner_sha256=deployment['runner_sha256'])
    if a.mode=='cpu':
        assert a.script and a.cpu_only
        old=json.loads((REPO/'lc_rec/letter/evidence/autograd_20261110/jobs/letter-20261110-autograd-cpu-r0/request.json').read_text())
        env=dict(old['env']);env['CUDA_VISIBLE_DEVICES']='';env.pop('DS_ACCELERATOR',None)
        cmd=['env','-u','DS_ACCELERATOR',*[k+'='+v for k,v in env.items()],fc.PYTHON,release+'/lc_rec/letter/scripts/'+a.script,*json.loads(a.args)]
        print(t.ssh(shlex.join(cmd),timeout=a.timeout));return
    assert a.job and a.job.startswith('letter-attention-20261112-');fc.gridctl.valid(a.job)
    if a.mode=='submit':
        assert a.script and a.timeout<=1800
        pending=json.loads(t.python("from pathlib import Path\nimport json\np=Path('/data/bwwu/lc-rec/attention-20261112/ledger.json')\nprint(json.dumps(json.loads(p.read_text()) if p.exists() else {'attempts':[]}))"))
        if not a.cpu_only:
            assert all(x.get('ended_utc') for x in pending['attempts']), 'query existing job before another submission'
        old=json.loads((REPO/'lc_rec/letter/evidence/autograd_20261110/jobs/letter-20261110-autograd-gpu-r0/request.json').read_text())
        req={k:old[k] for k in ['schema','host','user','uid','namespace','deployment_sha256']}
        env=dict(old['env']);env['PYTHONUNBUFFERED']='1'
        if a.cpu_only:env['CUDA_VISIBLE_DEVICES']='';env.pop('DS_ACCELERATOR',None)
        args=[commit if x=='@RUN_COMMIT@' else x for x in json.loads(a.args)]
        if '--run-commit' in args:assert args[args.index('--run-commit')+1]==commit
        req.update(job_id=a.job,capture_version=commit,cwd=release,argv=[fc.PYTHON,release+'/lc_rec/letter/scripts/'+a.script,*args],env=env,timeout_seconds=a.timeout,
          inputs=[{'path':release+'/'+r['path'],'sha256':r['sha256']} for r in deployment['files'] if r['path'].startswith(('lc_rec/letter/scripts/','lc_rec/letter/config/','lc_rec/letter/review/attention_20261112/'))])
        path=fc.SUPPORT/'requests'/(a.job+'.json')
        if path.exists():assert json.loads(path.read_text())==req
        else:fc.transport.atomic(path,req)
        result=fc.gridctl.submit(t,path)
    else:
        result=fc.gridctl.query(t,a.job)
        if a.mode=='collect':
            code=f"""from pathlib import Path
import json,hashlib
roots=[(Path({t.profile['job_root']!r})/{a.job!r},'job'),(Path('/data/bwwu/lc-rec/attention-20261112')/{a.job!r},'run')]
out={{}}
for root,prefix in roots:
 for p in root.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.jsonl','.log'] and p.stat().st_size<8*1024**2:
   data=p.read_bytes();out[prefix+'/'+str(p.relative_to(root))]={{'text':data.decode(),'sha256':hashlib.sha256(data).hexdigest()}}
print(json.dumps(out))
"""
            for name,r in json.loads(t.python(code,timeout=60)).items():
                dest=EVIDENCE/'jobs'/a.job/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(r['text']);assert fc.transport.sha(dest)==r['sha256']
    save(a.mode+'-'+a.job+'.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts','worker_group','cleanup']},indent=2))

if __name__=='__main__':main()
