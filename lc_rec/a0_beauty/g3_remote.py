#!/usr/bin/env python3
"""WSL client for the existing Windows OpenSSH transport and accepted runner."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

SUPPORT=Path('/home/addaswsw/recommend-support/sid-extensions-p0')
sys.path.insert(0,str(SUPPORT/'scripts'))
from transport import Transport, atomic, sha
import gridctl

REPO=Path(__file__).resolve().parents[2]
REMOTE='/home/bwwu/grid-audits/sid-extensions-p0/g3-production-20261118'
MODEL='/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549'
PYTHON='/data/bwwu/envs/letter-20261110/bin/python'
ENV={'CUDA_HOME':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4',
     'LD_LIBRARY_PATH':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4/lib64:/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages/torch/lib',
     'DS_ACCELERATOR':'cuda','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2',
     'OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
     'TOKENIZERS_PARALLELISM':'false','TORCHINDUCTOR_COMPILE_THREADS':'1','PYTHONOPTIMIZE':'0'}


def selected():
    names=subprocess.check_output(['git','ls-files','lc_rec/a0_beauty','lc_rec/letter/scripts',
        'lc_rec/letter/vendor','lc_rec/shuffle','lc_rec/letter/config'],cwd=REPO,text=True).splitlines()
    keep=[]
    for n in names:
        p=Path(n)
        if '/evidence/' in n or '/review/' in n:continue
        if p.suffix in ('.py','.json','.yaml','.yml','.csv'):
            keep.append(n)
    keep.extend(['lc_rec/letter/evidence/formal_20261111/precision_approval.json',
        'lc_rec/a0_beauty/evidence/e2e-20261116-r0/continuous/epoch1_rows.json',
        'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
        'lc_rec/letter/review/final_audit_20261113/statistics_verify.py'])
    return sorted(set(keep))


def deploy(t):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    base=REMOTE+'/'+commit[:12];release=base+'/release'
    files=selected()
    dirty=subprocess.check_output(['git','diff','HEAD','--',*files],cwd=REPO)
    if dirty:raise ValueError('commit run source before deploying')
    local=SUPPORT/'packages'/('g3-'+commit[:12]+'.tar');local.parent.mkdir(exist_ok=True)
    rows=[]
    with tarfile.open(local,'w') as tar:
        for name in files:
            p=REPO/name
            # Git byte comparison catches accidental deployment of a different working copy.
            fixed=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO)
            if fixed!=p.read_bytes():raise ValueError('source differs from fixed commit '+name)
            rows.append({'path':release+'/'+name,'relative':name,'bytes':p.stat().st_size,'sha256':sha(p)})
            tar.add(p,arcname=name,recursive=False)
    manifest={'run_commit':commit,'release':release,'files':rows}
    mf=local.with_suffix('.manifest.json');atomic(mf,manifest)
    t.put(local,base+'/source.tar');t.put(mf,base+'/deployment.json')
    code=f"""import pathlib,tarfile,json,hashlib,os
base=pathlib.Path({base!r});release=base/'release';release.mkdir(exist_ok=True)
with tarfile.open(base/'source.tar') as tar:
 for m in tar.getmembers():
  p=pathlib.PurePosixPath(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts
 tar.extractall(release,filter='data')
manifest=json.loads((base/'deployment.json').read_text())
for row in manifest['files']:
 p=pathlib.Path(row['path']);assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
(release/'RUN_VERSION.json').write_text(json.dumps(manifest,indent=2)+'\\n')
data=pathlib.Path({(REMOTE+'/data/Beauty')!r});data.mkdir(parents=True,exist_ok=True)
original=pathlib.Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3/Beauty')
for kind in ('item','inter'):
 dst=data/('Beauty.'+kind+'.json');target=original/dst.name
 if not dst.exists():dst.symlink_to(target)
 assert dst.resolve()==target.resolve()
dst=data/'Beauty.index.json';target=release/'lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/Beauty.g3.index.json'
if dst.exists():assert dst.read_bytes()==target.read_bytes()
else:dst.write_bytes(target.read_bytes())
upstream={{'experiment_id':'grid-g3-beauty-lcrec-seed42-20261118','g3':{{'path':'/data/bwwu/sid-extensions-assets/563dc50b1c7e/experiments/beauty_rkmeans_qwen_20261010_012253/semantic_ids.pt','sha256':'d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18'}},'bridge':{{'path':str(release/'lc_rec/a0_beauty/grid_sid_preparation/identity_audit_abfe/bridge.csv'),'sha256':'10a7ca66dba43b69232a7c96b48d3287df9314a70a9ec9083ff00acfcf2a8c4f'}}}}
up=data/'Beauty.g3.upstream.json'
if up.exists():
 old=json.loads(up.read_text());assert old['g3']==upstream['g3'] and old['bridge']['sha256']==upstream['bridge']['sha256']
else:up.write_text(json.dumps(upstream,indent=2)+'\\n')
print(json.dumps({{'release':str(release),'files':len(manifest['files']),'run_commit':manifest['run_commit'],'data_root':str(data.parent)}}))
"""
    print(t.python(code));atomic(SUPPORT/'packages/g3-current-deployment.json',manifest)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('mode',choices=['deploy','submit','query','probe','collect'])
    p.add_argument('--stage',choices=['cpu','train','generate','compare','evaluate','dryrun'])
    p.add_argument('--job-id')
    p.add_argument('--output',type=Path)
    a=p.parse_args();t=Transport('new-server')
    if a.mode=='deploy':deploy(t);return
    if a.mode=='probe':
        print(t.ssh('nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; df -B1 /data/bwwu /home/bwwu; free -b'))
        return
    if a.mode=='query':print(json.dumps(gridctl.query(t,a.job_id),indent=2));return
    if a.mode=='collect':
        terminal=gridctl.query(t,a.job_id)
        if terminal['status'] not in ('SUCCEEDED','FAILED','TIMED_OUT','CANCELLED') or not terminal.get('cleanup_verified') or terminal.get('worker_alive') or terminal.get('workgroup_live'):
            raise RuntimeError('collect requires terminal job and verified empty workgroup; use query meanwhile')
        # Bounded lightweight evidence only. Each retrieved member is hash checked.
        remote=t.profile['job_root']+'/'+a.job_id
        code=f"""import pathlib,json,hashlib
root=pathlib.Path({remote!r});rows=[]
for f in root.rglob('*'):
 if not f.is_file() or f.is_symlink() or f.stat().st_size>16*1024**2:continue
 if f.suffix not in ('.json','.jsonl','.log','.txt') and f.name not in ('stdout','stderr'):continue
 rows.append({{'path':str(f),'relative':str(f.relative_to(root)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}})
print(json.dumps(rows))
"""
        rows=json.loads(t.python(code));need_output=a.output
        if need_output is None:raise ValueError('--output required')
        for row in rows:t.get_verified(row['path'],need_output/row['relative'],row)
        atomic(need_output/'collection.json',rows);print(json.dumps({'files':len(rows),'output':str(need_output)}));return
    if not a.stage or not a.job_id:raise ValueError('submit requires --stage and --job-id')
    dep=json.loads((SUPPORT/'packages/g3-current-deployment.json').read_text());rel=dep['release']
    scripts=rel+'/lc_rec/a0_beauty';data=REMOTE+'/data';large='/data/bwwu/lc-rec/g3-preparation-single-20261118/'+dep['run_commit'][:12]
    out=t.profile['job_root']+'/'+a.job_id+'/output';gpu=a.stage in ('train','generate');cap=3300 if a.stage=='train' else 1500 if gpu else 3600
    if a.stage=='cpu':argv=[PYTHON,scripts+'/grid_sid_preparation/g3_conversion/production_cpu_check.py','--repo',rel,'--data-root',data,'--model',MODEL,'--config',scripts+'/config/formal_g3_beauty_seed42.json','--output',out+'/cpu']
    elif gpu:argv=[PYTHON,scripts+'/g3_preflight_driver.py','--release',rel,'--data-root',data,'--model',MODEL,'--large-root',large,'--run-commit',dep['run_commit'],'--mode',a.stage,'--wall-seconds',str(cap-90),'--cpu-validation',t.profile['job_root']+'/g3-lcrec-single-cpu-20261118-r2/output/cpu/result.json']
    elif a.stage=='compare':argv=[PYTHON,scripts+'/boundary_compare.py','--continuous',large+'/continuous','--early',large+'/early_evidence','--resumed',large+'/recovery','--output',out+'/boundary_compare.json']
    elif a.stage=='evaluate':argv=[PYTHON,scripts+'/g3_evaluate_preflight.py','--release',rel,'--data-root',data,'--training-root',large+'/continuous','--generation-root',large+'/generation','--run-commit',dep['run_commit'],'--output',out+'/evaluation']
    else:argv=[PYTHON,scripts+'/g3_dry_run.py','--release',rel,'--data-root',data,'--model',MODEL,'--run-commit',dep['run_commit'],'--output',out+'/dry_run.json']
    env=dict(ENV,CUDA_VISIBLE_DEVICES='1' if gpu else '',DS_ACCELERATOR='cuda' if gpu else 'cpu')
    req={'schema':1,'host':t.profile['hostname'],'user':t.profile['user'],'uid':t.profile['uid'],
         'namespace':t.profile['namespace'],'deployment_sha256':t.profile['deployment_sha256'],
         'job_id':a.job_id,'capture_version':dep['run_commit'],'cwd':rel,'argv':argv,'env':env,
         'timeout_seconds':cap,'inputs':[{'path':r['path'],'sha256':r['sha256']} for r in dep['files']],
         'g3_preparation_experiment':'grid-g3-beauty-lcrec-seed42-20261118','g3_gpu_devices':1 if gpu else 0,
         'execution_topology':'SINGLE_GPU','execution_profile':{'world_size':1,'per_device_train_batch_size':8,'gradient_accumulation_steps':8,'global_batch':64},'physical_gpu_index':1 if gpu else None,
         'single_gpu_supplement':'User explicitly authorized single-GPU topology; no formal launch authorization.',
         'g3_preparation_budget':{'gpu_device_seconds':14400,'cpu_seconds':14400,'correction_round':2}}
    local=SUPPORT/'requests'/(a.job_id+'.json');atomic(local,req)
    print(json.dumps(gridctl.submit(t,local),indent=2))


if __name__=='__main__':main()
