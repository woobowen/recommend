import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
repo=Path('/home/addaswsw/recommend-worktrees/lc-rec')
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
job='lc-rec-offload-20261109-bounded'
release='/home/bwwu/lc-rec/releases/'+commit
profile=json.loads((root/'profiles.json').read_text())['new-server']
inputs=[]
for name in ['scripts/run_bounded_candidates.py','scripts/resource_guard.py','scripts/offload_probe.py','config/probe_optimizer_only.json','config/probe_optimizer_parameter.json','config/ds_optimizer_only.json','config/ds_optimizer_parameter.json']:
 p='lc_rec/offload/'+name
 b=subprocess.check_output(['git','show',commit+':'+p],cwd=repo)
 inputs.append({'path':release+'/'+p,'sha256':hashlib.sha256(b).hexdigest()})
request={'schema':1,'host':profile['hostname'],'user':profile['user'],'uid':profile['uid'],
 'namespace':profile['namespace'],'deployment_sha256':profile['deployment_sha256'],
 'capture_version':commit,'job_id':job,
 'argv':['/data/bwwu/envs/lc-rec-offload-20261108/bin/python',release+'/lc_rec/offload/scripts/run_bounded_candidates.py',
         '--release',release,'--output',profile['job_root']+'/'+job+'/output','--job-id',job,'--run-commit',commit],
 'cwd':release,'timeout_seconds':5400,
 'env':{'CUDA_VISIBLE_DEVICES':'0,1','DS_ACCELERATOR':'cuda','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1',
         'CUDA_HOME':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4',
         'PATH':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4/bin:/data/bwwu/envs/lc-rec-offload-20261108/bin:/usr/local/bin:/usr/bin:/bin',
         'LD_LIBRARY_PATH':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4/lib64:/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages/torch/lib',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'},'inputs':inputs}
p=root/'requests'/f'{job}.json'
assert not p.exists()
p.write_text(json.dumps(request,indent=2)+'\n')
print(str(p))
