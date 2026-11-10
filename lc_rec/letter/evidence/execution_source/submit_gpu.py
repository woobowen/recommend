import hashlib,json,sys
from pathlib import Path
from transport import Transport
from gridctl import submit
commit=sys.argv[1];phase=sys.argv[2];attempt=sys.argv[3];t=Transport('new-server');p=t.profile;base=p['remote_support'];release=base+'/releases/'+commit;repo=Path('/home/addaswsw/recommend-worktrees/lc-rec');src=release+'/lc_rec/letter';job='letter-20261110-'+phase+'-r'+attempt;jobbase=p['job_root']+'/'+job;out=jobbase+'/output';py='/data/bwwu/envs/letter-20261110/bin/python';audit=p['job_root']+'/letter-20261110-data-r1/output/audit';model='/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549'
assert phase in ['gpu','inference'] and attempt in ['0','1','2','3']
if phase=='gpu':
 child=[py,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',src+'/scripts/gpu_probe.py','--source-root',src+'/vendor/LETTER/LETTER-LC-Rec','--data-root',src+'/vendor/LETTER/data','--model',model,'--audit-dir',audit,'--output',out+'/probe','--source-commit',commit]
else:
 adapter=sys.argv[4]
 child=[py,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',src+'/scripts/inference_probe.py','--source',src+'/vendor/LETTER/LETTER-LC-Rec','--data-root',src+'/vendor/LETTER/data','--model',model,'--adapter',adapter,'--audit-output',audit,'--output',out+'/inference','--run-commit',commit]
argv=[py,src+'/scripts/resource_guard.py','--config',src+'/config/guard.json','--output',out+'/resources','--job-id',job,'--',*child]
inputs=[]
for file in ['scripts/gpu_probe.py','scripts/compat.py','scripts/inference_probe.py','scripts/resource_guard.py','config/guard.json','config/probe_instruments.json']:
 q=repo/'lc_rec/letter'/file;inputs.append({'path':src+'/'+file,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
env={'CUDA_VISIBLE_DEVICES':'0,1','DS_ACCELERATOR':'cuda','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','TORCHINDUCTOR_COMPILE_THREADS':'1','CUDA_HOME':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4','LD_LIBRARY_PATH':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4/lib64:/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages/torch/lib'}
r={'schema':1,'job_id':job,'capture_version':commit,'argv':argv,'cwd':release,'env':env,'inputs':inputs,'timeout_seconds':3540,**{k:(p['hostname'] if k=='host' else p[k]) for k in ['host','uid','user','namespace','deployment_sha256']}}
f=Path('/home/addaswsw/letter-task-20261110/control/requests')/(job+'.json');f.write_text(json.dumps(r,indent=2)+'\n');print(submit(t,str(f)))
