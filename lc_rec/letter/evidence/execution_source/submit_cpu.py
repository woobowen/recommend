import hashlib,json,sys
from pathlib import Path
from transport import Transport
from gridctl import submit
commit='7a7365c6033864c232a4c2b042bcc9c494c938b3';root=Path('/home/addaswsw/letter-task-20261110/control');t=Transport('new-server');p=t.profile
release=p['remote_support']+'/releases/'+commit;job='letter-20261110-data-r1';output=p['job_root']+'/'+job+'/output/audit';src=release+'/lc_rec/letter'
argv=['/data/bwwu/envs/lc-rec-offload-20261108/bin/python',src+'/scripts/audit_data.py','--source',src+'/vendor/LETTER/LETTER-LC-Rec','--a-data',src+'/vendor/LETTER/data','--b-data','/data/bwwu/lc-rec/assets/data','--model','/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549','--output',output]
rows=[]
repo=Path('/home/addaswsw/recommend-worktrees/lc-rec')
for q in [repo/'lc_rec/letter/scripts/audit_data.py',*list((repo/'lc_rec/letter/vendor').rglob('*'))]:
 if q.is_file() and '__pycache__' not in str(q):rows.append({'path':release+'/'+str(q.relative_to(repo)),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
r={'schema':1,'job_id':job,'capture_version':commit,'argv':argv,'cwd':release,'env':{'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'},'inputs':rows,'timeout_seconds':1260,**{k:(p['hostname'] if k=='host' else p[k]) for k in ['host','uid','user','namespace','deployment_sha256']}}
path=root/'requests'/f'{job}.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(r,indent=2)+'\n');print(submit(t,str(path)))
