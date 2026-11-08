#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess,sys
import gridctl,transport
root=pathlib.Path(__file__).resolve().parents[1];repo=pathlib.Path('/home/addaswsw/recommend-worktrees/lc-rec')
commit='53a0f618fc68dc2c936b4e5e10e1b5f0bbf282ad';release='/home/bwwu/lc-rec/releases/'+commit
t=transport.Transport('new-server');p=t.profile;job='lc-rec-20261108-cpu-audit-r1'
source=json.loads((root/'evidence'/('deploy-'+commit+'.json')).read_text())['version']['files']
inputs=[{'path':release+'/'+v['path'],'sha256':v['sha256']} for v in source if v['path'].startswith('lc_rec/vendor/original/') or v['path'] in ['lc_rec/scripts/audit_real_data.py','lc_rec/config/task.json']]
code='''import pathlib,json,hashlib
paths=list(pathlib.Path('/data/bwwu/lc-rec/assets/data/Instruments').glob('Instruments.*'))
paths += [pathlib.Path('/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549/tokenizer.model')]
print(json.dumps([{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]))
'''
assets=json.loads(t.python(code,timeout=45));(root/'evidence/cpu_input_asset_hashes.json').write_text(json.dumps(assets,indent=2)+'\n');inputs+=assets
req={'schema':1,'host':p['hostname'],'user':p['user'],'uid':p['uid'],'namespace':p['namespace'],'deployment_sha256':p['deployment_sha256'],'capture_version':commit,'job_id':job,'argv':['/data/bwwu/envs/lc-rec-20261108/bin/python','-I','-B',release+'/lc_rec/scripts/audit_real_data.py','--source',release+'/lc_rec/vendor/original','--data-root','/data/bwwu/lc-rec/assets/data','--tokenizer','/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549','--output',p['job_root']+'/'+job+'/output/real_data_audit.json','--seed','42'],'cwd':release,'timeout_seconds':900,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'},'inputs':inputs}
request=root/'requests'/f'{job}.json';assert not request.exists(),'inspect existing job before any resubmission';request.write_text(json.dumps(req,indent=2)+'\n')
result=gridctl.submit(t,request);print(json.dumps({k:result.get(k) for k in ['job_id','status','exit_code','submission']},indent=2))
