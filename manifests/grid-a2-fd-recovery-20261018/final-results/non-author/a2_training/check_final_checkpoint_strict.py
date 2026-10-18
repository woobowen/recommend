#!/usr/bin/env python3
"""Complete direct strict=True model loading for the final recovery checkpoint."""
from pathlib import Path
import datetime,json,sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
P=Path(__file__).resolve().parent
code='''import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from pathlib import Path
import hashlib,json,sys
root=Path('/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/22286966cd544773')
job=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261018-sportsa2-train-fd-recovery-v1')
status=json.loads((job/'status.json').read_text());assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
result=json.loads((job/'output/training_result.json').read_text());asset=result['final_recovery']['checkpoint']
assert asset['sha256']=='74770b7d1166fa16df1a931b6ffa221bbf9e6f9b3eb84fa2a9d2ad8055b55676'
assert hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest()==asset['sha256']
sys.path.insert(0,str(root))
from experiments.beauty_extensions.common import bind_source,load_checkpoint
bind_source('A')
import src.utils.custom_hydra_resolvers
import hydra,torch
from omegaconf import OmegaConf
torch.set_num_threads(2)
cfg=OmegaConf.load(job/'output/effective_config.yaml');model=hydra.utils.instantiate(cfg.model)
state=load_checkpoint(asset['path']);assert state['global_step']==7900
loaded=model.load_state_dict(state['state_dict'],strict=True);actual=model.state_dict()
assert not loaded.missing_keys and not loaded.unexpected_keys and set(actual)==set(state['state_dict'])
assert all(v.shape==state['state_dict'][k].shape and torch.equal(v.cpu(),state['state_dict'][k].cpu()) for k,v in actual.items())
assert all(torch.isfinite(v).all() for v in actual.values() if v.is_floating_point())
sid_path=Path(result['sid']['path']);assert hashlib.sha256(sid_path.read_bytes()).hexdigest()==result['sid']['sha256']
sid=torch.load(sid_path,map_location='cpu',weights_only=True);assert torch.equal(state['hyper_parameters']['codebooks'].cpu(),sid)
enc,dec=[model.get_embedding_table(n).weight for n in ['encoder','decoder']]
assert enc.data_ptr()!=dec.data_ptr() and not torch.equal(enc,dec) and tuple(enc.shape)==tuple(dec.shape)==(1024,128)
print(json.dumps({'status':'A2_FINAL_RECOVERY_CHECKPOINT_STRICT_LOAD_PASS','checkpoint':asset,'global_step':state['global_step'],'state_tensors':len(actual),'missing_keys':list(loaded.missing_keys),'unexpected_keys':list(loaded.unexpected_keys),'keys_shapes_values_exact':True,'all_finite':True,'embedded_sid_exact':True,'independent_tables':True,'CPU_model_instantiations':1,'GPU_or_forward_calls':0,'filesystem_mutations':0,'best_checkpoint_reloaded':False}))
'''
r=json.loads(Transport('new-server').ssh('CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /data/bwwu/envs/grid-20261009-e61baf3/bin/python -B -',data=code.encode(),timeout=60))
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');target=P/('final-checkpoint-strict-review-'+stamp+'.json');target.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'path':str(target),'status':r['status'],'GPU_or_forward_calls':r['GPU_or_forward_calls']}))
