"""CPU-only import validation of the already-installed isolated LETTER environment."""
import importlib.metadata as m,inspect,json,os,sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'DS_ACCELERATOR' not in os.environ
import torch,transformers,peft,bitsandbytes,flash_attn,fastchat,einops,deepspeed
from fastchat.train.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from peft import prepare_model_for_int8_training
r={'python':sys.executable,'versions':{p:m.version(p) for p in ['torch','transformers','peft','bitsandbytes','flash-attn','fschat','einops','deepspeed','accelerate']},'torch_cuda_context':torch.cuda.is_initialized(),'torch_cuda_build':torch.version.cuda,'bnb_gpu_libraries':[p.name for p in Path(bitsandbytes.__file__).parent.glob('*cuda124*')],'fastchat_patch_source':inspect.getsourcefile(replace_llama_attn_with_flash_attn),'read_only_reuse':['/data/bwwu/envs/lc-rec-offload-20261108','/data/bwwu/envs/grid-20261009-e61baf3'],'status':'CPU_IMPORT_PASS_GPU_NOT_EXECUTED'}
assert not r['torch_cuda_context'] and r['bnb_gpu_libraries']
print(json.dumps(r,indent=2),flush=True)
