"""CPU-only import gate for the Torch Trainer; optional TF/Flax are disabled."""
import hashlib,inspect,json,os,sys
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('USE_TF')=='0' and os.environ.get('USE_FLAX')=='0'
from transformers import Trainer,TrainingArguments,Qwen3ForCausalLM
from transformers.utils import import_utils
import torch
assert import_utils.is_tf_available() is False and import_utils.is_flax_available() is False
assert not torch.cuda.is_initialized()
p=inspect.getsourcefile(import_utils)
print(json.dumps({'status':'PASS_CPU_TORCH_TRAINER_IMPORT','python':sys.executable,'Trainer_source':inspect.getsourcefile(Trainer),'import_utils_source':p,'import_utils_sha256':hashlib.sha256(open(p,'rb').read()).hexdigest(),'optional_backends':{'USE_TF':os.environ['USE_TF'],'USE_FLAX':os.environ['USE_FLAX']},'cuda_initialized':False}))
