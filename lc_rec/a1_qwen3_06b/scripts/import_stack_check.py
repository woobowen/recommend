"""Extended CPU-only import smoke check, no model allocation or GPU context."""
import inspect,json,os,sys
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('USE_TF')=='0' and os.environ.get('USE_FLAX')=='0'
from transformers import Trainer,TrainerCallback,TrainingArguments,Qwen3ForCausalLM
from transformers.utils.import_utils import is_tf_available,is_flax_available,is_torch_available
from peft import PeftModel,get_peft_model
from accelerate import Accelerator
from accelerate.utils.deepspeed import DeepSpeedSchedulerWrapper
from accelerate.data_loader import DataLoaderShard,SeedableRandomSampler
from deepspeed.runtime.engine import DeepSpeedEngine
from bitsandbytes.nn import Linear8bitLt
import torch
assert not is_tf_available() and not is_flax_available() and is_torch_available() and not torch.cuda.is_initialized()
classes=[Trainer,TrainerCallback,TrainingArguments,Qwen3ForCausalLM,PeftModel,get_peft_model,Accelerator,DeepSpeedSchedulerWrapper,DataLoaderShard,SeedableRandomSampler,DeepSpeedEngine,Linear8bitLt]
print(json.dumps({'status':'PASS_CPU_FULL_TRAINER_STACK_IMPORT','python':sys.executable,'objects':{x.__name__:inspect.getsourcefile(x) for x in classes},'TF':False,'Flax':False,'Torch':True,'CUDA_context':False}))
