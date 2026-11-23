"""Test 2080 Ti int8/FP16 backward on a tiny random model, not model results."""
import json
from pathlib import Path
import sys
import torch
from hardware_compat import disable_optional_bnb_triton, restore_legacy_bnb_state
disable_optional_bnb_triton()
import bitsandbytes as bnb
from transformers import LlamaConfig, LlamaForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/LETTER/LETTER-LC-Rec'))
from modeling_letter import LETTER

torch.manual_seed(42)
base_dir=HERE/'kernel_random_base'
if not (base_dir/'model.safetensors').exists():
    LETTER(LlamaConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4)).save_pretrained(base_dir)
base=LETTER.from_pretrained(base_dir,quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
base=prepare_model_for_kbit_training(base,gradient_checkpointing_kwargs={'use_reentrant':False})
base.config.use_cache=False
base.set_hyper(.8)
restore_legacy_bnb_state(base)
model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    modules_to_save=['embed_tokens','lm_head'],task_type='CAUSAL_LM'))
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002,foreach=False)
ids=torch.randint(0,128,(1,32),device='cuda:0')
labels=ids.clone();labels[:,:24]=-100
scaler=torch.amp.GradScaler('cuda',init_scale=128.0)
with torch.autocast('cuda',dtype=torch.float16):loss=model(input_ids=ids,labels=labels).loss
assert torch.isfinite(loss)
scaler.scale(loss).backward();scaler.unscale_(optimizer)
grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
assert grads and all(torch.isfinite(g).all().item() for g in grads), [
    n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all().item()]
scaler.step(optimizer);scaler.update()
model.eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
    reference=model(input_ids=ids).logits.detach().float().cpu()
adapter=HERE/'kernel_random_adapter'
model.save_pretrained(adapter,safe_serialization=True)
reloaded_base=LlamaForCausalLM.from_pretrained(base_dir,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),torch_dtype=torch.float16,
    device_map={'':0},attn_implementation='sdpa')
restore_legacy_bnb_state(reloaded_base)
reloaded_base=prepare_model_for_kbit_training(reloaded_base,use_gradient_checkpointing=False)
reloaded=PeftModel.from_pretrained(reloaded_base,adapter)
reloaded.eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
    actual=reloaded(input_ids=ids).logits.detach().float().cpu()
assert torch.allclose(reference,actual,atol=.01,rtol=.005),(reference-actual).abs().max().item()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
    generated=reloaded.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
        num_beams=2,num_return_sequences=2,max_new_tokens=3,pad_token_id=0,eos_token_id=2,
        prefix_allowed_tokens_fn=lambda _,x: [4,5] if x.shape[0]<34 else [2])
assert generated.shape==(2,35)
result={'passed':True,'random_model_only':True,'loss':loss.item(),
        'gpu':torch.cuda.get_device_name(),'compute_capability':torch.cuda.get_device_capability(),
        'int8_fp16_backward':True,'adamw_update':True,'save_reload_checked':True,
        'reload_max_logit_error':(reference-actual).abs().max().item(),'native_cached_generation_checked':True}
(HERE/'kernel_preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
