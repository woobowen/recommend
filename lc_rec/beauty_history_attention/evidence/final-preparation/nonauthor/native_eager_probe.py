"""Independent CPU native-eager parity probes on the pinned HF LLaMA implementation."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
from lc_rec.beauty_history_attention.inference_attention import install_eager_inference

torch.set_num_threads(1)
torch.manual_seed(515)
NATIVE_SOURCE_SHA = '2abf84a5dd3644330f93c88e036831578e407de7a9812b5454ef1bb8a18c3422'
assert hashlib.sha256(Path(inspect.getsourcefile(LlamaAttention)).read_bytes()).hexdigest() == NATIVE_SOURCE_SHA

def decoder(dtype):
    return LlamaForCausalLM(LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128,
                                      num_attention_heads=4, num_hidden_layers=3, max_position_embeddings=64,
                                      bos_token_id=1, eos_token_id=2, pad_token_id=0)).to(dtype=dtype).eval()

def add_eager(model, mode):
    result = install_eager_inference(model, mode)
    assert result['backend'] == 'native_eager'
    return result

def explicit_mask(valid, history, query_span, dtype):
    """Scalar-loop oracle in sequence coordinates, never imports experimental mask code."""
    bsz, seq = valid.shape
    additive = torch.full((bsz, 1, query_span, seq), torch.finfo(dtype).min, dtype=dtype)
    for b in range(bsz):
        for q in range(query_span):
            physical_query = seq - query_span + q
            if not bool(valid[b, physical_query]):
                additive[b,0,q,0] = 0
                continue
            for k in range(seq):
                if bool(valid[b,k]) and (k <= physical_query or bool(history[b,k] and history[b,physical_query])):
                    additive[b,0,q,k] = 0
    return additive

def maximum_delta(a, b):
    return (a.float()-b.float()).abs().max().item()

def positions(mask):
    return (mask.cumsum(-1)-1).masked_fill(mask.eq(0),1)

def run_subject(model, ids, valid, history, mode, **kw):
    return model(input_ids=ids, attention_mask=valid, history_sid_mask=history, **kw)

def compare(dtype):
    model = decoder(dtype)
    causal = copy.deepcopy(model)
    target = copy.deepcopy(model)
    oracle = copy.deepcopy(model)
    label = add_eager(causal, 'causal')
    label_b = add_eager(target, 'bidirectional')
    raw = torch.tensor([[0,0,1,11,12,19,21,22,20,40,41,2],
                        [0,1,13,14,19,23,24,20,25,42,43,2],
                        [0,0,0,0,0,0,0,0,0,0,0,0]])
    valid = raw.ne(0)
    h = torch.tensor([[i in (3,4,6,7) for i in range(12)],
                      [i in (2,3,5,6) for i in range(12)],[False]*12])
    pos = positions(valid.long())
    original = model(raw, attention_mask=valid, use_cache=False, position_ids=pos)
    patched = run_subject(causal, raw, valid, h, 'causal', use_cache=False, position_ids=pos)
    assert torch.equal(original.logits[valid], patched.logits[valid])
    def oracle_mask(self, binary, input_shape, embeds, past_length):
        assert past_length == 0
        assert torch.equal(binary.bool(),valid)
        return explicit_mask(valid, h, input_shape[-1], embeds.dtype)
    oracle.model._prepare_decoder_attention_mask = MethodType(oracle_mask, oracle.model)
    expected = oracle(raw, attention_mask=valid, use_cache=False, position_ids=pos)
    observed = run_subject(target, raw,valid,h,'bidirectional',use_cache=False,position_ids=pos,
                           output_hidden_states=True)
    assert torch.isfinite(observed.logits).all()
    assert torch.equal(expected.logits[valid], observed.logits[valid])
    changed = raw.clone()
    changed[0,10], changed[1,10] = 73, 71
    res = run_subject(target,changed,valid,h,'bidirectional',use_cache=False,position_ids=pos,output_hidden_states=True)
    assert torch.equal(observed.logits[:2,:10][valid[:2,:10]],res.logits[:2,:10][valid[:2,:10]])
    for before,after in zip(observed.hidden_states,res.hidden_states,strict=True):
        assert torch.equal(before[h],after[h])
    future_h = raw.clone()
    future_h[0,7] = 75
    causal_later = run_subject(causal,future_h,valid,h,'causal',use_cache=False,position_ids=pos).logits
    assert torch.equal(causal_later[0,3],patched.logits[0,3])
    b_later = run_subject(target,future_h,valid,h,'bidirectional',use_cache=False,position_ids=pos).logits
    assert not torch.equal(b_later[0,3],observed.logits[0,3])
    raw,valid,h,pos = raw[:2],valid[:2],h[:2],pos[:2]
    width = 9
    prompt = raw[:,:width]
    current = run_subject(target,prompt,valid[:,:width],h[:,:width],'bidirectional',
                          use_cache=True,position_ids=pos[:,:width])
    differences=[]
    for q_end in range(width + 1,raw.shape[1]+1):
        current = target(raw[:,q_end-1:q_end], attention_mask=valid[:,:q_end], history_sid_mask=h[:,:width],
                         position_ids=pos[:,q_end-1:q_end], past_key_values=current.past_key_values,use_cache=True)
        actual = current.logits[:,-1]
        recompute = target(raw[:,:q_end],attention_mask=valid[:,:q_end],history_sid_mask=h[:,:width],
                           position_ids=pos[:,:q_end],use_cache=False).logits[:,-1]
        dif=maximum_delta(actual,recompute)
        differences.append(dif)
        assert dif < (1e-6 if dtype == torch.float32 else .035)
    def callback(batch_id, prefix):
        step = prefix.shape[0] - width
        return [2] if step == 4 else list(range(50+step*6+batch_id,56+step*6+batch_id))
    settings = dict(attention_mask=valid[:,:width].long(), num_beams=20,num_return_sequences=20,max_new_tokens=10,
                    do_sample=False,early_stopping=True,output_scores=True,return_dict_in_generate=True,
                    prefix_allowed_tokens_fn=callback)
    old_beam=model.generate(prompt,use_cache=True,**settings)
    causal_beam=causal.generate(prompt,history_sid_mask=h[:,:width],use_cache=True,**settings)
    assert torch.equal(old_beam.sequences,causal_beam.sequences)
    assert torch.equal(old_beam.sequences_scores,causal_beam.sequences_scores)

    # Exercise PEFT's actual generate delegation on tracked cache history.
    peft=get_peft_model(target,LoraConfig(task_type=TaskType.CAUSAL_LM,target_modules=['q_proj','v_proj'],r=2,
                                          lora_alpha=4,lora_dropout=0.,inference_mode=True)).eval()
    if hasattr(peft,'base_model_prepare_inputs_for_generation'):
        peft.base_model_prepare_inputs_for_generation=target.prepare_inputs_for_generation
    generated=peft.generate(input_ids=prompt,history_sid_mask=h[:,:width],use_cache=True,**settings)
    repeat=peft.generate(input_ids=prompt,history_sid_mask=h[:,:width],use_cache=True,**settings)
    assert torch.equal(generated.sequences,repeat.sequences)
    assert torch.equal(generated.sequences_scores,repeat.sequences_scores)
    assert generated.sequences.shape == (40,width+5)
    for i,row in enumerate(generated.sequences):
        assert row[-1].item()==2
        assert all(row[width+k].item() in callback(i//20,row[:width+k]) for k in range(4))
    return {'dtype':str(dtype),'valid_causal_and_oracle_valid_B_exact':True,
            'future_answer_invariance_each_layer_exact':True,'cross_item_earlier_H_delta':maximum_delta(b_later[0,3],observed.logits[0,3]),
            'cached_uncached_next_logit_max_abs':max(differences),
            'causal_native_20beam_scores_and_ids_exact':True,'two_sample_b_repeat_20beam_scores_and_ids_exact':True,
            'native_backend_reports':[label,label_b]}
with torch.inference_mode():
    result=[compare(dtype) for dtype in (torch.float32,torch.bfloat16)]

print(json.dumps({'status':'PASS','installed_llama_sha256':NATIVE_SOURCE_SHA,'subject_source_sha256':source_shas,
                  'torch':torch.__version__,'cuda_visible':torch.cuda.device_count(),'result':result},indent=2,allow_nan=False))
