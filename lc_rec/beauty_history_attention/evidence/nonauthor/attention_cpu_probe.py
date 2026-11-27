import torch, copy, json, hashlib, os
from transformers import LlamaForCausalLM, LlamaConfig
from peft import get_peft_model, LoraConfig, TaskType
from review_subject import install_attention, allowed_edges
torch.set_num_threads(1)
torch.manual_seed(1703)
cfg=LlamaConfig(vocab_size=41, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                num_hidden_layers=3, max_position_embeddings=48, pad_token_id=0, eos_token_id=2, bos_token_id=1)
native=LlamaForCausalLM(cfg).eval()
ids=torch.tensor([[0,0,1,14,15,9,16,17,10,20,21,2],[1,6,14,15,9,16,17,10,20,21,2,0],[0]*12])
valid=ids.ne(0)
hist=torch.tensor([[i in (3,4,6,7) for i in range(12)],[i in (2,3,5,6) for i in range(12)],[False]*12])
mut=ids.clone(); mut[0,10:]=torch.tensor([23,24]); mut[1,9:11]=torch.tensor([25,26])
bmask=torch.zeros(3,12,12,dtype=torch.bool)
cmask=bmask.clone()
for b in range(3):
    for i in range(12):
        for j in range(12):
            cmask[b,i,j]=valid[b,i] and valid[b,j] and j<=i
            bmask[b,i,j]=valid[b,i] and valid[b,j] and (j<=i or hist[b,i] and hist[b,j])
assert torch.equal(allowed_edges(valid,hist,12,'bidirectional'),bmask)
assert torch.equal(allowed_edges(valid,hist,12,'causal'),cmask)
assert torch.equal(allowed_edges(valid,hist,1,'bidirectional'),bmask[:,-1:])
def pos(mask):
    p=mask.long().cumsum(-1)-1
    return p.masked_fill(mask.eq(0),1)
def run(model, ids, mask, h, **opts):
    return model(input_ids=ids,attention_mask=mask,history_sid_mask=h,**opts)
def model(mode):
    m=copy.deepcopy(native)
    install_attention(m,mode)
    return m
results={}
with torch.no_grad():
    control=model('causal')
    ref=native(ids,attention_mask=valid,use_cache=False).logits
    actual=run(control,ids,valid,hist,use_cache=False).logits
    results['causal_native_valid_logits_max_abs']=(actual[valid]-ref[valid]).abs().max().item()
    assert torch.allclose(actual[valid],ref[valid],atol=2e-7,rtol=2e-6)
    assert actual.isfinite().all()
    m=model('bidirectional')
    a=run(m,ids,valid,hist,output_hidden_states=True,use_cache=False)
    b=run(m,mut,valid,hist,output_hidden_states=True,use_cache=False)
    for i,last in [(0,10),(1,9)]:
        assert torch.equal(a.logits[i,:last],b.logits[i,:last])
        for x,y in zip(a.hidden_states,b.hidden_states,strict=True): assert torch.equal(x[i,hist[i]],y[i,hist[i]])
    assert a.logits.isfinite().all()
    future=ids.clone(); future[0,7]=31
    out=run(m,future,valid,hist,use_cache=False).logits
    assert (out[0,3]-a.logits[0,3]).abs().max().item()>1e-5
    results['bidirectional_earlier_history_logit_delta']=(out[0,3]-a.logits[0,3]).abs().max().item()
    for mode in ('causal','bidirectional'):
        m=model(mode)
        prev=run(m,ids[:2,:8],valid[:2,:8],hist[:2,:8],position_ids=pos(valid[:2,:8]),use_cache=True)
        errs=[]
        for end in range(9,12):
            q=run(m,ids[:2,end-1:end],valid[:2,:end],hist[:2,:8],position_ids=pos(valid[:2,:end])[:,-1:],use_cache=True,past_key_values=prev.past_key_values)
            full=run(m,ids[:2,:end],valid[:2,:end],hist[:2,:8],position_ids=pos(valid[:2,:end]),use_cache=False)
            diff=(q.logits[:,-1]-full.logits[:,-1]).abs().max().item()
            assert diff<1e-6
            errs.append(diff); prev=q
        results[mode+'_cached_uncached_max_abs']=max(errs)
    p=run(control,ids[:1,:8],valid[:1,:8],hist[:1,:8],use_cache=True)
    try:
        run(model('bidirectional'),ids[:1,8:9],valid[:1,:9],hist[:1,:8],past_key_values=p.past_key_values,use_cache=True)
        raise AssertionError('foreign cache accepted')
    except ValueError: pass
    for mode in ('causal','bidirectional'):
        m=model(mode)
        peft=get_peft_model(m,LoraConfig(task_type=TaskType.CAUSAL_LM,target_modules=['q_proj','v_proj'],r=2,lora_alpha=4,lora_dropout=0.))
        peft.eval()
        cfg2=dict(max_new_tokens=3,num_beams=20,num_return_sequences=20,do_sample=False,early_stopping=True,return_dict_in_generate=True,output_scores=True)
        a=peft.generate(input_ids=ids[:2,:8],attention_mask=valid[:2,:8],history_sid_mask=hist[:2,:8],use_cache=True,**cfg2)
        b=peft.generate(input_ids=ids[:2,:8],attention_mask=valid[:2,:8],history_sid_mask=hist[:2,:8],use_cache=False,**cfg2)
        assert a.sequences.shape==b.sequences.shape and torch.equal(a.sequences,b.sequences)
        score=(a.sequences_scores-b.sequences_scores).abs().max().item()
        assert score<1e-6
        results[mode+'_PEFT_20_beam_cache_score_error']=score
labels=ids[:2].clone()
labels[:,:9]=-100
labels[~valid[:2]]=-100
h2=hist[:2].flip(0).clone()
h2 &= valid[:2]
for mode in ('causal','bidirectional'):
    eager=model(mode).train(); ckpt=model(mode).train()
    ckpt.gradient_checkpointing_enable()
    def f(m, x, h):
        return run(m,x,valid[:2],h,labels=labels,use_cache=False).loss
    loss_e=f(eager,ids[:2],hist[:2])+0.5*f(eager,mut[:2],h2)
    loss_c=f(ckpt,ids[:2],hist[:2])+0.5*f(ckpt,mut[:2],h2)
    loss_e.backward()
    loss_c.backward()
    diffs=[]
    for x,y in zip(eager.parameters(),ckpt.parameters(),strict=True):
        assert x.grad.isfinite().all() and y.grad.isfinite().all()
        diffs.append((x.grad-y.grad).abs().max().item())
    results[mode+'_checkpoint_interleaved_grad_max_abs']=max(diffs)
    assert max(diffs)<1e-6 and torch.equal(loss_e,loss_c)
print(json.dumps({'status':'PASS','identity':__subject_sha,'torch':torch.__version__,'gpu_used':False,'results':results}))
