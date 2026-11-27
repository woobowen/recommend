import hashlib
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from lc_rec.beauty_history_attention.attention import install_attention, reference_attention
from lc_rec.beauty_history_attention.regression import compare_real_contexts

torch.set_num_threads(1)
torch.manual_seed(9719)
q = torch.randn(1,1,4,16,requires_grad=True)
k = torch.randn(1,1,4,16,requires_grad=True)
v = torch.randn(1,1,4,16,requires_grad=True)
mask = torch.zeros(1,1,4,4,dtype=torch.bool)
mask[0,0,2,1:3] = True
output = reference_attention(q,k,v,mask)
assert torch.isfinite(output).all()
assert torch.equal(output[:,:,0:2], torch.zeros_like(output[:,:,0:2]))
assert torch.equal(output[:,:,3:], torch.zeros_like(output[:,:,3:]))
output.square().sum().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q,k,v))
assert q.grad[:,:,2].abs().sum() > 0
assert q.grad[:,:,0:2].abs().sum() == q.grad[:,:,3:].abs().sum() == 0
assert k.grad[:,:,0].abs().sum() == k.grad[:,:,3].abs().sum() == 0
assert v.grad[:,:,0].abs().sum() == v.grad[:,:,3].abs().sum() == 0
dims = LlamaConfig(vocab_size=41,hidden_size=64,intermediate_size=128,num_attention_heads=4,
                   num_hidden_layers=3,max_position_embeddings=48,pad_token_id=0,eos_token_id=2,bos_token_id=1)
base = LlamaForCausalLM(dims).eval()
install_attention(base,'bidirectional')
ids = torch.tensor([[1,11,12,7,13,14,8,10]])
valid = ids.ne(0)
history = torch.tensor([[False,True,True,False,True,True,False,False]])
with torch.inference_mode():
    contexts = compare_real_contexts(base,ids,valid,history)
assert set(contexts) == {'0','2'}
assert max(e['max_abs'] for e in contexts.values()) < 1e-6
root = Path('lc_rec/beauty_history_attention')
print(json.dumps(dict(result='PASS',inactive_reference_gradients='finite and zero outside allowed edges',
                      observed_layer_errors=contexts, source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                                      for name in ('attention.py','regression.py')})))
