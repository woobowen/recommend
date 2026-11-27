import json
import math
from pathlib import Path
import sys

inference_path, train_path, full_run_path = (Path(x) for x in sys.argv[1:])
inference = json.loads(inference_path.read_text())
train = json.loads(train_path.read_text())
assert inference['adapter_files']['adapter_model.bin']['sha256'] == 'dd9f3d2983af7ce85ace99534bd23df9362d9cd91ac39d0b356b219cb790b1c1'
ca = inference['eager_vs_experimental_causal']
assert ca['attention']['mode'] == 'causal'
assert ca['attention']['layer_count'] == 32 and ca['attention']['q1_calls'] > 0
assert ca['attention']['qkv_dtypes'].keys() == {'torch.bfloat16/torch.bfloat16/torch.bfloat16'}
assert len(ca['eager_candidates']) == len(ca['experimental_candidates']) == 20
old_candidates = [tuple(c['generated_token_ids']) for c in ca['eager_candidates']]
new_candidates = [tuple(c['generated_token_ids']) for c in ca['experimental_candidates']]
assert (old_candidates == new_candidates) == ca['same_candidate_order']
assert len(set(old_candidates).intersection(new_candidates)) == ca['shared_candidate_ids']
assert any('_scaled_dot_product_efficient_attention' in x for x in inference['observed_attention_operators'])
assert set(inference['fused_vs_explicit_fp32_contexts']) == {'0','31'}
b = inference['bidirectional']
assert b['earlier_logits_equal'] and b['history_equal_each_layer']
assert b['attention']['mode'] == 'bidirectional' and b['attention']['q1_calls'] > 0
assert b['cached_vs_uncached_next_logits']['rms'] / b['cached_vs_uncached_next_logits']['ref_rms'] < .03
for layer in inference['fused_vs_explicit_fp32_contexts'].values():
    assert layer['rms'] / layer['ref_rms'] < .015
assert train['status'] == 'REGRESSION_MEASURED'
assert train['historical_commit'] == '6e1b211d9c37f358c43b1540b40b17f6703150d2'
assert train['old_flash']['enabled'] is True and train['old_flash']['calls']['extension_varlen'] > 0
assert train['old_flash']['qkv_dtypes'].keys() == {'torch.bfloat16/torch.bfloat16/torch.bfloat16'}
assert train['new_attention']['mode'] == 'causal'
assert train['new_attention']['layer_count'] == 32 and train['new_attention']['layer_calls'] == 64
assert train['new_attention']['qkv_dtypes'].keys() == {'torch.bfloat16/torch.bfloat16/torch.bfloat16'}
assert train['old_causal']['loss_dtype'] == train['new_causal']['loss_dtype'] == 'torch.float32'
assert math.isclose(train['loss_abs_error'], abs(train['old_causal']['loss']-train['new_causal']['loss']), rel_tol=0., abs_tol=1e-10)
assert len(train['gradient_errors']['parameters']) == 450
families = train['gradient_errors']['families']
assert sum(f['elements'] for f in families.values()) == 290037760
assert train['gradient_errors']['unchanged_weights_and_trainable_membership']
with (full_run_path / 'events.rank0.jsonl').open() as stream:
    first = next(e for line in stream if (e:=json.loads(line))['event']=='batch')
assert train['real_batch']['sha256']['input_ids'] == first['input_sha256']
assert train['real_batch']['sha256']['labels'] == first['label_sha256']
assert train['real_batch']['sha256']['history_sid_mask'] == first['history_sid_mask_sha256']
result = {'status':'PASS','inference_path':str(inference_path),'train_causal_path':str(train_path),
          'inference': {'logit_errors':ca['all_logits'],'supervised_logit_errors':ca['supervised_logits'],
                        'gradient_errors':ca['gradient'], 'beam_same_order':ca['same_candidate_order'],
                        'shared_candidates':ca['shared_candidate_ids'],'context_errors':inference['fused_vs_explicit_fp32_contexts'],
                        'B_decode':b['cached_vs_uncached_next_logits'],'operators':inference['observed_attention_operators'],
                        'eager_temperature_loss':ca['temperature_0p8_shifted_ce_eager'],
                        'causal_temperature_loss':ca['temperature_0p8_shifted_ce_experimental']},
          'training':{'real_batch_matches_preflight_first_batch':True,'causal_old_call_counts':train['old_flash']['calls'],
                      'causal_new_operators':train['new_causal']['backend_ops'],'logits':train['logit_errors'],
                      'gradient_families':families, 'old_loss':train['old_causal']['loss'],
                      'new_loss':train['new_causal']['loss'],'loss_abs_error':train['loss_abs_error']}}
print(json.dumps(result, indent=2, allow_nan=False))
