"""Synthetic CPU model verifies the new evaluation entry with real held-out input/callback."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from lc_rec.beauty_history_attention.attention import install_attention
from lc_rec.beauty_history_attention.evaluate import (
    REPLAY_DIR, generation_settings, paired_inputs, predict, read_replay,
)
from lc_rec.shuffle_v2 import beauty, replay

torch.set_num_threads(2)
torch.manual_seed(42)
config = json.loads(beauty.CONFIG.read_text())
source = beauty.frozen_module(Path(config['baseline']['entry']))
_, rows, _ = read_replay(REPLAY_DIR, config)
data, tokenizer, collate, vocab = replay.frozen_data(config)
i = 734
uid = rows[i]['user_id']
seq = data.inters[uid]
data_input, _ = collate([data[i]])
mask = data_input['attention_mask']
hs, modes = paired_inputs(data_input['input_ids'][0].tolist(), mask[0].tolist(),
                          [[vocab[sid] for sid in data.indices[str(item)]] for item in seq[:-1][-20:]],
                          set(vocab.values()), rows[i])
h = torch.tensor([hs], dtype=torch.bool)
llama_config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=4, pad_token_id=0,
                           eos_token_id=2, bos_token_id=1)
base = LlamaForCausalLM(llama_config)
peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj', 'v_proj'],
                         modules_to_save=['embed_tokens', 'lm_head'], task_type='CAUSAL_LM')
model = get_peft_model(base, peft_config)
attention = install_attention(base, 'bidirectional')
model.base_model_prepare_inputs_for_generation = base.prepare_inputs_for_generation
model.eval().requires_grad_(False)
allowed = data.get_prefix_allowed_tokens_fn(tokenizer)
settings, _ = generation_settings(model)
captured = {}
with torch.no_grad(), patch.object(torch.cuda, 'synchronize'):
    for mode in ('normal', 'shuffle3a', 'shuffle3b', 'shuffle4'):
        ids, evidence = modes[mode]
        candidates, order, _, boundary = predict(
            model, base, tokenizer, source, allowed, torch.tensor([ids]), mask, h, settings, True)
        assert boundary['input_sha256'] == evidence['shuffled_sha256']
        assert boundary['history_sid_mask_sha256'] == evidence['history_sid_mask_sha256']
        captured[mode] = (candidates, order, boundary)
assert captured['normal'] == captured['shuffle3a'] == captured['shuffle3b']
assert attention['prefill_calls'] == 4 and attention['cached_calls'] == 16
print(json.dumps({'status': 'PASS_TINY_PEFT_CPU_BOUNDARY', 'row_idx': i, 'modes': list(captured),
                  'attention': attention, 'raw_nonnumeric_boundary': captured['normal'][2],
                  'normal_and_two_identity_shuffles_exact': True}, indent=2))
