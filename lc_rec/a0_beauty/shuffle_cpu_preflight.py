#!/usr/bin/env python3
"""Full Beauty prompt/token-boundary precheck against the fixed Shuffle core.

This uses the reconstructed base tokenizer, not a trained adapter; it does not
substitute for the post-training CPU/GPU preacceptance against the saved adapter.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

source, data_root, model, core_root = map(Path, sys.argv[1:])
sys.path[:0] = [str(source), str(core_root)]
from data import SeqRecDataset
from core import transform, canonical_bytes
from transformers import LlamaTokenizer

args = SimpleNamespace(dataset='Beauty', data_path=str(data_root), max_his_len=20,
                       his_sep=', ', index_file='.index.json', add_prefix=False)
data = SeqRecDataset(args, mode='test', sample_num=-1)
data.set_prompt(0)
tok = LlamaTokenizer.from_pretrained(str(model), local_files_only=True, model_max_length=2048,
                                     padding_side='left')
tok.pad_token_id = 0
symbols = data.get_new_tokens()
assert tok.add_tokens(symbols) == 965 and len(tok) == 32965
token_ids = {s: tok.encode(s, add_special_tokens=False)[0] for s in symbols}
assert len(set(token_ids.values())) == 965
assert all(tok.encode(s, add_special_tokens=False) == [v] for s, v in token_ids.items())
assert len(data) == 22363
axis = list(data.inters)
manifest_hash = hashlib.sha256()
total_groups = 0
for row, uid in enumerate(axis):
    items = data.inters[uid]
    groups = [[token_ids[t] for t in data.indices[str(item)]] for item in items[:-1][-20:]]
    sample = data[row]
    assert sample['labels'] == ''.join(data.indices[str(items[-1])])
    encoded = tok(sample['input_ids'], return_attention_mask=True)
    ids, mask = encoded['input_ids'], encoded['attention_mask']
    manifests = {}
    for n in (0, 3, 4):
        new, evidence = transform(ids, mask, groups, set(token_ids.values()), n, row)
        assert len(new) == len(ids)
        assert n == 0 or evidence['input_changed']
        manifests[str(n)] = {'input_sha256': evidence['shuffled_sha256'],
                             'permutations': [g['permutation'] for g in evidence['groups']]}
    total_groups += len(groups)
    manifest_hash.update(canonical_bytes([row, uid, sample['labels'], manifests]) + b'\n')
print(json.dumps({'status':'PASS_PRELIMINARY_CPU', 'users':len(data), 'groups':total_groups,
                  'manifest_sha256':manifest_hash.hexdigest(), 'sid_tokens':len(symbols),
                  'core_sha256':hashlib.sha256((core_root/'core.py').read_bytes()).hexdigest(),
                  'scope':'Reconstructed base tokenizer; adapter and GPU checks remain required.'}, indent=2))
