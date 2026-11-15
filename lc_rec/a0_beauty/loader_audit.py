#!/usr/bin/env python3
"""CPU audit of the actual fixed LETTER dataset and LLaMA tokenizer."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

source, root, model = map(Path, sys.argv[1:])
sys.path.insert(0, str(source))
from data import SeqRecDataset
from utils import load_datasets, set_seed
from collator import Collator
from transformers import LlamaTokenizer

args = SimpleNamespace(dataset='Beauty', data_path=str(root), max_his_len=20,
    his_sep=', ', index_file='.index.json', add_prefix=False, tasks='seqrec',
    train_prompt_sample_num='1', train_data_sample_num='0', sample_valid=True,
    valid_prompt_id=0, valid_prompt_sample_num=2, only_train_response=True)
set_seed(42)
train, valid = load_datasets(args)
test = SeqRecDataset(args, mode='test', sample_num=-1)
test.set_prompt(0)
index = test.indices
inter = test.inters
assert len(train) == 131413 and len(valid) == 44726 and len(test) == 22363
assert list(test.remapped_inters) == list(inter) == list(valid.remapped_inters)
rows = train.datasets[0].inter_data
cursor = 0
for row, (uid, items) in enumerate(inter.items()):
    sid = [''.join(index[str(item)]) for item in items]
    for i in range(1, len(sid)-2):
        assert rows[cursor] == {'item': sid[i], 'inters': ', '.join(sid[:i][-20:])}
        cursor += 1
    assert valid.inter_data[row] == {'item': sid[-2], 'inters': ', '.join(sid[:-2][-20:])}
    assert test.inter_data[row] == {'item': sid[-1], 'inters': ', '.join(sid[:-1][-20:])}
    assert test[row]['labels'] == sid[-1]
assert cursor == len(rows)
tokenizer = LlamaTokenizer.from_pretrained(str(model), model_max_length=2048,
                                          padding_side='right', local_files_only=True)
tokenizer.pad_token_id = 0
tokens = train.datasets[0].get_new_tokens()
added = tokenizer.add_tokens(tokens)
assert added == len(tokens) == 965 and len(tokenizer) == 32965
ids = [tokenizer.encode(token, add_special_tokens=False) for token in tokens]
assert all(len(x) == 1 for x in ids) and len({x[0] for x in ids}) == len(tokens)
assert all(tokenizer.convert_ids_to_tokens(x[0]) == token for token, x in zip(tokens, ids))
batch = [train[i] for i in (0, 1, 17, len(train)-1)]
collated = Collator(args, tokenizer)(batch)
for i, row in enumerate(batch):
    prompt = tokenizer(row['input_ids'], add_special_tokens=True)['input_ids']
    full = tokenizer(row['labels'] + tokenizer.eos_token, add_special_tokens=True)['input_ids']
    labels = collated['labels'][i].tolist()
    assert all(x == -100 for x in labels[:len(prompt)])
    assert labels[len(prompt):len(full)] == full[len(prompt):]
    assert all(x == -100 for x in labels[len(full):])
axis = hashlib.sha256(json.dumps(list(inter), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
mapping = dict(zip(tokens, (x[0] for x in ids)))
mapping_sha = hashlib.sha256(json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
print(json.dumps({'status': 'PASS', 'source': str(source), 'data_root': str(root),
                  'train': len(train), 'valid': len(valid), 'test': len(test),
                  'all_split_rows_verified': True, 'prompt0_test_targets_verified': True,
                  'response_mask_examples': len(batch), 'added_tokens': added,
                  'vocab_size': len(tokenizer), 'token_mapping_sha256': mapping_sha,
                  'user_axis_sha256': axis}, indent=2))
