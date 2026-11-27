from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import hashlib
import itertools
import json
import sys

source = Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/beauty-infer-63f13def364d/release/lc_rec/letter/vendor/LETTER/LETTER-LC-Rec')
model = '/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549'
data = '/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3'
evidence = Path('/data/bwwu/lc-rec/letter-shuffle-v2-beauty/cpu-replay-r3')
sys.path.insert(0, str(source))
from transformers import LlamaTokenizer
from data import SeqRecDataset

tok = LlamaTokenizer.from_pretrained(model, model_max_length=2048, padding_side='left', local_files_only=True)
tok.pad_token_id = 0
options = SimpleNamespace(dataset='Beauty', data_path=data, max_his_len=20, his_sep=', ', index_file='.index.json', add_prefix=False)
test = SeqRecDataset(options, mode='test', prompt_id=0)
tok.add_tokens(test.get_new_tokens())
audit = json.loads((evidence/'audit.json').read_text())
vocab = tok.get_vocab()
sid_ids = set(tok.get_added_vocab().values())

def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def semantic(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

assert semantic(vocab) == audit['tokenizer_vocab_sha256']
users = list(test.inters)
assert semantic(users) == audit['user_axis_sha256']
modes = {'shuffle2': (2,3), 'shuffle3a': (0,1,2), 'shuffle3b': (1,2,3), 'shuffle4': (0,1,2,3)}
perms = {n: tuple(itertools.permutations(range(n))) for n in (2,3,4)}
streams = {mode: hashlib.sha256() for mode in modes}
identity_counts = Counter()
groups = user_count = 0
with (evidence/'inputs.jsonl').open() as stream:
    for i, line in enumerate(stream):
        record = json.loads(line)
        uid = users[i]
        row = test[i]
        tokenized = tok(row['input_ids'], max_length=tok.model_max_length, truncation=True, return_attention_mask=True)
        ids, mask = tokenized['input_ids'], tokenized['attention_mask']
        assert record['row_idx'] == i and record['user_id'] == uid and record['target'] == row['labels']
        assert record['original_sha256'] == digest(ids) and record['attention_mask_sha256'] == digest(mask)
        assert all(mask)
        offsets = [j for j, token in enumerate(ids) if token in sid_ids]
        visible = test.inters[uid][:-1][-20:]
        assert len(offsets) == 4 * len(visible) and record['visible_history_length'] == len(visible)
        groups += len(visible)
        token_groups = []
        for k, item in enumerate(visible):
            at = offsets[4*k:4*k+4]
            assert at == list(range(at[0], at[0] + 4))
            assert [ids[j] for j in at] == [vocab[token] for token in test.indices[str(item)]]
            token_groups.append(at)
        for mode, active in modes.items():
            choices = perms[len(active)]
            bound = 2**256 - 2**256%len(choices)
            changed = ids.copy()
            permutation_ids = []
            for occurrence, at in enumerate(token_groups):
                counter = 0
                while True:
                    key = ['GRID-LETTER-SHUFFLE-V2', 42, len(active), i, occurrence, counter]
                    number = int.from_bytes(hashlib.sha256(canonical(key)).digest(), 'big')
                    if number < bound:
                        break
                    counter += 1
                index = number % len(choices)
                permutation_ids.append(index)
                order = choices[index]
                for j, slot in enumerate(active):
                    changed[at[slot]] = ids[at[active[order[j]]]]
            expected = record['conditions'][mode]
            assert expected['input_sha256'] == digest(changed)
            assert expected['input_changed'] == (changed != ids)
            assert expected['permutation_ids'] == permutation_ids
            if changed == ids:
                identity_counts[mode] += 1
            streams[mode].update(canonical([i, changed]) + b'\n')
        user_count += 1
observed = {mode: h.hexdigest() for mode, h in streams.items()}
assert observed == audit['input_stream_sha256']
assert user_count == 22363 and groups == 159966
print(json.dumps(dict(result='PASS', users=user_count, groups=groups, identity=dict(identity_counts), sha256=observed)))
