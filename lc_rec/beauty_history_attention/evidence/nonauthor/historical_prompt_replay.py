from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import random
import sys
import torch

hist = Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/formal-run/training')
data = Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3')
source = Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/beauty-infer-63f13def364d/release/lc_rec/letter/vendor/LETTER/LETTER-LC-Rec')
base = '/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549'
sys.path.insert(0, str(source))

from transformers import LlamaTokenizer
from data import SeqRecDataset
from collator import Collator
args = SimpleNamespace(dataset='Beauty', data_path=str(data), max_his_len=20, his_sep=', ', index_file='.index.json',
                       add_prefix=False, only_train_response=True)
train = SeqRecDataset(args, mode='train')
tok = LlamaTokenizer.from_pretrained(base, model_max_length=2048, padding_side='right', local_files_only=True)
tok.pad_token_id = 0
tok.add_tokens(train.get_new_tokens())
collate = Collator(args, tok)
def digest(t):
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()

epoch_rows = []
for epoch in range(1,5):
    epoch_rows.append(json.loads((hist / f'epoch{epoch}_rows.json').read_text())['ranks'])
for rank in range(2):
    random.seed(42)
    captured = {}
    with (hist / f'events.rank{rank}.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] == 'batch':
                k = event['epoch_index'], event['loader_yielded']
                if k in captured: assert event['input_sha256'] == captured[k]['input_sha256']
                captured[k] = event
    compared = 0
    mismatches = []
    for epoch, rank_rows in enumerate(epoch_rows):
        rows = rank_rows[rank]
        for idx in range(0, len(rows), 8):
            key = epoch, idx//8 + 1
            if key in captured:
                expected = captured[key]
                observed = collate([train[row] for row in rows[idx:idx+8]])
                if (digest(observed['input_ids']), digest(observed['labels'])) != (expected['input_sha256'], expected['label_sha256']):
                    mismatches.append(key)
                compared += 1
            else:
                for _ in range(8):
                    random.randint(0, 16)
    assert compared == len(captured)
    print(json.dumps(dict(rank=rank, compared=compared, mismatches=len(mismatches),
                          first_mismatches=mismatches[:5], epoch_counts=[sum(k[0]==e for k in captured) for e in range(4)])))
