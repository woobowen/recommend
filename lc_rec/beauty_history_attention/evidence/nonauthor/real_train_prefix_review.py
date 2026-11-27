import json
from pathlib import Path

old = Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/formal-run/training')
newer = Path('/data/bwwu/lc-rec/beauty-history-attention-20261127/train-r1/training/events.rank0.jsonl')
known = {}
for rank in (0, 1):
    with (old / f'events.rank{rank}.jsonl').open() as s:
        for line in s:
            v = json.loads(line)
            if v['event'] == 'batch' and v['epoch_index'] == 0:
                known[rank, v['loader_yielded']] = v
with newer.open() as s:
    data = [r for line in s if (r:=json.loads(line))['event']=='batch' and r['epoch_index']==0]
assert len(data) >= 20
for i, r in enumerate(data[:20]):
    assert r['loader_yielded'] == i + 1 and r['shape'][0] == 8
    assert min(r['history_tokens_per_row']) >= 4 and max(r['history_tokens_per_row']) <= 80
    k = i % 2, (i//2)+1
    frozen = known[k]
    assert (r['input_sha256'],r['label_sha256'],r['shape']) == (frozen['input_sha256'],frozen['label_sha256'],frozen['shape'])
print(json.dumps({'validated':20,'scope':'first bounded preflight epoch; compare matching old A real GPU inputs after rank interleave','equal_fields':['input_ids hash','labels hash','tensor shape'],'train_job_id':data[0]['attempt_job_id'],'min_history_tokens':min(min(r['history_tokens_per_row']) for r in data[:20]),'max_history_tokens':max(max(r['history_tokens_per_row']) for r in data[:20])}))
