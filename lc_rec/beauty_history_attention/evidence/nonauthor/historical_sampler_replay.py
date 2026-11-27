from collections import Counter
from pathlib import Path
import hashlib,json,torch
from review_sampler import build_serial_sampler, execution_schedule
root=Path('/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/formal-run/training')
n=131413
sampler=build_serial_sampler(range(n))
print(json.dumps(dict(generator_seed=sampler.generator.initial_seed(), schedule=execution_schedule(n))))
all_rows=0
for epoch in range(1,5):
    path=root/f'epoch{epoch}_rows.json'
    blob=path.read_bytes()
    r=json.loads(blob)['ranks']
    flat=[x for j in range(0,len(r[0]),8) for rk in r for x in rk[j:j+8]]
    actual=list(sampler)
    count=Counter(flat)
    assert sorted(count)==list(range(n)) and Counter(count.values())==Counter({1:131402,2:11})
    all_rows+=len(flat)
    print(json.dumps(dict(epoch=epoch, matches_old_archived_order=actual==flat, sample_differing_positions=sum(a!=b for a,b in zip(actual,flat,strict=True)), receipt_sha256=hashlib.sha256(blob).hexdigest(), rows=len(flat), unique=len(count), reuse_examples=all_rows%64, updates=all_rows//64)))
assert all_rows==525696 and all_rows%64==0
