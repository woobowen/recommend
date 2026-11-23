"""Regression checks against the previous LETTER-LC-Rec shuffle protocol."""
from pathlib import Path
import hashlib
import json
import random
from shuffle_protocol import PROTOCOL_VERSION, permutation, permutation_catalog, transform

HERE=Path(__file__).resolve().parent
assert PROTOCOL_VERSION=='GRID-LETTER-SHUFFLE-V1'
assert len(permutation_catalog(3))==5 and len(permutation_catalog(4))==23
fixtures=[(3,0,0,(2,1,0)),(3,0,1,(0,2,1)),(3,1,0,(1,0,2)),(3,1,1,(0,2,1)),
          (4,0,0,(1,3,2,0)),(4,0,1,(0,3,2,1)),(4,1,0,(2,1,3,0)),(4,1,1,(0,2,3,1))]
for n,row,item,expected in fixtures:assert permutation(n,row,item,42)==expected
groups=[[100,101,102,103],[104,105,106,107]]
original=[1,2]+groups[0]+[99]+groups[1]+[3]
mask=[1]*len(original);vocab=set(range(100,108))
count=0
for n in (0,3,4):
    for row in range(1000):
        output,audit=transform(original,mask,groups,vocab,n,row,seed=42)
        assert all(output[i]==original[i] for i in [0,1,6,11])
        for g in audit['groups']:
            assert sorted(g['shuffled_ids'])==sorted(g['original_ids'])
            assert g['nonidentity']==bool(n)
            if n in (0,3):assert g['shuffled_ids'][3]==g['original_ids'][3]
        random.seed(row);random.random()
        assert output==transform(original,mask,groups,vocab,n,row,seed=42)[0]
        count+=len(groups)
try:transform(original+[100],mask+[1],groups,vocab,3,0,seed=42)
except ValueError:pass
else:raise AssertionError('Extra SID token was not rejected')
manifest=json.loads((HERE/'shuffle_alignment.json').read_text(encoding='utf-8'))
assert hashlib.sha256((HERE/'shuffle_protocol.py').read_bytes()).hexdigest()==manifest['source_sha256']
report={'passed':True,'protocol':PROTOCOL_VERSION,'seed':42,'golden_fixtures':len(fixtures),
        'history_group_checks':count,'identity_allowed':False,'nonhistory_preserved':True,
        'global_rng_independent':True,'extra_sid_rejected':True}
(HERE/'shuffle_alignment_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
