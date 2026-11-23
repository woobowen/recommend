"""Compare directly with the frozen GRID method body, without model imports."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import itertools
import json
import random
from shuffle_grid_protocol import PROTOCOL_VERSION, permutation, permutation_catalog, transform

HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'plan_v1_1/protocol.py'
source=REFERENCE.read_text(encoding='utf-8')
tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ProtocolMixin')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='permutation')
code=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
env={'hashlib':hashlib,'itertools':itertools,'torch':SimpleNamespace(tensor=lambda rows,dtype:rows,long='long')}
exec(compile(code,str(REFERENCE),'exec'),env)
assert len(permutation_catalog(3))==6 and len(permutation_catalog(4))==24
count=0;identity={3:0,4:0}
for dataset in ['beauty','instruments']:
    dummy=SimpleNamespace(eval_seed=2026,dataset_name=dataset,_permutation_cache={},num_hierarchies=4)
    for uid in range(1000):
        for n in [3,4]:
            expected=env['permutation'](dummy,uid,30,n)
            for pos in range(30):
                actual=list(permutation(n,uid,pos,dataset,2026))+list(range(n,4))
                assert actual==expected[pos],(dataset,uid,n,pos)
                identity[n]+=actual==[0,1,2,3];count+=1
groups=[[100,101,102,103],[104,105,106,107]]
original=[0,1]+groups[0]+[99]+groups[1]+[2,0]
mask=[0]+[1]*11+[0]
assert len(original)==len(mask)
for n in [0,3,4]:
    for uid in range(100):
        out,audit=transform(original,mask,groups,set(range(100,108)),n,uid,'beauty',2026)
        assert all(out[i]==original[i] for i in [0,1,6,11,12])
        if n==0:assert out==original
        for group in audit['groups']:
            assert sorted(group['shuffled_ids'])==sorted(group['original_ids'])
            if n in [0,3]:assert group['shuffled_ids'][3]==group['original_ids'][3]
        random.seed(uid);random.random()
        assert out==transform(original,mask,groups,set(range(100,108)),n,uid,'beauty',2026)[0]
assert identity[3]>0 and identity[4]>0
report={'passed':True,'protocol':PROTOCOL_VERSION,'seed':2026,'identity_allowed':True,
    'reference_file':'codex/plan_v1_1/protocol.py','reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
    'compared_permutations':count,'identity_counts':identity,'nonhistory_and_padding_preserved':True,
    'comparison':'unmodified original permutation method body executed via AST'}
(HERE/'grid_alignment_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
