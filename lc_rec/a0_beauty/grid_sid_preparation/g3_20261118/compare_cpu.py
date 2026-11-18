#!/usr/bin/env python3
import collections, hashlib, json, pathlib, torch
root=pathlib.Path('/data/bwwu/sid-extensions-assets/563dc50b1c7e/experiments/beauty_rkmeans_qwen_20261010_012253')
paths={'G3':root/'semantic_ids.pt','merged':root/'sid_generation/pickle/merged_predictions_tensor.pt'}
expected={'G3':'d41adcfb22a9a47d5e0424c068f4c74f33cc9fdbaf5db08540fbfa0e3d720e18','merged':'d5241c43708cb9e5ce0c1e812b965a0b5f92e38d94aead8246bcebeea62e385f'}
out={'assets':{},'comparison':{}}; ts={}
for name,p in paths.items():
    b=p.read_bytes(); h=hashlib.sha256(b).hexdigest(); assert h==expected[name]
    t=torch.load(p,map_location='cpu',weights_only=True); assert t.shape==(4,12101) and t.dtype==torch.int64
    rows=[tuple(x) for x in t.T.tolist()]; count=collections.Counter(rows)
    out['assets'][name]={'sha256':h,'bytes':len(b),'shape':list(t.shape),'dtype':str(t.dtype),'stride':list(t.stride()),'position_range':[[int(t[i].min()),int(t[i].max())] for i in range(4)],'position_distinct':[len(set(t[i].tolist())) for i in range(4)],'unique_prefixes':[len(set(r[:k] for r in rows)) for k in range(1,5)],'collision_groups':sum(n>1 for n in count.values()),'collision_items':sum(n for n in count.values() if n>1)}; ts[name]=t
x,y=ts.values(); delta=x!=y
out['comparison']={'equal_file':expected['G3']==expected['merged'],'equal_position_items':int((~delta.any(dim=0)).sum()),'different_items':int(delta.any(dim=0).sum()),'per_digit_differences':delta.sum(dim=1).tolist(),'all_values_equal':torch.equal(x,y),'same_tuple_set':set(map(tuple,x.T.tolist()))==set(map(tuple,y.T.tolist()))}
print(json.dumps(out,indent=2))
