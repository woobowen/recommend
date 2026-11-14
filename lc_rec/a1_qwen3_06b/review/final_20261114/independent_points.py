"""Reviewer a1_non_author_audit's independent formulas, executed through bridge.

Scope is frozen A1 point metrics and seven groups; no GPU or model loading.
Main controller supplies explicit paths; no imports from primary formulas.
"""
import collections as C
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ip,hp,tp,p0,p1,tokp,cp,mp,gp,op = sys.argv[1:]
N,EOS = 24772,151643
M = ([f'hit@{k}' for k in (1,5,10,20)] +
     [f'{m}@{k}' for m in ('ndcg','mrr') for k in (5,10,20)] +
     [f'{m}@{k}' for m in ('ItemHit','ItemNDCG') for k in (5,10,20)])
EXPECTED = ['33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd',
 '0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743',
 '1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2',
 '790506cf834bca27f4bb01d35d7d4048d494872f74ff4d9a8bcdac516c475b4d',
 '028a74dbab9ba2afaa018994b1518c465d20d58ce05218720634e5c42a4a2a45']
def load(p): return json.loads(Path(p).read_bytes())
def identity(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return {'path':str(Path(p).resolve()),'bytes':Path(p).stat().st_size,'sha256':h.hexdigest()}
def eq(a,b,label):
    assert math.isclose(float(a),float(b),abs_tol=5e-12,rel_tol=1e-10),(label,a,b)
ident=[identity(p) for p in [ip,hp,tp,p0,p1,tokp,cp,mp,gp]]
assert [v['sha256'] for v in ident[:5]]==EXPECTED
index,inter,items=load(ip),load(hp),load(tp)
assert len(index)==len(items)==9922 and len(inter)==N and set(index)==set(items)
sizes=C.Counter(''.join(v) for v in index.values())
assert len(sizes)==9897 and sum(v>1 for v in sizes.values())==23
added={x['content']:x['id'] for x in load(tokp)['added_tokens']}
layers=[{added[s[j]] for s in index.values()} for j in range(4)]
symbols={added[s]:s for v in index.values() for s in v}
assert list(map(len,layers))==[150,256,256,256] and len(symbols)==918 and EOS not in symbols
users=list(inter.items());freq=C.Counter(str(i) for seq in inter.values() for i in seq[:-2])
positive=sorted(freq[i] for i in index if freq[i]>0)
assert (positive[len(positive)//3],positive[(2*len(positive)-1)//3])==(5,11)
def metrics(sids,target):
    hits=[int(s==target) for s in sids]
    first=next((i+1 for i,x in enumerate(hits) if x),None)
    out={f'hit@{k}':float(any(hits[:k])) for k in (1,5,10,20)}
    for k in (5,10,20):
        out[f'ndcg@{k}']=math.fsum(x/math.log2(i+2) for i,x in enumerate(hits[:k]))
        out[f'mrr@{k}']=1/first if first is not None and first<=k else 0.
    before=0
    for s in sids:
        if s==target:break
        before+=sizes.get(s,1)
    g=sizes[target]
    for k in (5,10,20):
        m=min(g,max(0,k-before)) if first is not None else 0
        out[f'ItemHit@{k}']=m/g
        out[f'ItemNDCG@{k}']=math.fsum(1/math.log2(before+i+1) for i in range(1,m+1))/g
    return out
rows={};rankcounts=[]
for rank,p in enumerate((p0,p1)):
    count=0
    with open(p) as f:
        for line in f:
            r=json.loads(line);i=r['row_idx'];assert type(i) is int and 0<=i<N and i not in rows
            uid,seq=users[i];seq=list(map(str,seq));target=''.join(index[seq[-1]])
            assert str(r['user_id'])==str(uid) and str(r['target_item_id'])==seq[-1] and r['target']==target
            assert r['rank']==rank and r['prompt_id']==0 and r['original_callback_verified_all_tokens'] is True
            cs=r['candidates'];assert len(cs)==20
            for j,c in enumerate(cs):
                assert c['generation_order']==j and math.isfinite(c['sequence_score'])
                assert c['raw_decoded'].split('Response:')[-1].strip().replace(' ','')==c['sid']
                ts=c['generated_token_ids'];assert c['eos_position']==4 and len(ts)>=5 and ts[4]==EOS
                assert all(t==EOS for t in ts[5:]) and all(ts[z] in layers[z] for z in range(4))
                assert ''.join(symbols[t] for t in ts[:4])==c['sid']
            order=sorted(range(20),key=lambda j:cs[j]['sequence_score'],reverse=True)
            assert order==r['sorted_generation_order']
            sids=[cs[j]['sid'] for j in order];assert r['relevance']==[int(s==target) for s in sids]
            hlen=len(seq[:-1][-20:]);fq=freq[seq[-1]]
            rows[i]={'user_id':str(uid),'target_item_id':seq[-1],'target_sid':target,
             'visible_history_length':hlen,'history_group':'0-5' if hlen<=5 else '6-10' if hlen<=10 else '11-20',
             'target_training_frequency':fq,'popularity_group':'UNSEEN' if fq==0 else 'LONG_TAIL' if fq<=5 else 'MID' if fq<=11 else 'POPULAR',
             'metrics':metrics(sids,target)}
            count+=1
    assert count==12386;rankcounts.append(count)
assert set(rows)==set(range(N))
seen=set()
with open(cp) as f:
    for r in csv.DictReader(f):
        i=int(r['row_idx']);assert i in rows and i not in seen;seen.add(i);a=rows[i]
        for k in ('user_id','target_item_id','target_sid','history_group','popularity_group'):assert str(r[k])==str(a[k]),(i,k)
        for k in ('visible_history_length','target_training_frequency'):assert int(r[k])==a[k],(i,k)
        for k in M:eq(a['metrics'][k],r[k],('per_user',i,k))
assert seen==set(range(N))
def mean(rr):return {k:math.fsum(r['metrics'][k] for r in rr)/len(rr) for k in M}
overall=mean(list(rows.values()))
for k in M:eq(overall[k],load(mp)['metrics'][k],('overall',k))
group_results={};actual_groups=load(gp)['groups']
for field,names in [('history_group',('0-5','6-10','11-20')),('popularity_group',('UNSEEN','LONG_TAIL','MID','POPULAR'))]:
    group_results[field]={}
    for name in names:
        rr=[r for r in rows.values() if r[field]==name];got=actual_groups[field][name]
        assert got['users']==len(rr)
        values=mean(rr)
        for k in M:eq(values[k],got['metrics'][k],('group',field,name,k))
        group_results[field][name]={'users':len(rr),'metrics':values}
report={'status':'PASS','scope':'independent A1 frozen points and seven groups','not_final_a1_audit':True,
 'execution':'reviewer-authored formulas; main-controller bridge','source':identity(__file__),'inputs':ident,
 'users':N,'rank_counts':rankcounts,'candidates':N*20,'per_user_metric_values_compared':N*16,
 'overall':overall,'groups':group_results,'tolerance':{'absolute':5e-12,'relative':1e-10},
 'not_checked':['bootstrap','sampler order','Prompt retokenization','model/adapter/training identity','GPU score recomputation','runner resources']}
with open(op,'x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':report['status'],'users':N,'values_compared':N*16}))
