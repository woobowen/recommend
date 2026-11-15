#!/usr/bin/env python3
"""Replay the 32-user inference evidence independently of the runtime's PASS flags."""
import argparse,collections,hashlib,itertools,json,math
from pathlib import Path

def perm(n,row,item):
    cat=[p for p in itertools.permutations(range(n)) if p!=tuple(range(n))]
    cap=((1<<256)//len(cat))*len(cat)
    counter=0
    while True:
        raw=json.dumps(['GRID-LETTER-SHUFFLE-V1',42,n,row,item,counter],ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        draw=int.from_bytes(hashlib.sha256(raw).digest(),'big')
        if draw<cap:return cat[draw%len(cat)]+((3,) if n==3 else ())
        counter+=1

def main():
    p=argparse.ArgumentParser()
    for n in ('normal','shuffle','audit','manifest','data-root','model-config','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();audit=json.loads(a.audit.read_text());manifest=json.loads(a.manifest.read_text());assert hashlib.sha256(a.manifest.read_bytes()).hexdigest()=='714450a0e9479ed272266a5b4d3c2150d83742e4b70c38f80da888d4fa0aaa39'
    for kind,expected in {'index':'be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82','inter':'8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5','item':'74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330'}.items():
        assert hashlib.sha256((a.data_root/'Beauty'/('Beauty.'+kind+'.json')).read_bytes()).hexdigest()==expected
    index=json.loads((a.data_root/'Beauty/Beauty.index.json').read_text());inter=json.loads((a.data_root/'Beauty/Beauty.inter.json').read_text());uids=list(inter);lookup=audit['token_ids'];vocab=set(lookup.values());allowed=[set(lookup[t[i]] for t in index.values()) for i in range(4)];eos=json.loads(a.model_config.read_text())['eos_token_id']
    seen=set();counts=collections.Counter();max_diff=0.
    for rank in (0,1):
        wanted=manifest['ranks'][str(rank)];paths=a.normal/f'predictions.rank{rank}.jsonl';rows=[json.loads(x) for x in paths.read_text().splitlines()]
        assert len(rows)==len(wanted)==16
        checks=json.loads((a.shuffle/f'acceptance.rank{rank}.json').read_text())['samples'];assert len(checks)==16
        for original,w,record in zip(rows,wanted,checks,strict=True):
            row=w['row_idx'];uid=uids[row];seq=inter[uid];assert row not in seen;seen.add(row)
            assert original['row_idx']==record['row_idx']==row and original['user_id']==w['user_id']==uid and original['target']==w['target']==''.join(index[str(seq[-1])])
            raw=original['input_token_ids']; assert original['input_nonpadding_tokens']==len(raw)
            pos=[i for i,t in enumerate(raw) if t in vocab];groups=[[lookup[t] for t in index[str(item)]] for item in seq[:-1][-20:]];assert len(pos)==4*len(groups)
            for j,g in enumerate(groups):
                at=pos[4*j:4*j+4];assert at==list(range(at[0],at[0]+4)) and [raw[k] for k in at]==g
            by=record['conditions'];assert set(by)=={'original','OFF','S3','S4'}
            base_seqs=[x['generated_token_ids'] for x in original['candidates']];base_scores=[x['sequence_score'] for x in original['candidates']];assert len(base_seqs)==len(base_scores)==20
            assert sorted(range(20),key=lambda i:base_scores[i],reverse=True)==original['sorted_generation_order']
            assert by['original']['sequences']==by['OFF']['sequences']==base_seqs
            assert by['original']['scores']==by['OFF']['scores']
            for x,y in zip(by['OFF']['scores'],base_scores,strict=True):
                assert math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-5+1e-5*abs(y);max_diff=max(max_diff,abs(x-y))
            for label,n in [('original',0),('OFF',0),('S3',3),('S4',4)]:
                c=by[label];expected=list(raw)
                for j,g in enumerate(groups):
                    pp=perm(n,row,j) if n else (0,1,2,3)
                    for k,t in zip(pos[4*j:4*j+4],[g[x] for x in pp]):expected[k]=t
                assert c['input_ids']==expected and (not n or expected!=raw)
                assert c['prefill']['current_input_ids']==expected and c['prefill']['initial_cache_length']==0 and c['prefill']['position_ids']==list(range(len(raw)))
                assert len(c['sequences'])==len(c['scores'])==20 and all(math.isfinite(x) for x in c['scores'])
                for seqout in c['sequences']:
                    assert len(seqout)>=5 and seqout[4]==eos and all(seqout[i] in allowed[i] for i in range(4))
                counts[label]+=1
    assert len(seen)==32
    report={'status':'PASS','scope':'independently replayed saved input IDs, permutation, position/cache receipts, candidates and scores; does not independently observe in-memory embedding/weight hooks','users':32,'conditions':dict(counts),'normal_candidates':640,'max_cross_process_off_score_abs_difference':max_diff,'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()}
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
