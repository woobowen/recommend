import json, math, pathlib, collections, csv
root=pathlib.Path('/data/bwwu/lc-rec/a0-beauty-20261115/SYNTHETIC_CPU_INTEGRATION_ONLY_271df1e33054_r0')
data=pathlib.Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3/Beauty')
idx=json.loads((data/'Beauty.index.json').read_text()); sizes=collections.Counter(''.join(s) for s in idx.values());out={}
for name,offset in [('Normal',0),('S3',1),('S4',2)]:
    m=json.loads((root/name/'primary/metrics.json').read_text()); sums=collections.Counter()
    for i in range(22363):
        positions=([1],[5],[10],[20],[],[1,2])[(i+offset)%6]
        for k in (1,5,10,20): sums[f'hit@{k}']+=int(any(pos<=k for pos in positions))
        for k in (5,10,20):
            sums[f'ndcg@{k}']+=sum(1/math.log2(pos+1) for pos in positions if pos<=k)
            sums[f'mrr@{k}']+=(1/positions[0] if positions and positions[0]<=k else 0)
    delta={key:abs(m['metrics'][key]-value/22363) for key,value in sums.items()}
    samples={};rank_counts=[]
    for rank in (0,1):
        p=root/name/'synthetic_predictions'/f'predictions.rank{rank}.jsonl'
        with p.open() as f:
            for count,line in enumerate(f,1):
                if count>50: continue
                row=json.loads(line); ordered=[row['candidates'][i]['sid'] for i in row['sorted_generation_order']]; target=row['target'];g=sizes[target]
                try: index=ordered.index(target)
                except ValueError:index=None
                prefix=sum(sizes.get(s,1) for s in ordered[:index]) if index is not None else 0
                values={}
                for k in (5,10,20):
                    span=min(g,max(k-prefix,0)) if index is not None else 0
                    values[f'ItemHit@{k}']=span/g
                    values[f'ItemNDCG@{k}']=sum(1/math.log2(prefix+t+1) for t in range(1,span+1))/g
                samples[str(row['row_idx'])]=values
        rank_counts.append(count)
    errors=[];checked=0
    with (root/name/'primary/per_user.csv').open() as f:
        for row in csv.DictReader(f):
            if row['row_idx'] in samples:
                errors.extend(abs(float(row[k])-v) for k,v in samples[row['row_idx']].items());checked+=1
    assert checked==100 and len(errors)==600 and max(errors)<=1e-13
    assert rank_counts==[11182,11181] and max(delta.values())<=1e-13
    out[name]={'rows_per_rank':rank_counts,'candidates':sum(rank_counts)*20,'metric_count':len(m['metrics']),'independent_ten_metric_abs_errors':delta,'ten_metric_max_abs_error':max(delta.values()),'cce_sample_users':checked,'cce_sample_comparisons':len(errors),'cce_sample_max_abs_error':max(errors),'reported_official_cce_full_user_comparisons':m['official_cce_full_user_comparisons']}
summary=json.loads((root/'summary.json').read_text())
assert summary['designation']=='SYNTHETIC_CPU_INTEGRATION_ONLY' and summary['formal_paired_entry_executed'] is False
print(json.dumps({'status':'PASS_BOUNDED_INDEPENDENT_STATISTICS_REVIEW','scope':'Synthetic ranking expected values on full user axis; first 50 records per rank per condition for independent CCE; counts; no resampling replay or formal paired entry','designation':summary['designation'],'formal_paired_entry_executed':summary['formal_paired_entry_executed'],'conditions':out},indent=2))
