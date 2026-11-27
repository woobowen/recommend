"""CPU-only group-size census of the four fixed SID tables."""
from pathlib import Path
import collections,json
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
sources={
 'beauty_qwen':ROOT/'codex/llm2vec_sid_hierarchy_20261113/inputs/beauty/item_sids.jsonl',
 'instruments_qwen':ROOT/'codex/instrument_semantic_hierarchy_20261115/inputs/instrument/item_sids.jsonl'}
tables={name:[json.loads(x)['sid'] for x in path.read_text().splitlines()] for name,path in sources.items()}
for domain in ['Beauty','Instruments']:
 path=ROOT/f'codex/letter_lc_rec_smoke_20261121/vendor/LETTER/data/{domain}/{domain}.index.json'
 data=json.loads(path.read_text());tables[domain.lower()+'_letter']=[[int(t.split('_')[1][:-1]) for t in data[str(i)]] for i in range(len(data))]
out={}
for name,codes in tables.items():
 counts={d:collections.Counter(tuple(c[:d]) for c in codes) for d in [1,2,3]}
 out[name]={'items':len(codes),'levels':{d:dict(nodes=len(cnt),evaluable_nodes=sum(n>=2 for n in cnt.values()),covered_items=sum(n for n in cnt.values() if n>=2),coverage=sum(n for n in cnt.values() if n>=2)/len(codes),max_items=max(cnt.values())) for d,cnt in counts.items()},
  'proper_edges':{str(d-1)+'_to_'+str(d):sum(2<=n<counts[d-1][p[:-1]] for p,n in counts[d].items()) for d in [2,3]},
  'top3_three_prefix_counts':counts[3].most_common(3)}
print(json.dumps(out,indent=2))
