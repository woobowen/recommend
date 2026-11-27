"""Read-only schema, category coverage and SID-axis inventory; no GPU work."""
from pathlib import Path
import collections,json
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
AUTHOR=ROOT/'codex/letter_lc_rec_smoke_20261121/vendor/LETTER/data'
out={}
for domain in ['Beauty','Instruments']:
 rows=json.loads((AUTHOR/domain/(domain+'.item.json')).read_text())
 index=json.loads((AUTHOR/domain/(domain+'.index.json')).read_text())
 first=next(iter(rows.values()))
 out[domain]=dict(items=len(rows),sid_items=len(index),field_counts=dict(collections.Counter(k for r in rows.values() for k in r)),
  sample={k:str(v)[:900] for k,v in first.items()},
  category_nonempty={k:sum(bool(r.get(k)) for r in rows.values()) for k in ['categories','category','asin','item_id','title']},
  unique_sids=len(set(tuple(s) for s in index.values())))
for name,p in [('qwen_beauty',ROOT/'codex/llm2vec_sid_hierarchy_20261113/inputs/beauty'),('qwen_instruments',ROOT/'codex/instrument_semantic_hierarchy_20261115/inputs/instrument')]:
 out[name]={'files':[str(x) for x in p.glob('*')]}
 for f in p.glob('*.jsonl'):
  with f.open() as stream:first=json.loads(next(stream))
  out[name][f.name]={'first':{k:str(v)[:900] for k,v in first.items()}}
print(json.dumps(out,ensure_ascii=False,indent=2))
