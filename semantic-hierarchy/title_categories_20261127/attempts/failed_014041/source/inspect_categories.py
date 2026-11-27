import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
from pathlib import Path
import collections,json,re
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
import tensorflow as tf
tf.config.set_visible_devices([],'GPU');opts=tf.data.Options();opts.threading.private_threadpool_size=1
rows={}
for raw in tf.data.TFRecordDataset([str(p) for p in sorted((ROOT/'original_assets/data/amazon_data/beauty/items').glob('*.tfrecord.gz'))],compression_type='GZIP').with_options(opts):
 r=tf.train.Example.FromString(raw.numpy()).features.feature
 i=r['id'].int64_list.value[0];text=r['text'].bytes_list.value[0].decode();rows[i]=text
cats=[];missing=[];examples=[]
for i,s in rows.items():
 m=re.search(r'; Categories: (.*); Price:',s)
 if m and m.group(1).strip().lower() not in ['', 'unknown','nan','none','[]']:cats.append(m.group(1))
 else:missing.append(i)
 if len(examples)<3:examples.append(dict(item_id=i,text=s))
old=ROOT/'codex/instrument_semantic_hierarchy_20261115/inputs/instrument/items_metadata.jsonl'
instrument=[json.loads(s) for s in old.read_text().splitlines()]
print(json.dumps(dict(beauty=dict(items=len(rows),category_nonempty=len(cats),category_missing=len(missing),missing_examples=missing[:10],unique_category_text=len(set(cats)),examples=examples),qwen_instruments=dict(items=len(instrument),category_nonempty=sum(bool(r.get('category')) for r in instrument))),ensure_ascii=False,indent=2))
