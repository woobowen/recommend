#!/usr/bin/env python3
"""Check published GRID item-text fingerprints against fixed historical item shards."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
import csv,hashlib,json
from pathlib import Path
import tensorflow as tf
root=Path('/home/bwwu/grid-audits/sid-extensions-p0/g3-identity-abfe'); data=Path('/data/bwwu/sid-extensions-assets/563dc50b1c7e/original_assets/data/amazon_data/beauty')
manifest=json.loads((root/'dataset_source_manifest.json').read_text()); paths=[data/x['path'].removeprefix('amazon_data/beauty/') for x in manifest['files'] if x['path'].startswith('amazon_data/beauty/items/')]
items={}
for path in paths:
 for raw in tf.compat.v1.io.tf_record_iterator(str(path),options=tf.io.TFRecordOptions(compression_type='GZIP')):
  f=tf.train.Example.FromString(raw).features.feature; i=list(f['id'].int64_list.value); text=list(f['text'].bytes_list.value)
  if len(i)!=1 or len(text)!=1 or i[0] in items: raise ValueError('bad or duplicate item')
  items[i[0]]=hashlib.sha256(text[0]).hexdigest()
with (root/'grid_item_text_fingerprints.csv').open(newline='') as f: rows=list(csv.DictReader(f))
if set(items)!=set(range(12101)) or len(rows)!=12101 or any(int(r['grid_item_id'])!=i or r['utf8_text_sha256']!=items[i] for i,r in enumerate(rows)):raise ValueError('fingerprints differ')
result={'status':'PASS','matched_item_text_sha256':len(items),'fingerprint_sha256':hashlib.sha256((root/'grid_item_text_fingerprints.csv').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(root/'item_fingerprint_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
