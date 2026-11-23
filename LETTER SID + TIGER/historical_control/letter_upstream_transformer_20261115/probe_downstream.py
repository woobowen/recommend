import os
os.environ.update(CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3',
                  TF_NUM_INTEROP_THREADS='1', TF_NUM_INTRAOP_THREADS='1', OMP_NUM_THREADS='2')
import collections
import json
from pathlib import Path
import tensorflow as tf

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
data = ROOT / 'original_assets/data/amazon_data/beauty'
for split in ['training','evaluation','testing','items']:
    files = sorted((data / split).glob('*.gz'))
    print('SPLIT',split,'files',len(files),flush=True)
    if not files: continue
    ds = tf.data.TFRecordDataset([str(files[0])], compression_type='GZIP')
    for raw in ds.take(1):
        ex=tf.train.Example.FromString(raw.numpy())
        print({k: (list(v.int64_list.value)[:40] if v.HasField('int64_list') else
                   list(v.float_list.value)[:5] if v.HasField('float_list') else
                   [x.decode(errors='replace')[:500] for x in v.bytes_list.value[:2]])
               for k,v in ex.features.feature.items()},flush=True)
for path in sorted((ROOT / 'experiments/sid_order_5seed_v1/beauty/shuffle').rglob('config.yaml'))[:1]:
    print('LATEST CONFIG',path,flush=True)
    print(path.read_text(),flush=True)
src=ROOT/'third_party/LETTER-8d0154e/data/Beauty'
it=json.loads((src/'Beauty.inter.json').read_text())
sid=json.loads((src/'Beauty.index.json').read_text())
counts=collections.Counter(tuple(v) for v in sid.values())
print('AUTHOR COUNTS',len(it),len(sid),sum(map(len,it.values())),dict(collections.Counter(counts.values())),flush=True)
