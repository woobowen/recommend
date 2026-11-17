#!/usr/bin/env python3
"""Compare existing item text at equal IDs; this does not create a mapping."""
import json
import os
from collections import Counter
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
root = Path('/data/bwwu/sid-extensions-assets/563dc50b1c7e/original_assets/data/amazon_data/beauty/items')
letter = json.loads(Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3/Beauty/Beauty.item.json').read_text())
rows = {}
opts = tf.data.Options(); opts.threading.private_threadpool_size = 1
for path in sorted(root.glob('*.tfrecord.gz')):
    for raw in tf.data.TFRecordDataset(str(path), compression_type='GZIP').with_options(opts):
        f = tf.train.Example.FromString(raw.numpy()).features.feature
        idx = int(f['id'].int64_list.value[0]); val = f['text'].bytes_list.value
        if idx in rows or len(val) != 1: raise ValueError('duplicate ID or missing text')
        rows[idx] = val[0].decode('utf-8')
same = sum(rows[i] == letter[str(i)]['title'] for i in rows if str(i) in letter)
contains = sum(letter[str(i)]['title'] in rows[i] for i in rows if str(i) in letter and letter[str(i)]['title'])
titles = Counter(v['title'] for v in letter.values())
mismatch = [i for i in sorted(rows) if str(i) in letter and (not letter[str(i)]['title'] or letter[str(i)]['title'] not in rows[i])]
print(json.dumps({'grid_count': len(rows), 'letter_count': len(letter), 'same_numeric_keys': sum(str(i) in letter for i in rows), 'exact_title': same, 'title_contained': contains, 'duplicate_title_groups': sum(v > 1 for v in titles.values()), 'duplicate_title_items': sum(v for v in titles.values() if v > 1), 'mismatch': [{'id': i, 'grid_text': rows[i][:500], 'letter_title': letter[str(i)]['title']} for i in mismatch], 'examples': [{'id': i, 'grid_text': rows[i][:500], 'letter_title': letter[str(i)]['title']} for i in sorted(rows)[:3] if str(i) in letter], 'limitation': 'Text overlap is evidence only; not a proven item mapping or identity.'}, indent=2))
