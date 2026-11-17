"""CPU-only preparation. Never loads the classifier or produces semantic labels."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3',
                  TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys
import time

from common import HERE, roots, sha, write_json, write_jsonl, prefix_groups


def main():
    import numpy as np
    import tensorflow as tf
    import torch
    tf.config.set_visible_devices([], 'GPU')
    torch.set_num_threads(2)
    cfg, root, prep, output = roots()
    audit = {'prepared_unix': time.time(), 'experiment_started': False, 'datasets': {}}
    for dataset, spec in cfg['datasets'].items():
        text_rows = {}
        files = sorted((root / 'original_assets/data/amazon_data' / dataset / 'items').glob('*.tfrecord.gz'))
        assert files, dataset
        source_files = {}
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        for path in files:
            source_files[str(path.relative_to(root))] = sha(path)
            for raw in tf.data.TFRecordDataset(str(path), compression_type='GZIP').with_options(options):
                fields = tf.train.Example.FromString(raw.numpy()).features.feature
                item_id = int(fields['id'].int64_list.value[0])
                texts = fields['text'].bytes_list.value
                assert item_id not in text_rows and len(texts) == 1
                text_rows[item_id] = {'item_id': item_id, 'text': texts[0].decode('utf-8', errors='strict')}
        n = len(text_rows)
        assert n == spec['expected_items'] and sorted(text_rows) == list(range(n))
        sid_path = root / 'experiments' / spec['baseline'] / 'semantic_ids.pt'
        tensor = torch.load(sid_path, map_location='cpu', weights_only=True)
        assert tuple(tensor.shape) == (4, n), tensor.shape
        assert torch.equal(tensor, tensor.long()) and tensor.min() >= 0 and tensor[:3].max() < 256
        assert torch.unique(tensor.T, dim=0).shape[0] == n
        sids = tensor.T.long().tolist()
        dataset_prep = prep / dataset
        rows = [text_rows[i] for i in range(n)]
        # This is the only item table consumed by the classifier: no SID fields.
        write_jsonl(dataset_prep / 'classifier_items.jsonl', rows)
        rng = random.Random(cfg['seed'])
        discovery_ids = sorted(rng.sample(range(n), min(n, cfg['discovery_items'])))
        review_ids = sorted(rng.sample(range(n), min(n, cfg['review_items'])))
        write_jsonl(dataset_prep / 'discovery_items.jsonl', [text_rows[i] for i in discovery_ids])
        write_jsonl(dataset_prep / 'review_items.jsonl', [text_rows[i] for i in review_ids])
        write_jsonl(dataset_prep / 'item_sids.jsonl', [{'item_id': i, 'sid': sids[i]} for i in range(n)])
        groups = prefix_groups(sids)
        summary = {}
        for length, mapping in groups.items():
            ranked = sorted(mapping.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            write_jsonl(dataset_prep / f'prefix_groups_l{length}.jsonl', [
                {'prefix': list(prefix), 'count': len(ids), 'item_ids': ids} for prefix, ids in ranked])
            counts = np.array([len(ids) for _, ids in ranked])
            summary[length] = {
                'observed_prefixes': len(ranked), 'singleton_prefixes': int((counts == 1).sum()),
                'shared_prefixes': int((counts >= 2).sum()), 'min': int(counts.min()),
                'median': float(np.median(counts)), 'p90': float(np.quantile(counts, .9)),
                'p99': float(np.quantile(counts, .99)), 'max': int(counts.max()),
                'thresholds': {str(m): {'groups': int((counts >= m).sum()),
                    'items': int(counts[counts >= m].sum()), 'coverage': float(counts[counts >= m].sum()/n)}
                    for m in [2, 5, 10, 20]},
                'top_k_shared': {str(k): {'groups': len([v for v in counts if v >= 2][:k]),
                    'items': int(sum([v for v in counts if v >= 2][:k])),
                    'coverage': float(sum([v for v in counts if v >= 2][:k])/n)} for k in cfg['top_k']},
                'top_k_min5': {str(k): {'groups': len([v for v in counts if v >= 5][:k]),
                    'items': int(sum([v for v in counts if v >= 5][:k])),
                    'coverage': float(sum([v for v in counts if v >= 5][:k])/n)} for k in cfg['top_k']}}
        audit['datasets'][dataset] = {
            'items': n, 'blank_texts': sum(not r['text'].strip() for r in rows),
            'duplicate_texts': n - len({r['text'] for r in rows}),
            'text_characters': {'min': min(len(r['text']) for r in rows),
                'median': statistics.median(len(r['text']) for r in rows),
                'max': max(len(r['text']) for r in rows)},
            'item_source_sha256': source_files, 'sid_path': str(sid_path), 'sid_sha256': sha(sid_path),
            'classifier_items_sha256': sha(dataset_prep / 'classifier_items.jsonl'),
            'discovery_ids': discovery_ids, 'review_ids': review_ids, 'prefix_summary': summary}
        print(dataset, n, json.dumps(summary), flush=True)
    audit['source_sha256'] = {p.name: sha(p) for p in HERE.glob('*.py')}
    audit['config_sha256'] = sha(HERE / 'config.json')
    write_json(prep / 'data_preparation_verified.json', audit)
    print('preparation_complete; no semantic classification or experiment started', flush=True)


if __name__ == '__main__':
    main()
