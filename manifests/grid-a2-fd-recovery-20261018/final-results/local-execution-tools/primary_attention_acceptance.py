#!/usr/bin/env python3
"""Read the terminal A2 attention arrays and verify their saved numerical content.

No model, Torch, GPU, production statistics or generation code is imported.
This primary check supplements the separate nonauthor result audit.
"""
from pathlib import Path
from itertools import combinations
import datetime as dt
import hashlib
import json
import math
import struct

import numpy as np

BASE = Path(__file__).resolve().parent
RAW = BASE/'deliveries/final-evidence-staging-v2/raw/a2'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def job(phase):
    return RAW/('ga-20261018-sportsa2-'+phase+'-fd-recovery-v1')

def bundle(path):
    record = read(path)
    archive = path.with_name(record['arrays_file'])
    assert archive.name == 'arrays.npz' and sha(archive) == record['arrays_sha256']
    used = set()
    with np.load(archive, allow_pickle=False) as arrays:
        def decode(value):
            if isinstance(value, dict) and '__array__' in value:
                key = value['__array__']; array = arrays[key].copy(); used.add(key)
                assert list(array.shape) == value['shape'] and str(array.dtype) == value['dtype']
                return array
            if isinstance(value, dict):
                return {key: decode(child) for key, child in value.items()}
            if isinstance(value, list):
                return [decode(child) for child in value]
            return value
        decoded = decode(record['bundle'])
        assert used == set(arrays.files)
    return decoded

def metrics(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    cosine = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
    xc, yc = x-x.mean(), y-y.mean()
    denom = np.linalg.norm(xc)*np.linalg.norm(yc)
    p, q = x/x.sum(), y/y.sum(); middle = (p+q)/2
    def divergence(vector):
        positive = vector > 0
        return np.sum(vector[positive]*np.log(vector[positive]/middle[positive]))
    return dict(cosine=float(cosine), pearson=float(np.dot(xc, yc)/denom) if denom else None,
                total_variation=float(abs(p-q).sum()/2),
                jensen_shannon_nats=float(max(0, (divergence(p)+divergence(q))/2)))

def main():
    cohort_path = job('metadata')/'output/fixed_attention_samples.json'
    cohort = read(cohort_path)
    stats_job = job('stats'); stats = read(stats_job/'output/RESULT.json')
    assert read(stats_job/'status.json')['status'] == 'SUCCEEDED' and read(stats_job/'cleanup.json')['verified']
    conditions = {}
    for condition, n in [('Normal', 0), ('Shuffle3', 3), ('Shuffle4', 4)]:
        directory = job('attention-'+str(n)); output = directory/'output'
        status = read(directory/'status.json'); result = read(output/'RESULT.json')
        assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
        assert read(directory/'cleanup.json')['verified'] and result['samples'] == 32 and result['formal']
        assert result['cohort']['sha256'] == sha(cohort_path)
        reference = next(r for r in stats['jobs'] if r['job_id'] == directory.name)
        pairs = {(r['sample_key'], r['path'], tuple(r['query_pair'])): r
                 for r in reference['per_sample_query_pairs']}
        errors = []; max_sum_error = 0.; users = []; pngs = 0; npzs = 0; masks = 0.
        for batch in range(4):
            prefix = output/f'batch-{batch:02d}'
            record = prefix/'bundle/record.json'; data = bundle(record); npzs += 1
            keys = cohort['sample_keys'][batch*8:batch*8+8]
            assert data['capture']['sample_keys'] == keys and data['capture']['encoder_calls'] == 1
            users += keys
            assert np.array_equal(data['input_sids'], cohort['input_ids'][batch*8:batch*8+8])
            assert np.array_equal(data['input_mask'], cohort['attention_mask'][batch*8:batch*8+8])
            mask = data['encoder_mask'].astype(bool)
            assert mask.shape == (8, 150)
            assert np.array_equal(mask.reshape(8, 30, 5)[:, :, :4].reshape(8, 120), data['input_mask'])
            steps = data['steps']; assert len(steps) == 4
            previous = None
            for h, step in enumerate(steps):
                heads, ids = step['cross_heads'], step['generated_ids']
                assert heads.shape == (8, 1 if h == 0 else 10, 6, 150)
                assert ids.shape == (8, 10, h+1) and heads.dtype == np.float32
                assert np.isfinite(heads).all() and (heads >= 0).all()
                max_sum_error = max(max_sum_error, float(abs(heads.sum(-1)-1).max()))
                masks = max(masks, float((heads * (~mask[:, None, None, :])).max()))
                if previous is None:
                    parents = np.zeros((8, 10), np.int64)
                else:
                    matching = (ids[:, :, None, :-1] == previous[:, None, :, :]).all(-1)
                    assert (matching.sum(-1) == 1).all(); parents = matching.argmax(-1)
                assert np.array_equal(parents, step['parent_indices']); previous = ids
            local = np.stack([step['cross_heads'][:, 0] for step in steps], axis=1)
            final = np.empty_like(local); indices = np.empty((8, 4), np.int64)
            for sample in range(8):
                child = 0
                for h in range(3, -1, -1):
                    parent = int(steps[h]['parent_indices'][sample, child])
                    indices[sample, h] = parent
                    final[sample, h] = steps[h]['cross_heads'][sample, parent]
                    assert steps[h]['generated_ids'][sample, child, -1] == data['final_generated_ids'][sample, 0, h]
                    child = parent
            assert np.array_equal(indices, data['paths']['final_parent_indices'])
            for label, head_values in [('local_first', local), ('final_top1', final)]:
                assert np.array_equal(head_values, data['paths'][label+'_heads'])
                average = head_values.mean(2)
                assert np.array_equal(average, data['paths'][label+'_cross'])
                for sample, key in enumerate(keys):
                    for first, second in combinations(range(4), 2):
                        calculated = metrics(average[sample, first, mask[sample]], average[sample, second, mask[sample]])
                        expected = pairs[(key, label+'_cross', (first+1, second+1))]
                        for metric, value in calculated.items():
                            if value is None:
                                assert expected[metric] is None
                            else:
                                errors.append(abs(value-expected[metric]))
            receipt = read(prefix/'figures/plot_receipt.json')
            assert receipt['source_record_sha256'] == sha(record) and len(receipt['products']) == 8
            for product in receipt['products']:
                image = prefix/'figures'/product['file']; raw = image.read_bytes()
                assert raw[:8] == b'\x89PNG\r\n\x1a\n'
                width, height = struct.unpack('>II', raw[16:24])
                assert [width, height] == [product['width'], product['height']] == [2880, 1280]
                assert len(raw) == product['bytes'] and sha(image) == product['sha256']; pngs += 1
        assert users == cohort['sample_keys'] and npzs == 4 and pngs == 32
        assert masks == 0 and max_sum_error < 2e-6 and max(errors) < 1e-12
        conditions[condition] = dict(users=32, NPZ=npzs, PNG=pngs, queries=4, heads=6,
             paths=['final_top1_cross', 'local_first_cross'], query_pair_rows=len(pairs),
             maximum_statistic_delta=max(errors), maximum_head_row_sum_error=max_sum_error,
             maximum_masked_weight=masks, exact_parent_path_reconstruction=True,
             statistics_summary=reference['summary'])
    report = dict(status='PRIMARY_A2_RAW_ATTENTION_CONTENT_ACCEPTANCE_PASS', author='/root',
                  checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  cohort_sha256=sha(cohort_path), cohort_split='evaluation',
                  conditions=conditions, total_NPZ=12, total_PNG=96,
                  model_or_GPU_calls=0, final_full_task_review_pass=False,
                  PNG_scope='All PNG hashes, source-record receipts and dimensions checked; no independent pixel rerender claim')
    destination = BASE/'primary-result-acceptance/a2-attention.json'
    assert not destination.exists()
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'conditions'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
