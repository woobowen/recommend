"""Read original G3 Beam20 candidates, independently score and verify saved metrics."""
from argparse import ArgumentParser
from collections import Counter
from csv import DictReader
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np

ORIGINAL = {
    'index': '420fc060caba3a22b0f03900dfe26abc67da8753f3bac2803f7f3c8703a05c22',
    'inter': '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5',
    'item': '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330',
    'axis': '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f',
    'normal': 'a2247ec7016c4d88c08957fea6c0338515a124ca316245e9592ac260e8df730a',
    'csv': '31331f3da0ee9f96e0e15600ced0e8d8cb9cbb4419e9e840c467ad3a925fec9a',
    'npz': 'c58d2d80c059fa48bf1262e9006fb3406cca25a623e4f4c82be119f5f22d7fd3',
    'metrics': '74a967a8cd636a238658bd7d256d45ab70905789f4f925a816d6a109ff652698',
    'rank': '82a026533a9ea619d38ba4f3baa632eee9270860b2d3e43030873b9626a94e1f',
    'adapter': '3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915',
    'added_tokens': 'a0ce503305df0d477c5c62450a72c363aa35c4d82b38abfcf309ab72f704bec1',
    'preflight': '8d72a6525b53be97fee4dc3283f75df68b6cba87fc08b1dfe6ce265ebcad77d0',
}
MODES = ('shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4')
KEYS = tuple(f'{metric}@{cutoff}' for metric in ('hit', 'ndcg', 'mrr', 'ItemHit', 'ItemNDCG')
             for cutoff in ((1, 5, 10, 20) if metric == 'hit' else (5, 10, 20)))
CONTRASTS = tuple((mode, 'normal') for mode in MODES) + (('shuffle3b', 'shuffle3a'),)
DECAY = [0.0] + [1.0 / math.log2(rank + 1) for rank in range(1, 21)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_digest(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def compact_digest(sequence):
    return sha256(json.dumps(sequence, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def row_metrics(ranked, target, groups):
    hits = [rank for rank, sid in enumerate(ranked, 1) if sid == target]
    # The first occurrence gets uniformly distributed credit over the matched item group.
    # Every earlier occurrence, including repeats and unknown SIDs, occupies its own slots.
    cce_g = groups[target]
    cce_first = 1 + sum(groups.get(sid, 1) for sid in ranked[:hits[0] - 1]) if hits else 21
    credits = range(cce_first, min(21, cce_first + cce_g))
    metrics = {}
    for k in (1, 5, 10, 20):
        metrics[f'hit@{k}'] = float(bool(hits) and hits[0] <= k)
    for k in (5, 10, 20):
        metrics[f'ndcg@{k}'] = math.fsum(DECAY[rank] for rank in hits if rank <= k)
        metrics[f'mrr@{k}'] = 1.0 / hits[0] if hits and hits[0] <= k else 0.0
        metrics[f'ItemHit@{k}'] = sum(rank <= k for rank in credits) / cce_g
        metrics[f'ItemNDCG@{k}'] = math.fsum(DECAY[rank] for rank in credits if rank <= k) / cce_g
    return [metrics[key] for key in KEYS]


def check_fixture():
    # Unknown and repeated prefix blocks occupy slots. SID and Item aggregation differ on repeated targets.
    ranked = ['unknown', 'a', 'a', 't', 't']
    result = dict(zip(KEYS, row_metrics(ranked, 't', {'a': 2, 't': 3})))
    assert result['hit@5'] == 1 and result['mrr@5'] == 1 / 4
    assert result['ndcg@5'] == math.fsum((DECAY[4], DECAY[5]))
    assert result['ItemHit@5'] == result['ItemNDCG@5'] == 0
    assert result['ItemHit@10'] == 1
    assert result['ItemNDCG@10'] == math.fsum(DECAY[6:9]) / 3
    repeated_target = dict(zip(KEYS, row_metrics(['t', 't'], 't', {'t': 1})))
    assert repeated_target['ndcg@5'] > 1 and repeated_target['ItemNDCG@5'] == 1


def prediction_matrix(path, mode, expected, users, targets, groups, token_names, legal_ids):
    content = sha256()
    records = {}
    totals = Counter()
    covered = set()
    seen = []
    first = Counter()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    original = signature(path.stat())
    with path.open('rb', buffering=1024**2) as stream:
        assert signature(os.fstat(stream.fileno())) == original
        for raw in stream:
            content.update(raw)
            row = json.loads(raw)
            i = row['row_idx']
            assert type(i) is int and 0 <= i < len(users) and i not in records, (mode, i)
            seen.append(i)
            item, target = targets[i]
            assert (row['user_id'], row['target_item_id'], row['target']) == (users[i], item, target), (mode, i)
            assert row['world_size'] == 1 and row['rank'] == row['prompt_id'] == 0, (mode, i)
            assert row['condition'] == mode and row['scope'] == 'FORMAL', (mode, i)
            assert row['model_identity_sha256'] == ORIGINAL['adapter'], (mode, i)
            assert mode == 'normal' or row['V2_RUN_COMMIT'] == V2_RUN
            beams = row['candidates']
            assert len(beams) == 20
            for pos, candidate in enumerate(beams):
                ids = candidate['generated_token_ids']
                assert (candidate['generation_order'] == pos and len(ids) >= 5 and candidate['eos_position'] == 4
                        and ids[4] == 2 and all(ids[k] in legal_ids[k] for k in range(4))
                        and all(token in (0, 2) for token in ids[5:])), (mode, i, pos)
                raw_sid = candidate['raw_decoded'].rsplit('Response:', 1)[-1].strip().replace(' ', '')
                assert raw_sid == candidate['sid'] == ''.join(token_names[token] for token in ids[:4]), (mode, i, pos)
                assert math.isfinite(candidate['sequence_score']), (mode, i, pos)
            order = sorted(range(20), key=lambda j: (-beams[j]['sequence_score'], j))
            ranked = [beams[j]['sid'] for j in order]
            assert order == row['sorted_generation_order'] and [int(sid == target) for sid in ranked] == row['relevance'], (mode, i)
            counts = Counter(ranked)
            invalid = sum(sid not in groups for sid in ranked)
            totals.update(rows=1, candidates=20, invalid_candidates=invalid,
                          affected_users=int(invalid > 0), duplicate_occurrences=20 - len(counts),
                          users_with_duplicates=int(len(counts) < 20))
            covered.update(sid for sid in ranked if sid in groups)
            records[i] = row_metrics(ranked, target, groups)
            first[next((pos for pos, sid in enumerate(ranked, 1) if sid == target), 0)] += 1
            totals['identity_inputs'] += int(row['input_token_ids'] == row['original_input_token_ids'])
        assert signature(os.fstat(stream.fileno())) == original
    assert signature(path.stat()) == original
    assert seen == expected, (mode, len(seen), len(expected))
    totals['catalog_coverage'] = len(covered)
    matrix = np.array([records[i] for i in sorted(seen)], dtype=np.float64)
    histogram = {str(i): first[i] for i in range(1, 21)} | {'miss': first[0]}
    return matrix, totals, content.hexdigest(), histogram


def archived_compare(matrix, archive, users, targets):
    csv_path, npz_path, metrics_path = [archive / name for name in ('per_user.csv', 'per_user_metrics.npz', 'metrics.json')]
    assert [file_digest(path) for path in (csv_path, npz_path, metrics_path)] == [ORIGINAL[k] for k in ('csv', 'npz', 'metrics')]
    with np.load(npz_path, allow_pickle=False) as stored:
        keys = stored['keys'].tolist()
        assert tuple(keys) == KEYS and str(stored['user_axis_sha256']) == ORIGINAL['axis']
        assert np.array_equal(stored['row_idx'], np.arange(len(users)))
        diff_npz = float(np.max(np.abs(matrix - stored['values'])))
        assert diff_npz <= 1e-14
    with csv_path.open(newline='') as csv_file:
        rows = list(DictReader(csv_file))
    assert len(rows) == len(users)
    for i, row in enumerate(rows):
        assert (row['row_idx'], row['user_id'], row['target_item_id'], row['target_sid'], row['user_axis_sha256']) == (
            str(i), users[i], *targets[i], ORIGINAL['axis'])
    csv_matrix = np.asarray([[float(row[k]) for k in KEYS] for row in rows])
    diff_csv = float(np.max(np.abs(matrix - csv_matrix)))
    assert diff_csv <= 1e-14
    archive_metrics = read(metrics_path)
    means = {key: math.fsum(matrix[:, col]) / len(matrix) for col, key in enumerate(KEYS)}
    diff_avg = max(abs(means[key] - archive_metrics['metrics'][key]) for key in KEYS)
    assert diff_avg <= 1e-14
    return dict(means=means, max_abs_npz_difference=diff_npz, max_abs_csv_difference=diff_csv,
                max_abs_mean_difference=diff_avg)


V2_RUN = '86660be5a62c8325caf74cf2810581850ca12803'
CPU_RUN = 'cb3b1f36d69ef0c1a7f554ef5455026523635f78'
MATRIX_TOLERANCE = 1e-14
BOOTSTRAP_TOLERANCE = 1e-13
USERS = 22363
CONDITIONS = ('normal', *MODES)
CPU_MANIFEST = '2329944065d81f07c02041dbf5a62daf2609af7795e71804ed854db8faf58906'
ASSIGNMENT = '22ee58fe9f948f50e5cbc92160b6dc7df7e2f633cb9e8bf3121da5fb35bcb088'
ROOT = Path('/data/bwwu/lc-rec/g3-beauty-v2')
G3_ROOT = Path('/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b')
DATA = Path('/home/bwwu/grid-audits/sid-extensions-p0/g3-production-20261118/data/Beauty')
NORMAL_TEST = G3_ROOT / 'attempts/normal/a0/test'
ARCHIVE = Path('/home/bwwu/grid-audits/sid-extensions-p0/releases/g3-beauty-v2') / CPU_RUN / (
    'lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/production_preparation/'
    'cpu_closure/results/compute/original/normal/primary')


def compare_published(matrix, evaluated, mode, users, targets, totals, digest, count_bytes, histogram):
    summary_file = evaluated / 'summary.json'
    csv_file = evaluated / 'primary/per_user.csv'
    npz_file = evaluated / 'primary/per_user_metrics.npz'
    summary = read(summary_file)
    hashes = {path.name: file_digest(path) for path in (summary_file, csv_file, npz_file)}
    assert (summary['scope'], summary['condition'], summary['users'], summary['candidate_count']) == ('FORMAL', mode, USERS, USERS*20)
    assert (summary['prediction_sha256'], summary['prediction_bytes'], summary['model_identity_sha256']) == (digest, count_bytes, ORIGINAL['adapter'])
    assert summary['assignment_sha256'] == ASSIGNMENT and summary['user_axis_sha256'] == ORIGINAL['axis']
    assert (summary['items'], summary['unique_sids'], summary['collision_groups'], summary['official_cce_rows']) == (12101, 12101, 0, USERS)
    if mode != 'normal':
        assert summary['run_commit'] == V2_RUN and summary['cpu_manifest_sha256'] == CPU_MANIFEST
        assert all(hashes[key] == value for key,value in summary['per_user_files'].items())
        assert len(summary['natural_identity_rows']) == totals['identity_inputs']
    else:
        assert summary['frozen_archive_csv_npz_and_means_exact'] is True
    assert summary['first_target_rank_users'] == histogram
    stats = summary['sid_statistics']['20']
    assert (stats['candidates'], stats['invalid'], stats['users_invalid'], stats['duplicates'], stats['users_duplicate'],
            stats['unique_valid_sids'], stats['colliding'], stats['users_colliding']) == (
        totals['candidates'], totals['invalid_candidates'], totals['affected_users'], totals['duplicate_occurrences'],
        totals['users_with_duplicates'], totals['catalog_coverage'], 0, 0)
    with np.load(npz_file, allow_pickle=False) as npz:
        assert npz['keys'].tolist() == list(KEYS) and str(npz['user_axis_sha256']) == ORIGINAL['axis']
        assert np.array_equal(npz['row_idx'], np.arange(USERS))
        npz_max = float(np.max(np.abs(npz['values'] - matrix)))
    with csv_file.open(newline='') as stream:
        records = list(DictReader(stream))
    assert len(records) == USERS
    for i, record in enumerate(records):
        assert (record['row_idx'], record['user_id'], record['target_item_id'], record['target_sid'],
                record['user_axis_sha256']) == (str(i), users[i], *targets[i], ORIGINAL['axis'])
    values = np.asarray([[float(row[key]) for key in KEYS] for row in records], dtype=np.float64)
    csv_max = float(np.max(np.abs(values - matrix)))
    averages = {key: math.fsum(matrix[:, j]) / USERS for j, key in enumerate(KEYS)}
    assert set(averages) == set(summary['metrics'])
    mean_max = max(abs(averages[key] - summary['metrics'][key]) for key in KEYS)
    assert max(npz_max, csv_max, mean_max) <= MATRIX_TOLERANCE
    return {'comparison_max_abs': {'csv': csv_max, 'npz': npz_max, 'means': mean_max},
            'file_sha256': hashes, 'means': averages, 'sid_top20': dict(totals),
            'first_target_rank_users': histogram}


def bootstrap_by_multiplicity(matrices):
    """Index digest from PCG64; draw means via weighted dot products."""
    deltas = [matrices[left] - matrices[right] for left, right in CONTRASTS]
    stacked = np.column_stack(deltas)
    means = np.empty((5000, len(KEYS) * len(CONTRASTS)), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(20261113))
    digest = sha256()
    for start in range(0, 5000, 32):
        count = min(32, 5000 - start)
        index = rng.integers(0, USERS, (count, USERS), dtype=np.int64)
        digest.update(index.astype('<i8', copy=False).tobytes(order='C'))
        frequency = np.vstack([np.bincount(row, minlength=USERS) for row in index])
        means[start:start+count] = frequency.astype(np.float64) @ stacked / USERS
    draws = means.reshape(5000, len(CONTRASTS), len(KEYS)).transpose(1, 0, 2)
    return digest.hexdigest(), deltas, draws


def review_paired(root, matrices):
    summary_file, csv_file, npz_file = (root / member for member in ('summary.json','paired.csv','arrays.npz'))
    summary = read(summary_file)
    assert (summary['status'], summary['designation'], summary['scope'], summary['users']) == ('PASS_G3_V2_PAIRED', 'FORMAL_PAIRED', 'FORMAL', USERS)
    assert summary['row_idx'] == list(range(USERS)) and summary['user_axis_sha256'] == ORIGINAL['axis']
    assert summary['v2_run_commit'] == V2_RUN and summary['cpu_manifest_sha256'] == CPU_MANIFEST
    assert summary['normal_prediction_sha256'] == ORIGINAL['normal']
    digest, deltas, draws = bootstrap_by_multiplicity(matrices)
    assert summary['bootstrap'] == {'rng': 'numpy Generator PCG64', 'seed': 20261113,
        'replicates': 5000, 'replacement': True, 'shared_draws': True, 'percentiles': [2.5, 97.5],
        'method': 'linear', 'multiple_comparison_adjustment': False, 'draw_indices_sha256_little_endian': digest}
    names = [f'{left}-{right}' for left, right in CONTRASTS]
    with np.load(npz_file, allow_pickle=False) as npz:
        assert npz['comparison'].tolist() == names and npz['conditions'].tolist() == list(CONDITIONS)
        assert npz['metric'].tolist() == list(KEYS) and np.array_equal(npz['row_idx'], np.arange(USERS))
        maxima = {k: float(np.max(np.abs(npz[k] - expected))) for k, expected in {
            'per_user_values': [matrices[n] for n in CONDITIONS],
            'per_user_differences': deltas,
            'bootstrap_difference_means': draws,
        }.items()}
    with csv_file.open(newline='') as stream:
        table = list(DictReader(stream))
    assert len(table) == len(summary['comparisons']) == 80
    means = {mode: {key: math.fsum(values[:, col]) / USERS for col,key in enumerate(KEYS)}
             for mode, values in matrices.items()}
    ci = np.percentile(draws, (2.5, 97.5), axis=1, method='linear')
    maxima['summary_means'] = max(abs(means[m][key] - summary['metrics'][m][key]) for m in CONDITIONS for key in KEYS)
    maxima['pair_csv_vs_summary'] = 0.0
    for field in ('left', 'right', 'difference', 'relative_change', 'ci95_low', 'ci95_high'):
        maxima[field] = 0.0
    for i,(left_name,right_name) in enumerate(CONTRASTS):
        comparison = names[i]
        for col,key in enumerate(KEYS):
            offset = i*len(KEYS)+col
            official, csv_row = summary['comparisons'][offset], table[offset]
            assert official['comparison'] == csv_row['comparison'] == comparison
            assert official['metric'] == csv_row['metric'] == key
            left, right = means[right_name][key], means[left_name][key]
            difference = right - left
            assert abs(difference - math.fsum(deltas[i][:, col])/USERS) <= MATRIX_TOLERANCE
            reconstructed = {'left': left, 'right': right, 'difference': difference,
                             'relative_change': difference/left if left else None,
                             'ci95_low':ci[0,i,col], 'ci95_high':ci[1,i,col]}
            for field, value in reconstructed.items():
                if value is None:
                    assert official[field] is None and csv_row[field] == ''
                else:
                    maxima[field] = max(maxima[field],abs(value-official[field]))
                    maxima['pair_csv_vs_summary'] = max(maxima['pair_csv_vs_summary'],abs(float(csv_row[field])-official[field]))
    assert max(maxima.values()) <= BOOTSTRAP_TOLERANCE,maxima
    return {'records':80,'draws':5000,'shared_draw_indices_sha256_little_endian':digest,
            'independent_bootstrap_reduction':'per-draw frequency and weighted dot product',
            'comparison_max_abs':maxima,'file_sha256':{p.name:file_digest(p) for p in (summary_file,csv_file,npz_file)}}


def main():
    started = monotonic()
    p = ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--normal-test',type=Path,default=NORMAL_TEST)
    p.add_argument('--archive',type=Path,default=ARCHIVE)
    p.add_argument('--added-tokens',type=Path,default=G3_ROOT/'training/best_adapter/added_tokens.json')
    args = p.parse_args()
    check_fixture()
    paths = {key:args.data/f'Beauty.{key}.json' for key in ('index','inter','item')}
    assert all(file_digest(path) == ORIGINAL[key] for key,path in paths.items())
    data = {key:read(path) for key,path in paths.items()}
    assert len(data['index']) == 12101 and data['index'].keys() == data['item'].keys()
    sid_of = {item:''.join(tokens) for item,tokens in data['index'].items()}
    groups = Counter(sid_of.values())
    assert len(groups) == 12101 and set(groups.values()) == {1}
    users = list(data['inter'])
    assert len(users) == USERS and compact_digest(users) == ORIGINAL['axis']
    targets = [(str(seq[-1]),sid_of[str(seq[-1])]) for seq in data['inter'].values()]
    assert file_digest(args.added_tokens) == ORIGINAL['added_tokens']
    added = read(args.added_tokens)
    assert len(added) == 786 and set(added.values()) == set(range(32000,32786))
    legal = [{added[tokens[slot]] for tokens in data['index'].values()} for slot in range(4)]
    names = {token:name for name,token in added.items()}
    normal_rank = args.normal_test/'rank0.json'
    assert file_digest(normal_rank) == ORIGINAL['rank']
    assigned = read(normal_rank)['assigned_row_indices']
    assert sorted(assigned) == list(range(USERS)) and compact_digest(assigned) == ASSIGNMENT
    matrices, outputs = {}, {}
    for mode in CONDITIONS:
        raw_root = args.normal_test if mode == 'normal' else args.root/f'gpu-formal-{mode}-r0'
        evaluated = args.root/('cpu-normal-r0' if mode == 'normal' else f'cpu-formal-eval-{mode}-r0')
        path = raw_root/'predictions.rank0.jsonl'
        raw_rank = read(raw_root/'rank0.json')
        assert raw_rank['assigned_row_indices'] == assigned and raw_rank['assignment_sha256'] == ASSIGNMENT
        assert (raw_rank['world_size'],raw_rank['rank'],raw_rank['scope'],raw_rank['completed_users']) == (1,0,'FORMAL',USERS)
        matrix, totals, digest, histogram = prediction_matrix(path,mode,assigned,users,targets,groups,names,legal)
        if mode == 'normal':
            assert digest == ORIGINAL['normal']
            frozen = archived_compare(matrix,args.archive,users,targets)
        gpu_stage = read(raw_root/'metrics.json')
        file_row, = gpu_stage['prediction_files']
        assert file_row['sha256'] == digest and file_row['bytes'] == path.stat().st_size
        assert totals['rows'] == USERS and totals['candidates'] == USERS*20
        checked = compare_published(matrix,evaluated,mode,users,targets,totals,digest,path.stat().st_size,histogram)
        outputs[mode] = {'source_path':str(path),'prediction_sha256':digest,'prediction_bytes':path.stat().st_size,**checked}
        matrices[mode] = matrix
        print(json.dumps({'verified':mode,'rows':totals['rows'],'candidates':totals['candidates'],
                          'elapsed_seconds':monotonic()-started}),flush=True)
    paired = review_paired(args.root/'cpu-formal-paired-r0',matrices)
    out = {'status':'PASS','direction':'B','scope':'five full formal raw predictions',
           'method':'stream original JSONL; independent stable SID and frozen CCE formulas; shared draws weighted reductions',
           'users_per_condition':USERS,'candidates_per_condition':USERS*20,
           'total_candidates':USERS*20*len(CONDITIONS),'numpy':np.__version__,
           'candidate_metrics':outputs,'frozen_normal_archive':frozen,
           'fixtures':['unknown_prefix','repeated_collision_prefix','repeated_target'],
           'paired':paired,'elapsed_seconds':monotonic()-started}
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(out,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'PASS','paired':paired['comparison_max_abs'],'elapsed_seconds':out['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
