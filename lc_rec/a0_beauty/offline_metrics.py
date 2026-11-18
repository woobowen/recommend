#!/usr/bin/env python3
"""CPU-only frozen Beam20 evaluator. Never writes into original experiment paths."""
import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time

N = 22363
# CPU-verified torch 2.6 seed-0 randperm, rank-strided, without padding.
RANK_ASSIGNMENT_SHA = ('8eb669c968e313c79df1bba3f16b16bd94ce1e0d6697728ff528c1da3de01363',
                       'ea9c57674be765e0424ac991e986823a5b8587cfee3d162c3b38a46deaf4f173')
from g3_identity import ASSETS, EXPERIMENT_ID
CATALOG_SIDS = 12101
CATALOG_COLLISION_GROUPS = 0

KEYS = ([f'hit@{k}' for k in (1, 5, 10, 20)] +
        [f'{m}@{k}' for m in ('ndcg', 'mrr') for k in (5, 10, 20)] +
        [f'{m}@{k}' for m in ('ItemHit', 'ItemNDCG') for k in (5, 10, 20)])


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def load(path):
    return json.loads(Path(path).read_bytes())


def identity(path):
    return {'path': str(path), 'bytes': Path(path).stat().st_size, 'sha256': sha(path)}


def module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def normalize(raw):
    return raw.split('Response:')[-1].strip().replace(' ', '')


def ranking(sids, target):
    positions = [i + 1 for i, sid in enumerate(sids) if sid == target]
    first = positions[0] if positions else 0
    values = {f'hit@{k}': float(0 < first <= k) for k in (1, 5, 10, 20)}
    for k in (5, 10, 20):
        values[f'ndcg@{k}'] = math.fsum(1 / math.log2(r + 1) for r in positions if r <= k)
        values[f'mrr@{k}'] = 1 / first if 0 < first <= k else 0.0
    return values, first


def cce(sids, target, sizes):
    result = {}
    r = sids.index(target) if target in sids else None
    g = sizes.get(target, 1)
    p = sum(sizes.get(sid, 1) for sid in sids[:r]) if r is not None else 0
    for k in (5, 10, 20):
        m = min(g, max(0, k - p)) if r is not None else 0
        result[f'ItemHit@{k}'] = m / g
        result[f'ItemNDCG@{k}'] = math.fsum(1 / math.log2(p + i + 1) for i in range(1, m + 1)) / g
    return result


def selftest(official):
    assertions = 0
    for r in (1, 5, 6, 10, 11, 20, 0):
        sids = [f'valid{i}' for i in range(20)]
        if r:
            sids[r - 1] = 'target'
        values, first = ranking(sids, 'target')
        require(first == r, 'first rank')
        for k in (5, 10, 20):
            hit = 0 < r <= k
            require(values[f'hit@{k}'] == float(hit), 'hit cutoff')
            require(values[f'ndcg@{k}'] == (1 / math.log2(r + 1) if hit else 0), 'NDCG cutoff')
            require(values[f'mrr@{k}'] == (1 / r if hit else 0), 'MRR cutoff')
            assertions += 3
    values, _ = ranking(['t', 't'], 't')
    require(values['ndcg@5'] == 1 + 1 / math.log2(3) and values['mrr@5'] == 1, 'duplicate semantics')
    require(sorted(range(4), key=[0, 1, 1, 0].__getitem__, reverse=True) == [1, 2, 0, 3], 'stable ties')
    require(normalize('Response: old Response: <a> <b>\n') == '<a><b>', 'parser')
    cases = [('no_collision', ['a', 'b', 'c', 't'], {'t': 1}),
             ('target2', ['a', 'b', 'c', 't'], {'t': 2}),
             ('figure2_target3_partial', ['a', 'b', 'c', 't'], {'t': 3}),
             ('group_outside', ['a', 't'], {'a': 5, 't': 3}),
             ('absent', ['a'], {'t': 2}),
             ('prefix_collision', ['a', 't'], {'a': 4, 't': 1}),
             ('invalid_prefix', ['INVALID', 't'], {'t': 1}),
             ('duplicate_prefix', ['a', 'a', 't'], {'a': 2, 't': 1}),
             ('duplicate_target', ['t', 't'], {'t': 2}),
             ('unique_pushed_out', ['a', 'b', 't'], {'a': 4, 'b': 2, 't': 1})]
    receipts = []
    for name, sids, sizes in cases:
        mine = cce(sids, 't', sizes)
        theirs = official.compute_cce_metrics(sids, 't', sizes, [5, 10, 20])
        require(all(abs(mine[k] - theirs[k]) <= 1e-14 for k in mine), name)
        receipts.append({'case': name, 'result': mine})
        assertions += len(mine)
    figure = receipts[2]['result']
    require(figure['ItemHit@5'] == 2/3 and abs(figure['ItemNDCG@5'] - (1/math.log2(5)+1/math.log2(6))/3) < 1e-15, 'Figure2 numerical fixture')
    return {'status': 'PASS', 'numerical_assertions': assertions + 4, 'cce_cases': receipts}


def verify_bootstrap_config(cfg):
    frozen = {'unit': 'user', 'replacement': True, 'repetitions': 5000,
              'seed': 20261113, 'rng': 'numpy Generator PCG64',
              'percentiles': [2.5, 97.5], 'percentile_method': 'linear'}
    require(all(cfg.get(k) == v for k, v in frozen.items()), 'Frozen bootstrap protocol mismatch')


def evaluate(args):
    import numpy as np
    started = time.monotonic()
    scope = getattr(args, 'scope', 'FORMAL')
    require(scope in ('FORMAL', 'PREFLIGHT'), 'Unknown evaluation scope')
    preflight = scope == 'PREFLIGHT'
    selection_path = getattr(args, 'preflight_users', None)
    require(preflight == (selection_path is not None), 'Explicit preflight user selection required only for PREFLIGHT')
    if preflight:
        selection = load(selection_path)
        require(selection['asset_sha256'] == ASSETS, 'Preflight selection asset identity')
        rank_indices = [[row['row_idx'] for row in selection['ranks'][str(rank)]] for rank in (0, 1)]
        selected_indices = sorted(i for shard in rank_indices for i in shard)
        require(len(selected_indices) == len(set(selected_indices)) == 32 and
                all(type(i) is int and 0 <= i < N for i in selected_indices), 'Fixed 32-user preflight population')
    else:
        selected_indices = list(range(N))
    population = len(selected_indices)
    dense_index = {row: i for i, row in enumerate(selected_indices)}
    output = args.output
    require(not output.exists(), 'Preserve existing result directory')
    require(args.test_dir.resolve() not in output.resolve().parents and output.resolve() != args.test_dir.resolve(), 'Original results are read only')
    output.mkdir(parents=True)
    cfg = load(args.config)
    verify_bootstrap_config(cfg['bootstrap'])
    require(cfg['protocol_version'] == 'letter_beauty_beam20_v1' and cfg['original_run_commit'] == args.run_commit, 'Protocol identity')
    require(len(args.run_commit) == len(args.evaluation_commit) == 40 and all(c in '0123456789abcdef' for c in args.run_commit + args.evaluation_commit), 'Commit identity')
    require(cfg['dataset'] == 'Beauty' and cfg['users'] == population and cfg['asset_sha256'] == ASSETS and cfg['sid_coverage_denominator'] == CATALOG_SIDS, 'Beauty configuration identity')
    require(sha(args.official_cce) == cfg['official_cce_sha256'], 'Official CCE identity')
    official = module(args.official_cce)
    write(output/'unit_tests.json', selftest(official))
    stage = load(args.stage_a_report)
    require(stage['status'] == 'PASS' and stage.get('run_commit', stage.get('RUN_COMMIT')) == args.run_commit and stage['actual_test_users'] == population, 'Original audit population gate')
    require(stage.get('scope', 'FORMAL') == scope, 'Stage evaluation scope mismatch')
    require(not preflight or stage.get('preflight_only') is True, 'PREFLIGHT stage must be explicitly labeled')
    data, assets = {}, {}
    for kind, expected in ASSETS.items():
        path = args.data_root/'Beauty'/f'Beauty.{kind}.json'
        assets[kind] = identity(path)
        require(assets[kind]['sha256'] == expected, 'Asset identity')
        data[kind] = load(path)
    index, inter = data['index'], data['inter']
    require(len(index) == 12101 and len(inter) == N and set(index) == set(data['item']), 'Frozen axes')
    sid_for = {item: ''.join(tokens) for item, tokens in index.items()}
    sizes = Counter(sid_for.values())
    require(len(sizes) == CATALOG_SIDS and sum(g > 1 for g in sizes.values()) == CATALOG_COLLISION_GROUPS, 'Beauty catalog identity')
    users = list(inter)
    user_axis_sha = hashlib.sha256(json.dumps(users, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    require(user_axis_sha == cfg['user_axis_sha256'], 'Beauty user axis identity')
    freq = Counter({item: 0 for item in index})
    for seq in inter.values():
        freq.update(str(item) for item in seq[:-2])
    positive = sorted(f for f in freq.values() if f > 0)
    q1, q2 = positive[math.ceil(len(positive)/3)-1], positive[math.ceil(2*len(positive)/3)-1]
    def popularity(f):
        return 'UNSEEN' if f == 0 else 'LONG_TAIL' if f <= q1 else 'MID' if f <= q2 else 'POPULAR'
    matrix = np.zeros((population, len(KEYS)), dtype=np.float64)
    seen, peruser = set(), [None]*population
    first_counts, target_collision = Counter(), 0
    diag = {k: Counter() for k in (5, 10, 20)}
    coverage = {k: set() for k in diag}
    examples = {'invalid': [], 'target_collision': [], 'prefix_collision_changes_cce': []}
    manifests, generation = [], np.zeros(population, dtype=np.float64)
    for rank in (0, 1):
        path = args.test_dir/f'predictions.rank{rank}.jsonl'
        before = path.stat()
        inherited = stage['prediction_files'][rank]
        require(inherited['bytes'] == before.st_size and inherited['sha256'] == sha(path), 'Frozen prediction identity')
        rank_report = load(args.test_dir/f'rank{rank}.json')
        require(rank_report.get('run_commit', rank_report.get('RUN_COMMIT')) == args.run_commit, 'Rank RUN identity')
        require(rank_report.get('scope', 'FORMAL') == scope and
                rank_report.get('preflight_only', False) == preflight, 'Rank evaluation scope mismatch')
        require(rank_report['data_identity'].keys() == ASSETS.keys() and
                {k: v['sha256'] for k, v in rank_report['data_identity'].items()} == ASSETS, 'Rank asset identity')
        model_identity = {k: rank_report[k] for k in ('adapter_files', 'generation', 'allowed_callback_sha256')}
        if rank == 0:
            first_model_identity = model_identity
        else:
            require(model_identity == first_model_identity, 'Ranks used different model/tokenizer/generation identities')
        if 'adapter_model_sha256' in stage:
            require(rank_report['adapter_files']['adapter_model.bin']['sha256'] == stage['adapter_model_sha256'], 'Stage model binding')
        if preflight:
            require(rank_report.get('condition') == stage.get('condition'), 'Shuffle condition binding')
        assignments = []
        with path.open('rb') as stream:
            for raw in stream:
                row = json.loads(raw)
                i = row['row_idx']
                require(type(i) is int and i in dense_index and i not in seen and row['rank'] == rank and row['prompt_id'] == 0, 'User coverage/rank/prompt')
                seq = inter[users[i]]
                item, target = str(seq[-1]), sid_for[str(seq[-1])]
                require(row['user_id'] == users[i] and row['target_item_id'] == item and row['target'] == target, 'User/target pairing')
                require(row.get('scope', scope) == scope, 'Prediction scope mismatch')
                if preflight:
                    require(row.get('condition') == stage.get('condition'), 'Prediction Shuffle condition mismatch')
                candidates = row['candidates']
                require(len(candidates) == 20, 'Beam20')
                for j, candidate in enumerate(candidates):
                    require(candidate['generation_order'] == j and normalize(candidate['raw_decoded']) == candidate['sid'] and math.isfinite(candidate['sequence_score']), 'Candidate integrity')
                order = sorted(range(20), key=lambda j: candidates[j]['sequence_score'], reverse=True)
                sids = [normalize(candidates[j]['raw_decoded']) for j in order]
                require(order == row['sorted_generation_order'] and [int(s == target) for s in sids] == row['relevance'], 'Stored relevance/order')
                values, first = ranking(sids, target)
                corrected = cce(sids, target, sizes)
                ref = official.compute_cce_metrics(sids, target, sizes, [5, 10, 20])
                require(all(abs(corrected[k] - ref[k]) <= 1e-14 for k in ref), 'Full-user official CCE comparison')
                values.update(corrected)
                matrix[dense_index[i]] = [values[k] for k in KEYS]
                first_counts[str(first) if first else 'miss'] += 1
                target_collision += sizes[target] > 1
                for k in diag:
                    subset = sids[:k]
                    valid = sum(s in sizes for s in subset)
                    dup = k-len(set(subset))
                    coll = sum(sizes.get(s, 0) > 1 for s in subset)
                    diag[k].update(candidates=k, valid=valid, invalid=k-valid, duplicates=dup, colliding=coll,
                                   users_invalid=int(valid<k), users_duplicate=int(dup>0), users_colliding=int(coll>0))
                    coverage[k].update(s for s in subset if s in sizes)
                elapsed = float(row['generation_seconds'])
                require(math.isfinite(elapsed) and elapsed >= 0, 'Generation time')
                generation[dense_index[i]] = elapsed
                hlen = min(20, len(seq)-1)
                hgroup = '0-5' if hlen <= 5 else '6-10' if hlen <= 10 else '11-20'
                peruser[dense_index[i]] = {'row_idx': i, 'user_id': users[i], 'user_axis_sha256': user_axis_sha,
                              'target_item_id': item, 'target_sid': target, 'target_group_size': sizes[target],
                              'original_interaction_length': len(seq), 'visible_history_length': hlen,
                              'history_group': hgroup, 'target_training_frequency': freq[item],
                              'popularity_group': popularity(freq[item]), 'first_target_rank': first,
                              'generation_seconds': elapsed, 'protocol_version': cfg['protocol_version'], **values}
                sample = {'row_idx': i, 'target_sid': target, 'first_target_rank': first, 'ranked_sids': sids, 'metrics': values}
                for condition, label in [(any(s not in sizes for s in sids), 'invalid'), (sizes[target]>1, 'target_collision'),
                                         (sizes[target]==1 and any(abs(values[f'ndcg@{k}']-corrected[f'ItemNDCG@{k}'])>1e-15 for k in (5,10,20)), 'prefix_collision_changes_cce')]:
                    if condition and len(examples[label]) < 5:
                        examples[label].append(sample)
                seen.add(i)
                assignments.append(i)
        after = path.stat()
        require((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'Original prediction changed')
        assignment_sha = hashlib.sha256(json.dumps(assignments,separators=(',',':')).encode()).hexdigest()
        require(assignment_sha == rank_report['assignment_sha256'] and len(assignments)==(len(rank_indices[rank]) if preflight else (N+1-rank)//2), 'Original rank assignment identity')
        require(assignments == rank_indices[rank] if preflight else assignment_sha == RANK_ASSIGNMENT_SHA[rank], 'Frozen rank sampler order mismatch')
        manifests.append({**inherited, 'mtime_ns': after.st_mtime_ns, 'ctime_ns': after.st_ctime_ns, 'assignment_sha256': assignment_sha, 'hash_recomputed': True})
    require(seen == set(selected_indices) and sum(first_counts.values()) == population, 'Complete population')
    means = {key: math.fsum(matrix[:,j])/population for j,key in enumerate(KEYS)}
    deltas = {key: means[key]-stage['metrics'][key] for key in stage['metrics']}
    require(all(abs(d)<1e-13 for d in deltas.values()), 'Original metric regression')
    sid_stats = {}
    for k, counts in diag.items():
        sid_stats[str(k)] = {**counts, 'valid_rate':counts['valid']/(population*k), 'invalid_rate':counts['invalid']/(population*k),
            'duplicate_rate':counts['duplicates']/(population*k), 'collision_candidate_rate':counts['colliding']/(population*k),
            'users_invalid_rate':counts['users_invalid']/population, 'users_duplicate_rate':counts['users_duplicate']/population,
            'users_colliding_rate':counts['users_colliding']/population, 'unique_valid_sids':len(coverage[k]),
            'coverage_denominator':len(sizes), 'unique_valid_sid_coverage':len(coverage[k])/len(sizes)}
    require(diag[20]['invalid']==stage['anomalies']['invalid_complete_sid_candidates'] and diag[20]['colliding']==stage['anomalies']['colliding_sid_candidate_occurrences'], 'Historical anomalies')
    for k in (5,10,20):
        require(sum(first_counts[str(r)] for r in range(1,k+1)) == round(means[f'hit@{k}']*population), 'Hit histogram')
    perpath=output/'per_user.csv'
    with perpath.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(peruser[0]));writer.writeheader();writer.writerows(peruser)
    np.savez_compressed(output/'per_user_metrics.npz', values=matrix, keys=np.asarray(KEYS), row_idx=np.asarray(selected_indices), user_axis_sha256=np.asarray(user_axis_sha))
    group_results={}
    for field in ('history_group','popularity_group'):
        group_results[field]={}
        labels=cfg['history_groups'] if field=='history_group' else ['UNSEEN','LONG_TAIL','MID','POPULAR']
        for label in labels:
            indices=[i for i,row in enumerate(peruser) if row[field]==label]
            group_results[field][label]={'users':len(indices),'metrics':{key:math.fsum(matrix[indices,j])/len(indices) if indices else None for j,key in enumerate(KEYS)}}
    bootstrap_started=time.monotonic()
    rng=np.random.Generator(np.random.PCG64(cfg['bootstrap']['seed']))
    draws=np.empty((5000,len(KEYS)),dtype=np.float64)
    for begin in range(0,5000,8):
        selected=rng.integers(0,population,size=(min(8,5000-begin),population),dtype=np.int64)
        draws[begin:begin+len(selected)]=matrix[selected].mean(axis=1)
    intervals=np.percentile(draws,[2.5,97.5],axis=0,method='linear')
    np.save(output/'bootstrap_draw_means.npy',draws)
    bootstrap={**cfg['bootstrap'],'users':population,'numpy_version':np.__version__,'elapsed_seconds':time.monotonic()-bootstrap_started,
               'intervals':{key:{'estimate':means[key],'low':float(intervals[0,j]),'high':float(intervals[1,j])} for j,key in enumerate(KEYS)},
               'metric_array_sha256':sha(output/'per_user_metrics.npz')}
    write(output/'bootstrap.json',bootstrap)
    write(output/'groups.json',{'designation':'SUPPLEMENTARY_ANALYSIS','q1':q1,'q2':q2,'training_visible_interactions':sum(freq.values()),
                               'catalog_groups':dict(Counter(popularity(f) for f in freq.values())), 'groups':group_results,
                               'limits':'Overall bootstrap is not a subgroup CI. UNSEEN is small; subgroup estimates descriptive only.'})
    write(output/'examples.json',examples)
    genstats={'definition':'per-user synchronized model.generate seconds; excludes loading/CPU serialization',
              'p50_seconds':float(np.percentile(generation,50)),'p95_seconds':float(np.percentile(generation,95)),
              'mean_seconds':float(generation.mean()),'sum_user_seconds':float(generation.sum()),
              'sum_user_seconds_is_not_end_to_end_wall':True}
    write(output/'metrics.json',{'status':'PASS','protocol_version':cfg['protocol_version'],'run_commit':args.run_commit,'evaluation_commit':args.evaluation_commit,
         'users':population,'metrics':means,'original_metric_deltas':deltas,'sid_statistics':sid_stats,
         'first_target_rank_users':{str(r):first_counts[str(r)] for r in range(1,21)}|{'miss':first_counts['miss']},
         'collision_catalog':{'items':len(index),'unique_sids':len(sizes),'groups':sum(g>1 for g in sizes.values()),
             'affected_items':sum(g for g in sizes.values() if g>1),'maximum_group_size':max(sizes.values()),
             'group_size_distribution':dict(Counter(g for g in sizes.values() if g>1)),'target_collision_users':target_collision},
         'generation_cost':genstats,'user_axis_sha256':user_axis_sha,'assets':assets,'prediction_files':manifests,
         'official_cce_full_user_comparisons':population,'config':identity(args.config),'script':identity(__file__),
         'official_cce':identity(args.official_cce),'per_user_files':[identity(perpath),identity(output/'per_user_metrics.npz')],
         'elapsed_seconds':time.monotonic()-started,'cpu_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         'python':sys.executable,'argv':sys.argv,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
         'gpu_used':False,'scope':scope,'preflight_only':preflight,'experiment_id':EXPERIMENT_ID,
         'row_indices':selected_indices,'evidence_scope':'Frozen predictions only. No tokenizer/model reload; recorded token callback evidence inherited.'})
    return output/'metrics.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','test-dir','stage-a-report','config','official-cce','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--evaluation-commit',required=True)
    p.add_argument('--run-commit',required=True)
    p.add_argument('--scope', choices=('FORMAL', 'PREFLIGHT'), default='FORMAL')
    p.add_argument('--preflight-users', type=Path)
    args=p.parse_args()
    print(evaluate(args),flush=True)


if __name__=='__main__':
    main()
