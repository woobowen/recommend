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

RUN = None
N = 24772
PRED = None
ASSETS = {'index': '33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd',
          'inter': '0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743',
          'item': '1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2'}
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


def evaluate(args):
    import numpy as np
    started = time.monotonic()
    output = args.output
    require(not output.exists(), 'Preserve existing result directory')
    require(args.test_dir.resolve() not in output.resolve().parents and output.resolve() != args.test_dir.resolve(), 'Original results are read only')
    output.mkdir(parents=True)
    cfg = load(args.config)
    require(cfg['protocol_version'] == 'letter_instruments_beam20_v1' and cfg['original_run_commit'] == RUN, 'Protocol identity')
    require(sha(args.official_cce) == cfg['official_cce_sha256'], 'Official CCE identity')
    official = module(args.official_cce)
    write(output/'unit_tests.json', selftest(official))
    stage = load(args.stage_a_report)
    require(stage['status'] == 'PASS' and stage['run_commit'] == RUN and stage['actual_test_users'] == N, 'Original full audit gate')
    data, assets = {}, {}
    for kind, expected in ASSETS.items():
        path = args.data_root/'Instruments'/f'Instruments.{kind}.json'
        assets[kind] = identity(path)
        require(assets[kind]['sha256'] == expected, 'Asset identity')
        data[kind] = load(path)
    index, inter = data['index'], data['inter']
    require(len(index) == 9922 and len(inter) == N and set(index) == set(data['item']), 'Frozen axes')
    sid_for = {item: ''.join(tokens) for item, tokens in index.items()}
    sizes = Counter(sid_for.values())
    users = list(inter)
    user_axis_sha = hashlib.sha256(json.dumps(users, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    freq = Counter({item: 0 for item in index})
    for seq in inter.values():
        freq.update(str(item) for item in seq[:-2])
    positive = sorted(f for f in freq.values() if f > 0)
    q1, q2 = positive[math.ceil(len(positive)/3)-1], positive[math.ceil(2*len(positive)/3)-1]
    def popularity(f):
        return 'UNSEEN' if f == 0 else 'LONG_TAIL' if f <= q1 else 'MID' if f <= q2 else 'POPULAR'
    matrix = np.zeros((N, len(KEYS)), dtype=np.float64)
    seen, peruser = set(), [None]*N
    first_counts, target_collision = Counter(), 0
    diag = {k: Counter() for k in (5, 10, 20)}
    coverage = {k: set() for k in diag}
    examples = {'invalid': [], 'target_collision': [], 'prefix_collision_changes_cce': []}
    manifests, generation = [], np.zeros(N, dtype=np.float64)
    for rank in (0, 1):
        path = args.test_dir/f'predictions.rank{rank}.jsonl'
        before = path.stat()
        inherited = stage['prediction_files'][rank]
        require((before.st_size, inherited['sha256']) == PRED[rank] and inherited['bytes'] == before.st_size, 'Frozen prediction identity')
        rank_report = load(args.test_dir/f'rank{rank}.json')
        require(rank_report.get('run_commit', rank_report.get('RUN_COMMIT')) == RUN, 'Rank RUN identity')
        assignments = []
        with path.open('rb') as stream:
            for raw in stream:
                row = json.loads(raw)
                i = row['row_idx']
                require(type(i) is int and 0 <= i < N and i not in seen and row['rank'] == rank and row['prompt_id'] == 0, 'User coverage/rank/prompt')
                seq = inter[users[i]]
                item, target = str(seq[-1]), sid_for[str(seq[-1])]
                require(row['user_id'] == users[i] and row['target_item_id'] == item and row['target'] == target, 'User/target pairing')
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
                matrix[i] = [values[k] for k in KEYS]
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
                generation[i] = elapsed
                hlen = min(20, len(seq)-1)
                hgroup = '0-5' if hlen <= 5 else '6-10' if hlen <= 10 else '11-20'
                peruser[i] = {'row_idx': i, 'user_id': users[i], 'user_axis_sha256': user_axis_sha,
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
        require(assignment_sha == rank_report['assignment_sha256'] and len(assignments)==N//2, 'Original rank assignment identity')
        manifests.append({**inherited, 'mtime_ns': after.st_mtime_ns, 'ctime_ns': after.st_ctime_ns, 'assignment_sha256': assignment_sha, 'hash_reused_from_stage_a': True})
    require(len(seen) == N and sum(first_counts.values()) == N, 'Complete population')
    means = {key: math.fsum(matrix[:,j])/N for j,key in enumerate(KEYS)}
    deltas = {key: means[key]-stage['metrics'][key] for key in stage['metrics']}
    require(all(abs(d)<1e-13 for d in deltas.values()), 'Original metric regression')
    sid_stats = {}
    for k, counts in diag.items():
        sid_stats[str(k)] = {**counts, 'valid_rate':counts['valid']/(N*k), 'invalid_rate':counts['invalid']/(N*k),
            'duplicate_rate':counts['duplicates']/(N*k), 'collision_candidate_rate':counts['colliding']/(N*k),
            'users_invalid_rate':counts['users_invalid']/N, 'users_duplicate_rate':counts['users_duplicate']/N,
            'users_colliding_rate':counts['users_colliding']/N, 'unique_valid_sids':len(coverage[k]),
            'coverage_denominator':len(sizes), 'unique_valid_sid_coverage':len(coverage[k])/len(sizes)}
    require(diag[20]['invalid']==stage['anomalies']['invalid_complete_sid_candidates'] and diag[20]['colliding']==stage['anomalies']['colliding_sid_candidate_occurrences'], 'Historical anomalies')
    for k in (5,10,20):
        require(sum(first_counts[str(r)] for r in range(1,k+1)) == round(means[f'hit@{k}']*N), 'Hit histogram')
    perpath=output/'per_user.csv'
    with perpath.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(peruser[0]));writer.writeheader();writer.writerows(peruser)
    np.savez_compressed(output/'per_user_metrics.npz', values=matrix, keys=np.asarray(KEYS), row_idx=np.arange(N), user_axis_sha256=np.asarray(user_axis_sha))
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
        selected=rng.integers(0,N,size=(min(8,5000-begin),N),dtype=np.int64)
        draws[begin:begin+len(selected)]=matrix[selected].mean(axis=1)
    intervals=np.percentile(draws,[2.5,97.5],axis=0,method='linear')
    np.save(output/'bootstrap_draw_means.npy',draws)
    bootstrap={**cfg['bootstrap'],'users':N,'numpy_version':np.__version__,'elapsed_seconds':time.monotonic()-bootstrap_started,
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
    write(output/'metrics.json',{'status':'PASS','protocol_version':cfg['protocol_version'],'run_commit':RUN,'evaluation_commit':args.evaluation_commit,
         'users':N,'metrics':means,'original_metric_deltas':deltas,'sid_statistics':sid_stats,
         'first_target_rank_users':{str(r):first_counts[str(r)] for r in range(1,21)}|{'miss':first_counts['miss']},
         'collision_catalog':{'items':len(index),'unique_sids':len(sizes),'groups':sum(g>1 for g in sizes.values()),
             'affected_items':sum(g for g in sizes.values() if g>1),'maximum_group_size':max(sizes.values()),
             'group_size_distribution':dict(Counter(g for g in sizes.values() if g>1)),'target_collision_users':target_collision},
         'generation_cost':genstats,'user_axis_sha256':user_axis_sha,'assets':assets,'prediction_files':manifests,
         'official_cce_full_user_comparisons':N,'config':identity(args.config),'script':identity(__file__),
         'official_cce':identity(args.official_cce),'per_user_files':[identity(perpath),identity(output/'per_user_metrics.npz')],
         'elapsed_seconds':time.monotonic()-started,'cpu_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         'python':sys.executable,'argv':sys.argv,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
         'gpu_used':False,'scope':'Frozen predictions only. No tokenizer/model reload; recorded token callback evidence inherited.'})
    return output/'metrics.json'


def main():
    global RUN, PRED
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','test-dir','stage-a-report','config','official-cce','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--evaluation-commit',required=True)
    p.add_argument('--manifest',type=Path,required=True)
    args=p.parse_args()
    manifest=load(args.manifest)
    require(manifest['schema']=='a1_frozen_prediction_manifest_v1' and manifest['status']=='FROZEN_INPUTS', 'A1 manifest schema')
    RUN=manifest['gpu_run_commit']
    require(len(RUN)==40 and all(c in '0123456789abcdef' for c in RUN), 'A1 GPU RUN')
    require(manifest['test_dir']==str(args.test_dir.resolve()), 'Manifest test directory')
    PRED=[(r['bytes'],r['sha256']) for r in manifest['predictions']]
    require(len(PRED)==2, 'Two frozen rank shards')
    for rank,row in enumerate(manifest['predictions']):
        require(row['rank']==rank and row['path']==str((args.test_dir/f'predictions.rank{rank}.jsonl').resolve()), 'A1 shard manifest order')
        require(sha(row['path'])==row['sha256'], 'A1 frozen prediction full SHA')
    print(evaluate(args),flush=True)


if __name__=='__main__':
    main()
