"""Read-only independent verification of archived Shuffle V2 metrics and bootstrap."""
import csv
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import fsum, isfinite, log2
from pathlib import Path
import time

import numpy as np

STARTED_UTC = datetime.now(timezone.utc)
WALL_START = time.monotonic()
CPU_START = time.process_time()
ROOT = Path('/data/bwwu/lc-rec/letter-shuffle-v2-20261120')
MODES = ('normal', 'shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4')
CUTOFFS = (5, 10, 20)
KEYS = [f'hit@{k}' for k in (1, *CUTOFFS)] + [f'{m}@{k}' for m in ('ndcg','mrr','ItemHit','ItemNDCG') for k in CUTOFFS]
PAIRS = [(0, 1), (0, 2), (0, 3), (0, 4), (2, 3)]
NAMES = [f'{MODES[r]}-{MODES[l]}' for l,r in PAIRS]
COUNT = 24772
RUN = 'e2f54ebd2d7709821de1071693f91644f5ed9a30'
NORMAL = '4c6a83d9ba42dc2fe737792759e410ef1874da76'
TOL = 1e-13
DISCOUNT = {r: 1/log2(r+1) for r in range(1,21)}


def load(path):
    return json.loads(path.read_bytes())


def h(path):
    return sha256(path.read_bytes()).hexdigest()


def jsonsha(values):
    return sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def score(sids, target, sizes):
    phases = [(r,s) for r,s in enumerate(sids,1) if s == target]
    first = phases[0][0] if phases else len(sids)+1
    # Expand saved ranked SID occurrences into individual item slots. An
    # unknown group consumes one slot. Only the earliest target group earns
    # inclusion credit, uniformly divided among its member items.
    item_target_slots = []
    expanded_position = 1
    target_seen = False
    for sid in sids:
        width = sizes.get(sid, 1)
        if sid == target and not target_seen:
            item_target_slots = list(range(expanded_position,expanded_position+width))
            target_seen = True
        expanded_position += width
    item_tot = sizes[target]
    return ([float(first <= k) for k in (1, *CUTOFFS)]
        + [fsum(DISCOUNT[r] for r,_ in phases if r <= k) for k in CUTOFFS]
        + [1/first if first <= k else 0.0 for k in CUTOFFS]
        + [sum(r <= k for r in item_target_slots)/item_tot for k in CUTOFFS]
        + [fsum(DISCOUNT[r] for r in item_target_slots if r <= k)/item_tot for k in CUTOFFS])


def incoming(manifest, users, interactions, sid_for, sizes, evalrow):
    path = Path(manifest['path'])
    assert path.is_file() and not path.is_symlink()
    before = path.stat()
    size = 0
    samples = 0
    rawsha = sha256()
    with path.open('rb') as f:
        for raw in f:
            rawsha.update(raw)
            size += len(raw)
            row = json.loads(raw)
            idx = row['row_idx']
            assert type(idx) is int and 0 <= idx < len(users) and row['rank'] == manifest['rank'] and row['prompt_id'] == 0
            uid, target_item = users[idx], str(interactions[users[idx]][-1])
            target = sid_for[target_item]
            assert (row['user_id'], row['target_item_id'], row['target']) == (uid, target_item, target)
            candidates = row['candidates']
            assert len(candidates) == 20 and row['original_callback_verified_all_tokens'] is True
            values = []
            for j, candidate in enumerate(candidates):
                raw_sid = candidate['raw_decoded'].rsplit('Response:',1)[-1].strip().replace(' ','')
                assert candidate['generation_order'] == j and raw_sid == candidate['sid'] and isfinite(candidate['sequence_score'])
                values.append((-candidate['sequence_score'],j,raw_sid))
            ordered = sorted(values)
            sids = [c[2] for c in ordered]
            assert [c[1] for c in ordered] == row['sorted_generation_order']
            assert [int(sid == target) for sid in sids] == row['relevance']
            evalrow(idx, sids, target)
            samples += 1
    after = path.stat()
    assert (before.st_ino, before.st_size,before.st_mtime_ns,before.st_ctime_ns) == (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    assert rawsha.hexdigest() == manifest['sha256'] and size == manifest['bytes'] == after.st_size
    return {'path':str(path), 'bytes':size, 'rows':samples, 'sha256':rawsha.hexdigest()}


def main():
    normal_metrics = load(ROOT/'cpu-eval-normal-r0'/'primary'/'metrics.json')
    assets = normal_metrics['assets']
    for asset in assets.values():
        f = Path(asset['path'])
        assert f.stat().st_size == asset['bytes'] and h(f) == asset['sha256']
    index=load(Path(assets['index']['path'])); interactions=load(Path(assets['inter']['path'])); items=load(Path(assets['item']['path']))
    assert len(index) == 9922 and set(items) == set(index)
    users = list(interactions)
    axis = jsonsha(users)
    assert len(users) == COUNT and axis == normal_metrics['user_axis_sha256']
    sid_for={item: ''.join(sid) for item,sid in index.items()}
    sizes=Counter(sid_for.values())
    assert len(sizes)==9897 and sum(x>1 for x in sizes.values())==23
    means = np.zeros((5,16))
    all_metrics = np.zeros((5,COUNT,16))
    checked = {}
    series_dir = ROOT/'cpu-summary-r0'
    series = load(series_dir/'summary.json')
    bootstrap = series['bootstrap']
    assert (bootstrap['rng'],bootstrap['seed'],bootstrap['replacement'],bootstrap['replicates'],bootstrap['percentile_method'],bootstrap['bounds'],bootstrap['multiple_comparison_adjustment']) == ('NumPy PCG64',20261113,True,5000,'linear',[2.5,97.5],False)
    assert series['users']==COUNT and series['user_axis_sha256']==axis
    for c, mode in enumerate(MODES):
        mode_dir = ROOT/f'cpu-eval-{mode}-r0'
        manifest = load(mode_dir/'prediction_manifest.json')
        metrics = load(mode_dir/'primary'/'metrics.json')
        assert manifest['condition']==mode and manifest['gpu_run_commit']==(NORMAL if mode=='normal' else RUN)
        seen = set()
        diag=Counter()
        def check_row(idx, sids, target):
            assert idx not in seen
            seen.add(idx)
            diag.update(invalid=sum(s not in sizes for s in sids), duplicates=20-len(set(sids)), colliding=sum(sizes.get(s,0)>1 for s in sids))
            all_metrics[c,idx]=score(sids,target,sizes)
        files=[incoming(f,users,interactions,sid_for,sizes,check_row) for f in manifest['predictions']]
        assert seen==set(range(COUNT))
        assert [f['rows'] for f in files] == ([COUNT//2]*2 if mode=='normal' else [COUNT])
        reference_diag=metrics['sid_statistics']['20']
        assert all(diag[k]==reference_diag[k] for k in diag)
        with np.load(mode_dir/'primary'/'per_user_metrics.npz', allow_pickle=False) as f:
            assert f['keys'].tolist()==KEYS and f['user_axis_sha256'].item()==axis and np.array_equal(f['row_idx'],np.arange(COUNT))
            discrepancy=float(np.max(np.abs(f['values']-all_metrics[c])))
            assert discrepancy < TOL, (mode, discrepancy)
        means[c] = [fsum(all_metrics[c,:,col])/COUNT for col in range(16)]
        aggregate_error=max(abs(x-reported[k]) for reported in (metrics['metrics'], series['metrics'][mode]) for x,k in zip(means[c],KEYS,strict=True))
        assert aggregate_error < TOL
        assert {str(p): h(p) for p in [mode_dir/'summary.json', mode_dir/'primary'/'per_user.csv', mode_dir/'primary'/'per_user_metrics.npz']}==series['inputs'][mode]
        checked[mode]={'rows':len(seen), 'candidates':20*len(seen), 'raw_shards':files,
                       'sid_count_diagnostics':dict(diag),'max_abs_per_user_error':discrepancy,'max_abs_aggregate_error':aggregate_error}
    deltas=np.stack([all_metrics[r]-all_metrics[l] for l,r in PAIRS])
    with np.load(series_dir/'arrays.npz',allow_pickle=False) as archived:
        assert archived['keys'].tolist()==KEYS and archived['condition'].tolist()==list(MODES) and archived['comparison'].tolist()==NAMES
        assert archived['user_axis_sha256'].item()==axis and np.array_equal(archived['row_idx'],np.arange(COUNT))
        assert np.max(np.abs(archived['per_user_values']-all_metrics)) < TOL
        assert np.max(np.abs(archived['per_user_differences']-deltas)) < TOL
        saved_draws=archived['bootstrap_difference_means']
    expected=deltas.transpose(1,0,2).reshape(COUNT,80).copy()
    rng=np.random.Generator(np.random.PCG64(20261113))
    draw_sha=sha256(); reps=np.empty((5000,80))
    # The production implementation gathers all selected rows. This audit
    # instead converts row draws to multiplicities, and computes a weighted dot
    # product on our independently scored raw-candidate matrix.
    for off in range(0,len(reps),20):
        indices=rng.integers(0,COUNT,size=(min(20,len(reps)-off),COUNT),dtype=np.int64)
        draw_sha.update(indices.astype('<i8',copy=False).tobytes())
        weights=np.asarray([np.bincount(batch,minlength=COUNT) for batch in indices],dtype=np.float64)
        reps[off:off+len(indices)]=weights.dot(expected)/COUNT
    assert draw_sha.hexdigest()==bootstrap['draw_indices_sha256_little_endian']
    computed = reps.reshape(5000,5,16).transpose(1,0,2)
    all_draw_diff=np.max(np.abs(computed-saved_draws),axis=(1,2))
    sorted_reps=np.sort(computed,axis=1)
    comparison = {}
    with (series_dir/'paired.csv').open(newline='') as f:
        csv_paired = list(csv.DictReader(f))
    assert len(csv_paired)==80
    for compare, (left,right) in enumerate(PAIRS):
        label=NAMES[compare]
        assert all_draw_diff[compare] < TOL
        quantiles=[]
        # Explicit linear interpolation on the sorted 5,000 replicate means.
        for pct in (0.025,0.975):
            idx=(len(reps)-1)*pct
            j=int(idx); fraction=idx-j
            quantiles.append(sorted_reps[compare,j]*(1-fraction)+sorted_reps[compare,j+1]*fraction)
        max_ci=0.0
        max_point=0.0
        published = series['comparisons'][label]
        assert len(published)==16
        for col,key in enumerate(KEYS):
            vals=published[col]
            row=csv_paired[compare*16+col]
            assert row['comparison']==label and row['metric']==key and vals['metric']==key
            for field in ('left','right','absolute_difference','difference_CI95_low','difference_CI95_high','relative_decrease_fraction'):
                assert vals[field]==(float(row[field]) if row[field] else None)
            pt=fsum(deltas[compare,:,col])/COUNT
            l,r=means[left,col],means[right,col]
            assert abs(vals['left']-l)<TOL and abs(vals['right']-r)<TOL
            max_point=max(max_point,abs(pt-vals['absolute_difference']))
            if l:
                assert abs(-pt/l-vals['relative_decrease_fraction'])<TOL
            else:
                assert vals['relative_decrease_fraction'] is None
            max_ci=max(max_ci,abs(quantiles[0][col]-vals['difference_CI95_low']),abs(quantiles[1][col]-vals['difference_CI95_high']))
        assert max_ci < TOL and max_point < TOL
        comparison[label]={'draws':5000,'metrics':16,'max_abs_draw_error':float(all_draw_diff[compare]),'max_abs_ci_error':float(max_ci),'max_abs_delta_error':float(max_point)}
    with (series_dir/'metrics.csv').open(newline='') as f:
        csv_metrics=list(csv.DictReader(f)); assert len(csv_metrics)==5
    for mode,row in zip(MODES,csv_metrics,strict=True):
        assert row['condition']==mode and all(float(row[k])==series['metrics'][mode][k] for k in KEYS)
    print(json.dumps({'status':'PASS','started_utc':STARTED_UTC.isoformat(),'ended_utc':datetime.now(timezone.utc).isoformat(),
        'wall_s':time.monotonic()-WALL_START,'process_cpu_s':time.process_time()-CPU_START,'np_version': np.__version__,
        'users':COUNT,'axis_sha256':axis,'draw_index_sha256':draw_sha.hexdigest(),'full_read':checked,'paired':comparison,
        'statistic_artifacts':{n:h(series_dir/n) for n in ('metrics.csv','paired.csv','summary.json','arrays.npz')}},indent=2,allow_nan=False))


if __name__=='__main__': main()
