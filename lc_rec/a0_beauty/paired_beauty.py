"""Beauty adaptation of frozen paired-user statistics; no formal execution implied."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time


def module(path):
    spec=importlib.util.spec_from_file_location('independent_statistics',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
    import numpy as np
    p=argparse.ArgumentParser()
    for n in ['normal-primary','shuffle-primary','normal-test','shuffle-test','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False)
    started=time.monotonic();roots=[Path(a.normal_primary),Path(a.shuffle_primary)]
    rows=[];matrices=[];keys=None;identities={}
    for root in roots:
        with (root/'per_user.csv').open() as f:rows.append(list(csv.DictReader(f)))
        with np.load(root/'per_user_metrics.npz',allow_pickle=False) as z:
            newkeys=z['keys'].tolist();matrix=z['values'].copy()
            assert keys is None or keys==newkeys;keys=newkeys
            assert matrix.shape==(22363,16) and matrix.dtype==np.float64
            assert np.array_equal(z['row_idx'],np.arange(22363))
            assert str(z['user_axis_sha256'])=='7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f'
            assert np.array_equal(matrix,np.asarray([[float(r[k]) for k in keys] for r in rows[-1]]))
            matrices.append(matrix)
        identities[str(root)]={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['per_user.csv','per_user_metrics.npz']}
    fields=['row_idx','user_id','user_axis_sha256','target_item_id','target_sid','target_group_size',
            'original_interaction_length','visible_history_length','history_group',
            'target_training_frequency','popularity_group','protocol_version']
    for i,(left,right) in enumerate(zip(*rows,strict=True)):
        assert left['row_idx']==right['row_idx']==str(i)
        assert all(left[f]==right[f] for f in fields)
    difference=matrices[1]-matrices[0]
    rng=np.random.Generator(np.random.PCG64(20261113));draws=np.empty((5000,16),dtype=np.float64)
    for begin in range(0,5000,8):
        selected=rng.integers(0,22363,size=(min(8,5000-begin),22363),dtype=np.int64)
        draws[begin:begin+len(selected)]=difference[selected].mean(axis=1)
    low,high=np.percentile(draws,[2.5,97.5],axis=0,method='linear')
    np.save(out/'paired_bootstrap_difference_means.npy',draws)
    # Distinct previously reviewed implementation, column-wise rather than joint gather.
    independent_path=Path(__file__).resolve().parents[1]/'letter/review/final_audit_20261113/statistics_verify.py'
    intervals,check_draws=module(independent_path).bootstrap_columns(difference)
    max_draw=float(np.max(np.abs(draws-check_draws)));max_ci=float(np.max(np.abs(np.asarray([low,high])-intervals)))
    assert max_draw<=1e-13 and max_ci<=1e-13
    metrics={}
    for j,key in enumerate(keys):
        normal=math.fsum(matrices[0][:,j])/22363;shuffle=math.fsum(matrices[1][:,j])/22363;delta=shuffle-normal
        metrics[key]={'Normal':normal,'Shuffle':shuffle,'absolute_difference':delta,
                      'relative_difference':delta/normal if normal else None,
                      'difference_CI95_low':float(low[j]),'difference_CI95_high':float(high[j])}
    normal_candidates={}
    for rank in [0,1]:
        with (Path(a.normal_test)/('predictions.rank%d.jsonl'%rank)).open() as stream:
            for line in stream:
                row=json.loads(line);cs=row['candidates'];assert row['row_idx'] not in normal_candidates
                normal_candidates[row['row_idx']]=[cs[j]['sid'] for j in row['sorted_generation_order']]
    assert set(normal_candidates)==set(range(22363))
    changed={'users':22363,'top1_changed':0,'top20_ordered_changed':0,'top20_set_changed':0,'hit_transitions':{}}
    for key in [k for k in keys if k.startswith('hit@')]:
        j=keys.index(key);changed['hit_transitions'][key]={f'{left}->{right}':int(np.sum((matrices[0][:,j]==left)&(matrices[1][:,j]==right))) for left in [0,1] for right in [0,1]}
    seen=set()
    for rank in [0,1]:
        with (Path(a.shuffle_test)/('predictions.rank%d.jsonl'%rank)).open() as stream:
            for line in stream:
                row=json.loads(line);i=row['row_idx'];assert i not in seen;seen.add(i)
                cs=row['candidates'];now=[cs[j]['sid'] for j in row['sorted_generation_order']];old=normal_candidates[i]
                changed['top1_changed']+=old[0]!=now[0]
                changed['top20_ordered_changed']+=old!=now
                changed['top20_set_changed']+=set(old)!=set(now)
    assert seen==set(range(22363))
    for key in ['top1_changed','top20_ordered_changed','top20_set_changed']:changed[key+'_fraction']=changed[key]/22363
    result={'status':'PASS','users':22363,'identity_fields_exact':fields,'metrics':metrics,
      'bootstrap':{'repetitions':5000,'seed':20261113,'rng':'PCG64','replacement':True,'percentile':'95% linear',
       'scope':'User uncertainty conditional on fixed trained model and fixed shuffle realization; not training-seed or permutation variance.'},
      'independent_verification':{'source':str(independent_path),'source_sha256':hashlib.sha256(independent_path.read_bytes()).hexdigest(),
       'maximum_draw_difference':max_draw,'maximum_interval_difference':max_ci,'tolerance':1e-13},
      'inputs':identities,'changes':changed,'elapsed_seconds':time.monotonic()-started}
    (out/'paired_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','users':22363}))


if __name__=='__main__':main()
