"""Beauty adaptation of frozen paired-user statistics; no formal execution implied."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
from offline_metrics import ASSETS, KEYS

if not __debug__:
    raise RuntimeError('Pairing identity checks require Python assertions enabled')

def validate_metric_matrix(np, keys, matrix, scope="FORMAL"):
    assert keys==KEYS, 'canonical metric columns and order required'
    assert scope in ("FORMAL", "PREFLIGHT")
    assert matrix.shape==((32 if scope=="PREFLIGHT" else 22363),16) and matrix.dtype==np.float64
    assert np.isfinite(matrix).all()


def validate_provenance(primary, test, source_context=None):
    """Bind evaluated files to completed formal rank reports; no training proof implied."""
    metrics=json.loads((primary/'metrics.json').read_text())
    scope=source_context['scope'] if source_context else 'FORMAL'
    population=32 if scope=='PREFLIGHT' else 22363
    assert scope in ('FORMAL','PREFLIGHT')
    assert metrics.get('scope','FORMAL')==scope
    assert metrics['status']=='PASS' and metrics['users']==population
    run=metrics['run_commit']
    assert isinstance(run,str) and len(run)==40 and all(c in '0123456789abcdef' for c in run)
    assert {k:v['sha256'] for k,v in metrics['assets'].items()}==ASSETS
    per_user_files=metrics['per_user_files']
    assert len(per_user_files)==2
    saved_by_name={Path(v['path']).name:v for v in per_user_files}
    assert set(saved_by_name)=={'per_user.csv','per_user_metrics.npz'}
    for name,saved in saved_by_name.items():
        path=primary/name
        assert saved['bytes']==path.stat().st_size and saved['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    prediction_files=metrics['prediction_files']
    assert len(prediction_files)==2
    identities=[]
    if source_context is not None:
        from production_identity import verify_predictions, verify_training
        supplied=source_context['best']
        fresh=verify_training(supplied['training_root'],supplied['run_commit'],supplied['config_sha256'],source_context['scope'])
        assert fresh==supplied, 'source context does not match current training artifacts'
        condition='normal'
        first=json.loads((test/'rank0.json').read_text())
        if scope=='PREFLIGHT':condition=first['condition']
        elif 'shuffle' in first:condition='s'+str(first['shuffle']['n'])
        verify_predictions(test,run,source_context['best'],condition,source_context['scope'])
    for rank in (0,1):
        report=json.loads((test/f'rank{rank}.json').read_text())
        assert report['status'] in ('PASS','PREFLIGHT_INTERFACE_PASS') and report['rank']==rank and report['preflight_only'] == (scope=='PREFLIGHT') and report.get('scope','FORMAL')==scope
        assert report['RUN_COMMIT']==run and report['completed_users']==(16 if scope=='PREFLIGHT' else (11182 if rank==0 else 11181))
        assert {k:v['sha256'] for k,v in report['data_identity'].items()}==ASSETS
        pred=test/f'predictions.rank{rank}.jsonl'; saved=prediction_files[rank]
        assert saved['bytes']==pred.stat().st_size and saved['sha256']==hashlib.sha256(pred.read_bytes()).hexdigest()
        identities.append(report['adapter_files'])
    assert identities[0]==identities[1]
    assert identities[0]['adapter_model.bin']['sha256']!='e56abd90bd3cc55d3786df04971a670bc12c54578110d255b5e7f3f4ec76af07'
    return {'run_commit':run,'adapter_files':identities[0]}


def module(path):
    spec=importlib.util.spec_from_file_location('independent_statistics',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def paired_bootstrap(np, difference, scope="FORMAL"):
    """Frozen paired-user calculation shared by the production entry and CPU tests."""
    population=32 if scope=="PREFLIGHT" else 22363
    if scope not in ("FORMAL","PREFLIGHT") or difference.shape != (population, 16) or difference.dtype != np.float64:
        raise ValueError('expected full Beauty user-by-metric float64 matrix')
    if not np.isfinite(difference).all():
        raise ValueError('nonfinite paired differences')
    rng=np.random.Generator(np.random.PCG64(20261113))
    draws=np.empty((5000,16),dtype=np.float64)
    for begin in range(0,5000,8):
        selected=rng.integers(0,population,size=(min(8,5000-begin),population),dtype=np.int64)
        draws[begin:begin+len(selected)]=difference[selected].mean(axis=1)
    low,high=np.percentile(draws,[2.5,97.5],axis=0,method='linear')
    return draws,low,high


def process(a, source_context=None):
    if source_context is None:
        raise ValueError('positive training source context required')
    import numpy as np
    scope=source_context['scope']
    assert scope in ('FORMAL','PREFLIGHT')
    population=32 if scope=='PREFLIGHT' else 22363
    out=Path(a.output);out.mkdir(exist_ok=False)
    started=time.monotonic();roots=[Path(a.normal_primary),Path(a.shuffle_primary)]
    provenance=[validate_provenance(roots[0],Path(a.normal_test),source_context),
                validate_provenance(roots[1],Path(a.shuffle_test),source_context)]
    assert provenance[0]['adapter_files']==provenance[1]['adapter_files'], 'Adapter identity mismatch'
    primary_metadata=[json.loads((root/'metrics.json').read_text()) for root in roots]
    expected_rows=primary_metadata[0].get('row_indices',list(range(22363)))
    assert len(expected_rows)==population and len(set(expected_rows))==population
    assert expected_rows==primary_metadata[1].get('row_indices',list(range(22363)))
    assert scope=='PREFLIGHT' or expected_rows==list(range(22363))
    rows=[];matrices=[];keys=None;identities={}
    for root in roots:
        with (root/'per_user.csv').open() as f:rows.append(list(csv.DictReader(f)))
        with np.load(root/'per_user_metrics.npz',allow_pickle=False) as z:
            newkeys=z['keys'].tolist();matrix=z['values'].copy()
            validate_metric_matrix(np,newkeys,matrix,scope)
            assert keys is None or keys==newkeys;keys=newkeys
            assert np.array_equal(z['row_idx'],np.asarray(expected_rows))
            assert str(z['user_axis_sha256'])=='7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f'
            assert np.array_equal(matrix,np.asarray([[float(r[k]) for k in keys] for r in rows[-1]]))
            matrices.append(matrix)
        identities[str(root)]={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['per_user.csv','per_user_metrics.npz']}
    fields=['row_idx','user_id','user_axis_sha256','target_item_id','target_sid','target_group_size',
            'original_interaction_length','visible_history_length','history_group',
            'target_training_frequency','popularity_group','protocol_version']
    for i,(left,right) in enumerate(zip(*rows,strict=True)):
        assert left['row_idx']==right['row_idx']==str(expected_rows[i])
        assert all(left[f]==right[f] for f in fields)
    difference=matrices[1]-matrices[0]
    draws,low,high=paired_bootstrap(np,difference,scope)
    np.save(out/'paired_bootstrap_difference_means.npy',draws)
    # Distinct previously reviewed implementation, column-wise rather than joint gather.
    independent_path=Path(__file__).resolve().parents[1]/'letter/review/final_audit_20261113/statistics_verify.py'
    intervals,check_draws=module(independent_path).bootstrap_columns(difference)
    max_draw=float(np.max(np.abs(draws-check_draws)));max_ci=float(np.max(np.abs(np.asarray([low,high])-intervals)))
    assert max_draw<=1e-13 and max_ci<=1e-13
    metrics={}
    for j,key in enumerate(keys):
        normal=math.fsum(matrices[0][:,j])/population;shuffle=math.fsum(matrices[1][:,j])/population;delta=shuffle-normal
        metrics[key]={'Normal':normal,'Shuffle':shuffle,'absolute_difference':delta,
                      'relative_difference':delta/normal if normal else None,
                      'difference_CI95_low':float(low[j]),'difference_CI95_high':float(high[j])}
    normal_candidates={}
    for rank in [0,1]:
        with (Path(a.normal_test)/('predictions.rank%d.jsonl'%rank)).open() as stream:
            for line in stream:
                row=json.loads(line);cs=row['candidates'];assert row['row_idx'] not in normal_candidates
                normal_candidates[row['row_idx']]=[cs[j]['sid'] for j in row['sorted_generation_order']]
    assert set(normal_candidates)==set(expected_rows)
    changed={'users':population,'top1_changed':0,'top20_ordered_changed':0,'top20_set_changed':0,'hit_transitions':{}}
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
    assert seen==set(expected_rows)
    for key in ['top1_changed','top20_ordered_changed','top20_set_changed']:changed[key+'_fraction']=changed[key]/population
    result={'scope':source_context['scope'] if source_context else 'LEGACY_PROVENANCE_ONLY','status':'PASS','users':population,'preflight_only':scope=='PREFLIGHT','row_indices':expected_rows,'identity_fields_exact':fields,'metrics':metrics,
      'bootstrap':{'repetitions':5000,'seed':20261113,'rng':'PCG64','replacement':True,'percentile':'95% linear',
       'scope':'User uncertainty conditional on fixed trained model and fixed shuffle realization; not training-seed or permutation variance.'},
      'independent_verification':{'source':str(independent_path),'source_sha256':hashlib.sha256(independent_path.read_bytes()).hexdigest(),
       'maximum_draw_difference':max_draw,'maximum_interval_difference':max_ci,'tolerance':1e-13},
      'inputs':identities,'prediction_provenance':provenance,'changes':changed,'elapsed_seconds':time.monotonic()-started}
    (out/'paired_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','users':population}))


def main():
    p=argparse.ArgumentParser()
    for n in ['normal-primary','shuffle-primary','normal-test','shuffle-test','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--training-root',required=True)
    p.add_argument('--training-run-commit',required=True)
    p.add_argument('--training-config-sha256',required=True)
    p.add_argument('--scope', choices=('FORMAL','PREFLIGHT'), default='FORMAL')
    a=p.parse_args()
    from production_identity import verify_training
    best=verify_training(a.training_root,a.training_run_commit,a.training_config_sha256,a.scope)
    process(a,source_context={'scope':a.scope,'best':best})

if __name__=='__main__':main()
