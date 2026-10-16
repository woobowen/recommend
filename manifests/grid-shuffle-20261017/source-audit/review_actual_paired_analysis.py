"""Exhaustive nonauthor comparison of paired reports with original saved rows."""
import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def identity(path):
    data = Path(path).read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(Path(path).read_bytes())


def equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == {str(key) for key in expected}
        for key, value in expected.items(): equal(actual[str(key)], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected): equal(first, second)
    elif isinstance(expected, float):
        assert math.isfinite(actual) and abs(actual-expected) < 1e-12, (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def metric(rank, k, discount=False):
    return (1/math.log2(rank+1) if discount else 1.0) if rank is not None and rank <= k else 0.0


def metrics(rank):
    return {f'{name}@{k}':metric(rank,k,name=='ndcg') for k in (5,10) for name in ('recall','ndcg')}


def distribution(values):
    return {'n':len(values),'min':min(values),'median':statistics.median(values),
        'mean':math.fsum(values)/len(values),'max':max(values),
        'negative':sum(value<0 for value in values),'zero':sum(value==0 for value in values),
        'positive':sum(value>0 for value in values)}


def prediction_map(path, accepted, expected):
    assert identity(path)['sha256'] == accepted['prediction_sha256']
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert Counter((row['sample_key'],tuple(row['target'])) for row in rows) == expected
    assert len({row['sample_key'] for row in rows}) == len(rows)
    for row in rows:
        assert len(row['sids']) == len(row['scores']) == 10
        assert len(set(map(tuple,row['sids']))) == 10
    return {row['sample_key']:row for row in rows}


def rank(row):
    return next((index+1 for index,value in enumerate(row['sids']) if value==row['target']), None)


def audit(args):
    plan, report = read(args.plan), read(args.report)
    assert report['status'] == 'DESCRIPTIVE_PAIRED_ANALYSIS_COMPLETE'
    assert report['new_model_GPU_or_evaluator_calls'] == 0
    assert report['original_predictions_or_accepted_reviews_modified'] is False
    for record in report['inputs']:
        assert identity(record['path']) == record
    assert identity(report['program']['path']) == report['program']
    checked = []
    for route in plan['routes']:
        domain, family = route['domain'], route['route']
        targets = Counter()
        for line in Path(route['targets']).read_text().splitlines():
            row = json.loads(line); targets[(row['sample_key'],tuple(row['target']))]+=1
        count = sum(targets.values()); assert count and len(targets)==count
        legal = set(map(tuple,read(route['legal_sids'])['legal_raw_sids']))
        accepts = {n:read(route['conditions'][str(n)]['review']) for n in (0,3,4)}
        data = {n:prediction_map(route['conditions'][str(n)]['predictions'], accepts[n],targets) for n in (0,3,4)}
        assert all(accepts[n]['status']=='FORMAL_EVALUATION_ACCEPTANCE_PASS' for n in (0,3,4))
        for name in ('config','sid','checkpoint','test_index','source_commit','snapshot_sha256'):
            assert accepts[0][name]==accepts[3][name]==accepts[4][name]
        for n in (3,4):
            current = next(row for row in report['comparisons'] if (row['domain'],row['route'],row['Normal_to_Shuffle_n'])==(domain,family,n))
            assert current['paired_users']==count and current['target_multiset_preserved']
            paired_identity = current['all_user_paired_observations']; assert identity(paired_identity['path'])==paired_identity
            changes = Counter(); hits = {5:Counter(),10:Counter()}
            ranks = [[0]*11 for _ in range(11)]; categories = Counter()
            common = Counter(); changed_positions = Counter(); invalid_changes = Counter()
            invalid_total = [0,0]; overlaps = []; jaccards=[]; rank_shift=[]; retained=[]
            deltas = {name:[] for name in metrics(None)}
            ordered_cases=[]; lost=[]; gained=[]
            with gzip.open(paired_identity['path'],'rt') as gz:
                for key in sorted(data[0]):
                    before,after=data[0][key],data[n][key]; assert before['target']==after['target']
                    first,second=rank(before),rank(after)
                    ranks[first or 0][second or 0]+=1
                    if first is None and second is None: label='absent_both_top10'
                    elif first is None: label='entered_top10'
                    elif second is None: label='left_top10'
                    else:
                        label='same_rank' if first==second else 'improved_rank' if second<first else 'worsened_rank'
                        retained.append(second-first)
                    categories[label]+=1
                    expected={'sample_key':key,'target_unchanged':True,'normal_target_rank':first,'shuffle_target_rank':second}
                    for k in (5,10):
                        b=first is not None and first<=k; a=second is not None and second<=k
                        hits[k][('both_hit' if a else 'lost_hit') if b else ('gained_hit' if a else 'both_miss')]+=1
                        expected.update({f'normal_hit{k}':b,f'shuffle_hit{k}':a})
                    ids0,ids1=list(map(tuple,before['sids'])),list(map(tuple,after['sids']))
                    overlap=set(ids0)&set(ids1)
                    size=len(overlap); positions=sum(a!=b for a,b in zip(ids0,ids1))
                    same1=ids0[0]==ids1[0]; same_order=ids0==ids1; same_set=set(ids0)==set(ids1)
                    changes['top1_same']+=same1; changes['order_same']+=same_order; changes['set_same']+=same_set
                    common[size]+=1; changed_positions[positions]+=1; overlaps.append(size); jaccards.append(size/(20-size))
                    rank_shift.extend(abs(ids0.index(value)-ids1.index(value)) for value in overlap)
                    bad0=sum(value not in legal for value in ids0);bad1=sum(value not in legal for value in ids1)
                    invalid_total[0]+=bad0;invalid_total[1]+=bad1;invalid_changes[bad1-bad0]+=1
                    difference={name:metrics(second)[name]-metrics(first)[name] for name in deltas}
                    for name,value in difference.items():deltas[name].append(value)
                    expected.update(top1_same=same1,ordered_top10_same=same_order,top10_set_same=same_set,
                        common_candidates_at10=size,changed_positions_at10=positions,normal_invalid_count=bad0,shuffle_invalid_count=bad1,
                        normal_duplicate_count=0,shuffle_duplicate_count=0,independent_metric_deltas=difference)
                    actual=json.loads(gz.readline()); equal(actual,expected)
                    case={**expected,'target':before['target'],'normal_sids':before['sids'],'shuffle_sids':after['sids']}
                    ordered_cases.append(case)
                    if expected['normal_hit10'] and not expected['shuffle_hit10']:lost.append(case)
                    if expected['shuffle_hit10'] and not expected['normal_hit10']:gained.append(case)
                assert not gz.readline()
            equal(current['paired_hits'],{str(k):{name:hits[k][name] for name in ('both_hit','both_miss','lost_hit','gained_hit')} for k in (5,10)})
            equal(current['target_rank_transition_counts'],ranks);equal(current['target_rank_change_counts'],dict(categories))
            equal(current['mean_rank_delta_when_target_present_in_both'],math.fsum(retained)/len(retained) if retained else None)
            lists={'top1_same_users':changes['top1_same'],'top1_changed_users':count-changes['top1_same'],
                'top1_changed_fraction':(count-changes['top1_same'])/count,'ordered_top10_same_users':changes['order_same'],
                'ordered_top10_changed_users':count-changes['order_same'],'same_top10_set_users':changes['set_same'],
                'top10_set_changed_users':count-changes['set_same'],'same_set_but_changed_order_users':changes['set_same']-changes['order_same'],
                'intersection_size_at10_histogram':dict(common),'mean_common_candidates_at10':math.fsum(overlaps)/count,
                'mean_jaccard_at10':math.fsum(jaccards)/count,'changed_original_candidate_positions_histogram':dict(changed_positions),
                'mean_abs_common_candidate_rank_shift':math.fsum(rank_shift)/len(rank_shift) if rank_shift else None}
            equal(current['recommendation_list_changes'],lists)
            equal(current['invalid_and_duplicate_behavior'],{'invalid_candidates_Normal':invalid_total[0],'invalid_candidates_Shuffle':invalid_total[1],
                'per_user_invalid_count_delta_histogram':dict(invalid_changes),'duplicate_candidates_Normal':0,'duplicate_candidates_Shuffle':0,
                'candidate_filtering_deduplication_or_rescoring_applied':False})
            equal(current['independent_per_user_metric_difference_distribution'],{name:distribution(values) for name,values in deltas.items()})
            examples={'greatest_original_list_change_first_three':sorted(ordered_cases,key=lambda row:(-row['changed_positions_at10'],row['common_candidates_at10'],row['sample_key']))[:3],
                'sample_key_first_three_lost_hit10':lost[:3],'sample_key_first_three_gained_hit10':gained[:3]}
            equal(current['prespecified_examples'],examples)
            for name,base in accepts[0]['metrics'].items():
                after=accepts[n]['metrics'][name];difference=after-base
                equal(current['metric_differences_from_accepted_original_float32_metrics'][name],{
                    'Normal':base,'Shuffle':after,'absolute_difference':difference,'absolute_difference_percentage_points':100*difference,
                    'relative_difference':difference/base if base else None,'relative_difference_percent':100*difference/base if base else None})
                assert abs(math.fsum(deltas[name])/count-difference)<=1e-7
            checked.append({'domain':domain,'route':family,'shuffle_n':n,'complete_users':count,
                'every_saved_paired_user_record_checked':True,'every_summary_and_declared_case_checked':True,
                'all_original_rows_checked':True,'paired_file':paired_identity})
    assert len(checked)==len(report['comparisons'])
    receipt={'status':'INDEPENDENT_COMPLETE_PAIRED_ANALYSIS_PASS','reviewer_task':'/root/source_audit',
        'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report':identity(args.report),
        'plan':identity(args.plan),'program':identity(Path(__file__)),'comparisons':checked,
        'method':'Independently construct every paired row from original prediction maps, all rank/hit/list/set/invalid distributions, summaries, global deltas and declared three-case selection; compare all compressed per-user rows exhaustively.',
        'new_GPU_or_model_or_evaluator_calls':0,'scientific_invariance_or_multi_seed_claim':None,
        'original_evidence_writes':0}
    assert not args.output.exists();args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(identity(args.output)|{'status':receipt['status'],'comparisons':len(checked)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    audit(parser.parse_args())
