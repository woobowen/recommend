"""Nonauthor inspection and analytical CPU cases for new result audit scripts."""
import ast
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile

ROOT = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
AUTHORED = Path('/home/addaswsw/recommend-support/grid-shuffle-20261017/results-audit')
OUT = Path(__file__).parent/'analysis-programs-independent-review.json'


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def function(path, name):
    tree = ast.parse(path.read_bytes())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


origins = ROOT/'manifests/grid-formal-20261015/attention-results-review'
same = []
for name, symbols in [('verify_attention_exports.py', ['verify_png_pixels']),
                      ('verify_attention_statistics.py', ['identity', 'close', 'pair_values'])]:
    for symbol in symbols:
        assert ast.dump(function(origins/name, symbol), include_attributes=False) == ast.dump(function(AUTHORED/name, symbol), include_attributes=False)
        same.append({'file': name, 'symbol': symbol, 'AST_equal_to_accepted_Beauty': True})

spec = importlib.util.spec_from_file_location('paired_under_review', AUTHORED/'paired_shuffle_analysis.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

cases = {
    0: [[0,1,2,3,4,5,6,7,8,9], list(range(101,111)), [201,202,203,204,205,200,206,207,208,209]],
    3: [list(range(1,11)), list(range(101,110))+[100], [201,202,200,203,204,205,206,207,208,209]],
    4: [[1,0,2,3,4,5,6,7,8,9], list(range(101,111)), [201,202,203,204,205,200,206,207,209,208]],
}
metrics = {
    0: {'recall@5': 1/3, 'recall@10': 2/3, 'ndcg@5': 1/3, 'ndcg@10': (1+1/math.log2(7))/3},
    3: {'recall@5': 1/3, 'recall@10': 2/3, 'ndcg@5': 1/6, 'ndcg@10': (1/math.log2(11)+1/2)/3},
    4: {'recall@5': 1/3, 'recall@10': 2/3, 'ndcg@5': 1/(3*math.log2(3)), 'ndcg@10': (1/math.log2(3)+1/math.log2(7))/3},
}
keys = ['a'*24, 'b'*24, 'c'*24]
target_ids = [0,100,200]
sid = lambda value: [value,0,0,0]
with tempfile.TemporaryDirectory(prefix='shuffle-pair-analysis-review-') as tmp:
    tmp = Path(tmp)
    targets = tmp/'targets.jsonl'
    targets.write_text(''.join(json.dumps({'sample_key': key, 'target': sid(value)})+'\n' for key,value in zip(keys,target_ids)))
    legal = tmp/'legal.json'
    legal.write_text(json.dumps({'legal_raw_sids': [sid(value) for value in range(301) if value != 209]}))
    plan = {'domain':'toys','route':'B','targets':str(targets),'legal_sids':str(legal),'conditions':{}}
    for n, rows in cases.items():
        predictions = tmp/f'predictions-{n}.jsonl'
        predictions.write_text(''.join(json.dumps({'sample_key':key,'target':sid(target),
            'sids':[sid(value) for value in values], 'scores':list(range(10,0,-1))})+'\n'
            for key,target,values in zip(keys,target_ids,rows)))
        acceptance = {'status':'FORMAL_EVALUATION_ACCEPTANCE_PASS','domain':'toys','route':'B','shuffle_n':n,
            'samples':3,'targets_sha256':identity(targets)['sha256'],'legal_vocabulary':identity(legal),
            'prediction_sha256':identity(predictions)['sha256'],'metrics':metrics[n],'invalid_candidate_count':1,
            'config':{},'sid':{},'checkpoint':{},'test_index':{},'source_commit':'synthetic_not_real','snapshot_sha256':'synthetic_not_real'}
        receipt = tmp/f'accepted-{n}.json';receipt.write_text(json.dumps(acceptance))
        plan['conditions'][str(n)] = {'review':str(receipt),'predictions':str(predictions)}
    comparisons, _ = module.analyze_route(plan,tmp)
    assert len(comparisons)==2
    first, second = comparisons
    assert first['paired_hits']['5'] == {'both_hit':0,'both_miss':1,'lost_hit':1,'gained_hit':1}
    assert first['paired_hits']['10'] == {'both_hit':1,'both_miss':0,'lost_hit':1,'gained_hit':1}
    changes = first['recommendation_list_changes']
    assert changes['top1_changed_users']==1 and changes['ordered_top10_changed_users']==3
    assert changes['top10_set_changed_users']==2 and changes['intersection_size_at10_histogram']=={9:2,10:1}
    assert abs(changes['mean_common_candidates_at10']-28/3)<1e-15
    assert second['recommendation_list_changes']['top1_changed_users']==1
    assert second['recommendation_list_changes']['same_top10_set_users']==3
    assert second['recommendation_list_changes']['ordered_top10_changed_users']==2
    assert second['recommendation_list_changes']['same_set_but_changed_order_users']==2
    assert first['target_rank_transition_counts'][1][0]==1 and first['target_rank_transition_counts'][0][10]==1
    assert first['target_rank_transition_counts'][6][3]==1
    for comparison,n in zip(comparisons,[3,4]):
        for name in metrics[0]:
            summary=comparison['metric_differences_from_accepted_original_float32_metrics'][name]
            expected=metrics[n][name]-metrics[0][name]
            assert abs(summary['absolute_difference']-expected)<1e-15
            assert abs(summary['relative_difference']-expected/metrics[0][name])<1e-15
        assert comparison['invalid_and_duplicate_behavior']['invalid_candidates_Normal']==1
        assert comparison['invalid_and_duplicate_behavior']['invalid_candidates_Shuffle']==1
    analytical = {'users':3,'conditions':[0,3,4], 'all_four_hit_transition_buckets':True,
        'manual_target_rank_transition_cases':True, 'same_global_Recall_with_changed_user_lists':True,
        'same_set_changed_order_cases':True, 'original_invalid_candidates_retained':True,
        'metric_absolute_relative_and_per_user_distribution_agree_with_manual_calculation':True,
        'new_model_GPU_or_evaluator_calls':0, 'synthetic_example_not_formal_result':True}

programs = ['verify_formal_shuffle_result.py','paired_shuffle_analysis.py','verify_attention_exports.py','verify_attention_statistics.py']
for name in programs:
    ast.parse((AUTHORED/name).read_bytes(),filename=name)
report = {'status':'INDEPENDENT_ANALYSIS_PROGRAM_STATIC_AND_ANALYTICAL_CPU_PASS',
    'reviewer_task':'/root/source_audit','reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'programs':[identity(AUTHORED/name) for name in programs], 'accepted_function_identity':same,
    'analytical_paired_cases':analytical,
    'reviewed_new_logic':['Result verifier independently recomputes all per-user Recall/NDCG, refuses mixed relevance score ties, validates exact user-target coverage and native trace/lifecycle identity.',
        'A and B switch fields checked by actual model configuration; absent should_check_prefix follows byte-bound constructor False default, explicit field must remain False.',
        'Batch count follows actual lifecycle batch records; workers0 additionally requires ceil complete coverage.',
        'Attention verifier infers actual history_slots from effective val sequence_length and uses full H+SEP width; inherited parent/head/mask/PNG-cell checks unchanged.',
        'Paired analysis uses exact within-domain/route user-target identity; original ordered candidates unchanged; set operations only describe differences; manual three-user cases cover lost/gained/same hits, rank and list changes.'],
    'limits':['These are audit and analysis scripts authored by /root/results_audit, independently reviewed here. No new formal result or raw attention output is accepted by this static/program review.',
        'The paired-analysis cases use explicitly synthetic saved rows and receipts to exercise calculation paths; they do not certify real experiment identities or empirical outcomes.',
        'Real paired reports and all formal evidence require later exhaustive data-level checks.'],
    'review_program':identity(Path(__file__)),'new_GPU_or_model_calls_by_reviewer':0,'source_or_Git_writes_by_reviewer':0}
assert not OUT.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(identity(OUT)|{'status':report['status']}))
