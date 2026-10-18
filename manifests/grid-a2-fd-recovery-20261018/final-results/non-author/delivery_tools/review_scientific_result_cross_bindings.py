"""Cross-review completed scientific receipts; final publication/ZIP remain separate."""
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


refs_checked = {}


def check_ref(ref):
    path = Path(ref['path'])
    if path.is_absolute() and str(path).startswith('/home/addaswsw/'):
        assert path.is_file(), str(path)
        assert sha(path) == ref['sha256'], str(path)
        if 'bytes' in ref:
            assert path.stat().st_size == ref['bytes']
        refs_checked[str(path)] = identity(path)


def walk_refs(node):
    if isinstance(node, dict):
        if isinstance(node.get('path'), str) and isinstance(node.get('sha256'), str):
            check_ref(node)
        for value in node.values():
            walk_refs(value)
    elif isinstance(node, list):
        for value in node:
            walk_refs(value)


scene = read(BASE / 'final-scene/scene-20261018T043428336611Z.json')
terminal_cross = read(OUT / 'terminal-scene-cross-review.json')
assert terminal_cross['status'] == 'TERMINAL_SCENE_RESOURCES_SOURCE_AND_TRAINING_CROSS_REVIEW_PASS'
reports = {
    'fd_engineering': BASE / 'reviews/fd/source-repair-admission.json',
    'quantization': BASE / 'reviews/fd/quantization-result-review.json',
    'A2_training': BASE / 'reviews/a2_training/final_training_review.json',
    'Toys_both_candidates': BASE / 'reviews/toys/toys-independent-review.json',
    'Toys_lifecycle': BASE / 'reviews/toys/toys-supplemental-review.json',
    'Sports_A1': BASE / 'reviews/sports_a1/audit_result.json',
    'A2_outputs': BASE / 'reviews/a2_outputs/A2_OUTPUTS_NONAUTHOR_REVIEW.json',
    'A2_output_bindings': BASE / 'reviews/a2_outputs/a2-final-output-binding-checks.json',
    'A2_PNG_cells': BASE / 'reviews/a2_outputs/a2-png-pixel-consistency.json',
    'A2_output_delivery_validation': BASE / 'reviews/a2_outputs/A2_REVIEW_DELIVERY_VALIDATION.json',
}
docs = {key: read(path) for key, path in reports.items()}
assert docs['quantization']['status'] == 'QUANTIZATION_AND_SID_INDEPENDENT_REVIEW_PASS'
assert docs['A2_training']['status'] == 'A2_TRAINING_NONAUTHOR_REVIEW_PASS'
assert docs['Toys_both_candidates']['status'] == 'INDEPENDENT_TOYS_SAVED_RESULT_REVIEW_PASS'
assert docs['Toys_lifecycle']['status'] == 'SUPPLEMENTAL_LIFECYCLE_SOURCE_REVIEW_PASS'
assert docs['Sports_A1']['status'] == 'SPORTS_A1_NONAUTHOR_REVIEW_PASS'
a2 = docs['A2_outputs']
assert a2['status'] == 'A2_EVALUATION_ATTENTION_STATISTICS_NONAUTHOR_REVIEW_PASS'
assert a2['reviewer'] == '/root/toys_result_audit'
assert docs['A2_training']['reviewer'] == docs['Sports_A1']['reviewer'] == '/root/sports_a1_result_audit'
assert docs['Toys_both_candidates']['reviewer'] == '/root/toys_result_audit'
assert a2['scientific_capture'] == scene['source_commit'] + '/' + scene['source_snapshot_sha256']
assert a2['best_checkpoint'] == docs['A2_training']['best']['checkpoint']
assert a2['new_SID_sha256'] == docs['A2_training']['sid']['sha256'] == '06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0'
assert a2['cohort_split'] == 'evaluation' and a2['formal_metrics_split'] == 'testing'
assert a2['overall_final_internal_review_pass'] is False
for key in ['A2_outputs', 'A2_output_bindings', 'A2_PNG_cells', 'A2_output_delivery_validation']:
    walk_refs(docs[key])
primary = read(BASE / 'primary-result-acceptance/all-four.json')
primary_a2 = next(route for route in primary['routes'] if route['route'] == 'Sports-A2')
primary_attention = read(BASE / 'primary-result-acceptance/a2-attention.json')
condition_names = {'0': 'Normal', '3': 'Shuffle3', '4': 'Shuffle4'}
for suffix, label in condition_names.items():
    evaluation = a2['evaluations'][suffix]
    assert evaluation['metrics_original'] == primary_a2['conditions'][label]['original_metrics']
    assert evaluation['samples'] == 35598 and evaluation['coverage_exact']
    assert evaluation['prediction']['sha256'] == primary_a2['conditions'][label]['raw_prediction_sha256']
    assert evaluation['identity']['strict_load']['all_parameter_keys_shapes_values_equal']
    assert not evaluation['identity']['strict_load']['missing_keys'] and not evaluation['identity']['strict_load']['unexpected_keys']
    assert evaluation['identity']['checkpoint'] == a2['best_checkpoint']
    assert evaluation['identity']['sid']['sha256'] == a2['new_SID_sha256']
    assert evaluation['intervention_calls'] == (0 if suffix == '0' else 8900)
    for key in ['invalid_candidates', 'duplicate_candidates']:
        assert evaluation[key] == primary_a2['conditions'][label][key]
    attention = a2['attention'][suffix]
    assert attention['users'] == 32 and attention['batches'] == len(attention['NPZ_files']) == 4
    assert attention['PNG_count'] == 32 and attention['raw_heads'] == 6 and attention['autoregressive_queries'] == 4
    assert attention['both_paths_reconstructed_exactly'] and attention['maximum_masked_mass'] == 0
    rows = attention['all_query_pair_rows']
    assert len(rows) == len({(row['sample_key'], row['path'], tuple(row['query_pair'])) for row in rows}) == 384
    assert len({row['sample_key'] for row in rows}) == 32
    assert set(row['path'] for row in rows) == {'final_top1_cross', 'local_first_cross'}
    assert set(tuple(row['query_pair']) for row in rows) == {(a,b) for a in range(1,5) for b in range(a+1,5)}
    for path, values in attention['statistics'].items():
        subset = [row for row in rows if row['path'] == path]
        assert len(subset) == 192
        for metric, summary in values.items():
            numbers = [row[metric] for row in subset]
            assert all(math.isfinite(number) for number in numbers)
            expected = dict(n=192, min=min(numbers), max=max(numbers), mean=statistics.mean(numbers), median=statistics.median(numbers))
            for key, value in expected.items():
                assert abs(summary[key] - value) < 3e-15
                assert abs(summary[key] - primary_attention['conditions'][label]['statistics_summary'][path][metric][key]) < 3e-15
    assert attention['fixed32']['sha256'] == primary_attention['cohort_sha256'] == '0a2bbfbd62ab5bbd82b3032330e1f0f411af015d674fccc6b5f23c0ddacafe01'
    if suffix != '0':
        pair = a2['paired_predictions'][suffix]
        original = primary_a2['paired'][label]
        for key in ['samples', 'top1_changed', 'top10_order_changed', 'top10_set_changed']:
            assert pair[key] == original[key]
        for k, transitions in pair['hit_migration'].items():
            for transition, n in transitions.items():
                assert original['hit' + k + '_' + transition.replace('->','to')] == n

pixel = docs['A2_PNG_cells']
assert pixel['status'] == 'A2_ALL_PNG_RAW_MATRIX_CELL_CONSISTENCY_PASS'
assert pixel['total_images'] == 96 and pixel['total_heatmap_cells'] == 115200 and pixel['total_mask_cells'] == 14400
assert all(row['maximum_RGB_difference_0_to_255'] == 0 for row in pixel['images'])
binding = docs['A2_output_bindings']
assert binding['status'] == 'A2_OUTPUT_SOURCE_BINDING_AND_SELECTOR_PASS'
assert len(binding['accepted_requests']) == 7 and binding['status_artifact_hashes_independently_cross_checked'] == 226
assert binding['statistics_raw_source_hashes_exact'] == 24 and len(binding['least_three_selectors']) == 6
assert all(row['least_three_exact_under_declared_rule'] for row in binding['least_three_selectors'])
validation = docs['A2_output_delivery_validation']
assert validation['status'] == 'A2_OUTPUT_REVIEW_FINAL_PRODUCTS_VALIDATED'
assert validation['source_condition_rows_exact'] == 24 and validation['old_non_A2_conditions_unchanged'] == 21
assert validation['original_pending_index_unchanged'] and validation['whole_task_FINAL_INTERNAL_REVIEW_not_signed_here']

# Bind every exported A-route NPZ/PNG to the actual terminal runner artifact map.
raw_attention_counts = {}
for route in primary['routes']:
    raw_root = BASE / ('reviews/toys/raw' if route['route'].startswith('Toys') else 'reviews/sports_a1/raw' if route['route'] == 'Sports-A1' else 'reviews/a2_outputs/terminal_outputs/raw/jobs')
    counts = collections.Counter()
    for condition in route['conditions'].values():
        name = condition['job_id'].replace('-eval-', '-attention-')
        artifacts = scene['jobs'][name]['raw_status']['artifacts']
        selected = [row for row in artifacts if row['path'].endswith(('.npz', '.png'))]
        assert sum(row['path'].endswith('.npz') for row in selected) == 4
        assert sum(row['path'].endswith('.png') for row in selected) == 32
        for row in selected:
            path = raw_root / name / row['path']
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
            counts[path.suffix] += 1
    raw_attention_counts[route['route']] = dict(NPZ=counts['.npz'], PNG=counts['.png'])
assert sum(row['NPZ'] for row in raw_attention_counts.values()) == 48
assert sum(row['PNG'] for row in raw_attention_counts.values()) == 384

cross_root = BASE / 'reviews/cross_domain'
old_index = read(cross_root / 'cross-domain-evidence-index.json')
index_path = cross_root / 'cross-domain-evidence-index-with-a2.json'
index = read(index_path)
assert index['status'] == 'CROSS_DOMAIN_ORIGINAL_OUTPUT_EVIDENCE_COMPLETE'
assert index['available_condition_count'] == 24 and index['pending_condition_count'] == 0
assert index['overall_final_review_pass'] is False
assert index['old_evidence_fixed_commit'] == '9ab3d0ea5d32db00e278f74c599a1943885453db'
for old in old_index['routes']:
    if old['route_id'] != 'Sports-A2':
        assert old == next(row for row in index['routes'] if row['route_id'] == old['route_id'])
metrics = read(cross_root / 'cross-domain-original-metrics-with-a2.json')
assert len(metrics['rows']) == 24
metric_map = {(row['route'], row['condition']): row for row in metrics['rows']}
for route in index['routes']:
    assert len(route['conditions']) == 3
    for condition in route['conditions']:
        row = metric_map[(route['route_id'], condition['condition'])]
        assert row['samples'] == condition['samples'] and row['job_id'] == condition['job_id']
        assert all(row[key] == condition['metrics_original'][key] for key in ['recall@5', 'ndcg@5', 'recall@10', 'ndcg@10'])
for route in primary['routes']:
    for label, condition in route['conditions'].items():
        row = metric_map[(route['route'], label)]
        assert all(row[key] == value for key,value in condition['original_metrics'].items())
with (cross_root / 'cross-domain-original-metrics-with-a2.csv').open(newline='') as stream:
    for row in csv.DictReader(stream):
        original_row = metric_map[(row['route'], row['condition'])]
        assert all(float(row[key]) == original_row[key] for key in ['recall@5', 'ndcg@5', 'recall@10', 'ndcg@10'])

report = dict(
    reviewer='/root/fd_repair_reviewer', reviewed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    status='ALL_AUTHORIZED_SCIENTIFIC_RESULTS_NONAUTHOR_CROSS_REVIEW_PASS',
    all_authorized_calculations_and_scientific_result_reviews_complete=True,
    final_internal_review_pass=False, final_GitHub_and_ZIP_delivery_approved=False,
    scope='Cross-bind actual completed A2 quantization/training/testing/attention/statistics and original Toys two candidates/SportsA1 reviews to source, checkpoint, SID, original predictions, terminal scene and final cross-domain index. Dedicated nonauthors own their documented direct CPU raw-result checks; this is an additional source/result cross-review, not another model run.',
    source_commit=scene['source_commit'], source_snapshot_sha256=scene['source_snapshot_sha256'],
    executed_review_roles=[dict(reviewer='/root/fd_repair_reviewer',scope='FD implementation/equivalence and real quantizer/SID; terminal/resource/source and result cross-bindings'),
                           dict(reviewer='/root/sports_a1_result_audit',scope='SportsA1 raw results and direct CPU checkpoint; new A2 full training and strict best/final checkpoint CPU loads'),
                           dict(reviewer='/root/toys_result_audit',scope='Both ToysA raw result/attention audits; new A2 original prediction, vector shuffle, raw NPZ/beam/statistics/PNG audits; fixed-source index')],
    scientific_review_receipts={key:identity(path) for key,path in reports.items()},
    actual_terminal_and_resource_cross_review=identity(OUT / 'terminal-scene-cross-review.json'),
    A2_result_summary=dict(testing_targets_each=35598, raw_prediction_rows=106794, best_step=6900, final_step=7900,
                           SID_sha256=a2['new_SID_sha256'], best_checkpoint=a2['best_checkpoint'], fixed32_evaluation_sha256=primary_attention['cohort_sha256'],
                           NPZ=12, PNG=96, query_pair_statistics_rows=1152, two_paths=['final_top1_cross','local_first_cross'],
                           independent_statistic_max_error=max(row['maximum_independent_statistic_error'] for row in a2['attention'].values()),
                           all_PNG_heatmap_cells_verified_by_nonauthor=115200, all_PNG_mask_cells_verified_by_nonauthor=14400),
    all_A_attention_exports_hash_bound_to_actual_runner_status=raw_attention_counts,
    output_local_identity_references_checked=len(refs_checked),
    final_cross_domain_index=identity(index_path), final_cross_domain_original_metrics=identity(cross_root / 'cross-domain-original-metrics-with-a2.json'),
    cross_domain_scope=dict(complete_routes=8, complete_conditions=24, prior_non_A2_conditions_exactly_preserved=21,
                            inherited_old_Beauty_AB_ToysB_SportsB_conditions=12, current_ToysA5300_ToysA5000_SportsA1_SportsA2_conditions=12,
                            original_pending_index_kept=True, no_old_research_reruns=True),
    limits=['Toys upstream historical producer remains non-unique; both5300/5000 candidates stay separately reported and neither is selected by testing metrics.',
            'A1/A2 are new Sports method reproductions with independently trained generators, not recovered historical author Sports weights or strict one-factor SID controls.',
            'One seed42 and preselected evaluation fixed32 give descriptive observations; aggregate similarity cannot prove permutation invariance, cross-seed robustness or decoder target-prefix nonuse.',
            'Formal recommendation metrics use testing; attention uses evaluation/validation. User-key equality across splits does not equate histories/targets.',
            'Strict checkpoint values are supported by original strict-load records and named direct CPU checkpoint reviewer; this cross-review adds no model load.',
            'Large model/embedding/data assets remain on the server. Their final restore manifest and published commit/readback/ZIP require separate delivery review.'],
    pending_delivery_requirements=['Review final report and published fixed wbw commit/readback plus actual execution-versus-delivery identity mapping.',
                                   'Review final complete staging/restore manifest and actual final single ZIP path/hash/CRC/member/path/secret verification.'],
    new_remote_operations=0, GPU_calls=0, model_loads=0, dependencies_installed=[], GPT_UPLOAD_REQUIRED='YES')
target = OUT / 'scientific-results-cross-review.json'
assert not target.exists()
target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(path=str(target),sha256=sha(target),status=report['status'],A_route_NPZ=48,A_route_PNG=384,
                     complete_cross_domain_conditions=24,local_identity_references=len(refs_checked),final_internal_pass=False)))
