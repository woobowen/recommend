"""Independent prepublication / fixed-publication content and evidence review.

Reads original accepted exports, every prediction and paired row, all decoder
query companions, final activity evidence and existing non-author receipts.
No model, generator, external write, Git mutation or GPU operation.
"""
import argparse
import ast
from collections import Counter
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import numpy as np

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
JOBS = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
PREFIX = 'manifests/grid-shuffle-20261017/'
ANCHOR = 'd0bcacfee4bab4276861573f7746405fb63c54f2'
METRICS = ('recall@5', 'ndcg@5', 'recall@10', 'ndcg@10')


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(path.read_bytes())


def verify_identity(row):
    assert identity(Path(row['path'])) == row, ('referenced bytes changed', row['path'])


def metrics_at(rank):
    return {f'{name}@{k}': (0.0 if rank is None or rank > k else
                           1.0 if name == 'recall' else 1.0 / math.log2(rank + 1))
            for k in (5, 10) for name in ('recall', 'ndcg')}


def compare_numbers(actual, expected):
    assert abs(actual - expected) <= 2e-12, (actual, expected)


def git_bytes(commit, name):
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=REPO)


def review_predictions(condition):
    domain, n = condition['domain'], condition['shuffle_n']
    root = JOBS / condition['evaluation_job_id'] / 'retrieved'
    review = read(Path(condition['independent_evaluation_review']['path']))
    verify_identity(condition['independent_evaluation_review'])
    assert review['status'] == 'FORMAL_EVALUATION_ACCEPTANCE_PASS'
    assert (review['domain'], review['route'], review['shuffle_n'], review['job_id']) == (domain, 'B', n, condition['evaluation_job_id'])
    assert all(review[key] == condition[key] for key in ('metrics', 'samples', 'unique_users', 'config', 'sid', 'checkpoint', 'prediction_sha256'))
    for row in review['inputs']:
        verify_identity(row)
    result, invocation = read(root / 'output/RESULT.json'), read(root / 'output/invocation.json')
    assert result['metrics']['metrics'] == condition['metrics']
    assert invocation['phase'] == 'evaluate'
    spec = invocation['spec']
    assert all(spec[key] == condition[key] for key in ('domain', 'route', 'shuffle_n', 'config', 'sid', 'checkpoint', 'test_index', 'attention_cohort'))
    used = read(root / 'output/used_config.json')
    assert used['model']['top_k_for_generation'] == 10
    assert used['model'].get('should_check_prefix', False) is False
    prefix = review['prefix_constraint_configuration']
    assert prefix['effective_value'] is False
    verify_identity(prefix['source'])
    source = ast.parse(Path(prefix['source']['path']).read_bytes())
    model_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'SemanticIDEncoderDecoder')
    constructor = next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    defaults = dict(zip([argument.arg for argument in constructor.args.args][-len(constructor.args.defaults):], constructor.args.defaults))
    assert isinstance(defaults['should_check_prefix'], ast.Constant) and defaults['should_check_prefix'].value is False
    assert used['trainer']['precision'] == '32-true'
    index_path = JOBS / f'gs-20261017-{domain}-b-preflight-r2/retrieved/output/test_target_index.jsonl'
    assert identity(index_path)['sha256'] == spec['test_index']['sha256']
    targets = Counter((row['sample_key'], tuple(row['target'])) for row in map(json.loads, index_path.read_text().splitlines()))
    verify_identity(review['legal_vocabulary'])
    vocabulary = read(Path(review['legal_vocabulary']['path']))
    assert vocabulary['source']['sha256'] == condition['sid']['sha256']
    legal = {tuple(row) for row in vocabulary['legal_raw_sids']}
    predictions = {}
    sums = {name: [] for name in METRICS}
    invalid = duplicates = 0
    prediction_path = root / 'output/test/predictions.jsonl'
    assert identity(prediction_path)['sha256'] == condition['prediction_sha256']
    with prediction_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key, target = row['sample_key'], tuple(row['target'])
            assert key not in predictions and len(target) == 4 and target in legal
            sids = [tuple(candidate) for candidate in row['sids']]
            scores = row['scores']
            assert len(sids) == len(scores) == 10 and all(len(sid) == 4 for sid in sids)
            assert all(math.isfinite(score) for score in scores)
            assert all(left >= right for left, right in zip(scores, scores[1:]))
            rank = sids.index(target) + 1 if target in sids else None
            if rank:
                target_score = scores[rank - 1]
                assert all(score != target_score or sid == target for score, sid in zip(scores, sids)), 'rank ambiguous under tied positive/negative scores'
            metric = metrics_at(rank)
            invalid_here = sum(sid not in legal for sid in sids)
            duplicate_here = len(sids) - len(set(sids))
            invalid += invalid_here
            duplicates += duplicate_here
            for name in METRICS:
                sums[name].append(metric[name])
            predictions[key] = {'target': target, 'sids': sids, 'rank': rank, 'metrics': metric,
                                'invalid': invalid_here, 'duplicates': duplicate_here}
    expected = Counter((key, row['target']) for key, row in predictions.items())
    assert expected == targets
    assert len(predictions) == condition['samples'] == condition['unique_users'] == {'toys': 19412, 'sports': 35598}[domain]
    for name in METRICS:
        compare_numbers(math.fsum(sums[name]) / len(predictions), review['independent_stdlib_metrics'][name])
    assert invalid == condition['invalid_candidate_count'] and duplicates == condition['duplicate_candidate_count'] == 0
    return predictions, {'domain': domain, 'shuffle_n': n, 'original_predictions_read': len(predictions),
                         'four_metrics_independently_crosschecked': True, 'exact_key_target_multiset': True,
                         'invalid_candidates': invalid, 'duplicate_candidates': duplicates,
                         'prediction_sha256': condition['prediction_sha256']}


def review_pairs(summary, maps):
    report_path = SUPPORT / 'results-audit/toys-sports-paired-shuffle-analysis.json'
    report = read(report_path)
    assert summary['paired_comparisons'] == report['comparisons'] and len(report['comparisons']) == 4
    independent = read(Path(summary['paired_analysis_independent_review']['path']))
    verify_identity(summary['paired_analysis_independent_review'])
    assert independent['status'] == 'INDEPENDENT_COMPLETE_PAIRED_ANALYSIS_PASS'
    verify_identity(independent['report'])
    checked = 0
    for comparison in report['comparisons']:
        domain, n = comparison['domain'], comparison['Normal_to_Shuffle_n']
        normal, shuffled = maps[(domain, 0)], maps[(domain, n)]
        assert normal.keys() == shuffled.keys()
        source = comparison['all_user_paired_observations']
        verify_identity(source)
        changes = Counter()
        seen = set()
        with gzip.open(source['path'], 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                key = row['sample_key']; assert key not in seen
                seen.add(key)
                a, b = normal[key], shuffled[key]
                assert a['target'] == b['target'] and row['target_unchanged'] is True
                assert row['normal_target_rank'] == a['rank'] and row['shuffle_target_rank'] == b['rank']
                for label, source_row in [('normal', a), ('shuffle', b)]:
                    for k in (5, 10):
                        assert row[f'{label}_hit{k}'] == bool(source_row['metrics'][f'recall@{k}'])
                    assert row[f'{label}_invalid_count'] == source_row['invalid']
                    assert row[f'{label}_duplicate_count'] == source_row['duplicates']
                for field, value in [('top1_same', a['sids'][0] == b['sids'][0]),
                                     ('ordered_top10_same', a['sids'] == b['sids']),
                                     ('top10_set_same', set(a['sids']) == set(b['sids'])),
                                     ('common_candidates_at10', len(set(a['sids']) & set(b['sids']))),
                                     ('changed_positions_at10', sum(x != y for x, y in zip(a['sids'], b['sids'])))]:
                    assert row[field] == value
                    if field in ('top1_same', 'ordered_top10_same', 'top10_set_same'):
                        changes[field] += bool(value)
                for name in METRICS:
                    compare_numbers(row['independent_metric_deltas'][name], b['metrics'][name] - a['metrics'][name])
                checked += 1
        assert seen == normal.keys() and len(seen) == comparison['paired_users']
        change_report = comparison['recommendation_list_changes']
        assert change_report['top1_changed_users'] == len(seen) - changes['top1_same']
        assert change_report['ordered_top10_changed_users'] == len(seen) - changes['ordered_top10_same']
        assert change_report['top10_set_changed_users'] == len(seen) - changes['top10_set_same']
    assert checked == 110020
    return {'comparisons': 4, 'original_paired_rows_read_and_key_fields_crosschecked': checked,
            'report': identity(report_path), 'complete_distribution_and_declared_examples_review': summary['paired_analysis_independent_review']}


def review_query_companions(summary):
    module_path = SUPPORT / 'delivery-audit/audit_raw_attention.py'
    module_spec = importlib.util.spec_from_file_location('raw_attention_final_review', module_path)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    master = read(Path(summary['decoder_query_identity_independent_review']['path']))
    verify_identity(summary['decoder_query_identity_independent_review'])
    assert master['status'] == 'INDEPENDENT_DECODER_QUERY_IDENTITIES_PASS'
    assert master['condition_count'] == 6 and master['total_query_identity_records'] == 1536
    all_checked = 0
    for condition in summary['conditions']:
        verify_identity(condition['decoder_query_identity'])
        document = read(Path(condition['decoder_query_identity']['path']))
        assert document['status'] == 'DERIVED_DECODER_QUERY_TOKEN_IDENTITY_COMPLETE'
        assert document['job_id'] == condition['attention_job_id']
        assert not document['T5Stack_inputs_embeds_direct_capture_claimed'] and not document['raw_NPZ_or_PNG_modified']
        assert document['total_query_identity_records'] == 256 and len(document['sample_query_identities']) == 32
        for row in document['unchanged_raw_bundle_inputs'] + document['byte_bound_source_mapping_inputs']:
            verify_identity(row)
        bundles = {}
        for row in document['sample_query_identities']:
            record = row['source_record']
            if record not in bundles:
                bundles[record] = module.load_direct(Path(record))[0]
            bundle = bundles[record]
            sample = bundle['capture']['sample_keys'].index(row['sample_key'])
            assert row['final_Top1_raw_SID'] == bundle['final_generated_ids'][sample, 0].tolist()
            for path in ('final_top1', 'local_first'):
                identities = row['paths'][path]
                assert len(identities) == 4
                for h, token in enumerate(identities):
                    parent = int(bundle['paths']['final_parent_indices'][sample, h]) if path == 'final_top1' else 0
                    prefix = [] if h == 0 else bundle['steps'][h - 1]['generated_ids'][sample, parent].tolist()
                    assert token['preselection_parent_beam_index'] == parent and token['raw_parent_SID_prefix'] == prefix
                    assert token['target_generated_code_1based'] == h + 1
                    assert token['actual_observed_cross_attention_query_length'] == bundle['steps'][h]['query_length_before_last_selection']
                    assert token['actual_observed_cross_attention_last_query_index'] == bundle['steps'][h]['last_query_index']
                    if h == 0:
                        assert token['raw_query_SID_integer'] is None and token['derived_embedding_lookup_index'] is None
                        assert token['query_source_code_position_0based'] is None and 'bos_token' in token['query_kind']
                    else:
                        assert token['raw_query_SID_integer'] == prefix[-1]
                        assert token['query_source_code_position_0based'] == h - 1
                        assert token['derived_embedding_lookup_index'] == prefix[-1] + (h - 1) * 256
                    all_checked += 1
    assert all_checked == 1536
    return {'conditions': 6, 'derived_query_identities_independently_crosschecked': all_checked,
            'BOS_and_derived_vs_direct_capture_boundary_preserved': True}


def check(args):
    summary_path = REPO / PREFIX / 'EXECUTION_SUMMARY.json'
    report_path = REPO / PREFIX / 'FINAL_REPORT.md'
    summary_bytes, human_bytes = summary_path.read_bytes(), report_path.read_bytes()
    if args.commit:
        assert summary_bytes == git_bytes(args.commit, PREFIX + 'EXECUTION_SUMMARY.json')
        assert human_bytes == git_bytes(args.commit, PREFIX + 'FINAL_REPORT.md')
    summary = json.loads(summary_bytes)
    assert summary['status'] == 'PARTIAL' and summary['qualified_B_conditions_complete']
    assert summary['inherited_Beauty_anchor'] == ANCHOR and not summary['Beauty_rerun'] and not summary['new_training'] and not summary['SID_regenerated']
    assert summary['branch'] == 'wbw' and summary['active_job_ids'] == []
    assert summary['task_correction_rounds_used'] == 3 and summary['maximum_task_correction_rounds'] == 3
    assert summary['GPT_UPLOAD_REQUIRED'] == 'YES'
    assert len(summary['conditions']) == 6 and {(row['domain'], row['route'], row['shuffle_n']) for row in summary['conditions']} == {
        (domain, 'B', n) for domain in ('toys', 'sports') for n in (0, 3, 4)}
    verified_jobs = read(SUPPORT / 'delivery-audit/final-terminal-job-review.json')
    assert verified_jobs['status'] == 'PASS' and verified_jobs['final_complete_B_matrix_required']
    assert verified_jobs['job_count'] == 22 and verified_jobs['terminal_counts'] == {'FAILED': 6, 'SUCCEEDED': 16}
    job_reviews = {row['job_id']: row for row in verified_jobs['jobs']}
    final_query = read(SUPPORT / 'final-state-query.json')
    assert final_query['active_job_ids'] == [] and final_query['all_task_jobs_terminal_and_clean']
    assert set(final_query['jobs']) == set(job_reviews) == {row['job_id'] for row in summary['jobs']}
    for row in summary['jobs']:
        original, current = job_reviews[row['job_id']], final_query['jobs'][row['job_id']]
        assert (row['status'], row['exit_code']) == (original['status'], original['exit_code']) == (current['status'], current['exit_code'])
        assert current['cleanup_verified'] and not current['worker_alive'] and not current['workgroup_live'] and not current.get('worker_may_be_live', False)
        for name in ('request', 'status_record', 'export_manifest'):
            verify_identity(row[name])
        compare_numbers(row['worker_wall_seconds_including_CPU_preparation'], original['conservative_status_interval_seconds_including_cleanup'])
        assert row['argv'] == read(Path(row['local_evidence']) / 'request.json')['argv']
    seconds = verified_jobs['accounting']['conservative_status_interval_seconds_including_cleanup']
    compare_numbers(seconds / 3600, summary['budgets']['total_GPU_request_worker_wall_hours'])
    assert seconds <= 19800 and verified_jobs['accounting']['gpu_check_conservative_seconds'] <= 1800
    maps, prediction_checks = {}, []
    for condition in summary['conditions']:
        predictions, check_row = review_predictions(condition)
        maps[(condition['domain'], condition['shuffle_n'])] = predictions
        prediction_checks.append(check_row)
        attention = read(Path(condition['independent_attention_review']['path']))
        verify_identity(condition['independent_attention_review'])
        assert attention['status'] == 'INDEPENDENT_FORMAL_ATTENTION_RESULT_PASS'
        assert (attention['domain'], attention['route'], attention['shuffle_n'], attention['job_id']) == (condition['domain'], 'B', condition['shuffle_n'], condition['attention_job_id'])
        assert attention['samples'] == 32 and attention['raw_head_mean_and_both_paths_exactly_recomputed']
        assert attention['maximum_padding_attention_mass'] == 0
    for domain in ('toys', 'sports'):
        rows = [row for row in summary['conditions'] if row['domain'] == domain]
        normal = next(row for row in rows if row['shuffle_n'] == 0)
        assert all(all(row[key] == normal[key] for key in ('config', 'sid', 'checkpoint', 'test_index', 'attention_cohort')) for row in rows)
        for row in rows:
            for name in METRICS:
                diff = row['metrics'][name] - normal['metrics'][name]
                compare_numbers(row['differences_from_own_Normal'][name]['absolute'], diff)
                compare_numbers(row['differences_from_own_Normal'][name]['relative_percent'], 100 * diff / normal['metrics'][name])
    paired = review_pairs(summary, maps)
    del maps
    raw_attention = read(SUPPORT / 'delivery-audit/raw-attention-statistics-crosscheck.json')
    assert raw_attention['status'] == 'FINAL_RAW_ATTENTION_STATISTICS_CROSSCHECK_PASS'
    assert raw_attention['NPZ_count'] == 24 and raw_attention['all_pairs_checked'] == 2304
    verify_identity(raw_attention['statistics'])
    verify_identity(summary['attention_statistics']['report'])
    stats_review = read(Path(summary['attention_statistics']['independent_review']['path']))
    verify_identity(summary['attention_statistics']['independent_review'])
    assert stats_review['status'] == 'INDEPENDENT_ATTENTION_DESCRIPTIVE_STATISTICS_PASS'
    verify_identity(stats_review['report'])
    stats = read(Path(raw_attention['statistics']['path']))
    expected_stats = [{key: row[key] for key in ['job_id', 'samples', 'summary', 'least_similar_three_under_declared_mean_cosine', 'maximum_masked_mass', 'maximum_raw_row_sum_error']} for row in stats['jobs']]
    assert summary['attention_statistics']['conditions'] == expected_stats
    queries = review_query_companions(summary)
    asset = read(Path(summary['asset_audit']['path'])); verify_identity(summary['asset_audit'])
    assert asset['status'] == 'PASS_B_CORE_ASSET_ADMISSION_WITH_A_UNRESOLVED'
    assert asset['A']['toys']['status'] == 'IMPORTANT_CHECKPOINT_SELECTION_UNRESOLVED_NOT_MISSING_ALL_ASSETS'
    assert asset['A']['sports']['status'] == 'MATCHED_SOURCE_SID_CHECKPOINT_CHAIN_NOT_FOUND_IN_CAPTURED_SCOPE'
    for domain in ('toys', 'sports'):
        row = asset['B'][domain]
        assert row['status'] == 'PASS' and row['strict_model_values_and_embedded_sid']
        config = SUPPORT / 'asset-audit/B-original-records' / domain / 'tiger_train_test_val1600_20261010_112259'
        with (config / 'csv/version_0/metrics.csv').open(newline='') as stream:
            validation = [value for value in csv.DictReader(stream) if value['val/recall@5']]
        best = max(validation, key=lambda value: float(value['val/recall@5']))
        assert int(best['step']) + 1 == row['selected_checkpoint_source']['selected_step'] == {'toys': 6200, 'sports': 9500}[domain]
        stage = read(config / 'stage_result.json')
        assert Path(stage['checkpoint']).name == Path(row['assets']['checkpoint']['path']).name
        assert row['selected_checkpoint_source']['not_selected_by_largest_step_or_test_score']
    assert set(summary['extra_failed_CPU_evidence']) == {f'gs-20261017-{domain}-b-callback-audit-v1' for domain in ('toys', 'sports')}
    assert summary['inherited_Beauty']['conditions'] == read(REPO / 'manifests/grid-formal-20261015/results-review/beauty-six-formal-results-independent-summary.json')['conditions']
    old = ['GRID baseline/src', 'references/wmh-original/GRID/src', 'experiments/beauty_extensions', 'experiments/preflight_20261015', 'experiments/formal_20261015', 'manifests/grid-formal-20261015']
    committed_changes = subprocess.check_output(['git', 'diff', '--name-only', ANCHOR, 'HEAD', '--', *old], cwd=REPO, text=True).splitlines()
    assert committed_changes == ['experiments/formal_20261015/run.py']
    assert not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *old], cwd=REPO, text=True).strip()
    report_text = human_bytes.decode()
    for required in ('PARTIAL', 'Toys A', 'Sports A', '2304', '1536', '192', 'seed42', 'callback', '3/3', 'GPT_UPLOAD_REQUIRED: YES'):
        assert required in report_text, ('report omission', required)
    showcase = read(REPO / PREFIX / 'attention-showcase/IDENTITIES.json')
    assert len(showcase) == 6
    showcase_count = 0
    for row in showcase:
        root = JOBS / row['job_id'] / 'retrieved'
        assert len(row['all_32_original_PNGs']) == 32 and len(row['original_NPZ_files']) == 4
        for item in row['all_32_original_PNGs'] + row['original_NPZ_files']:
            source = root / item['relative_path']; assert identity(source)['sha256'] == item['sha256'] and source.stat().st_size == item['bytes']
        for name in row['selected_relative_paths']:
            published = REPO / PREFIX / 'attention-showcase' / row['job_id'] / name
            assert published.read_bytes() == (root / name).read_bytes()
            showcase_count += 1
    publication = None
    if args.commit:
        assert args.publication is not None
        publication = read(args.publication)
        assert publication['commit'] == args.commit and publication['branch'] == 'wbw'
        assert all(publication[key] for key in ('remote_ref_verified', 'GitHub_ref_and_tree_verified', 'key_file_bytes_verified', 'server_snapshot_mapping_verified'))
        assert not publication['other_branches_written'] and not publication['force_push']
        for row in publication['actual_GitHub_file_readbacks']:
            data = git_bytes(args.commit, row['path'])
            assert {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} == {key: row[key] for key in ('bytes', 'sha256')}
        by_root = {row['root']: row for row in publication['server_snapshots']}
        for row in summary['jobs']:
            if not row['snapshot_sha256']:
                continue
            manifest = by_root[row['snapshot_root']]
            request = read(Path(row['local_evidence']) / 'request.json')
            assert request['capture_version'] == manifest['manifest']['source_commit'] + '/' + manifest['manifest']['snapshot_sha256']
            assert {entry['path']: entry['sha256'] for entry in request['inputs']}[request['cwd'] + '/DEPLOYMENT.json'] == manifest['manifest_sha256']
    result = {'status': 'CONTENT_ACCEPTANCE_PASS' if args.commit else 'PREPUBLICATION_CONTENT_ACCEPTANCE_PASS',
              'reviewer_task': '/root/delivery_audit', 'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'content_commit': args.commit, 'task_overall_status': 'PARTIAL',
              'summary': identity(summary_path), 'human_report': identity(report_path),
              'publication': identity(args.publication) if args.publication else None,
              'original_terminal_jobs': 22, 'terminal_counts': verified_jobs['terminal_counts'], 'active_job_ids': [],
              'predictions': prediction_checks, 'paired': paired, 'queries': queries,
              'raw_attention_statistics_crosscheck': identity(SUPPORT / 'delivery-audit/raw-attention-statistics-crosscheck.json'),
              'all_24_NPZ_192_PNG_identity_maps_checked': True, 'original_byte_showcase_PNGs_checked': showcase_count,
              'asset_review': summary['asset_audit'], 'A_scope': {domain: asset['A'][domain]['status'] for domain in ('toys', 'sports')},
              'B_selected_checkpoint_steps': {'toys': 6200, 'sports': 9500},
              'conservative_GPU_requested_worker_wall_hours': seconds / 3600,
              'requested_timeout_cap_sum_hours': verified_jobs['accounting']['sum_requested_timeout_caps_seconds'] / 3600,
              'corrections_used': 3, 'mandatory_failed_attempts_preserved': 6,
              'Beauty_inherited_not_rerun': True, 'no_new_training_or_SID': True,
              'scientific_limits': summary['scientific_limits'], 'installation_log': summary['installation_log'],
              'new_model_or_GPU_calls': 0, 'original_evidence_or_Git_writes': 0,
              'remaining_delivery_steps': [] if args.commit else ['Fixed wbw content commit publication and actual GitHub/server readback', 'Actual ZIP full origin/container audit', 'Final delivery receipt commit publication/readback'],
              'scope_limit': 'Real original evidence/content review; A routes remain unexecuted. Internal acceptance is not GPT webpage review or a scientific invariance claim. Final ZIP/container and final delivery publication need separate actual acceptance.'}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit')
    p.add_argument('--publication', type=Path)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    try:
        report = check(args)
    except Exception as error:
        report = {'status': 'FAIL', 'reviewer_task': '/root/delivery_audit', 'exception': type(error).__name__, 'reason': str(error)}
    report['program'] = identity(Path(__file__).resolve())
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'content_commit', 'task_overall_status', 'reason')}))
    raise SystemExit(0 if report['status'] in ('PREPUBLICATION_CONTENT_ACCEPTANCE_PASS', 'CONTENT_ACCEPTANCE_PASS') else 1)


if __name__ == '__main__':
    main()
