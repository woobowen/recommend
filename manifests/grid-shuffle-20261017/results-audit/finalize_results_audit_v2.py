"""Freeze bounded visual/provenance checks and the qualified-B audit index.

No metrics, model, evaluator or statistics method is rerun here. This binds
already completed audits and actual view_image observations to current bytes.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path('/home/addaswsw/recommend-support/grid-shuffle-20261017')
AUDIT = ROOT/'results-audit'
REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
JOBS = Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')


def read(path):
    return json.loads(Path(path).read_bytes())


def identity(path):
    path = Path(path); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value):
    if path.exists():
        existing = read(path)
        time_keys = {'reviewed_at_utc', 'recorded_at_utc'}
        assert {k:v for k,v in existing.items() if k not in time_keys} == {k:v for k,v in value.items() if k not in time_keys}
        return
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def verify_identity(row):
    assert identity(row['path']) == row


def main():
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    stats_path = ROOT/'attention-statistics/attention-descriptive-statistics.json'
    stats, stats_review = read(stats_path), read(AUDIT/'attention-statistics-independent-review.json')
    assert stats_review['status'] == 'INDEPENDENT_ATTENTION_DESCRIPTIVE_STATISTICS_PASS'
    assert stats_review['report'] == identity(stats_path)
    old_path = REPO/'manifests/grid-formal-20261015/attention-statistics/attention-descriptive-statistics.json'
    old = read(old_path)
    assert old['invocation']['script'] == stats['invocation']['script']
    assert old['method'] == stats['method']
    verify_identity(stats['invocation']['script'])
    invocation_path = ROOT/'attention-statistics/invocation.json'
    execution_path = ROOT/'attention-statistics/execution-receipt.json'
    invocation, execution = read(invocation_path), read(execution_path)
    assert invocation['script'] == execution['script'] == stats['invocation']['script']
    assert invocation['plan'] == execution['plan'] == stats['invocation']['plan']
    assert execution['exit_code'] == 0 and execution['raw_inputs_unchanged']
    assert execution['conditions'] == 6 and execution['pair_path_records'] == 2304
    assert execution['result'] == identity(stats_path)
    assert invocation['model_calls'] == invocation['GPU_calls'] == 0
    assert invocation['statistics_formula_changed'] is False
    for row in invocation['raw_inputs_before']:
        verify_identity(row)
    provenance = {'status': 'INDEPENDENT_STATISTICS_SOURCE_PROVENANCE_PASS', 'reviewer': '/root/results_audit',
        'reviewed_at_utc': now, 'source_program': stats['invocation']['script'],
        'producer_identity_exact_to_previously_accepted_eight_condition_statistics': True,
        'method_document_exact_to_previously_accepted_statistics': True,
        'previously_accepted_report_reference': identity(old_path),
        'new_report': identity(stats_path), 'execution_receipt': identity(execution_path), 'invocation': identity(invocation_path),
        'all_recorded_pre_execution_raw_input_files_currently_identical': len(invocation['raw_inputs_before']),
        'independent_2304_row_numeric_verification': identity(AUDIT/'attention-statistics-independent-review.json'),
        'new_model_or_GPU_calls': 0, 'scope': 'Compare inherited accepted report identities and actual current bytes; no re-execution of old Beauty/MLP analyses and no Git operation.'}
    provenance_path = AUDIT/'statistics-source-provenance-review.json'
    write(provenance_path, provenance)
    report_by_job = {j['job_id']: j for j in stats['jobs']}
    observations = read(AUDIT/'visual-inspection-observations.json')
    checked_views = []
    for row in observations['observations']:
        current = dict(row); image_path = JOBS/row['job_id']/'retrieved'/row['relative']
        domain = row['job_id'].split('-')[2]
        condition = row['job_id'].split('-')[4]
        n = {'normal': 0, 'shuffle3': 3, 'shuffle4': 4}[condition]
        acceptance_path = AUDIT/f'{domain}-b-{n}-attention-review.json'
        acceptance = read(acceptance_path)
        assert acceptance['status'] == 'INDEPENDENT_FORMAL_ATTENTION_RESULT_PASS'
        match = [v for v in acceptance['sample_checks'] if v['figure']['path'] == str(image_path)]
        assert len(match) == 1 and match[0]['figure'] == identity(image_path)
        sample = match[0]
        if current.get('sample_key'):
            assert current['sample_key'] == sample['sample_key']
        current['sample_key'] = sample['sample_key']
        if current['selection'].startswith('lowest'):
            declared = report_by_job[row['job_id']]['least_similar_three_under_declared_mean_cosine']['final_top1_cross'][0]
            assert declared['sample_key'] == sample['sample_key'] and declared['plot_relative'] == row['relative']
            current['prespecified_least_sample_values'] = declared
        else:
            assert sample['batch'] == sample['index_in_batch'] == 0
        current['original_PNG'] = identity(image_path)
        current['exhaustive_raw_pixel_review'] = identity(acceptance_path)
        checked_views.append(current)
    assert len(checked_views) == 12 and len({v['job_id'] for v in checked_views}) == 6
    visual = {'status': 'BOUNDED_ACTUAL_VISUAL_INSPECTION_COMPLETE', 'reviewer': '/root/results_audit',
        'recorded_at_utc': now, 'viewing_tool': 'view_image', 'actual_images_viewed': len(checked_views),
        'selection_rule': observations['selection_rule'], 'statistics': identity(stats_path),
        'observations_input': identity(AUDIT/'visual-inspection-observations.json'), 'images': checked_views,
        'scope': 'Actual subjective inspection of first and least-similar fixed-user figures; exhaustive pixel/PNG identity validation covers all192 in separate original-attention audits. No crop, recoloring, row normalization or unsupported similarity/invariance threshold.',
        'new_model_or_GPU_calls': 0, 'raw_PNG_or_NPZ_modified': False}
    visual_path = AUDIT/'visual-inspection-review.json'
    write(visual_path, visual)
    queries_path = ROOT/'source-audit/decoder-query-identities-independent-review.json'
    queries = read(queries_path)
    assert queries['status'] == 'INDEPENDENT_DECODER_QUERY_IDENTITIES_PASS' and queries['total_query_identity_records'] == 1536
    conditions, count_rows, pixels, masks = [], 0, 0, 0
    for domain in ('toys', 'sports'):
        for n in (0, 3, 4):
            evaluation_path = AUDIT/f'{domain}-b-{n}-evaluation-review.json'
            attention_path = AUDIT/f'{domain}-b-{n}-attention-review.json'
            query_path = AUDIT/f'{domain}-b-{n}-decoder-query-identity.json'
            query_conditions = [v for v in queries['conditions'] if (v['domain'], v['route'], v['shuffle_n']) == (domain, 'B', n)]
            assert len(query_conditions) == 1
            query_condition = query_conditions[0]
            assert query_condition['report'] == identity(query_path) and query_condition['complete_query_identity_records'] == 256
            evaluation, attention = read(evaluation_path), read(attention_path)
            assert evaluation['status'] == 'FORMAL_EVALUATION_ACCEPTANCE_PASS'
            assert attention['status'] == 'INDEPENDENT_FORMAL_ATTENTION_RESULT_PASS'
            assert evaluation['domain'] == attention['domain'] == domain and evaluation['shuffle_n'] == attention['shuffle_n'] == n
            for row in attention['sample_checks']:
                verify_identity(row['figure'])
            for batch in attention['batches']:
                for name in ('bundle', 'arrays', 'figures', 'off_on'):
                    verify_identity(batch[name])
            count_rows += evaluation['samples']
            pixels += attention['figure_evidence']['both_panel_raw_attention_cell_center_colors_checked']
            masks += attention['figure_evidence']['mask_cell_center_colors_checked']
            conditions.append({'domain': domain, 'route': 'B', 'shuffle_n': n, 'evaluation': identity(evaluation_path), 'attention': identity(attention_path),
                'derived_query_identity': identity(query_path), 'derived_query_independent_review': identity(queries_path),
                'derived_query_independent_condition': query_condition,
                'evaluated_users': evaluation['samples'], 'fixed_users': attention['samples'], 'raw_PNGs': attention['figure_evidence']['PNGs_reopened']})
    paired_path = AUDIT/'toys-sports-paired-shuffle-analysis.json'
    paired_review_path = ROOT/'source-audit/paired-analysis-independent-review.json'
    assert read(paired_review_path)['status'] == 'INDEPENDENT_COMPLETE_PAIRED_ANALYSIS_PASS'
    assert read(paired_review_path)['report'] == identity(paired_path)
    queries_path = ROOT/'source-audit/decoder-query-identities-independent-review.json'
    queries = read(queries_path)
    assert queries['status'] == 'INDEPENDENT_DECODER_QUERY_IDENTITIES_PASS' and queries['total_query_identity_records'] == 1536
    index = {'status': 'QUALIFIED_B_RESULTS_ATTENTION_AUDIT_COMPLETE', 'reviewer': '/root/results_audit', 'completed_at_utc': now,
        'scope': 'Two qualified B routes only; this does not turn missing Toys/Sports A assets into completed conditions or decide overall project readiness.',
        'formal_conditions': 6, 'formal_prediction_user_rows': count_rows, 'attention_conditions': 6, 'fixed_user_condition_instances': 192,
        'raw_NPZ_files': 24, 'all_preserved_PNGs_current_hashes_verified': 192, 'all_PNG_heatmap_cells_independently_checked': pixels,
        'all_PNG_mask_cells_independently_checked': masks, 'descriptive_pair_path_rows_independently_verified': 2304,
        'derived_decoder_query_identities_independently_verified': 1536, 'conditions': conditions,
        'original_legal_SID_projection_independent_review': identity(AUDIT/'sid-dictionary-independent-review.json'),
        'paired_analysis': identity(paired_path), 'paired_analysis_nonauthor_review': identity(paired_review_path),
        'statistics_independent_review': identity(AUDIT/'attention-statistics-independent-review.json'),
        'statistics_source_provenance': identity(provenance_path), 'decoder_query_nonauthor_summary': identity(queries_path),
        'bounded_visual_review': identity(visual_path), 'program': identity(Path(__file__)), 'argv': sys.argv,
        'reviewer_author_scope': 'Authored new review/path adaptations, descriptive paired producer, query-identity companion and these final audit receipts. Did not author model/beam/evaluator/attention/statistics producer; new paired/query outputs are separately reviewed by source_audit.',
        'new_model_GPU_or_evaluator_calls_by_this_reviewer': 0, 'Git_operations_by_this_reviewer': 0,
        'new_dependency_installations_by_this_reviewer': [], 'raw_original_evidence_modified': False}
    index_path = AUDIT/'RESULTS_ATTENTION_AUDIT_INDEX.json'
    write(index_path, index)
    print(json.dumps({'index': identity(index_path), 'visual': identity(visual_path), 'provenance': identity(provenance_path), 'status': index['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
