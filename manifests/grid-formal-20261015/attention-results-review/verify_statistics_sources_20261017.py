"""CPU provenance wrapper around the existing independent numeric verifier.

No new statistical definition. Refuses to overwrite existing review outputs.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
sys.path.insert(0, str(REPO))
from experiments.beauty_extensions.attention.records import load_bundle

EXPECTED = {
    'gf-20261015-a-normal-attention-v1',
    'gf-20261015-a-shuffle3-attention-v1',
    'gf-20261015-a-shuffle4-attention-v1',
    'gf-20261015-b-normal-attention-v1',
    'gf-20261015-b-shuffle3-attention-v1',
    'gf-20261015-b-shuffle4-attention-v1',
    'gf-20261015-toys-mlp-attention-v1',
    'gf-20261015-sports-mlp-attention-v1',
}


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    destination = SUPPORT/'attention-results-review/attention-statistics-source-provenance-review-20261017.json'
    runtime = SUPPORT/'attention-results-review/attention-statistics-runtime-review.json'
    assert not destination.exists() and not runtime.exists()
    report = json.loads(args.report.read_bytes())
    plan = json.loads(Path(report['invocation']['plan']['path']).read_bytes())
    assert set(plan['jobs']) == EXPECTED
    assert len(report['jobs']) == 8
    method = report['method']
    assert method['cross_condition_matrix_comparison'].startswith('not performed')
    assert method['similarity_threshold_or_invariance_proof'] is None
    observed = {}
    jobs = []

    def remember(path):
        row = identity(path)
        observed[str(path)] = row
        return row

    for job_id, directory_string in plan['jobs'].items():
        directory = Path(directory_string)
        result_path = directory/'output/RESULT.json'
        result = json.loads(result_path.read_bytes())
        assert result['formal'] is True and result['status'] == 'COMPLETE'
        assert result['samples'] == 32 and len(result['batches']) == 4
        assert result['scientific_similarity_claim'] is None
        status_path = directory/'status.json'
        status = json.loads(status_path.read_bytes())
        assert status['job_id'] == job_id and status['status'] == 'SUCCEEDED'
        assert status['exit_code'] == 0 and not status['worker_may_be_live']
        assert status['cleanup']['verified']
        request_path = directory/'request.json'
        request = json.loads(request_path.read_bytes())
        assert request['job_id'] == job_id
        review_path = SUPPORT/'attention-results-review'/f'{job_id}-review.json'
        prior = json.loads(review_path.read_bytes())
        assert prior['job_id'] == job_id and prior['status'].endswith('_PASS')
        assert identity(result_path) == prior['result']
        manifest_path = directory/'export-manifest.json'
        prior_manifest = prior.get('original_export_manifest', prior.get('export_manifest'))
        assert identity(manifest_path) == prior_manifest
        manifest = json.loads(manifest_path.read_bytes())
        members = {row['path']: row for row in manifest['files']}
        assert len(members) == len(manifest['files'])
        for relative, expected in members.items():
            path = directory/relative
            assert not path.is_symlink()
            actual = remember(path)
            assert actual['bytes'] == expected['bytes'] and actual['sha256'] == expected['sha256']
        for path in (manifest_path, review_path, status_path, request_path):
            remember(path)
        cohort_path = Path(prior['cohort']['path'])
        cohort_identity = remember(cohort_path)
        assert cohort_identity == prior['cohort']
        cohort = json.loads(cohort_path.read_bytes())
        assert cohort_identity['sha256'] == result['cohort']['sha256']
        assert cohort['selected_users'] == 32 and cohort['model_outputs_inspected'] is False
        expected_keys = cohort['sample_keys']
        assert len(expected_keys) == len(set(expected_keys)) == 32
        assert [key for batch in result['batches'] for key in batch['sample_keys']] == expected_keys
        record_paths = sorted((directory/'output').glob('batch-*/bundle/record.json'))
        assert len(record_paths) == 4
        batches = []
        for index, record_path in enumerate(record_paths):
            bundle = load_bundle(record_path)
            assert bundle['metadata']['engineering_only'] is False
            capture = bundle['capture']
            assert capture['selected_indices'] == list(range(8))
            assert capture['sample_keys'] == expected_keys[index*8:(index+1)*8]
            assert capture['batch_size'] == 8
            assert capture['local_first_path'].startswith('preselection parent rank 0 independently')
            assert capture['final_top1_path'].startswith('final child rank 0 backtracked')
            assert bundle['metadata']['spec']['formal_approved'] is True
            for label in ('config', 'sid', 'checkpoint'):
                assert bundle['metadata']['spec'][label] == result[label]
            assert bundle['metadata']['spec']['attention_cohort'] == result['cohort']
            assert result['batches'][index]['bundle_sha256'] == identity(record_path)['sha256']
            mask = np.asarray(bundle['encoder_mask'])
            assert set(np.unique(mask)) <= {0, 1}
            shapes = {}
            for name in ('final_top1', 'local_first'):
                heads = np.asarray(bundle['paths'][name+'_heads'])
                mean = np.asarray(bundle['paths'][name+'_cross'])
                assert heads.shape[:3] == (8, 4, 6)
                assert mean.shape == (8, 4, mask.shape[-1])
                assert np.array_equal(heads.mean(axis=2), mean)
                shapes[name] = {'raw_heads': list(heads.shape), 'mean': list(mean.shape)}
            figures = sorted((record_path.parent.parent/'figures').glob('sample_*_cross_attention.png'))
            assert len(figures) == 8
            batches.append({'record': identity(record_path), 'arrays': identity(record_path.parent/'arrays.npz'),
                'selected_keys_exact': True, 'matrix_shapes': shapes,
                'six_heads_mean_equals_saved_cross_exactly': True,
                'original_eight_PNG_hashes_match_export': True})
        jobs.append({'job_id': job_id, 'result': identity(result_path),
            'request': identity(request_path), 'status': identity(status_path),
            'source_commit': request['capture_version'].split('/')[0],
            'deployment_snapshot': request['cwd'], 'export_manifest': identity(manifest_path),
            'prior_technical_review': identity(review_path),
            'prior_technical_review_scope_inherited': 'beam ancestry, observer OFF/ON, raw PNG pixel/mask correspondence; no model rerun',
            'fixed_preoutput_cohort': cohort_identity, 'checkpoint': result['checkpoint'],
            'sid': result['sid'], 'config': result['config'], 'batches': batches,
            'all_original_export_members_hash_verified': len(members)})
    verifier = REPO/'manifests/grid-formal-20261015/attention-results-review/verify_attention_statistics.py'
    command = [sys.executable, '-B', str(verifier), str(args.report)]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    process = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    ended = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for path_string, expected in observed.items():
        assert identity(Path(path_string)) == expected, path_string
    receipt = {'status': 'INDEPENDENT_ATTENTION_STATISTICS_SOURCE_AND_RUNTIME_PASS' if process.returncode == 0 else 'FAILED_RUNTIME_VERIFICATION',
        'reviewer_task': '/root/implementation_review',
        'reviewer_role': 'new runtime reviewer in 2026-10-17 final-closeout session, not author of production statistics script or its actual output; historical same task label is separately dated',
        'actual_report': identity(args.report), 'plan': identity(Path(report['invocation']['plan']['path'])),
        'source_program': identity(Path(__file__)), 'independent_numeric_program': identity(verifier),
        'invocation': {'argv': sys.argv, 'interpreter': sys.executable, 'cwd': str(Path.cwd()),
            'numeric_verifier_argv': command, 'numeric_verifier_cwd': str(REPO),
            'start_utc': started, 'end_utc': ended, 'exit_code': process.returncode,
            'stdout': process.stdout, 'stderr': process.stderr,
            'source_git_commit_at_review': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'numpy_version': np.__version__},
        'jobs': jobs, 'job_count': 8, 'pair_path_records_total': sum(len(row['per_sample_query_pairs']) for row in report['jobs']),
        'original_evidence_files_unchanged_before_after_verifier': len(observed),
        'complete_eight_formal_jobs_no_engineering_only': True,
        'no_unaligned_columns_or_similarity_threshold': True,
        'beauty_inherited_MLP': {'included_in_numeric_statistics': False,
            'reason': 'Inherited result uses PASS while unchanged production statistics contract requires COMPLETE; no original evidence or metric code altered to force inclusion.',
            'prior_formal_and_technical_review_retained': True},
        'limitations': ['Fixed32 evaluation cohorts, not population inference or causal proof.',
            'A/B and Shuffle raw columns are not aligned or directly compared.',
            'Original model weights and GPU execution inherit accepted technical audits; no model reloading or GPU invocation.',
            'No independent numerical statistics for inherited Beauty MLP protocol.'],
        'new_GPU_or_model_calls': 0, 'research_source_Git_or_original_evidence_writes': 0,
        'prior_research_revision_allowance': '3/3 and one limited config exception already used; no new substantive revision'}
    if process.returncode == 0:
        assert runtime.exists()
        receipt['numeric_runtime_receipt'] = identity(runtime)
        assert receipt['pair_path_records_total'] == 3072
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(identity(destination)|{'status': receipt['status'], 'jobs': 8, 'exit_code': process.returncode}))
    assert process.returncode == 0, process.stderr


if __name__ == '__main__':
    main()
