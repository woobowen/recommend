"""Prepare fixed32 plans from accepted existing formal MLP outputs, no GPU."""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
REMOTE = '/home/bwwu/grid-audits/sid-extensions-p0/grid-formal-20261015'
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport, atomic


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    t = Transport('new-server')
    t.identity()
    collection = SUPPORT/'collected-9a64fd31218d0c68'
    combined = SUPPORT/'mlp-results-review/mlp-formal-combined-review.json'
    accepted = json.loads(combined.read_text())
    assert accepted['status'] == 'TOYS_SPORTS_FORMAL_TRAINING_AND_FULL_TEST_ACCEPTED'
    remote_review = REMOTE+'/approvals/'+sha(combined)+'/formal-MLP-combined-review.json'
    t.put(combined, remote_review)
    destination = SUPPORT/'mlp-attention-specs'
    destination.mkdir(exist_ok=True)
    for domain in ('toys', 'sports'):
        job = f'gf-20261015-{domain}-mlp-formal-v1'
        base = collection/job
        result_path = base/'output/RESULT.json'
        result = json.loads(result_path.read_text())
        status_path = base/'status.json'
        status = json.loads(status_path.read_text())
        assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0 and status['cleanup']['verified']
        assert result['status'] == 'COMPLETE' and result['formal'] and result['training_complete'] and result['test_complete']
        source_spec = REPO/f'experiments/formal_20261015/specs/{domain}-mlp.json'
        source = json.loads(source_spec.read_text())
        spec = {key: source[key] for key in ('route', 'seed', 'generation_k', 'data_dir',
            'config', 'sid', 'stage', 'domain', 'formal_approved', 'test_index',
            'evaluation_index', 'attention_cohort')}
        spec.update({'checkpoint': result['best_checkpoint'], 'shuffle_n': 0,
            'phase': 'formal_attention_capture',
            'purpose': 'Approved fixed32 inference capture after independently accepted full scratch MLP training/test; no training or full-test repetition',
            'evaluation_result': {'path': f'/home/bwwu/grid-audits/sid-extensions-p0/jobs/{job}/output/RESULT.json', 'sha256': sha(result_path)},
            'evaluation_status': {'path': f'/home/bwwu/grid-audits/sid-extensions-p0/jobs/{job}/status.json', 'sha256': sha(status_path)},
            'evaluation_independent_review': {'path': remote_review, 'sha256': sha(combined)},
            'training_protocol_reference': {'job_id': job, 'approved_training_spec_sha256': sha(source_spec),
                'result_contains_actual_effective_protocol_and_best_selection': True},
            'attention_protocol': {'cohort': 'original first32 evaluation users fixed before training/model outputs',
                'kind': 'cross-attention', 'rows': 'four autoregressive SID generation queries',
                'columns': 'per-history-item aggregated encoder tokens, including padding slots',
                'layer': 'last decoder layer', 'display_heads': 'mean', 'retained_heads': 'all raw heads of last layer',
                'paths': ['actual generation final Top-1 ancestry', 'local first parent per step'],
                'scientific_similarity_claim': None},
            'attention_wall_seconds_limit': 1200})
        assert result['checkpoint'] == spec['checkpoint'] and all(result[key] == spec[key] for key in ('config', 'sid'))
        from datetime import datetime
        elapsed = (datetime.fromisoformat(status['ended_at'])-datetime.fromisoformat(status['started_at'])).total_seconds()
        assert elapsed+1200 < 43200
        spec['resource_allowance'] = {'domain_original_hard_limit_seconds': 43200,
            'actual_training_validation_and_test_worker_wall_seconds': elapsed,
            'remaining_before_attention_seconds': 43200-elapsed,
            'this_attention_request_max_seconds': 1200,
            'maximum_cumulative_worker_wall_with_capture_seconds': elapsed+1200,
            'accounting': 'actual per-domain runner worker wall for training/validation/test plus this capture; no cross-domain transfer or extension; idle waiting/query is not GPU work',
            'user_approval_reference': '2026-10-16 explicit continuation attachment section4.2 authorizes each domain fixed32 within original remaining approved resources'}
        atomic(destination/(domain+'.json'), spec)
    print(json.dumps({'MLP_fixed32_plans': 2, 'model_or_GPU_calls': 0, 'independent_gate_pending': True}))


if __name__ == '__main__':
    main()
