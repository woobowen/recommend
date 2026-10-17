"""Write the independent narrow outer-runner engineering review."""
import datetime as dt
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


raw = json.loads((OUT / 'runner-adaptation-readonly-tests.json').read_text())
blocked = json.loads((OUT / 'outer-request-readonly-verify-blocked.json').read_text())
assert raw['status'] == 'NARROW_OUTER_RUNNER_READONLY_POSITIVE_NEGATIVE_PASS'
assert raw['case_count'] == 16 and all(r['expected_pass'] == r['actual_pass'] for r in raw['cases'])
assert raw['candidate_full_source_sha256'] == sha(ROOT / 'experiments/launch_20261017/queue_runner.py')
assert blocked['verify_pass'] is False and blocked['job_directory_exists'] is False
value = {
    'reviewer': '/root/fd_repair_reviewer',
    'reviewed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
    'status': 'OUTER_RUNNER_ENGINEERING_ADAPTATION_PASS',
    'fd_engineering_status': 'FD_ENGINEERING_EQUIVALENCE_PASS',
    'source': {'path': 'experiments/launch_20261017/queue_runner.py', 'sha256': raw['candidate_full_source_sha256']},
    'base_source_commit': '0d18820025e7a28675ed7112be0b1b4ff0110025',
    'old_outer_runner_sha256': raw['original_runner_sha256'],
    'old_outer_runner_bytes_unchanged': raw['old_runner_full_source_sha256'] == raw['original_runner_sha256'],
    'author_separation': 'Reviewer did not author or edit the runner adaptation; independently read exact diff, proved all other source bytes unchanged, and ran read-only positive/negative checks.',
    'discovery': 'The old accepted outer runner allows >86400s only for its two exact historical CPU queue identities. The unsubmitted c31/45fa new controller request125069s was correctly rejected by actual old remote verify() line292.',
    'minimal_diff_scope': 'Add exact ga-20261018-a2-fd-queue-v1 / seven-element module-supervise---config argv recognition; require absolute hash-bound config, matching config queue_id and snapshot.root, exact original absolute deadline, timeout <= config.max_queue_seconds. Only CPU CUDA_VISIBLE_DEVICES empty qualifies for existing172800s long queue cap.',
    'unchanged': ['all old deployed runner files', 'generic <=86400s worker cap', 'old two CPU queue recognition paths', 'independent supervisor/session/at-most-once/cleanup/ownership code', 'scientific child runner398768a732bbfebde4308486a0cedd924595157fda6cbff6a88f0bfaa512beff', 'all eleven scientific stage methods and GPU request budgets', 'original absolute deadline2026-10-19T05:40:30.012007+00:00'],
    'tests': {
        'status': raw['status'], 'cases': raw['case_count'],
        'old_unchanged_runner_rejects_new_long_controller': True,
        'new_exact_CPU_controller_accepts': True,
        'long_non_CPU_wrong_ID_wrong_module_extra_arg_rejected': True,
        'unbound_or_wrong_hash_config_rejected': True,
        'above_config_wall_changed_deadline_wrong_snapshot_wrong_queue_ID_rejected': True,
        'both_old_controller_requests_still_accept_old_and_new_verify': True,
        'actual_remote_unchanged_config_and_input_hash_checks': True,
        'config_mutation_negative_cases': 'Explicit in-memory read fixtures only; no remote file changed.',
        'execution_scope': raw['source_execution_scope'],
        'new_job_absent_before_after': True,
        'submission_called': False, 'GPU_compute_requested': False,
        'harness_note': 'First transport script had a placeholder replacement syntax error before execution. Only the independent harness was corrected; candidate source did not change. The complete16-case suite then passed; transport retains both invocation records.',
    },
    'limits': ['The exception only affects long CPU supervisor admission. Existing generic short-worker behavior is preserved; this is not a claim that all short non-CPU jobs are rejected.', 'A new immutable source/outer runner deployment, sidecar and request binding still require final review. This component PASS alone does not approve any concrete submission.', 'Future scientific results remain unapproved.'],
    'formal_submission_approved_by_this_receipt': False,
    'scientific_result_status': 'SCIENTIFIC_RESULT_NOT_YET_AVAILABLE',
    'historical_candidate_status': 'c31/45fa remains preserved unsubmitted and NOT_ADMITTED due to old outer-runner wall-cap mismatch.',
    'repair_budget_boundary': 'Same concentrated pre-submission engineering recovery; zero new formal quantizer attempts. Original task3/3 history is unchanged.',
    'evidence': [{'path': str(OUT / name), 'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)} for name in ['outer-request-readonly-verify-blocked.json', 'runner-adaptation-readonly-tests.json', 'check_runner_adaptation_readonly.py']],
    'dependencies_installed': [], 'global_config_changes': [],
    'GPT_UPLOAD_REQUIRED': 'YES',
}
target = OUT / 'runner-adaptation-review.json'
assert not target.exists()
target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(path=str(target),sha256=sha(target),status=value['status'],candidate_source_sha256=value['source']['sha256'])))
