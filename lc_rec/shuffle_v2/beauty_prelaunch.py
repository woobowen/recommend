"""Bind two read-only reviewers and actual server receipts to a formal launch."""
import argparse
import hashlib
import json
from pathlib import Path

from lc_rec.shuffle_v2 import beauty, beauty_control, control, core
from lc_rec.shuffle_v2.replay import dump


def receive(connection, job, relative):
    root = beauty_control.ROOT
    return json.loads(control.remote_python(connection, f"""
from pathlib import Path
import hashlib,json
path=Path({f'{root}/{job}/{relative}'!r})
data=path.read_bytes()
print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),content=json.loads(data))))
"""))


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--commit', required=True)
    p.add_argument('--replay', required=True)
    p.add_argument('--preflight', required=True)
    p.add_argument('--normal', required=True)
    p.add_argument('--independent-normal', required=True)
    p.add_argument('--reviewer-receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interface', required=True)
    p.add_argument('--lifecycle', type=Path, required=True)
    p.add_argument('--assessment', type=Path, required=True)
    p.add_argument('--execution-review', type=Path, required=True)
    args = p.parse_args()
    connection = beauty_control.connect(beauty_control.runner_sha_at(args.commit))
    config = json.loads(beauty.CONFIG.read_text())
    limits = config['budget']
    states = control.job_states(connection, beauty_control.PREFIX)
    for job in (args.replay, args.preflight, args.normal, args.independent_normal, args.interface):
        beauty_control.require_success(states, job)
    replay = receive(connection, args.replay, 'audit.json')
    gpu = receive(connection, args.preflight, 'summary.json')
    normal = receive(connection, args.normal, 'summary.json')
    independent = receive(connection, args.independent_normal, '')
    assert replay['content']['status'] == 'PASS_CPU_V2_FULL' and replay['content']['dataset'] == 'Beauty'
    assert gpu['content']['RUN_COMMIT'] == args.commit and gpu['content']['status'] == 'PASS_V2_PREFLIGHT'
    assert normal['content']['frozen_normal_cpu_reproduction']['matrix_exact']
    assert independent['content']['status'] == 'PASS_INDEPENDENT_BEAUTY_NORMAL_REVIEW'
    review = json.loads(args.reviewer_receipt.read_text())
    assert review['status'] == 'PASS' and review['RUN_COMMIT'] == args.commit
    assert review['preflight_summary_sha256'] == gpu['sha256'] and review['replay_audit_sha256'] == replay['sha256']
    interface = receive(connection, args.interface, 'summary.json')
    assert interface['content']['status'] == 'PASS_BEAUTY_PREFLIGHT_EVALUATION_INTERFACE'
    assert interface['content']['RUN_COMMIT'] == args.commit and not interface['content']['formal_results']
    lifecycle = json.loads(args.lifecycle.read_text())
    assert lifecycle['status'] == 'PASS_BEAUTY_RUNNER_LIFECYCLE'
    assert lifecycle['runner_sha256'] == connection.profile['runner_sha256']
    assessment = json.loads(args.assessment.read_text())
    assert assessment['status'] == 'BEAUTY_PREFLIGHT_TIMING_ESTIMATE' and assessment['run_commit'] == args.commit
    estimates = {mode: assessment['modes'][mode]['formal_wall_projection_seconds'] for mode in core.MODES}
    assert assessment['fits_formal_attempt_under_sensitivity'] and assessment['fits_formal_total_under_sensitivity']
    assert args.execution_review.stat().st_size > 0
    assert max(estimates.values()) < limits['formal_attempt_wall_seconds']
    assert sum(estimates.values()) < limits['formal_total_device_seconds']
    cards = {row['cuda_devices']['uuid'] for row in gpu['content']['stages']}
    assert len(cards) == 1
    result = {'status': 'PASS', 'RUN_COMMIT': args.commit,
              'preflight_job_id': beauty_control.PREFIX + args.preflight,
              'preflight_summary_sha256': gpu['sha256'], 'replay_audit_sha256': replay['sha256'],
              'normal_summary_sha256': normal['sha256'], 'independent_normal_sha256': independent['sha256'],
              'nonauthor': review, 'nonauthor_receipt_sha256': hashlib.sha256(args.reviewer_receipt.read_bytes()).hexdigest(),
              'runner_sha256': connection.profile['runner_sha256'],
              'interface_sha256': interface['sha256'],
              'lifecycle_sha256': hashlib.sha256(args.lifecycle.read_bytes()).hexdigest(),
              'assessment_sha256': hashlib.sha256(args.assessment.read_bytes()).hexdigest(),
              'execution_review_sha256': hashlib.sha256(args.execution_review.read_bytes()).hexdigest(),
              'estimate_seconds': estimates, 'device_uuid': cards.pop(), 'protocol': core.PROTOCOL}
    dump(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
