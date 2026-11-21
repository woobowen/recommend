"""Drive the four approved V2 jobs through the existing server runner."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from lc_rec.shuffle_v2.core import MODES
from lc_rec.shuffle_v2.replay import CONFIG


def atomic(path, value):
    staged = path.with_suffix('.partial')
    staged.write_text(json.dumps(value, indent=2) + '\n')
    staged.replace(path)


def elapsed(records, prefix):
    total = 0.0
    for item in records.glob(prefix + '*'):
        status_file = item / 'status.json'
        if not status_file.is_file():
            continue
        r = json.loads(status_file.read_text())
        if not r.get('ended_at'):
            raise RuntimeError(f'Active sibling job prevents a new submission: {item.name}')
        total += (datetime.fromisoformat(r['ended_at']) - datetime.fromisoformat(r.get('started_at', r['received_at']))).total_seconds()
    return total


def success(state):
    return (state['status'] == 'SUCCEEDED' and state['exit_code'] == 0 and state['cleanup_verified']
            and not state['worker_alive'] and not state['workgroup_live'])


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    config = json.loads(Path(plan.get('config', CONFIG)).read_text())
    jobs = Path(plan['job_root'])
    requests = Path(plan['requests_root'])
    runner = Path(plan['runner'])
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == plan['runner_sha256']
    assert list(plan['modes']) == list(MODES)
    assert plan['formal_run_commit'] == plan['preflight_review']['RUN_COMMIT']
    assert plan['preflight_review']['status'] == 'PASS'
    os.environ.pop('GRID_JOB_ID', None)
    os.environ.pop('GRID_JOB_DIR', None)
    a.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'modes': {}, 'formal_run_commit': plan['formal_run_commit'], 'stages': []}

    def check(*argv):
        return json.loads(subprocess.check_output(['/usr/bin/python3', '-I', '-S', '-B', str(runner), *map(str, argv)],
                                                  text=True, timeout=45))

    def execute(request, category, expected_status):
        job_id = request['job_id']
        job = jobs / job_id
        request_path = requests / (job_id + '.json')
        if (job / 'request.json').exists():
            assert json.loads((job / 'request.json').read_text()) == request, 'Immutable child request mismatch'
        if (job / 'accepted.json').exists():
            state = check('query', job)
        else:
            kind = 'gpu-formal-' if category == 'gpu' else 'cpu-'
            budget = config['budget']['formal_total_wall_seconds' if category == 'gpu' else 'cpu_total_wall_seconds']
            if category == 'cpu':
                assert request['timeout_seconds'] <= config['budget']['cpu_attempt_wall_seconds']
            if category == 'gpu':
                budget = min(budget, config['budget']['formal_total_device_seconds'])
                assert request['timeout_seconds'] + 15 <= config['budget']['formal_attempt_wall_seconds']
                assert request['env']['CUDA_VISIBLE_DEVICES'] in config['gpus']
            assert elapsed(jobs, plan['prefix'] + kind) + request['timeout_seconds'] + 15 <= budget
            if request_path.exists():
                assert json.loads(request_path.read_text()) == request, 'Staged child request differs'
            else:
                with request_path.open('x') as stream:
                    json.dump(request, stream, indent=2)
                    stream.write('\n')
            state = check('submit', jobs, request_path)
        stage = {'job_id': job_id, 'request_path': str(request_path), 'state': state}
        result['stages'].append(stage)
        atomic(a.output / 'summary.json', result)
        while state['status'] in ('ACCEPTED', 'RUNNING'):
            time.sleep(30)
            state = check('query', job)
            stage['state'] = state
            atomic(a.output / 'summary.json', result)
        assert success(state), state
        destination = Path(request['argv'][request['argv'].index('--output') + 1])
        summary = json.loads((destination / 'summary.json').read_text())
        assert summary['status'] == expected_status, summary['status']
        stage['summary'] = str(destination / 'summary.json')
        atomic(a.output / 'summary.json', result)
        return stage['job_id']

    try:
        preflight = check('query', jobs / plan['preflight_review']['preflight_job_id'])
        assert success(preflight)
        assert success(check('query', jobs / plan['normal_job_id']))
        for mode, stages in plan['modes'].items():
            gpu, cpu = stages['gpu'], stages['cpu']
            assert gpu['job_id'].startswith(plan['prefix'] + 'gpu-formal-' + mode + '-')
            assert cpu['job_id'].startswith(plan['prefix'] + 'cpu-eval-' + mode + '-')
            assert gpu['capture_version'] == cpu['capture_version'] == plan['formal_run_commit']
            generated = execute(gpu, 'gpu', 'PASS_V2_GENERATION')
            evaluated = execute(cpu, 'cpu', 'PASS_V2_EVALUATION')
            result['modes'][mode] = {'generation': generated, 'evaluation': evaluated}
            atomic(a.output / 'summary.json', result)
        execute(plan['paired'], 'cpu', 'PASS_V2_PAIRED')
        result['status'] = 'PASS_V2_GROUP_COMPUTE'
    except BaseException as err:
        result['status'] = 'FAILED_GATE_OR_CHILD'
        result['error'] = repr(err)
        raise
    finally:
        atomic(a.output / 'summary.json', result)


if __name__ == '__main__':
    main()
