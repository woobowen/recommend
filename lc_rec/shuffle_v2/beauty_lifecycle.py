"""Small, CPU-only fixtures and receipt verification for the Beauty supervisor."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_identity(pid):
    record = Path('/proc') / str(pid)
    fields = (record / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': fields[19], 'pgid': int(fields[2]), 'session_id': int(fields[3]),
            'uid': record.stat().st_uid, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def child(output, ignore_term):
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    write(output / 'child.json', {**process_identity(os.getpid()),
                                  'job_id': os.environ['GRID_JOB_ID'], 'job_dir': os.environ['GRID_JOB_DIR'],
                                  'ignores_term': ignore_term})
    time.sleep(40)


def worker(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ['GRID_JOB_ID'].startswith('letter-beauty-v2-cpu-lifecycle-')
    args.output.mkdir(parents=True, exist_ok=False)
    parent = process_identity(os.getpid())
    write(args.output / 'worker.json', parent)
    if args.case == 'normal':
        time.sleep(0.5)
        return
    if args.case == 'decoy':
        time.sleep(30)
        write(args.output / 'survived.json', process_identity(os.getpid()))
        return
    argv = [sys.executable, '-B', '-m', 'lc_rec.shuffle_v2.beauty_lifecycle',
            '--output', str(args.output), '--case', 'child']
    if args.case == 'timeout':
        argv.append('--ignore-term')
    running = subprocess.Popen(argv, start_new_session=True)
    until = time.monotonic() + 4
    while not (args.output / 'child.json').is_file():
        assert running.poll() is None and time.monotonic() < until
        time.sleep(0.05)
    while args.case == 'timeout':
        time.sleep(1)
    time.sleep(0.5)


def review(job_root, output_root, attempt, runner_sha):
    reports = {}
    for case, terminal in {'normal': 'SUCCEEDED', 'orphan': 'WORKERS_TERMINATED', 'timeout': 'TIMED_OUT',
                           'decoy': 'SUCCEEDED'}.items():
        name = f'cpu-lifecycle-{case}-{attempt}'
        job = job_root / ('letter-beauty-v2-' + name)
        output = output_root / name
        status = json.loads((job / 'status.json').read_text())
        accepted = json.loads((job / 'accepted.json').read_text())
        cleanup = json.loads((job / 'cleanup.json').read_text())
        assert accepted['runner_sha256'] == runner_sha
        assert status['status'] == terminal and status['cleanup'] == cleanup
        assert cleanup['verified'] and cleanup['last_observation']['verified_empty']
        events = [event for event in cleanup['events'] if event['outcome'] == 'SENT']
        report = {'status': terminal, 'finish': status['ended_at'], 'start': status['started_at'],
                  'exit_code': status['exit_code'], 'cleanup_reason': cleanup['reason'], 'signals': events}
        if case in ('orphan', 'timeout'):
            marker = json.loads((output / 'child.json').read_text())
            parent = json.loads((output / 'worker.json').read_text())
            assert marker['job_id'] == job.name and marker['job_dir'] == str(job)
            assert parent['session_id'] != marker['session_id'] == marker['pgid'] == marker['pid']
            tagged = [event for event in events if event['member']['pid'] == marker['pid']]
            assert tagged and {key: tagged[0]['member'][key] for key in parent} == {key: marker[key] for key in parent}
            assert tagged[0]['member']['proof'] == 'BEAUTY_MARKED_WORKER_SESSION'
            assert signal.SIGTERM in {event['signal'] for event in tagged}
            if case == 'timeout':
                assert signal.SIGKILL in {event['signal'] for event in tagged}
            report['child'] = marker
        else:
            assert not events and status['exit_code'] == 0
        if case == 'decoy':
            report['decoy'] = json.loads((output / 'survived.json').read_text())
        reports[case] = report
    decoy = reports['decoy']
    for case in ('orphan', 'timeout'):
        assert decoy['start'] < reports[case]['start'] < reports[case]['finish'] < decoy['finish']
        assert all(event['member']['pid'] != decoy['decoy']['pid'] for event in reports[case]['signals'])
    return {'status': 'PASS_BEAUTY_RUNNER_LIFECYCLE', 'runner_sha256': runner_sha, 'cases': reports}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--case', choices=('normal', 'orphan', 'timeout', 'decoy', 'child'))
    parser.add_argument('--ignore-term', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--attempt')
    parser.add_argument('--runner-sha')
    args = parser.parse_args()
    if args.case == 'child':
        child(args.output, args.ignore_term)
    elif args.case:
        worker(args)
    else:
        write(args.output, review(args.job_root, args.output_root, args.attempt, args.runner_sha))


if __name__ == '__main__':
    main()
