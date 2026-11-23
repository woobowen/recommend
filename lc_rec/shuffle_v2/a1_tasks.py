"""Bounded A1 V2 runner workers for preflight and future single-mode generation."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from lc_rec.shuffle_v2 import core
from lc_rec.shuffle_v2.a1_replay import dump, load
from lc_rec.shuffle_v2.generate import bounded_resource_check, gpu_idle


def check_replay(directory):
    audit = load(directory / 'audit.json')
    single = load(directory / 'single-shard.json')
    assert audit['status'] == 'PASS_CPU_V2_FULL' and single['status'] == 'PASS_SINGLE_SHARD_CPU'
    assert audit['manifest_sha256'] == single['manifest_sha256']
    for asset in load(directory / 'assets.json'):
        assert asset['status'] == 'verified'
        p = Path(asset['path'])
        stat = p.stat()
        assert [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] == asset['stat_identity'], p
    return audit, single


def match_identity(cpu, output, selected):
    normal = {row['row_idx']: row for row in read_rows(cpu / 'normal-sample.jsonl')}
    observed = list(read_rows(output / 'predictions.rank0.jsonl'))
    assert list(row['row_idx'] for row in observed) == selected and set(normal) == set(selected)
    comparisons = []
    for row in observed:
        if row['shuffle']['input_changed']:
            continue
        frozen = normal[row['row_idx']]
        candidate_fields = ('sid', 'generated_token_ids', 'eos_position', 'generation_order', 'raw_decoded')
        old_candidates, new_candidates = frozen['candidates'], row['candidates']
        assert len(old_candidates) == len(new_candidates) == 20
        comparisons.append({'row_idx': row['row_idx'], 'input_exact': row['input_token_ids'] == frozen['input_token_ids'],
                            'generation_order_and_content_exact': all(
                                all(a[k] == b[k] for k in candidate_fields) for a, b in zip(old_candidates, new_candidates)),
                            'sorting_exact': row['sorted_generation_order'] == frozen['sorted_generation_order'],
                            'score_deltas': [b['sequence_score'] - a['sequence_score']
                                             for a, b in zip(old_candidates, new_candidates)]})
    return comparisons


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def monitor(mode, output, argv, deadline, uuid):
    started = time.monotonic()
    with (output / f'{mode}.log').open('x') as log, (output / f'{mode}.resources.jsonl').open('x') as resources:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        while proc.poll() is None:
            now = time.monotonic()
            observed = subprocess.check_output(
                ['nvidia-smi', '-i', uuid, '--query-gpu=uuid,memory.used,utilization.gpu',
                 '--format=csv,noheader,nounits'], text=True).strip()
            resources.write(json.dumps({'seconds': now - started, 'gpu': observed, **bounded_resource_check(output)}) + '\n')
            resources.flush()
            if now >= deadline:
                raise TimeoutError('The independent runner verifies and cleans up owned children on exit')
            try:
                proc.wait(timeout=min(5, deadline - now))
            except subprocess.TimeoutExpired:
                pass
    record = {'seconds': time.monotonic() - started, 'exit_code': proc.returncode}
    dump(output / f'{mode}.exit.json', record)
    assert record['exit_code'] == 0, record
    return record


def generate(args):
    config = load(args.config)
    job_dir = Path(os.environ['GRID_JOB_DIR'])
    request = load(job_dir / 'request.json')
    job_id = os.environ['GRID_JOB_ID']
    preflight = args.mode == 'preflight'
    expected_prefix = 'a1-instruments-v2-' + ('gpu-preflight-' if preflight else 'formal-gpu-')
    assert job_id.startswith(expected_prefix) and job_dir.name == job_id
    assert request['capture_version'] == args.run_commit and request['cwd'] == str(Path.cwd())
    assert request['timeout_seconds'] >= args.seconds + 15
    if preflight:
        assert request['timeout_seconds'] + 10 <= config['budget']['preflight_device_seconds']
    else:
        assert request['timeout_seconds'] <= config['budget']['formal_attempt_wall_seconds']
    assert sys.executable == config['baseline']['python']
    for key, value in config['gpu_environment'].items():
        assert request['env'][key] == os.environ[key] == value, key
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    _, single = check_replay(args.cpu_manifest)
    began = time.monotonic()
    deadline = began + args.seconds
    stages = {}
    with (output.parent / '.gpu.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = gpu_idle(config)
        for mode in core.MODES if preflight else (args.mode,):
            assert gpu_idle(config) == device
            test = output / (mode if preflight else 'test')
            remaining = math.floor(deadline - time.monotonic()) - 12
            assert remaining > 30
            argv = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node=1', '--module', 'lc_rec.shuffle_v2.a1_generate',
                    '--config', str(args.config), '--cpu-manifest', str(args.cpu_manifest), '--mode', mode,
                    '--output', str(test), '--run-commit', args.run_commit, '--max-seconds', str(remaining)]
            if preflight:
                argv.append('--preflight')
            dump(output / f'{mode}.request.json', {'argv': argv, 'mode': core.mode_metadata(mode),
                                                   'device': device, 'run_commit': args.run_commit})
            result = monitor(mode, output, argv, deadline, device['uuid'])
            rank = load(test / 'rank0.json')
            assert rank['status'] == ('PREFLIGHT_INTERFACE_PASS' if preflight else 'PASS')
            assert (rank['world_size'], rank['rank'], rank['local_rank']) == (1, 0, 0)
            assert rank['completed_users'] == (32 if preflight else config['users'])
            assert rank['assignment_sha256'] == single['preflight_assignment_sha256' if preflight else 'assignment_sha256']
            assert rank['RUN_COMMIT'] == args.run_commit and rank['shuffle']['mode'] == mode
            if preflight:
                result['natural_identity'] = match_identity(args.cpu_manifest, test, single['preflight_assignment'])
            result['model_load_seconds'] = rank['model_load_and_strict_verification_seconds']
            result['generation_seconds'] = rank['elapsed_generation_seconds']
            result['gpu_peak_allocated_bytes'] = rank['gpu_peak_allocated_bytes']
            result['gpu_peak_reserved_bytes'] = rank['gpu_peak_reserved_bytes']
            result['users'] = rank['completed_users']
            stages[mode] = result
        gpu_idle(config)
    discrepancies = [{'mode': mode, **row} for mode, stage in stages.items() for row in stage.get('natural_identity', ())
                     if not (row['input_exact'] and row['generation_order_and_content_exact']
                             and row['sorting_exact'] and all(x == 0 for x in row['score_deltas']))]
    dump(output / 'summary.json', {'status': 'IDENTITY_DISCREPANCY' if discrepancies else 'PASS',
                                  'job_id': job_id, 'run_commit': args.run_commit, 'device': device, 'stages': stages,
                                  'seconds': time.monotonic() - began, 'identity_discrepancies': discrepancies})
    assert not discrepancies, 'Natural identity must match frozen Normal; inspect summary.json'


def lifecycle(case, output):
    output.mkdir(parents=True, exist_ok=False)
    child = subprocess.Popen([sys.executable, '-c',
                              'import os,signal,sys,time; from pathlib import Path; '
                              'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                              'Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(180)',
                              str(output / 'child.pid')], start_new_session=True)
    if case == 'clean':
        # Also prove that own marked child reaping produces a normal zero exit.
        path = output / 'child.pid'
        deadline = time.monotonic() + 5
        while not path.is_file() and time.monotonic() < deadline:
            time.sleep(.05)
        assert int(path.read_text()) == child.pid
        child.send_signal(signal.SIGKILL)
        assert child.wait(timeout=5) == -signal.SIGKILL
    else:
        child.wait()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subs = parser.add_subparsers(dest='task', required=True)
    gpu = subs.add_parser('gpu')
    gpu.add_argument('--config', required=True, type=Path)
    gpu.add_argument('--cpu-manifest', required=True, type=Path)
    gpu.add_argument('--output', required=True, type=Path)
    gpu.add_argument('--mode', choices=(*core.MODES, 'preflight'), required=True)
    gpu.add_argument('--run-commit', required=True)
    gpu.add_argument('--seconds', required=True, type=int)
    life = subs.add_parser('lifecycle')
    life.add_argument('--case', choices=('clean', 'timeout'), required=True)
    life.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.task == 'gpu':
        generate(args)
    else:
        lifecycle(args.case, args.output)


if __name__ == '__main__':
    main()
