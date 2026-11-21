"""Run only approved A0 V2 modes inside the existing job supervisor."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from lc_rec.shuffle_v2 import core
from lc_rec.shuffle_v2.replay import CONFIG, dump, frozen


def gpu_inventory(config):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    observed = [list(map(str.strip, row.split(','))) for row in rows]
    assert [(i, uuid) for i, uuid, _ in observed] == list(zip(('0', '1'), config['gpus'], strict=True))
    allocations = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                          '--format=csv,noheader'], text=True)
    occupied = {row.split(',')[0].strip() for row in allocations.splitlines() if row.strip()}
    return [{'physical_index': int(i), 'uuid': uuid, 'memory_used_mib': int(used), 'has_compute_process': uuid in occupied}
            for i, uuid, used in observed]


def gpu_idle(config):
    selected = os.environ['CUDA_VISIBLE_DEVICES']
    matches = [card for card in gpu_inventory(config) if card['uuid'] == selected]
    assert len(matches) == 1
    card = matches[0]
    if card['has_compute_process'] or card['memory_used_mib'] >= 100:
        raise RuntimeError(f'Chosen GPU is occupied: {card}')
    return {**card, 'visible_device': 0, 'world_size': 1}


def bounded_resource_check(output):
    data = os.statvfs(output)
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    free = data.f_bavail * data.f_frsize
    assert free >= 20 * 1024**3 and available >= 40 * 1024**3
    return {'disk_free_bytes': free, 'host_mem_available_bytes': available}


def argv_for(mode, out, audit, single, commit, seconds, preflight, base, config_path=CONFIG, beauty_config=None):
    argv = [base['python'], '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
            '--module', 'lc_rec.shuffle_v2.entry', '--model-key', 'a0', '--shuffle-mode', mode,
            '--cpu-audit', str(audit), '--single-shard-audit', str(single), '--output', str(out), '--run-commit', commit,
            '--max-seconds', str(seconds)]
    if beauty_config:
        argv = [*argv[:7], 'lc_rec.shuffle_v2.beauty', '--variant', 'letter', '--source', base['entry'],
                '--config', str(config_path), '--shuffle-mode', mode,
                '--cpu-audit', str(audit), '--single-shard-audit', str(single),
                '--output', str(out), '--v2-run-commit', commit, '--max-seconds', str(seconds)]
    for key in ('source-root', 'data-root', 'model', 'adapter'):
        argv.extend(('--' + key, base[key]))
    if preflight:
        argv.extend(('--preflight-users-per-rank', '32'))
        if beauty_config:
            argv.extend(('--preflight-manifest', beauty_config['frozen_preflight']))
    return argv


def validate_preflight(config, output, single):
    wanted = set(single['preflight_assignment'])
    normal = {}
    for rank in range(2):
        path = Path(config['baseline']['normal_test_dir']) / f'predictions.rank{rank}.jsonl'
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['row_idx'] in wanted:
                    normal[row['row_idx']] = row
    assert set(normal) == wanted
    results = {}
    for mode in core.MODES:
        identity_users = []
        with (output / mode / 'predictions.rank0.jsonl').open() as stream:
            for index, line in zip(single['preflight_assignment'], stream, strict=True):
                row = json.loads(line)
                intervention = row['shuffle']
                assert index == row['row_idx'] and intervention['actual_prefill']['lookup_exact']
                assert all(candidate['eos_position'] == 4 for candidate in row['candidates'])
                if intervention['input_changed']:
                    continue
                old = normal[index]
                assert row['input_token_ids'] == old['input_token_ids']
                assert row['sorted_generation_order'] == old['sorted_generation_order']
                maximum = 0.0
                for left, right in zip(row['candidates'], old['candidates'], strict=True):
                    assert left['sid'] == right['sid'] and left['generated_token_ids'] == right['generated_token_ids']
                    a, b = left['sequence_score'], right['sequence_score']
                    maximum = max(maximum, abs(a - b))
                    assert abs(a - b) <= 1e-5 + 1e-5 * abs(b)
                identity_users.append({'row_idx': index, 'rank': 0, 'maximum_score_delta': maximum})
        results[mode] = {'users': 32, 'natural_unchanged_vs_archived_normal': identity_users}
    return results


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-audit', type=Path, required=True)
    parser.add_argument('--single-shard-audit', type=Path, required=True)
    parser.add_argument('--run-commit', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--mode', choices=(*core.MODES, 'preflight'), required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    is_beauty = config.get('dataset') == 'Beauty'
    job_id = os.environ['GRID_JOB_ID']
    job = Path(os.environ['GRID_JOB_DIR'])
    request = json.loads((job / 'request.json').read_text())
    assert job.name == job_id and request['capture_version'] == args.run_commit
    assert request['cwd'] == str(Path.cwd()) and args.seconds + 30 <= request['timeout_seconds']
    preflight = args.mode == 'preflight'
    if preflight and 'gpu_validation_attempt_seconds' in config['budget']:
        assert request['timeout_seconds'] <= config['budget']['gpu_validation_attempt_seconds']
    else:
        cap = config['budget']['preflight_device_seconds'] if preflight else config['budget']['formal_attempt_wall_seconds']
        assert request['timeout_seconds'] + 15 <= cap
    if is_beauty and 'gpu_environment' in config:
        assert all(os.environ.get(key) == request['env'][key] == value
                   for key, value in config['gpu_environment'].items())
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    cpu = json.loads(args.cpu_audit.read_text())
    assert cpu['status'] == 'PASS_CPU_V2_FULL'
    single = json.loads(args.single_shard_audit.read_text())
    assert single['status'] == 'PASS_SINGLE_SHARD_CPU' and single['manifest_sha256'] == cpu['manifest_sha256']
    assets_file = args.cpu_audit.with_name('assets.json')
    for asset in json.loads(assets_file.read_text()):
        p = Path(asset['path'])
        stat = p.stat()
        assert [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] == asset['stat_identity'], p
    begin = time.monotonic()
    results = []
    with (output.parent / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        devices = gpu_idle(config)
        modes = tuple(core.MODES) if preflight else (args.mode,)
        for mode in modes:
            remaining = math.floor(args.seconds - (time.monotonic() - begin))
            assert remaining > 60
            out = output / mode if preflight else output / 'test'
            argv = argv_for(mode, out, args.cpu_audit, args.single_shard_audit, args.run_commit, remaining - 15, preflight, config['baseline'], args.config, config if is_beauty else None)
            plan = {'status': 'READY', 'job_id': job_id, 'RUN_COMMIT': args.run_commit,
                    'mode': core.mode_metadata(mode), 'argv': argv, 'cuda_devices': devices,
                    'preflight': preflight, 'resources': bounded_resource_check(output)}
            dump(output / f'{mode}.request.json', plan)
            started = time.monotonic()
            with (output / f'{mode}.log').open('x') as log, (output / f'{mode}.resources.jsonl').open('x') as resources:
                proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                while proc.poll() is None:
                    tick = time.monotonic()
                    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
                                                   '--format=csv'], text=True)
                    resources.write(json.dumps({'elapsed': tick - started, 'gpu': gpu, **bounded_resource_check(output)}) + '\n')
                    resources.flush()
                    if tick >= begin + args.seconds:
                        raise TimeoutError('Runner owns process-group cleanup after bounded task expiration')
                    try:
                        proc.wait(timeout=5 if preflight else 30)
                    except subprocess.TimeoutExpired:
                        continue
                code = proc.returncode
            result = {'exit_code': code, 'seconds': time.monotonic() - started, 'output': str(out), **plan}
            dump(output / f'{mode}.exit.json', result)
            assert code == 0, mode
            report = json.loads((out / 'rank0.json').read_text())
            assert report['status'] == ('PREFLIGHT_INTERFACE_PASS' if preflight else 'PASS')
            assert report['completed_users'] == (32 if preflight else config['users'])
            assert report['world_size'] == 1 and report['rank'] == report['local_rank'] == 0
            assert report['assignment_sha256'] == single['preflight_assignment_sha256' if preflight and is_beauty else 'assignment_sha256']
            assert report['shuffle']['mode'] == mode and report['shuffle']['protocol'] == core.PROTOCOL
            assert report['RUN_COMMIT'] == args.run_commit
            results.append(result)
            gpu_idle(config)
    checks = validate_preflight(config, output, single) if preflight else {}
    if is_beauty and preflight:
        reports = {mode: json.loads((output / mode / 'rank0.json').read_text()) for mode in core.MODES}
        for mode, report in reports.items():
            per_user = report['elapsed_generation_seconds'] / 32
            checks[mode]['estimate_seconds'] = report['model_load_and_strict_verification_seconds'] + per_user * config['users']
            checks[mode]['preflight_elapsed_generation_seconds'] = report['elapsed_generation_seconds']
        checks['total_estimate_seconds'] = sum(checks[mode]['estimate_seconds'] for mode in core.MODES)
    dump(output / 'summary.json', {'status': 'PASS_V2_PREFLIGHT' if preflight else 'PASS_V2_GENERATION',
                                  'job_id': job_id, 'RUN_COMMIT': args.run_commit, 'stages': results,
                                  'preflight_checks': checks, 'seconds': time.monotonic() - begin})


if __name__ == '__main__':
    main()
