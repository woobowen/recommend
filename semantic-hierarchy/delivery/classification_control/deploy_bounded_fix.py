"""Versioned, evidence-preserving replacement of the Toys consolidation worker."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
OLD = ROOT/'codex/sid_prefix_semantics_v1'
NEW = ROOT/'codex/sid_prefix_semantics_v3_bounded'
OUT = ROOT/'experiments/sid_prefix_semantics_v1'
CONTROL = OUT/'single_replica_control'
PYTHON = ROOT/'environments/llama_sid_eval/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    status = json.loads((CONTROL/'status.json').read_text())
    assert status['state'] == 'running'
    assert (status['current_dataset'], status['current_stage']) == ('toys', 'discover')
    assert not (OUT/'toys/taxonomy_draft.json').exists()
    assert not (OUT/'toys/draft_shard0of1.jsonl').exists()
    assert sha(NEW/'config.json') == sha(OLD/'config.json')
    assert sha(NEW/'inference.py') == sha(OLD/'inference.py')
    assert sha(NEW/'evaluate.py') == sha(OLD/'evaluate.py')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    archive = OUT/'runtime_revisions'/('bounded_convergence_'+stamp)
    archive.mkdir(parents=True, exist_ok=False)
    preserved = {str(p.relative_to(OUT)): sha(p) for p in (OUT/'beauty').iterdir()
                 if p.is_file()}
    write(archive/'beauty_files_before.json', preserved)
    write(archive/'status_before.json', status)
    shutil.copy2(CONTROL/'authorization_and_plan.json', archive/'authorization_and_plan_before.json')
    shutil.copytree(CONTROL/'source_snapshot_convergence_v2', archive/'source_snapshot_v2')
    shutil.copy2(OUT/'toys/discovery_candidates.json', archive/'discovery_candidates_before.json')
    # Only terminate exact processes belonging to the original controller and Toys worker.
    expected = [(status['controller_pid'], b'run_full_pipeline_convergent.py'),
                (status['current_pid'], b'run_stage.py\x00discover')]
    for pid, marker in expected:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
        assert marker in cmd, (pid, cmd)
        assert b'sid_prefix_semantics' in cmd
    for pid, _ in expected:
        os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        alive = []
        for pid, _ in expected:
            stat = Path(f'/proc/{pid}/stat')
            if stat.exists() and stat.read_text().split(') ')[1].split()[0] != 'Z':
                alive.append(pid)
        if not alive:
            break
        time.sleep(.5)
    else:
        raise RuntimeError('Old worker failed to terminate; no duplicate worker launched')
    lock = (CONTROL/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(CONTROL/'status.json', dict(status, state='maintenance', current_pid=None,
        maintenance_reason='User-authorized versioned convergence correction', archive=str(archive)))
    raw = OUT/'toys/raw/discover_shard0of1'
    if raw.exists():
        raw.rename(archive/'raw_discovery_and_old_merges')
    prior = OUT/'toys/discover_shard0of1.manifest.json'
    old_manifest_sha = sha(prior)
    prior.rename(archive/prior.name)
    batches = {p.name: sha(p) for p in sorted((OUT/'toys/discovery_batches').glob('*.json'))}
    assert len(batches) == 43
    shutil.copy2(OUT/'toys/reused_discovery_provenance.json', archive/'reused_discovery_provenance_before.json')
    write(OUT/'toys/reused_discovery_provenance.json', {
        'original_manifest_sha256': old_manifest_sha, 'archive': str(archive),
        'batch_sha256': batches, 'original_science_directory': str(OLD),
        'statement': 'Reuse all 1500 completed text-only discovery items; discard no candidates.'})
    prep = NEW/'preparation'
    prep.mkdir(exist_ok=True)
    for source in (OLD/'preparation').iterdir():
        if source.name == 'readiness.json':
            continue
        dest = prep/source.name
        if not dest.exists():
            dest.symlink_to(source, target_is_directory=source.is_dir())
    tests = []
    for script in [ROOT/'codex/sid_prefix_semantics_control/test_bounded_convergence.py', NEW/'selftest.py']:
        result = subprocess.run([str(PYTHON), '-B', str(script)], cwd=NEW, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        tests.append({'script': str(script), 'exit_code': result.returncode, 'output': result.stdout})
        print(result.stdout, flush=True)
        if result.returncode:
            write(archive/'failed_regressions.json', tests)
            raise RuntimeError('Regression failed before relaunch')
    ready = json.loads((OLD/'preparation/readiness.json').read_text())
    ready.update(state='USER_AUTHORIZED_BOUNDED_V3_READY',
        source_sha256={p.name: sha(p) for p in NEW.glob('*.py')},
        regression_tests_passed=True, previous_readiness_sha256=sha(OLD/'preparation/readiness.json'))
    write(prep/'readiness.json', ready)
    record = {'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'authorization': 'User explicitly requested solving the convergence problem.',
        'old_status_archive': str(archive), 'science_directory': str(NEW),
        'reason': 'Full-definition merging fails to shrink; unbounded name-only generation also repeated.',
        'method': 'Model first selects category count; schema requires exactly that many leaves; then bounded definitions.',
        'no_fixed_category_count': True, 'no_sid_metrics_used_for_taxonomy': True,
        'config_unchanged': True, 'model_and_precision_unchanged': True,
        'all_discovery_batches_reused': len(batches), 'beauty_files_sha256': preserved,
        'source_sha256': ready['source_sha256'], 'tests': tests,
        'limitation': 'Toys uses v3 model-selected-count consolidation; Beauty remains v1. '
                      'No claim of human annotation validation or identical taxonomy construction.'}
    write(archive/'patch_provenance.json', record)
    write(prep/'convergence_patch.json', record)
    for path, digest in preserved.items():
        assert sha(OUT/path) == digest, 'Beauty changed during deployment'
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    controller = ROOT/'codex/sid_prefix_semantics_control/run_full_pipeline_bounded.py'
    logpath = ROOT/'codex/sid_prefix_semantics_control/full_pipeline_bounded.log'
    with logpath.open('ab') as log:
        process = subprocess.Popen([str(PYTHON), '-B', '-u', str(controller), '--gpus', '0,1', '--execute'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    write(archive/'relaunch.json', {'controller_pid': process.pid, 'log': str(logpath)})
    print('RELAUNCHED', process.pid, 'ARCHIVE', archive, flush=True)


if __name__ == '__main__':
    main()
