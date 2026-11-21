"""Reuse complete v6 mappings and losslessly compact the classifier table."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
OLD = ROOT/'codex/sid_prefix_semantics_v1'
NEW = ROOT/'codex/sid_prefix_semantics_v7_compact'
OUT = ROOT/'experiments/sid_prefix_semantics_v1'
CONTROL = OUT/'single_replica_control'
PYTHON = ROOT/'environments/llama_sid_eval/bin/python'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2))


def main():
    s = json.loads((CONTROL/'status.json').read_text())
    assert s['science_version'] == 'sid_prefix_semantics_v6_local'
    assert s['current_stage'] == 'discover'
    assert not (OUT/'toys/taxonomy_draft.json').exists()
    complete = list((OUT/'toys/local_consolidation_v6').rglob('pass1_complete.json'))
    assert len(complete) == 1, 'Do not interrupt incomplete mapping passes'
    assert json.loads(complete[0].read_text())['every_source_mapped_once']
    smoke = ROOT/'codex/sid_prefix_semantics_control/compact_classifier_smoke_verified.json'
    assert json.loads(smoke.read_text())['passed']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    archive = OUT/'runtime_revisions'/('compact_table_'+stamp)
    archive.mkdir(exist_ok=False)
    write(archive/'status_before.json', s)
    for pid, marker in [(s.get('controller_pid'), b'run_full_pipeline_local.py'),
                        (s.get('current_pid'), b'run_stage.py')]:
        path = Path(f'/proc/{pid}/cmdline')
        if pid and path.exists() and path.read_bytes():
            assert marker in path.read_bytes()
            os.kill(pid, signal.SIGTERM)
    time.sleep(1)
    lock = (CONTROL/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = OUT/'toys/discover_shard0of1.manifest.json'
    shutil.move(str(prior), str(archive/prior.name))
    reuse = OUT/'toys/reused_discovery_provenance.json'
    old_reuse = sha(reuse)
    shutil.copy2(reuse, archive/'reused_discovery_before.json')
    caches = {str(p.relative_to(OUT/'toys')): sha(p)
              for p in (OUT/'toys/local_consolidation_v6').rglob('*.json')}
    write(reuse, {'original_manifest_sha256': sha(archive/prior.name),
        'previous_reuse_sha256': old_reuse, 'archive': str(archive),
        'batch_sha256': {p.name: sha(p) for p in (OUT/'toys/discovery_batches').glob('*.json')},
        'normalization_checkpoint_sha256': caches,
        'statement': 'Reuse both completed v6 mappings; only classifier table serialization changes.'})
    assert sha(NEW/'config.json') == sha(OLD/'config.json')
    assert sha(NEW/'evaluate.py') == sha(OLD/'evaluate.py')
    prep = NEW/'preparation';prep.mkdir(exist_ok=True)
    for f in (OLD/'preparation').iterdir():
        if f.name != 'readiness.json' and not (prep/f.name).exists():
            (prep/f.name).symlink_to(f, target_is_directory=f.is_dir())
    tests = []
    for script in [ROOT/'codex/sid_prefix_semantics_control/test_compact_table.py', NEW/'selftest.py']:
        result = subprocess.run([str(PYTHON), '-B', str(script)], cwd=NEW,
            env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout, flush=True)
        tests.append({'script': str(script), 'exit_code': result.returncode, 'output': result.stdout})
        assert result.returncode == 0
    beauty_before = sorted((OUT/'runtime_revisions').glob('compact_convergence_*/beauty_files_before.json'))[0]
    for path, digest in json.loads(beauty_before.read_text()).items():
        assert sha(OUT/path) == digest, path
    ready = json.loads((OLD/'preparation/readiness.json').read_text())
    ready.update(state='USER_AUTHORIZED_COMPACT_V7_READY',
        source_sha256={p.name: sha(p) for p in NEW.glob('*.py')}, regression_tests_passed=True)
    write(prep/'readiness.json', ready)
    record = {'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'authorization': 'User requested resolving convergence, including necessary capacity corrections.',
        'normalization': 'Complete finite local keyed source mappings from v6; no taxonomy regeneration.',
        'classifier_table': 'Header plus rows, all category names/parents/definitions/nonempty boundaries retained.',
        'all_normalization_caches_sha256': caches, 'beauty_verified_unchanged': True,
        'config_model_sid_metrics_unchanged': True, 'tests': tests,
        'source_sha256': ready['source_sha256'],
        'protocol_limitation': 'Toys taxonomy construction differs from Beauty v1; no human accuracy claim.'}
    write(prep/'convergence_patch.json', record);write(archive/'patch_provenance.json', record)
    write(CONTROL/'status.json', dict(s, state='maintenance', current_pid=None,
        maintenance_reason='Lossless classifier table compression', repair_archive=str(archive)))
    fcntl.flock(lock, fcntl.LOCK_UN);lock.close()
    script = ROOT/'codex/sid_prefix_semantics_control/run_full_pipeline_compact.py'
    logpath = ROOT/'codex/sid_prefix_semantics_control/full_pipeline_compact.log'
    with logpath.open('ab') as log:
        process = subprocess.Popen([str(PYTHON), '-B', '-u', str(script), '--gpus', '0,1', '--execute'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(archive/'relaunch.json', {'pid': process.pid, 'log': str(logpath)})
    print('COMPACT_V7_RELAUNCHED', process.pid, flush=True)


if __name__ == '__main__':
    main()
