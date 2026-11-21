import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
OLD = ROOT/'codex/sid_prefix_semantics_v1'
NEW = ROOT/'codex/sid_prefix_semantics_v4_coverage'
OUT = ROOT/'experiments/sid_prefix_semantics_v1'
CONTROL = OUT/'single_replica_control'
PYTHON = ROOT/'environments/llama_sid_eval/bin/python'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2))


def main():
    status = json.loads((CONTROL/'status.json').read_text())
    assert status['state'] == 'maintenance'
    assert not (OUT/'toys/taxonomy_draft.json').exists()
    lock = (CONTROL/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    archive = Path(status['repair_archive'])
    receipt = json.loads((archive/'discover_shard0of1.manifest.json').read_text())
    write(archive/'coverage_v4_deployment_status.json', status)
    reuse = OUT/'toys/reused_discovery_provenance.json'
    old_reuse_sha = sha(reuse)
    shutil.copy2(reuse, archive/'reused_discovery_before_v4.json')
    batches = {p.name: sha(p) for p in (OUT/'toys/discovery_batches').glob('*.json')}
    assert len(batches) == 43
    write(reuse, {'original_manifest_sha256': sha(archive/'discover_shard0of1.manifest.json'),
        'previous_reuse_sha256': old_reuse_sha, 'archive': str(archive),
        'batch_sha256': batches, 'statement': 'All completed v1 discovery items reused; v3 incomplete taxonomy rejected.'})
    assert sha(NEW/'config.json') == sha(OLD/'config.json')
    assert sha(NEW/'evaluate.py') == sha(OLD/'evaluate.py')
    prep = NEW/'preparation';prep.mkdir(exist_ok=True)
    for f in (OLD/'preparation').iterdir():
        if f.name != 'readiness.json' and not (prep/f.name).exists():
            (prep/f.name).symlink_to(f, target_is_directory=f.is_dir())
    tests = []
    for script in [ROOT/'codex/sid_prefix_semantics_control/test_coverage_convergence.py', NEW/'selftest.py']:
        run = subprocess.run([str(PYTHON), '-B', str(script)], cwd=NEW,
            env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(run.stdout, flush=True)
        tests.append({'script': str(script), 'code': run.returncode, 'output': run.stdout})
        assert run.returncode == 0
    # Verify Beauty against the original pre-repair snapshot, including its label receipts.
    first = sorted((OUT/'runtime_revisions').glob('compact_convergence_*/beauty_files_before.json'))[0]
    beauty = json.loads(first.read_text())
    for rel, digest in beauty.items():
        assert sha(OUT/rel) == digest, rel
    ready = json.loads((OLD/'preparation/readiness.json').read_text())
    ready.update(state='USER_AUTHORIZED_COVERAGE_V4_READY',
        source_sha256={p.name: sha(p) for p in NEW.glob('*.py')}, regression_tests_passed=True)
    write(prep/'readiness.json', ready)
    provenance = {'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'authorization': 'User requested solving convergence problem.',
        'reason': 'v1 slow shrink, v2 repeated names, v3 alphabetic coverage omission rejected.',
        'method': 'Two finite normalization passes; exactly one explicit target for every source name; NEW category allowed.',
        'all_original_discovery_batches_preserved': 43, 'beauty_verified_unchanged': True,
        'no_sid_or_metrics_used': True, 'config_unchanged': True,
        'source_sha256': ready['source_sha256'], 'tests': tests,
        'protocol_limitation': 'Toys taxonomy protocol revised; Beauty v1 preserved. Mapping verifies source accounting, not human semantic accuracy.'}
    write(prep/'convergence_patch.json', provenance);write(archive/'coverage_v4_patch.json', provenance)
    fcntl.flock(lock, fcntl.LOCK_UN);lock.close()
    script = ROOT/'codex/sid_prefix_semantics_control/run_full_pipeline_coverage.py'
    logpath = ROOT/'codex/sid_prefix_semantics_control/full_pipeline_coverage.log'
    with logpath.open('ab') as log:
        p = subprocess.Popen([str(PYTHON), '-B', '-u', str(script), '--gpus', '0,1', '--execute'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(archive/'coverage_v4_relaunch.json', {'pid': p.pid, 'log': str(logpath)})
    print('V4_RELAUNCHED', p.pid, flush=True)


if __name__ == '__main__':
    main()
