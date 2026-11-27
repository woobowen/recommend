"""Exercise final CLI protection before any dependency reads or CUDA initialization."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


snapshot = Path(sys.argv[1]).resolve(strict=True)
commit = snapshot.name
assert len(commit) == 40
source = snapshot / 'lc_rec/beauty_history_attention/evaluate.py'
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
results = []
settings = os.environ | {
    'CUDA_VISIBLE_DEVICES': '',
    'PYTHONOPTIMIZE': '0',
    'PYTHONDONTWRITEBYTECODE': '1',
}
for name in ('GRID_JOB_ID', 'GRID_JOB_DIR', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
    settings.pop(name, None)

with tempfile.TemporaryDirectory(prefix='letter-beauty-preview-boundary-') as scratch:
    scratch = Path(scratch)
    protected = scratch / 'formal-b-review-destination'
    protected.mkdir()
    linked = scratch / 'misleading-preflight-output'
    linked.symlink_to(protected, target_is_directory=True)
    prefix = [sys.executable, '-B', '-m', 'lc_rec.beauty_history_attention.evaluate']
    gen = ['generate', '--adapter', str(scratch / 'missing-adapter'), '--run-commit', commit,
           '--max-seconds', '10', '--sample-budget', '8']
    metrics = ['metrics', '--predictions', str(scratch / 'missing-predictions'), '--evaluation-commit', commit]
    for argv in (gen, metrics):
        allowed = scratch / f'{argv[0]}-allowed-target'
        process = subprocess.run(prefix + argv + ['--purpose', 'preflight', '--output', str(allowed),
                                 '--config', str(scratch / 'missing-config')], cwd=snapshot,
                                 env=settings, timeout=20, text=True, capture_output=True)
        assert process.returncode != 0 and 'FileNotFoundError' in process.stderr and not allowed.exists(), (
            process.stdout, process.stderr)
        for alias in (protected, linked):
            output = alias / f'{argv[0]}-invalid-target'
            process = subprocess.run(prefix + argv + ['--purpose', 'preflight', '--output', str(output),
                                    '--config', str(scratch / 'missing-config')], cwd=snapshot,
                                     env=settings, timeout=20, text=True, capture_output=True)
            assert process.returncode != 0 and not output.exists(), (process.stdout, process.stderr)
            reasons = [line for line in process.stderr.splitlines() if line.startswith('ValueError:')]
            assert reasons and 'formal B' in reasons[-1] and ('cannot' in reasons[-1] or 'may not' in reasons[-1]), process.stderr
            assert 'FileNotFoundError' not in process.stderr, process.stderr
            results.append({'subcommand': argv[0], 'target_kind': 'symlink_alias' if alias == linked else 'literal',
                            'exit_code': process.returncode, 'target_absent': not output.exists(),
                            'diagnostic': reasons[-1]})
    assert list(protected.iterdir()) == []
print(json.dumps({'status': 'PASS', 'run_commit': commit, 'evaluator_sha256': source_sha,
                  'cuda_visible_devices': settings['CUDA_VISIBLE_DEVICES'],
                  'ordinary_paths_reach_config_read': 2, 'checks': results}, indent=2))
