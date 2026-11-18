#!/usr/bin/env python3
"""Read-only verification of one already-pushed preparation delivery."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--commit', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
repo = a.repo.resolve()
commit = subprocess.check_output(['git', 'rev-parse', a.commit + '^{commit}'], cwd=repo, text=True).strip()
ref = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/wbw-lc-rec'], cwd=repo, text=True, timeout=45).strip()
assert ref.split()[0] == commit
base = 'lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/production_preparation/single_gpu/'
git_bytes = lambda name: subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo)
runtime = json.loads(git_bytes(base + 'run_deployment.json'))
differences = []
for row in runtime['files']:
    data = git_bytes(row['relative'])
    digest = hashlib.sha256(data).hexdigest()
    if digest != row['sha256']:
        differences.append({'path': row['relative'], 'run_sha256': row['sha256'], 'delivery_sha256': digest})
expected = json.loads(git_bytes(base + 'runtime_delivery_source_comparison.json'))['differences']
assert differences == expected
critical = json.loads(git_bytes(base + 'runtime_github_readback.json'))['files']
names = [r['path'] for r in critical]
names += ['lc_rec/a0_beauty/g3_remote.py', 'lc_rec/a0_beauty/grid_sid_preparation/g3_conversion/production_preparation/README.md']
names += [base + name for name in (
    'acceptance.json', 'internal_review_final.json', 'review_training_agent.json',
    'review_generation_agent.json', 'review_cpu_controls_agent.json', 'final_job_ledger.json',
    'final_remote_observation.json', 'model_asset_identity.json', 'formal_preparation_plan.json',
    'jobs/g3-lcrec-single-cpu-20261118-r3-final/output/cpu/result.json',
    'jobs/g3-lcrec-single-dryrun-20261118-r3-final/output/dry_run.json',
    'jobs/g3-lcrec-single-train-20261118-r3/output/g3_preflight.json',
    'jobs/g3-lcrec-single-compare-20261118-r3/output/boundary_compare.json',
    'jobs/g3-lcrec-single-evaluation-20261118-r3/output/evaluation/summary.json',
)]
tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--', base], cwd=repo, text=True).splitlines()
for suffix in ('.npy', '.npz'):
    names.append(next(n for n in tracked if n.endswith(suffix)))

def read(name):
    url = 'https://raw.githubusercontent.com/woobowen/recommend/' + commit + '/' + name
    request = urllib.request.Request(url, headers={'User-Agent': 'G3-preparation-byte-verification'})
    with urllib.request.urlopen(request, timeout=25) as response:
        data = response.read()
        status = response.status
    expected = git_bytes(name)
    assert status == 200 and data == expected, name
    return {'path': name, 'url': url, 'status': status, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'fixed_git_match': True}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(read, dict.fromkeys(names)))
out = {'status': 'PASS', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'argv': sys.argv, 'cwd': str(Path.cwd()), 'interpreter': sys.executable,
       'verification_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'delivery_commit': commit, 'run_commit': runtime['run_commit'], 'remote_ref': ref,
       'runtime_files_compared': len(runtime['files']), 'runtime_delivery_differences': differences,
       'files': rows, 'formal_experiment_started': False, 'formal_launch_authorized': False}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k: v for k, v in out.items() if k != 'files'}, indent=2))
print('HTTP200 exact Git bytes:', len(rows))
