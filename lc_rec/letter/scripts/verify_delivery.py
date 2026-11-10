"""Verify saved task evidence and review links without model or GPU execution."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
repo = root.parents[1]
checked = []
for manifest_path in sorted((root / 'evidence/jobs').glob('*/*export-manifest.json')):
    manifest = json.loads(manifest_path.read_text())
    for member in manifest['files']:
        path = manifest_path.parent / member['path']
        assert path.is_file() and not path.is_symlink()
        data = path.read_bytes()
        assert len(data) == member['bytes']
        assert hashlib.sha256(data).hexdigest() == member['sha256'], str(path)
    checked.append({'manifest': str(manifest_path.relative_to(root)), 'members': len(manifest['files'])})

upstream = json.loads((root / 'evidence/upstream_materialization.json').read_text())
for entry in upstream['files']:
    data = (root / 'vendor/LETTER' / entry['path']).read_bytes()
    assert len(data) == entry['bytes']
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == entry['git_blob']

links = []
for doc in [root / 'README.md', root / 'REVIEW_ENTRY.md']:
    for link in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
        if '://' in link or link.startswith('#'):
            continue
        path = (doc.parent / link.split('#', 1)[0]).resolve()
        assert path.is_relative_to(repo) and path.exists(), (str(doc), link)
        links.append(link)

python_files = list(root.rglob('*.py'))
json_files = list(root.rglob('*.json'))
for path in python_files:
    ast.parse(path.read_text(), filename=str(path))
    compile(path.read_text(), str(path), 'exec')
for path in json_files:
    json.loads(path.read_text())
branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip()
assert branch == 'wbw-lc-rec'
changes = subprocess.check_output(['git', 'diff', '--name-only', '6b9f336d0252b44a31a48fbcc59817c9af1197e2'], cwd=repo, text=True).splitlines()
assert all(path.startswith('lc_rec/letter/') for path in changes)
status = json.loads((root / 'evidence/final_status.json').read_text())
assert status['GPU_optimizer_updates'] == 0 and status['96_hour_assessment'] == 'UNKNOWN'
assert status['overall'] == 'PARTIAL_TECHNICAL_BLOCKED'
result = {
    'status': 'PASS_DELIVERY_CONSISTENCY_ONLY',
    'gpu_technical_acceptance': 'FAIL', '96h': 'UNKNOWN',
    'export_manifests': checked, 'upstream_files_verified': len(upstream['files']),
    'local_review_links_verified': len(links),
    'python_syntax_files': len(python_files), 'json_files': len(json_files),
    'git_branch': branch, 'tracked_diff_limited_to_task': True,
    'no_new_model_gpu_or_tokenizer_execution': True,
}
(root / 'evidence/delivery_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
