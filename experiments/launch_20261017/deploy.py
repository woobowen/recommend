"""Build and verify a fixed Git export; never update a running snapshot."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from experiments.beauty_extensions.common import REPO, sha256, write_json

SUPPORT = Path('/home/addaswsw/recommend-support/grid-a-launch-20261017')
REMOTE = '/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017'
TRANSPORT = Path('/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
SOURCE_DIRS = ['GRID baseline/src', 'references/wmh-original/GRID/src',
    'experiments/beauty_extensions', 'experiments/preflight_20261015',
    'experiments/formal_20261015', 'experiments/launch_20261017',
    'experiments/launch_20261018_shared_gpu', 'tools/sid-extensions-prep']


def snapshot(t):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() == 'wbw'
    files = sorted(p for d in SOURCE_DIRS for p in (REPO / d).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.yaml'))
    files += [REPO / d / '.project-root' for d in ('GRID baseline', 'references/wmh-original/GRID')
        if (REPO / d / '.project-root').is_file()]
    records = []
    for path in files:
        relative = str(path.relative_to(REPO))
        data = subprocess.check_output(['git', 'show', head + ':' + relative], cwd=REPO)
        assert hashlib.sha256(data).hexdigest() == sha256(path), relative
        records.append({'path': relative, 'bytes': len(data), 'sha256': sha256(path)})
    digest = hashlib.sha256(json.dumps({'source_commit': head, 'files': records}, sort_keys=True).encode()).hexdigest()
    manifest = {'source_commit': head, 'snapshot_sha256': digest, 'files': records,
        'identity_kind': 'fixed Git export; manifest verified; original assets external and read only'}
    data = (json.dumps(manifest, indent=2) + '\n').encode()
    manifest_sha = hashlib.sha256(data).hexdigest()
    SUPPORT.mkdir(parents=True, exist_ok=True)
    archive = SUPPORT / ('snapshot-' + digest[:16] + '.tar.gz')
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as tf:
            for item in records:
                tf.add(REPO / item['path'], arcname=item['path'], recursive=False)
            entry = tarfile.TarInfo('DEPLOYMENT.json')
            entry.size = len(data)
            tf.addfile(entry, io.BytesIO(data))
    remote_archive = REMOTE + '/archives/' + archive.name
    t.put(archive, remote_archive)
    root = REMOTE + '/snapshots/' + digest[:16]
    code = '''import pathlib,tarfile,hashlib,json,os
p=pathlib.Path(ROOT);a=pathlib.Path(ARCHIVE)
assert hashlib.sha256(a.read_bytes()).hexdigest()==ARCHIVE_SHA
if not p.exists():
 p.mkdir(parents=True)
 with tarfile.open(a) as tf:
  members=tf.getmembers()
  assert len({m.name for m in members})==len(members)
  assert all(m.isfile() and not pathlib.Path(m.name).is_absolute() and '..' not in pathlib.Path(m.name).parts for m in members)
  tf.extractall(p,filter='data')
assert not p.is_symlink()
assert hashlib.sha256((p/'DEPLOYMENT.json').read_bytes()).hexdigest()==MANIFEST_SHA
d=json.loads((p/'DEPLOYMENT.json').read_text())
for r in d['files']:
 f=p/r['path'];assert not f.is_symlink() and f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
 os.chmod(f,0o444)
os.chmod(p/'DEPLOYMENT.json',0o444)
print(json.dumps({'root':str(p),'files_verified':len(d['files']),'manifest_sha256':MANIFEST_SHA}))
'''.replace('MANIFEST_SHA', repr(manifest_sha)).replace('ARCHIVE_SHA', repr(sha256(archive))).replace('ARCHIVE', repr(remote_archive)).replace('ROOT', repr(root))
    receipt = json.loads(t.python(code, timeout=60))
    result = {'root': root, 'manifest': manifest, 'manifest_sha256': manifest_sha, 'receipt': receipt}
    write_json(SUPPORT / (digest[:16] + '-deployment.json'), result)
    return result


def request(t, deployment, job, argv, timeout, inputs=(), gpu=''):
    root = deployment['root']
    profile = t.profile
    return {'schema': 1, 'job_id': job, 'host': profile['hostname'], 'user': profile['user'],
        'uid': profile['uid'], 'namespace': profile['namespace'], 'deployment_sha256': profile['deployment_sha256'],
        'capture_version': deployment['manifest']['source_commit'] + '/' + deployment['manifest']['snapshot_sha256'],
        'argv': argv, 'cwd': root, 'timeout_seconds': timeout,
        'env': {'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '4', 'TF_CPP_MIN_LOG_LEVEL': '3',
            'TF_NUM_INTRAOP_THREADS': '2', 'TF_NUM_INTEROP_THREADS': '2', 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': root, 'HF_HUB_OFFLINE': '1', 'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false'},
        'inputs': [{'path': root + '/DEPLOYMENT.json', 'sha256': deployment['manifest_sha256']}, *inputs]}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.path.insert(0, str(TRANSPORT))
    from transport import Transport
    t = Transport('new-server')
    t.identity()
    result = snapshot(t)
    print(json.dumps({k: result[k] for k in ('root', 'manifest_sha256', 'receipt')}, indent=2))


if __name__ == '__main__':
    main()
