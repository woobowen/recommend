"""Deploy immutable V2 sources and submit requests through the accepted runner."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from lc_rec.shuffle_v2.replay import CONFIG

REPO = Path(__file__).resolve().parents[2]
SUPPORT = Path('/home/addaswsw/recommend-support/letter-shuffle-v2-20261120')
BASE = '/home/bwwu/grid-audits/sid-extensions-p0'
ROOT = '/data/bwwu/lc-rec/letter-shuffle-v2-20261120'
PREFIX = 'letter-a0-20261112-v2-20261120-'
RUNNER_SHA = '1fefc09a15c1cb73878417f9eb50d0fe9e8fa1b2f60cc70b27f7cc4ed9cb0947'
PATHS = ('lc_rec/shuffle_v2', 'lc_rec/shuffle/core.py', 'lc_rec/shuffle/entry.py',
         'lc_rec/letter/scripts/formal_evaluate.py', 'lc_rec/a1_qwen3_06b/scripts/evaluate.py',
         'lc_rec/letter/config/a0_evaluation_v1.json', 'lc_rec/letter/EVALUATION_PROTOCOL_V1.md',
         'lc_rec/letter/review/formal_det_20261112/independent_metrics.py',
         'lc_rec/a1_qwen3_06b/scripts/offline_metrics.py', 'lc_rec/a1_qwen3_06b/review/independent_extended.py',
         'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
         'lc_rec/letter/review/final_audit_20261113/statistics_verify.py', 'lc_rec/shuffle/paired_and_changes.py')
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import gridctl
import transport


def connect():
    connection = transport.Transport('new-server')
    connection.profile = dict(connection.profile, runner=f'{BASE}/scripts/runner-a1-96h-{RUNNER_SHA[:16]}.py',
                              runner_sha256=RUNNER_SHA)
    return connection


def remote_python(t, code, timeout=45):
    return t.ssh(shlex.join(['python3', '-I', '-S', '-B', '-c', code]), timeout=timeout)


def deploy(t, commit):
    release = f'{BASE}/releases/letter-shuffle-v2/{commit}'
    archive = SUPPORT / f'{commit}.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', '-o', str(archive), commit, *PATHS], cwd=REPO, check=True)
    upload = f'{BASE}/transfers/letter-shuffle-v2/{commit}.tar.gz'
    sent = t.put(archive, upload)
    actual_runner = t.profile['runner']
    deployment_sha = t.profile['deployment_sha256']
    receipt = json.loads(remote_python(t, f'''from pathlib import Path, PurePosixPath
import hashlib,json,tarfile
from tempfile import TemporaryDirectory
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
archive=Path({upload!r}); release=Path({release!r}); runner=Path({actual_runner!r}); rows=[]
assert sha(archive)=={sent['sha256']!r}
assert sha(runner)=={RUNNER_SHA!r}
assert sha(Path(str(runner)+'.deployment.json'))=={deployment_sha!r}
release.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as tf:
 members=tf.getmembers(); assert len(members)==len({{m.name for m in members}})
 for item in members:
  name=PurePosixPath(item.name)
  assert not name.is_absolute() and '..' not in name.parts and (item.isdir() or item.isfile())
 def inspect(root):
  for item in members:
   if item.isfile():
    target=root/item.name; source=tf.extractfile(item).read()
    assert target.is_file() and not target.is_symlink() and target.read_bytes()==source
    rows.append(dict(path=item.name,bytes=len(source),sha256=hashlib.sha256(source).hexdigest()))
 if release.exists():
  inspect(release)
 else:
  with TemporaryDirectory(prefix='extract-',dir=release.parent) as staging:
   export=Path(staging)/'release';export.mkdir()
   tf.extractall(export,filter='data')
   inspect(export)
   export.rename(release)
result=dict(RUN_COMMIT={commit!r},release=str(release),files=rows,archive_sha256=sha(archive),runner_sha256=sha(runner),git_checkout=False)
manifest=release/'RUN_VERSION.json'
if manifest.exists(): assert json.loads(manifest.read_text())==result
else: manifest.write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))''', timeout=90))
    transport.atomic(SUPPORT / f'{commit}.deployment.json', receipt)
    return {'commit': commit, 'release': release, 'files': len(receipt['files']), 'runner_sha256': RUNNER_SHA}


def submit(t, commit, job, stage, attempt, timeout):
    profile = t.profile
    job_id = PREFIX + job + '-' + attempt
    deployment = json.loads((SUPPORT / f'{commit}.deployment.json').read_text())
    release = deployment['release']
    config = json.loads(CONFIG.read_text())
    base = config['baseline']
    request = {k: profile[k] for k in ('user', 'uid', 'namespace', 'deployment_sha256')}
    env = {'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONOPTIMIZE': '0', 'PYTHONUNBUFFERED': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'TRANSFORMERS_OFFLINE': '1', 'USE_FLAX': '0', 'USE_TF': '0'}
    request.update(schema=1, host=profile['hostname'], job_id=job_id, capture_version=commit,
                   cwd=release, env=env, timeout_seconds=timeout,
                   inputs=[{'path': f"{release}/{item['path']}", 'sha256': item['sha256']} for item in deployment['files']],
                   argv=[base['python'], '-B', '-m', stage, '--output', f'{ROOT}/{job}-{attempt}'])
    path = SUPPORT / f'{job_id}.request.json'
    if path.exists():
        if json.loads(path.read_text()) != request:
            raise ValueError('Existing request differs. Preserve its attempt and allocate a new ID.')
        return gridctl.query(t, job_id)
    transport.atomic(path, request)
    return gridctl.submit(t, path)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('action', choices=('deploy', 'replay', 'query', 'pull'))
    parser.add_argument('--commit')
    parser.add_argument('--job', default='cpu-replay')
    parser.add_argument('--attempt', default='r0', choices=('r0','r1','r2','r3'))
    parser.add_argument('--relative')
    args = parser.parse_args()
    SUPPORT.mkdir(parents=True, exist_ok=True)
    t = connect()
    if args.action == 'query':
        result = gridctl.query(t, PREFIX + args.job + '-' + args.attempt)
    elif args.action == 'pull':
        relative = Path(args.relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Expected relative path under the task output')
        remote = f'{ROOT}/{relative}'
        meta = json.loads(remote_python(t, f'''from pathlib import Path
import hashlib,json
p=Path({remote!r}); assert p.is_file() and p.stat().st_size<100*1024**2
print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))'''))
        destination = SUPPORT / 'evidence' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = t.get_verified(remote, destination, meta)
    else:
        commit = subprocess.check_output(['git', 'rev-parse', '--verify', (args.commit or 'HEAD') + '^{commit}'],
                                         cwd=REPO, text=True).strip()
        if args.action == 'deploy':
            result = deploy(t, commit)
        else:
            result = submit(t, commit, 'cpu-replay', 'lc_rec.shuffle_v2.replay', args.attempt, 600)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
