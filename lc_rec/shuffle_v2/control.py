"""Deploy immutable V2 sources and submit requests through the accepted runner."""
import argparse
import fcntl
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

from lc_rec.shuffle_v2.core import MODES
from lc_rec.shuffle_v2.replay import CONFIG

REPO = Path(__file__).resolve().parents[2]
BRANCH = 'wbw-lc-rec-shuffle-v2'
BASE_COMMIT = '7368d301d0d928f77d50551c3b53a5b30c23d63f'
SUPPORT = Path('/home/addaswsw/recommend-support/letter-shuffle-v2-20261120')
BASE = '/home/bwwu/grid-audits/sid-extensions-p0'
ROOT = '/data/bwwu/lc-rec/letter-shuffle-v2-20261120'
PREFIX = 'letter-a0-20261112-v2-20261120-'
RUNNER_SHA = '1fefc09a15c1cb73878417f9eb50d0fe9e8fa1b2f60cc70b27f7cc4ed9cb0947'
PATHS = ('lc_rec/shuffle_v2', 'lc_rec/shuffle/core.py', 'lc_rec/shuffle/entry.py',
         'lc_rec/letter/scripts/formal_evaluate.py', 'lc_rec/a1_qwen3_06b/scripts/evaluate.py',
         'lc_rec/qwen_preflight_20261110/scripts/common.py',
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


def readback(commit):
    result = subprocess.check_output(['git', 'ls-remote', '--exit-code', '--heads', 'origin', f'refs/heads/{BRANCH}'],
                                     cwd=REPO, text=True, timeout=45).strip().split()
    assert result == [commit, f'refs/heads/{BRANCH}'], result
    assert subprocess.run(['git', 'merge-base', '--is-ancestor', BASE_COMMIT, commit], cwd=REPO).returncode == 0
    transport.atomic(SUPPORT / f'{commit}.publication.json', {'commit': commit, 'branch': BRANCH, 'base_commit': BASE_COMMIT})


def job_states(t):
    return json.loads(remote_python(t, f'''from pathlib import Path
import json
base=Path({BASE!r})/'jobs'; states={{}}
for job in sorted(base.glob({(PREFIX+'*')!r})):
 path=job/'status.json'
 if path.is_file(): states[job.name]=json.loads(path.read_text())
print(json.dumps(states))'''))


def budget_seconds(states, kind, limit):
    total = 0.0
    for name, state in states.items():
        if not name.startswith(PREFIX + kind):
            continue
        if not state.get('ended_at'):
            raise RuntimeError(f'Budget is reserved by active attempt {name}')
        started = state.get('started_at', state.get('received_at'))
        total += (datetime.fromisoformat(state['ended_at']) - datetime.fromisoformat(started)).total_seconds()
    return int(limit - total)


def succeeded(states, name):
    record = states[PREFIX + name]
    assert record['status'] == 'SUCCEEDED' and record['exit_code'] == 0 and record['cleanup']['verified'], record


def stage_success(states, stage):
    candidates = [name.removeprefix(PREFIX) for name, value in states.items()
                  if name.startswith(PREFIX + stage + '-') and value['status'] == 'SUCCEEDED'
                  and name.removeprefix(PREFIX + stage + '-') in ('r0', 'r1', 'r2', 'r3')]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one accepted successful attempt for {stage}, found {candidates}')
    succeeded(states, candidates[0])
    return candidates[0]


def remote_hash(t, path):
    return remote_python(t, f'from pathlib import Path; import hashlib; print(hashlib.sha256(Path({path!r}).read_bytes()).hexdigest())').strip()


def request_parts(t, config, states, args, commit):
    mode = args.mode
    kind = args.action
    out = lambda job: f'{ROOT}/{job}'
    audit = out(args.replay + '/audit.json')
    inputs = []
    gpu = ''
    if kind in ('preflight', 'generate'):
        from lc_rec.shuffle_v2.collect import hardware
        snapshot = hardware(t)
        allocated = {r.split(',')[0].strip() for r in snapshot['processes'].splitlines()[1:]}
        eligible = []
        for line, expected in zip(snapshot['gpus'].splitlines()[1:], config['gpus'], strict=True):
            _, uuid, model, used, total, _ = map(str.strip, line.split(','))
            assert uuid == expected and model == 'NVIDIA GeForce RTX 4090' and int(total.split()[0]) >= 24000
            if uuid not in allocated and int(used.split()[0]) < 100:
                eligible.append(uuid)
        if not eligible:
            raise RuntimeError('No approved GPU is idle: ' + snapshot['processes'].strip())
        gpu = eligible[0]
    limits = config['budget']
    if gpu:
        readback(commit)
        succeeded(states, args.replay)
        shard_job = args.single_shard or stage_success(states, 'cpu-single-shard')
        succeeded(states, shard_job)
        shard = out(shard_job + '/audit.json')
        inputs.append({'path': shard, 'sha256': remote_hash(t, shard)})
        for filename in ('assets.json', 'audit.json', 'inputs.jsonl'):
            path = out(args.replay + '/' + filename)
            inputs.append({'path': path, 'sha256': remote_hash(t, path)})
        if kind == 'preflight':
            job = 'gpu-preflight'
            remaining = budget_seconds(states, job, limits['preflight_device_seconds'])
            timeout = min(3500, remaining - 30)
            mode = 'preflight'
        else:
            assert mode in MODES
            job = 'gpu-formal-' + mode
            succeeded(states, args.preflight)
            review = json.loads(Path(args.review).read_text())
            assert review['status'] == 'PASS' and review['RUN_COMMIT'] == commit
            assert review['preflight_job_id'] == PREFIX + args.preflight
            receipt = out(args.preflight + '/summary.json')
            assert remote_hash(t, receipt) == review['preflight_summary_sha256']
            sent = t.put(args.review, f'{BASE}/requests/{PREFIX}{job}-{args.attempt}.review.json')
            inputs.extend({'path': path, 'sha256': sha} for path, sha in ((sent['path'], sent['sha256']), (receipt, review['preflight_summary_sha256'])))
            ordinal = list(MODES).index(mode)
            for prior in list(MODES)[:ordinal]:
                stage_success(states, 'gpu-formal-' + prior)
            remaining = budget_seconds(states, 'gpu-formal-', min(limits['formal_total_wall_seconds'], limits['formal_total_device_seconds']))
            timeout = min(limits['formal_attempt_wall_seconds'] - 30, remaining - 30)
            assert timeout >= args.minimum_seconds, 'Remaining formal budget below the verified time estimate'
        assert timeout >= 90
        extra = ['--cpu-audit', audit, '--single-shard-audit', shard, '--run-commit', commit, '--mode', mode, '--seconds', str(timeout - 45)]
        return job, 'generate', extra, gpu, timeout, inputs
    timeout = min(900, budget_seconds(states, 'cpu-', limits['cpu_total_wall_seconds']) - 30)
    assert 90 <= timeout <= limits['cpu_attempt_wall_seconds']
    if kind == 'replay':
        return 'cpu-replay', 'replay', [], gpu, min(timeout, 600), inputs
    if kind == 'single-shard':
        succeeded(states, args.replay)
        return 'cpu-single-shard', 'single_shard', ['--cpu-audit', audit], gpu, min(timeout, 300), inputs
    if kind in ('fixtures', 'boundary-fixture', 'evaluation-fixture'):
        return 'cpu-' + kind, kind.replace('-', '_'), [], gpu, min(timeout, 300), inputs
    if kind == 'metrics':
        assert mode in ('normal', *MODES)
        run = config['baseline']['run-commit'] if mode == 'normal' else args.prediction_commit or commit
        extra = ['--mode', mode, '--run-commit', run, '--evaluation-commit', commit]
        if mode != 'normal':
            assert args.prediction_job.startswith(f'gpu-formal-{mode}-')
            succeeded(states, args.prediction_job)
            shard_job = args.single_shard or stage_success(states, 'cpu-single-shard')
            extra.extend(['--cpu-audit', audit, '--single-shard-audit', out(shard_job + '/audit.json'), '--test-dir', out(args.prediction_job + '/test')])
        return 'cpu-eval-' + mode, 'evaluate', extra, gpu, timeout, inputs
    assert kind == 'summarize'
    stages = ['cpu-eval-' + mode for mode in ('normal', *MODES)]
    completed = [stage_success(states, stage) for stage in stages]
    mode_dirs = [f'{ROOT}/{stage}' for stage in completed]
    return 'cpu-summary', 'statistics', mode_dirs, gpu, timeout, inputs


def submit(t, commit, args):
    profile = t.profile
    deployment = json.loads((SUPPORT / f'{commit}.deployment.json').read_text())
    release = deployment['release']
    config = json.loads(CONFIG.read_text())
    base = config['baseline']
    with (SUPPORT / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        states = job_states(t)
        key = {'replay': 'cpu-replay', 'fixtures': 'cpu-fixtures', 'evaluation-fixture': 'cpu-evaluation-fixture', 'single-shard': 'cpu-single-shard', 'boundary-fixture': 'cpu-boundary-fixture', 'preflight': 'gpu-preflight', 'generate': 'gpu-formal-' + args.mode,
               'metrics': 'cpu-eval-' + args.mode, 'summarize': 'cpu-summary'}[args.action]
        job_id = PREFIX + key + '-' + args.attempt
        path = SUPPORT / f'{job_id}.request.json'
        if job_id in states:
            return gridctl.query(t, job_id)
        if path.exists():
            # A connection can fail before remote acceptance. Inspect the original
            # ID and resend only its already persisted immutable request.
            accepted = f"{profile['job_root']}/{job_id}/accepted.json"
            exists = remote_python(t, f'from pathlib import Path; print(int(Path({accepted!r}).is_file()))').strip()
            if exists == '1':
                return gridctl.query(t, job_id)
            assert exists == '0'
            stored = json.loads(path.read_text())
            assert stored['capture_version'] == commit
            _, _, _, _, available_timeout, _ = request_parts(t, config, states, args, commit)
            assert stored['timeout_seconds'] <= available_timeout, 'Pending request exceeds the current allowance'
            return gridctl.submit(t, path)
        job, stage, extra, gpu, timeout, inputs = request_parts(t, config, states, args, commit)
        assert job == key
        request = {k: profile[k] for k in ('user', 'uid', 'namespace', 'deployment_sha256')}
        env = {'CUDA_VISIBLE_DEVICES': gpu, 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONOPTIMIZE': '0', 'PYTHONUNBUFFERED': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'TORCHINDUCTOR_COMPILE_THREADS': '1',
               'TRANSFORMERS_OFFLINE': '1', 'USE_FLAX': '0', 'USE_TF': '0'}
        if gpu:
            prior = json.loads((REPO / 'lc_rec/shuffle/results/letter-a0-20261112-shuffle3-instruments-seed42-20261114-r0/job/request.json').read_text())
            for name in ('CUDA_HOME', 'LD_LIBRARY_PATH', 'TORCH_EXTENSIONS_DIR'):
                env[name] = prior['env'][name]
        request.update(schema=1, host=profile['hostname'], job_id=job_id, capture_version=commit,
                       cwd=release, env=env, timeout_seconds=timeout,
                       inputs=inputs + [{'path': f"{release}/{item['path']}", 'sha256': item['sha256']} for item in deployment['files']],
                       argv=[base['python'], '-B', '-m', f'lc_rec.shuffle_v2.{stage}', '--output', f'{ROOT}/{job}-{args.attempt}', *extra])
        transport.atomic(path, request)
        return gridctl.submit(t, path)



def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('action', choices=('deploy', 'replay', 'query', 'pull', 'preflight', 'generate', 'metrics', 'summarize', 'fixtures', 'boundary-fixture', 'single-shard', 'evaluation-fixture'))
    parser.add_argument('--commit')
    parser.add_argument('--job', default='cpu-replay')
    parser.add_argument('--attempt', default='r0', choices=('r0','r1','r2','r3'))
    parser.add_argument('--relative')
    parser.add_argument('--mode', default='normal', choices=('normal', *MODES))
    parser.add_argument('--replay', default='cpu-replay-r1')
    parser.add_argument('--single-shard')
    parser.add_argument('--preflight', default='gpu-preflight-r0')
    parser.add_argument('--prediction-job')
    parser.add_argument('--prediction-commit')
    parser.add_argument('--review', type=Path)
    parser.add_argument('--minimum-seconds', type=int, default=21600)
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
            result = submit(t, commit, args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
