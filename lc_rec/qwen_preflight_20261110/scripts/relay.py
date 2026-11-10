#!/usr/bin/env python3
"""Bounded WSL official download -> fixed-chunk Windows SCP relay.

Use `make-request` to generate an immutable request for the existing local-runner.
Each model is an independent job. At most two download jobs and one SCP operate
at once. Completed chunks are SHA256-bound and reused after interruption. The
remote project runner performs the separate final assets.py publication step.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

GIB = 1024 ** 3
CHUNK = 128 * 1024 ** 2
RUNNER = Path('/home/addaswsw/recommend-support/sid-extensions-p0/scripts/local-runner.py')
SUPPORT = Path('/home/addaswsw/recommend-support/qwen-preflight-20261110')
WIN_STAGE = Path('/mnt/c/Users/lenovo/AppData/Local/Temp/qwen-preflight-20261110')
REMOTE = '/data/bwwu/lc-rec/qwen-preflight-20261110'
SSH = '/mnt/c/Windows/System32/OpenSSH/ssh.exe'
SCP = '/mnt/c/Windows/System32/OpenSSH/scp.exe'
OPTIONS = ['-F', 'C:/Users/lenovo/.ssh/config', '-o', 'BatchMode=yes', '-o',
           'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', '-o',
           'ClearAllForwardings=yes', '-o', 'PermitLocalCommand=no', '-o', 'User=bwwu']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for b in iter(lambda: src.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def live(pid, start):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19] == start
    except (OSError, IndexError):
        return False


class Relay:
    def __init__(self, args):
        self.args = args
        self.manifest = json.loads(Path(args.manifest).read_text())
        self.name = self.manifest['model_id'].split('/')[-1]
        self.revision = self.manifest['revision']
        self.local_root = Path(args.local_root).resolve()
        self.local_asset = self.local_root / self.name / self.revision
        self.remote_transfer = f'{REMOTE}/transfers/{self.name}/{self.revision}'
        self.remote_formal = f'{REMOTE}/models/{self.name}/{self.revision}'
        self.receipt = Path(args.receipt_root).resolve() / (args.job_id + '.relay.json')
        self.start = time.monotonic()
        self.pid_start = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        self.budget_file = SUPPORT / 'relay_budget.json'
        self.budget_lock = SUPPORT / 'relay_budget.lock'
        SUPPORT.mkdir(parents=True, exist_ok=True)
        self.result = {'job_id': args.job_id, 'run_commit': args.run_commit,
                       'model_id': self.manifest['model_id'], 'revision': self.revision,
                       'script': str(Path(__file__).resolve()), 'script_sha256': sha(__file__),
                       'argv': sys.argv, 'interpreter': sys.executable, 'cwd': os.getcwd(),
                       'started_utc': utc(), 'status': 'RUNNING', 'exit_code': None,
                       'local_asset': str(self.local_asset), 'remote_transfer': self.remote_transfer,
                       'remote_formal': self.remote_formal, 'transport': 'existing Windows OpenSSH/SCP',
                       'chunk_bytes': CHUNK, 'events': [], 'files': [],
                       'budget_limits': {'model_seconds': 21600, 'collection_seconds': 57600,
                                         'wsl_staging_bytes': 15 * GIB, 'remote_new_bytes': 60 * GIB,
                                         'download_concurrency': 2, 'scp_concurrency': 1}}

    def event(self, event, **values):
        obj = {'utc': utc(), 'event': event, **values}
        self.result['events'].append(obj)
        atomic(self.receipt, self.result)
        print(json.dumps(obj, sort_keys=True), flush=True)

    def budget(self, terminal=False):
        with self.budget_lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = json.loads(self.budget_file.read_text()) if self.budget_file.exists() else {'jobs': {}}
            elapsed = time.monotonic() - self.start
            ledger['jobs'][self.args.job_id] = {'model': self.name, 'seconds': elapsed,
                'started_monotonic': self.start, 'pid': os.getpid(), 'pid_start_ticks': self.pid_start,
                'running': not terminal, 'updated_utc': utc()}
            totals = {}
            for item in ledger['jobs'].values():
                amount = item['seconds']
                if item['running'] and live(item['pid'], item['pid_start_ticks']):
                    amount = max(amount, time.monotonic() - item['started_monotonic'])
                totals[item['model']] = totals.get(item['model'], 0) + amount
            ledger['observed_total_seconds'] = sum(totals.values())
            ledger['observed_model_seconds'] = totals
            atomic(self.budget_file, ledger)
            if not terminal and (totals[self.name] >= 21600 or sum(totals.values()) >= 57600):
                raise TimeoutError('Cumulative model or collection download/relay time budget exhausted')
        if not terminal:
            count = tree_bytes(self.local_root) + tree_bytes(WIN_STAGE)
            if count > 15 * GIB:
                raise RuntimeError('WSL/Windows relay staging exceeds 15 GiB bound')
            disk = shutil.disk_usage(self.local_root.parent)
            if disk.free < max(30 * GIB, int(disk.total * 0.10)):
                raise RuntimeError('WSL free space safety threshold reached')

    def command(self, argv, data=None, timeout=900):
        self.budget()
        begin = time.monotonic()
        self.event('command_started', argv=argv, stdin_sha256=hashlib.sha256(data).hexdigest() if data else None)
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # SSH snippets and their results are small; avoid pipe deadlocks and
        # preserve the one subprocess identity through budget polling.
        try:
            sent = False
            while True:
                try:
                    out, err = proc.communicate(input=data if not sent else None, timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    sent = True
                    self.budget()
                    if time.monotonic() - begin > timeout:
                        raise TimeoutError('Bounded relay command timeout; inspect same transfer paths before retry')
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            raise
        result = {'argv': argv, 'exit_code': proc.returncode,
                  'seconds': time.monotonic() - begin, 'stdout': out.decode('utf8', 'replace'),
                  'stderr': err.decode('utf8', 'replace')}
        self.event('command_finished', **result)
        if proc.returncode:
            raise RuntimeError(f'Command exited {proc.returncode}: {result["stderr"][:600]}')
        return result['stdout']

    def remote(self, code, timeout=180):
        return json.loads(self.command([SSH, *OPTIONS, '-T', 'new-server',
                                       'python3 -I -S -B -'], code.encode(), timeout))

    @contextlib.contextmanager
    def slot(self, kind, count):
        held = None
        try:
            while held is None:
                self.budget()
                for n in range(count):
                    lock = (SUPPORT / f'{kind}-{n}.lock').open('a')
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        held = lock
                        break
                    except BlockingIOError:
                        lock.close()
                if held is None:
                    time.sleep(5)
            self.event('slot_acquired', kind=kind)
            yield
        finally:
            if held:
                held.close()

    def remote_check(self, path, size, expected):
        code = f'''from pathlib import Path
import hashlib,json
p=Path({path!r})
ok=False
if p.is_file() and not p.is_symlink() and p.stat().st_size=={size}:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 ok=h.hexdigest()=={expected!r}
print(json.dumps({{'verified':ok,'exists':p.exists()}}))
'''
        return self.remote(code)['verified']

    def transfer_file(self, local, entry):
        target = self.remote_transfer + '/' + entry['path']
        if self.remote_check(target, entry['size'], entry['sha256']):
            self.event('transfer_file_verified_reuse', path=target)
            return {'path': entry['path'], 'size': entry['size'], 'sha256': entry['sha256'], 'reused': True}
        windows_dir = WIN_STAGE / self.name / self.revision
        windows_dir.mkdir(parents=True, exist_ok=True)
        chunks = []
        with local.open('rb') as src:
            for index in range((entry['size'] + CHUNK - 1) // CHUNK):
                self.budget()
                payload = src.read(CHUNK)
                digest = hashlib.sha256(payload).hexdigest()
                chunk_remote = target + f'.chunks/{index:05d}.{digest}'
                chunk = {'path': chunk_remote, 'size': len(payload), 'sha256': digest}
                chunks.append(chunk)
                if self.remote_check(chunk_remote, len(payload), digest):
                    self.event('transfer_chunk_verified_reuse', path=chunk_remote)
                    continue
                stage = windows_dir / (Path(entry['path']).name + f'.{index:05d}.{digest}.chunk')
                if not stage.exists():
                    stage.write_bytes(payload)
                if stage.stat().st_size != len(payload) or sha(stage) != digest:
                    raise RuntimeError('Windows staging chunk fails SHA; preserved')
                winpath = subprocess.check_output(['wslpath', '-w', str(stage)], text=True).strip().replace('\\', '/')
                self.remote(f'''from pathlib import Path
import json,shutil
root=Path({REMOTE!r});root.mkdir(parents=True,exist_ok=True)
disk=shutil.disk_usage(root)
size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
assert size+{len(payload)}<60*1024**3,'60GiB remote task bound'
assert disk.free-{len(payload)}>max(30*1024**3,int(disk.total*0.10)),'remote reserve'
p=Path({chunk_remote!r});assert root.resolve() in p.resolve().parents
p.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({{'remote_task_bytes':size,'free_bytes':disk.free}}))
''')
                for attempt in range(4):
                    partial = chunk_remote + f'.upload{attempt}.partial'
                    try:
                        self.command([SCP, *OPTIONS, winpath, 'new-server:' + shlex.quote(partial)], timeout=900)
                        if not self.remote_check(partial, len(payload), digest):
                            raise RuntimeError('SCP chunk identity failure; partial retained')
                        self.remote(f'''from pathlib import Path
import os,json
p=Path({partial!r});q=Path({chunk_remote!r})
assert not q.exists();os.rename(p,q)
print(json.dumps({{'verified_chunk_published':str(q)}}))
''')
                        self.event('transfer_chunk_verified', **chunk)
                        # This exact single-use Windows staging chunk has its
                        # verified remote counterpart; no source asset is deleted.
                        stage.unlink()
                        break
                    except Exception as exc:
                        self.event('scp_retry', path=chunk_remote, attempt=attempt,
                                   error=str(exc)[:600], valid_prior_chunks_retained=True)
                        if attempt == 3:
                            raise
                        time.sleep(min(30, 2 ** attempt))
        code = f'''from pathlib import Path
import hashlib,json,os,shutil
entries={chunks!r};target=Path({target!r});root=Path({REMOTE!r});size={entry['size']}
disk=shutil.disk_usage(root)
assert disk.free-size>max(30*1024**3,int(disk.total*.10)),'assembly reserve'
assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())+size<60*1024**3
partial=target.with_name(target.name+'.assembly.partial')
if target.exists():raise RuntimeError('Unverified existing transfer target preserved')
h=hashlib.sha256()
with partial.open('wb') as out:
 for e in entries:
  p=Path(e['path']);assert p.stat().st_size==e['size']
  ch=hashlib.sha256()
  with p.open('rb') as src:
   for b in iter(lambda:src.read(8*1024**2),b''):ch.update(b);h.update(b);out.write(b)
  assert ch.hexdigest()==e['sha256']
 out.flush();os.fsync(out.fileno())
assert partial.stat().st_size==size and h.hexdigest()=={entry['sha256']!r}
os.rename(partial,target)
print(json.dumps({{'path':str(target),'size':size,'sha256':h.hexdigest(),'chunks':len(entries),'verified':True}}))
'''
        return self.remote(code, timeout=300)

    def run(self):
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.event('accepted', manifest_sha256=sha(self.args.manifest))
        all_formal = True
        for entry in self.manifest['files']:
            if not self.remote_check(self.remote_formal + '/' + entry['path'], entry['size'], entry['sha256']):
                all_formal = False
                break
        if all_formal:
            self.event('formal_asset_verified_reuse', path=self.remote_formal)
        else:
            with self.slot('download', 2):
                argv = [sys.executable, '-B', str(Path(__file__).with_name('assets.py')), '--manifest',
                        str(Path(self.args.manifest).resolve()), '--model-root', str(self.local_root),
                        '--receipt-root', self.args.receipt_root, '--job-id', self.args.job_id + '-download',
                        '--run-commit', self.args.run_commit]
                self.command(argv, timeout=21600)
            with self.slot('scp', 1):
                for entry in self.manifest['files']:
                    source = self.local_asset / entry['path']
                    if source.stat().st_size != entry['size'] or sha(source) != entry['sha256']:
                        raise RuntimeError('Locally published model asset no longer matches pinned manifest')
                    self.result['files'].append(self.transfer_file(source, entry))
                    atomic(self.receipt, self.result)
        self.result.update(status='PASS', exit_code=0, ended_utc=utc(),
                           elapsed_seconds=time.monotonic() - self.start,
                           remote_publication='SEPARATE_PROJECT_RUNNER_ASSETS_JOB_REQUIRED',
                           cleanup_verified={'owned_children': [], 'download_assets_retained': str(self.local_asset),
                                             'verified_remote_chunks_retained': True})
        self.budget(terminal=True)
        atomic(self.receipt, self.result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['make-request', 'run'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--run-commit', required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--local-root', default=str(SUPPORT / 'models'))
    p.add_argument('--receipt-root', default=str(SUPPORT / 'receipts'))
    p.add_argument('--request-output')
    a = p.parse_args()
    if len(a.run_commit) != 40 or any(c not in '0123456789abcdef' for c in a.run_commit):
        p.error('A full immutable RUN_COMMIT is required')
    if a.mode == 'make-request':
        if not a.request_output:
            p.error('--request-output is required in make-request mode')
        deployment = RUNNER.with_name(RUNNER.name + '.deployment.json')
        d = json.loads(deployment.read_text())
        script = Path(__file__).resolve()
        inputs = [script, script.with_name('assets.py'), Path(a.manifest).resolve()]
        argv = [sys.executable, '-B', str(script), 'run', '--manifest', str(Path(a.manifest).resolve()),
                '--run-commit', a.run_commit, '--job-id', a.job_id,
                '--local-root', a.local_root, '--receipt-root', a.receipt_root]
        request = {'schema': 1, 'job_id': a.job_id, 'host': d['hostname'], 'uid': d['uid'],
                   'user': d['user'], 'namespace': d['namespace'], 'deployment_sha256': sha(deployment),
                   'capture_version': a.run_commit, 'argv': argv, 'cwd': str(script.parent.parent),
                   'timeout_seconds': 21600, 'env': {'PYTHONDONTWRITEBYTECODE': '1'},
                   'inputs': [{'path': str(v), 'sha256': sha(v)} for v in inputs],
                   'qwen_preflight': {'RUN_COMMIT': a.run_commit, 'route': 'official WSL download -> Windows SCP',
                                      'runner_path': str(RUNNER), 'runner_sha256': sha(RUNNER)}}
        atomic(a.request_output, request)
        print(json.dumps({'request': str(Path(a.request_output).resolve()),
                          'job_path': d['job_root'] + '/' + a.job_id, 'argv': argv}, indent=2))
        return 0
    worker = None
    try:
        worker = Relay(a)
        worker.run()
        return 0
    except Exception as exc:
        if worker:
            worker.result.update(status='BLOCKED', exit_code=1, ended_utc=utc(),
                                 elapsed_seconds=time.monotonic() - worker.start,
                                 error={'type': type(exc).__name__, 'message': str(exc)},
                                 valid_assets_and_fragments_retained=True)
            worker.budget(terminal=True)
            atomic(worker.receipt, worker.result)
            print(json.dumps(worker.result, indent=2), flush=True)
        else:
            print(json.dumps({'status': 'BLOCKED', 'error': str(exc)}), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
