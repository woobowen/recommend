"""Task-scoped bridge using the existing accepted Windows SSH/SCP transport."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[2]
SUPPORT = Path('/home/addaswsw/recommend-support/letter-shuffle-20261114')
REMOTE = '/home/bwwu/lc-rec/letter-shuffle-20261114'
DATA = '/data/bwwu/lc-rec/letter-shuffle-20261114'
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import transport
import gridctl
transport.Transport.python = lambda self, code, timeout=45: self.ssh(
    shlex.join(['python3', '-I', '-S', '-B', '-c', code]), timeout=timeout)


def connection():
    t = transport.Transport('new-server')
    runner = REPO / 'lc_rec/a1_qwen3_06b/scripts/runner.py'
    digest = transport.sha(runner)
    t.profile = dict(t.profile, runner=t.profile['remote_support'] + '/scripts/runner-a1-96h-' + digest[:16] + '.py',
                     runner_sha256=digest)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['deploy', 'query', 'submit', 'read'])
    p.add_argument('--commit')
    p.add_argument('--job')
    p.add_argument('--request')
    p.add_argument('--remote-file')
    p.add_argument('--local-file')
    a = p.parse_args()
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() == 'wbw-lc-rec'
    t = connection()
    if a.mode == 'query':
        result = gridctl.query(t, a.job)
    elif a.mode == 'submit':
        result = gridctl.submit(t, a.request)
    elif a.mode == 'read':
        assert a.remote_file and a.local_file
        meta = json.loads(t.python('from pathlib import Path\nimport hashlib,json\np=Path(' + repr(a.remote_file) + ')\nassert p.is_file() and p.stat().st_size<20*1024**2\nprint(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))'))
        result = t.get_verified(a.remote_file, a.local_file, meta)
    else:
        commit = subprocess.check_output(['git', 'rev-parse', a.commit or 'HEAD'], cwd=REPO, text=True).strip()
        release = REMOTE + '/releases/' + commit
        identity = t.identity()
        SUPPORT.mkdir(parents=True, exist_ok=True)
        archive = SUPPORT / (commit + '.tar.gz')
        paths = ['lc_rec/shuffle', 'lc_rec/letter/scripts/formal_evaluate.py',
                 'lc_rec/a1_qwen3_06b/scripts/evaluate.py', 'lc_rec/qwen_preflight_20261110/scripts/common.py',
                 'lc_rec/a1_qwen3_06b/scripts/offline_metrics.py',
                 'lc_rec/a1_qwen3_06b/review/independent_extended.py',
                 'lc_rec/letter/config/a0_evaluation_v1.json', 'lc_rec/letter/review/final_audit_20261113',
                 'lc_rec/a1_qwen3_06b/review/final_20261114']
        paths = [x for x in paths if subprocess.run(['git','cat-file','-e',commit+':'+x],cwd=REPO,capture_output=True).returncode==0]
        subprocess.run(['git', 'archive', '--format=tar.gz', '--output=' + str(archive), commit, *paths], cwd=REPO, check=True)
        upload = t.profile['remote_support'] + '/transfers/shuffle-' + commit + '.tar.gz'
        sent = t.put(archive, upload)
        result = json.loads(t.python(f'''from pathlib import Path,PurePosixPath
import hashlib,json,tarfile
p=Path({upload!r});root=Path({release!r});rows=[]
assert hashlib.sha256(p.read_bytes()).hexdigest()=={sent['sha256']!r}
with tarfile.open(p) as tf:
 members=tf.getmembers();assert len(members)==len(set(m.name for m in members))
 for m in members:assert not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts and (m.isdir() or m.isfile())
 if not root.exists():root.mkdir(parents=True);tf.extractall(root,filter='data')
 for m in members:
  if m.isfile():
   b=tf.extractfile(m).read();q=root/m.name;assert q.is_file() and not q.is_symlink() and q.read_bytes()==b
   rows.append(dict(path=m.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
r=dict(RUN_COMMIT={commit!r},release=str(root),not_a_git_checkout=True,files=rows)
v=root/'RUN_VERSION.json'
if v.exists():assert json.loads(v.read_text())==r
else:v.write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r))''', timeout=90))
        transport.atomic(SUPPORT / 'receipts' / ('deployment-' + commit + '.json'), result)
        print(json.dumps({'RUN_COMMIT': commit, 'release': release, 'files_verified': len(result['files']), 'identity': identity}))
        return
    transport.atomic(SUPPORT / 'receipts' / (a.mode + '-' + (a.job or 'file') + '.json'), result)
    print(json.dumps(result))


if __name__ == '__main__': main()
