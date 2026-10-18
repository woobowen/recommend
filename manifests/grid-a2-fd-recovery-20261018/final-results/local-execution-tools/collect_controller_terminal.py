#!/usr/bin/env python3
"""Capture the exact finished recovery controller and its own queue directory."""
from pathlib import Path, PurePosixPath
import datetime as dt
import hashlib
import io
import json
import subprocess
import sys
import tarfile

BASE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport

REMOTE = r'''
from pathlib import Path
import hashlib, io, json, sys, tarfile
c = json.loads(Path(CONFIG).read_text())
job = Path(c['runner']['job_root'])/c['queue_id']
state = Path(c['state_dir'])
assert c['queue_id'] == 'ga-20261018-a2-fd-queue-v1'
status = json.loads((job/'status.json').read_text())
cleanup = json.loads((job/'cleanup.json').read_text())
assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
assert cleanup['verified'] and cleanup['last_observation']['verified_empty']
assert json.loads((state/'queue-status.json').read_text())['status'] == 'TERMINAL_PENDING_RESULT_REVIEW'
files = []
for prefix, directory in [('controller', job), ('queue-state', state)]:
    for path in sorted(directory.rglob('*')):
        assert not path.is_symlink(), str(path)
        if not path.is_file() or path.name == 'evidence.tar.gz' or path.suffix in ['.lock', '.pyc']:
            continue
        assert path.stat().st_size < 50_000_000
        files.append((prefix+'/'+path.relative_to(directory).as_posix(), path))
assert sum(path.stat().st_size for _, path in files) < 60_000_000
records = []
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for relative, path in files:
        data = path.read_bytes()
        records.append(dict(path=relative, remote=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        info = tarfile.TarInfo(relative); info.size = len(data); info.mode = 0o444
        archive.addfile(info, io.BytesIO(data))
    data = json.dumps(dict(files=records, remote_mutations=0, scope='Exact terminal controller and its own queue-state only'), indent=2).encode()
    info = tarfile.TarInfo('REMOTE_MANIFEST.json'); info.size = len(data); info.mode = 0o444
    archive.addfile(info, io.BytesIO(data))
'''

def main():
    deployment = json.loads((BASE/'queue-deployment.json').read_text())
    code = REMOTE.replace('CONFIG', repr(deployment['remote_config']))
    destination = BASE/'controller-terminal'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True, exist_ok=False)
    archive_path = destination/'controller.tar.gz'; transport = Transport('new-server')
    argv = ['/mnt/c/Windows/System32/OpenSSH/ssh.exe', *transport.options, '-T', transport.host, 'python3 -I -S -B -']
    with archive_path.open('xb') as stream:
        result = subprocess.run(argv, input=code.encode(), stdout=stream, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers(); names = [m.name for m in members]
        assert len(names) == len(set(names))
        manifest = json.load(archive.extractfile('REMOTE_MANIFEST.json'))
        entries = {r['path']: r for r in manifest['files']}
        assert set(entries) | {'REMOTE_MANIFEST.json'} == set(names)
        for member in members:
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
            data = archive.extractfile(member).read()
            if member.name in entries:
                entry = entries[member.name]
                assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
            path = destination/'raw'/member.name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    receipt = dict(status='TERMINAL_CONTROLLER_ORIGINALS_VERIFIED', directory=str(destination),
                   members=len(members), payload_bytes=sum(r['bytes'] for r in entries.values()),
                   archive_bytes=archive_path.stat().st_size,
                   archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                   member_hashes_verified=True, safe_paths_verified=True, remote_mutations=0,
                   generated_remote_script_sha256=hashlib.sha256(code.encode()).hexdigest())
    (destination/'capture-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (destination/'remote-readonly-query.py').write_text(code)
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
