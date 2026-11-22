"""Read back G3 outputs and verify the single local review package."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import zipfile

from lc_rec.shuffle_v2 import g3_assets


def identity(receipt):
    return {key: receipt[key] for key in ('bytes', 'sha256')}


def remote_files(target, directory, names):
    from lc_rec.shuffle_v2 import control

    return json.loads(control.remote_python(target, f'''
import hashlib,json
from pathlib import Path
root=Path({directory!r})
out={{}}
for name in {names!r}:
 p=root/name
 digest=hashlib.sha256()
 with p.open("rb") as stream:
  for block in iter(lambda: stream.read(1024*1024), b""):digest.update(block)
 out[name]={{"bytes":p.stat().st_size,"sha256":digest.hexdigest()}}
print(json.dumps(out))
'''))


def fetch(target, remote, destination, names):
    receipts = remote_files(target, remote, names)
    for name, receipt in receipts.items():
        target.get_verified(f'{remote}/{name}', destination / name, receipt)
    return receipts


def get_gpu(mode, attempt, output):
    from lc_rec.shuffle_v2 import g3_control as gc

    target, _, _ = gc.connection()
    name = f'gpu-preflight-{mode}-{attempt}'
    gc.require_success(target, name)
    return fetch(target, f'{gc.ROOT}/{name}', output / name,
                 ('predictions.rank0.jsonl', 'rank0.json', 'metrics.json', 'v2-binding.rank0.json', 'derived.rank0.py', 'derived.rank0.diff'))


def inventory(target):
    from lc_rec.shuffle_v2 import g3_control as gc

    result = json.loads(gc.control.remote_python(target, f'''
from datetime import datetime,timezone
from pathlib import Path
import json
roots={{"jobs":Path({target.profile["job_root"]!r}),
       "requests":Path({gc.control.BASE!r})/"requests",
       "outputs":Path({gc.ROOT!r})}}
result={{key:[p.name for p in sorted(root.iterdir())
    if p.name.startswith({gc.PREFIX!r}) or key=="outputs"] for key,root in roots.items()}}
result["roots"]={{key:str(root) for key,root in roots.items()}}
result["checked_at_utc"]=datetime.now(timezone.utc).isoformat()
print(json.dumps(result))
'''))
    result['matching_formal'] = {key: [name for name in result[key] if 'formal' in name or 'group-series-' in name]
                                for key in ('jobs', 'requests', 'outputs')}
    assert not any(result['matching_formal'].values()), result
    return result


def verify_terminal(target):
    from lc_rec.shuffle_v2 import g3_control as gc

    observed = inventory(target)
    states = gc.control.job_states(target, gc.PREFIX)
    assert set(observed['jobs']) == set(states)
    receipts = {}
    for name, state in states.items():
        query = gc.control.gridctl.query(target, name)
        assert query['status'] == state['status'] and query['ended_at'] == state['ended_at']
        assert query['cleanup_verified'] and not any(query[key] for key in ('supervisor_alive', 'worker_alive', 'workgroup_live'))
        receipts[name] = {key: query.get(key) for key in
                          ('started_at', 'ended_at', 'status', 'exit_code', 'request_sha256', 'cleanup_verified',
                           'supervisor_alive', 'worker_alive', 'workgroup_live', 'workgroup_observation')}
    config = g3_assets.read(g3_assets.CONFIG)
    limits = config['budget']
    gpu = gc.usage(states, 'gpu-preflight-')
    cpu = gc.usage(states, 'cpu-')
    assert 0 <= gpu <= limits['preflight_device_seconds'] and 0 <= cpu <= limits['cpu_wall_seconds']
    return {'inventory': observed, 'terminal': receipts, 'preflight_gpu_device_seconds': gpu,
            'cpu_runner_worker_wall_seconds': cpu, 'formal_gpu_device_seconds_delta': 0,
            'preflight_gpu_remaining_seconds': limits['preflight_device_seconds'] - gpu,
            'cpu_remaining_seconds': limits['cpu_wall_seconds'] - cpu}


def get_export(commit, replay, normal, attempt, destination):
    from lc_rec.shuffle_v2 import g3_control as gc

    target, _, _ = gc.connection()
    name = 'cpu-review-export-r0'
    remote = f'{gc.ROOT}/{name}'
    tarname = name + '.tar.gz'
    config = g3_assets.read(g3_assets.CONFIG)
    payload = gc.request(target, config, commit, name, 480, 'g3_export',
                         ['--output', f'{remote}/{tarname}', '--job-root', target.profile['job_root'],
                          '--requests-root', f'{gc.control.BASE}/requests', '--replay', replay,
                          '--normal', normal, '--preflight-attempt', attempt])
    gc.launch(target, payload)
    gc.wait_terminal(target, name)
    gc.require_success(target, name)
    receipts = fetch(target, remote, destination, ('export-manifest.json', tarname))
    manifest = g3_assets.read(destination / 'export-manifest.json')
    assert identity(manifest) == receipts[tarname]
    stage = destination / 'evidence'
    stage.mkdir(exist_ok=False)
    with tarfile.open(destination / tarname) as archive:
        members = archive.getmembers()
        assert len(members) == len({member.name for member in members})
        assert {member.name for member in members} == set(manifest['members'])
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and '..' not in path.parts and member.isfile()
        archive.extractall(stage, filter='data')
    for member, receipt in manifest['members'].items():
        path = stage / member
        assert path.stat().st_size == receipt['bytes']
        g3_assets.verified(path, receipt['sha256'])
    return verify_terminal(target)


SECRET = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----|'
                    rb'\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|'
                    rb'AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{32,})\b')


def zip_members(root):
    for path in sorted(root.rglob('*')):
        if path.is_dir():
            assert not path.is_symlink()
            continue
        assert path.is_file() and not path.is_symlink()
        name = path.relative_to(root).as_posix()
        assert not any(part in ('.git', '__pycache__', '.ssh', '.env') for part in PurePosixPath(name).parts), name
        yield path, name


def build_zip(root, destination, extracted):
    expected = {}
    assert not destination.exists()
    assert not extracted.exists()
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in zip_members(root):
            before = path.stat()
            expected[name] = {'bytes': before.st_size, 'sha256': g3_assets.sha(path)}
            archive.write(path, name)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    extracted.mkdir()
    with zipfile.ZipFile(destination) as archive:
        rows = archive.infolist()
        assert archive.testzip() is None
        assert len(rows) == len({row.filename.casefold() for row in rows})
        assert set(expected) == {row.filename for row in rows}
        for row in rows:
            name = PurePosixPath(row.filename)
            assert not name.is_absolute() and all(part not in ('..', '.') for part in name.parts)
            assert '\\' not in row.filename and stat.S_IFMT(row.external_attr >> 16) == stat.S_IFREG
            digest = hashlib.sha256()
            tail = b''
            with archive.open(row) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    assert SECRET.search(tail + block) is None, f'Credential pattern in {row.filename}'
                    digest.update(block)
                    tail = block[-128:]
            assert expected[row.filename] == {'bytes': row.file_size, 'sha256': digest.hexdigest()}
        archive.extractall(extracted)
    return {'file': str(destination), 'bytes': destination.stat().st_size, 'sha256': g3_assets.sha(destination),
            'members': len(expected), 'uncompressed_bytes': sum(record['bytes'] for record in expected.values()),
            'crc_safe_paths_secret_scan_and_member_hashes': 'PASS', 'verified_at_utc': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subs = parser.add_subparsers(dest='action', required=True)
    gpu = subs.add_parser('gpu')
    gpu.add_argument('--mode', required=True)
    gpu.add_argument('--attempt', default='r0')
    gpu.add_argument('--output', required=True, type=Path)
    export = subs.add_parser('export')
    export.add_argument('--commit', required=True)
    export.add_argument('--replay', required=True)
    export.add_argument('--normal', required=True)
    export.add_argument('--attempt', default='r0')
    export.add_argument('--output', required=True, type=Path)
    archive = subs.add_parser('zip')
    for key in ('root', 'output', 'extracted'):
        archive.add_argument('--' + key, required=True, type=Path)
    options = parser.parse_args()
    if options.action == 'gpu':
        result = get_gpu(options.mode, options.attempt, options.output)
    elif options.action == 'export':
        result = get_export(options.commit, options.replay, options.normal, options.attempt, options.output)
    else:
        result = build_zip(options.root, options.output, options.extracted)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
