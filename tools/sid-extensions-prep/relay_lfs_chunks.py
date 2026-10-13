"""Sequential 32 MiB public-LFS ranges -> bounded WSL staging -> verified SCP.

Only this task's transport is imported. Complete checkpoints never reside in
WSL. Completed, remote-hash-verified chunks survive controller disconnects.
The whole LFS OID is checked by a separate detached server assembly job.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transport

CHUNK_BYTES = 32 * 1024**2


def relay(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    assert plan['baseline_sha'] == '563dc50b1c7e9e2e05cdfddd13e91a3d1888001e'
    t = transport.Transport('new-server')
    scratch = transport.ROOT / 'staging-lfs'
    scratch.mkdir(exist_ok=True)
    receipts = transport.ROOT / 'evidence/lfs-chunks'
    receipts.mkdir(parents=True, exist_ok=True)
    output = Path(os.environ['GRID_JOB_DIR']) / 'output'
    assembled = []
    for e in plan['entries']:
        assert e['bytes'] < 161 * 1024**2
        parts = []
        for index, start in enumerate(range(0, e['bytes'], CHUNK_BYTES)):
            end = min(start + CHUNK_BYTES, e['bytes']) - 1
            rp = receipts / (e['sha256'] + '-' + str(index) + '.json')
            if rp.exists():
                old = json.loads(rp.read_text())
                assert (old['start'], old['end'], old['object_sha256']) == (start, end, e['sha256'])
                got = json.loads(t.python('from pathlib import Path\nimport hashlib,json\np=Path(' + repr(old['remote']) + ')\nprint(json.dumps({"exists":p.is_file(),"bytes":p.stat().st_size if p.is_file() else None,"sha256":hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}))'))
                assert (got['exists'], got['bytes'], got['sha256']) == (True, old['bytes'], old['sha256'])
                parts.append(old)
                continue
            for root in [scratch, transport.STAGE.parent]:
                v = os.statvfs(root)
                assert v.f_bavail * v.f_frsize > 1024**3 + 3 * CHUNK_BYTES
            url = 'https://media.githubusercontent.com/media/woobowen/recommend/' + plan['baseline_sha'] + '/' + urllib.parse.quote(e['repository_path'], safe='/')
            # Distinct query avoids intermediaries caching a different Range.
            url += '?sid_extensions_range=' + str(start) + '-' + str(end)
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'sid-extensions-p0-file-prep'})
            p = scratch / (e['sha256'] + '-' + str(index) + '.part')
            assert not p.exists(), 'unfinished local chunk requires explicit inspection'
            h = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(request, timeout=45) as r, p.open('xb') as f:
                assert r.status == 206 and r.headers.get('Content-Range') == f'bytes {start}-{end}/{e["bytes"]}'
                for b in iter(lambda: r.read(1024 * 1024), b''):
                    count += len(b)
                    assert count <= end - start + 1
                    f.write(b)
                    h.update(b)
                f.flush()
                os.fsync(f.fileno())
            assert count == end - start + 1
            remote = t.profile['remote_support'] + '/inputs/lfs-chunks/' + e['sha256'] + '/' + f'{index:04}-' + h.hexdigest() + '.part'
            verified = t.put(p, remote)
            row = {'index': index, 'start': start, 'end': end, 'bytes': count,
                   'sha256': h.hexdigest(), 'object_sha256': e['sha256'],
                   'remote': remote, 'source_url': url, 'scp_verified': verified,
                   'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            transport.atomic(rp, row)
            p.unlink()  # only this task's completed temporary chunk
            parts.append(row)
            print(json.dumps({'object': e['sha256'], 'chunk': index, 'bytes': count, 'verified': True}), flush=True)
        assembled.append(dict(e, parts=parts))
        transport.atomic(output / 'progress.json', {'objects_completed': len(assembled), 'objects_total': len(plan['entries'])})
    transport.atomic(output / 'assembly-plan.json', {'stage_id': 'sid-extensions-p0', 'baseline_sha': plan['baseline_sha'], 'entries': assembled, 'chunk_bytes': CHUNK_BYTES})
    transport.atomic(output / 'RESULT.json', {'status': 'SUCCEEDED', 'objects': len(assembled), 'chunks': sum(len(e['parts']) for e in assembled),
        'bytes': sum(e['bytes'] for e in assembled), 'concurrency': 1, 'chunk_bytes': CHUNK_BYTES,
        'whole_object_identity': 'PENDING_SERVER_ASSEMBLY_SHA256', 'source_code_execution': False,
        'model_execution': False, 'recovery': 're-read committed per-chunk receipts and rehash remote chunks; never resume a .partial'})


if __name__ == '__main__':
    relay(sys.argv[1])
