"""Deploy hash-bound SID projection script/spec and submit two detached CPU jobs."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport, sha
import gridctl


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    script = HERE / 'export_sid_dictionary.py'
    script_sha = sha(script)
    previous = json.loads((HERE.parent / 'gs-20261017-toys-b-preflight-r1-request.json').read_text())
    commit = previous['capture_version'].split('/')[0]
    calculation_source = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0/manifests/grid-formal-20261015/export_sid_dictionary.py')
    assets = json.loads((HERE / 'candidate_specs.json').read_text())
    transport = Transport('new-server')
    transport.identity()
    remote_base = transport.profile['remote_support'] + '/grid-shuffle-20261017/asset-audit'
    remote_script = remote_base + '/scripts/' + script_sha + '/export_sid_dictionary.py'
    script_receipt = transport.put(script, remote_script)
    save(HERE / 'sid-dictionary-script-deployment.json', script_receipt)
    for domain, count in [('toys', 11924), ('sports', 18357)]:
        spec = {
            'domain': domain, 'route': 'B', 'sid': assets[domain]['sid'],
            'expected_item_count': count, 'source_commit': commit,
            'script': {'path': remote_script, 'sha256': script_sha, 'bytes': script.stat().st_size},
            'calculation_source': {'path': 'manifests/grid-formal-20261015/export_sid_dictionary.py',
                                   'sha256': sha(calculation_source),
                                   'method': 'sid.t().tolist(); sorted(set(tuple(row))); original four-code column axis'},
            'purpose': 'Read-only projection of fixed B SID for legality auditing; no new SID generation',
        }
        local_spec = HERE / 'sid-dictionary-specs' / (domain + '.json')
        save(local_spec, spec)
        spec_sha = sha(local_spec)
        remote_spec = remote_base + '/specs/' + spec_sha + '.json'
        spec_receipt = transport.put(local_spec, remote_spec)
        save(HERE / 'sid-dictionary-specs' / (domain + '-deployment.json'), spec_receipt)
        request = {k: previous[k] for k in ['schema', 'host', 'user', 'uid', 'namespace', 'deployment_sha256']}
        request.update({
            'job_id': f'gs-20261017-{domain}-b-sid-dictionary-v1',
            'capture_version': commit + '/sid-dictionary/' + script_sha + '/' + spec_sha,
            'argv': ['/data/bwwu/envs/grid-20261009-e61baf3/bin/python', '-B', remote_script, '--spec', remote_spec],
            'cwd': remote_base + '/scripts/' + script_sha,
            'timeout_seconds': 120,
            'env': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'},
            'inputs': [spec['script'], {'path': remote_spec, 'sha256': spec_sha, 'bytes': local_spec.stat().st_size}, spec['sid']],
        })
        request_path = HERE / 'sid-dictionary-requests' / (request['job_id'] + '.json')
        save(request_path, request)
        receipt = gridctl.submit(transport, request_path)
        save(HERE / 'sid-dictionary-requests' / (request['job_id'] + '-submission.json'), receipt)
        query = gridctl.query(transport, request['job_id'])
        save(HERE / 'sid-dictionary-requests' / (request['job_id'] + '-startup-query.json'), query)
        print(json.dumps({'job_id': request['job_id'], 'script_sha256': script_sha,
                          'spec_sha256': spec_sha, 'startup_query': query.get('status')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
