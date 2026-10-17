"""Independent Git/source/server-byte and exact 11-template plan review."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018')
COMPILED = SUPPORT / 'compiled-c31d2d3eaaafbd18'
REMOTE = '/home/bwwu/grid-audits/sid-extensions-p0/grid-a2-fd-recovery-20261018/compiled/c31d2d3eaaafbd18'
OUT = Path(__file__).resolve().parent


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_text())


def replace(value, mapping):
    if isinstance(value, str):
        for a, b in mapping:
            value = value.replace(a, b)
        return value
    if isinstance(value, list):
        return [replace(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: replace(v, mapping) for k, v in value.items()}
    return value


def plan_sha(template):
    return sha_bytes(json.dumps({k: v for k, v in template.items() if k != 'launch_review'}, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())


def main():
    deployment = read(SUPPORT / 'source-deployment.json')
    m = deployment['manifest']
    admission = read(OUT / 'source-repair-admission.json')
    assert admission['fd_engineering_status'] == 'FD_ENGINEERING_EQUIVALENCE_PASS'
    assert m['source_commit'] == '0d18820025e7a28675ed7112be0b1b4ff0110025'
    assert len(m['files']) == len({r['path'] for r in m['files']}) == 262
    assert sha_bytes(json.dumps({'source_commit': m['source_commit'], 'files': m['files']}, sort_keys=True).encode()) == m['snapshot_sha256']
    assert sha_bytes((json.dumps(m, indent=2) + '\n').encode()) == deployment['manifest_sha256']
    entries = {r['path']: r for r in m['files']}
    for row in m['files']:
        content = (ROOT / row['path']).read_bytes()
        assert len(content) == row['bytes'] and sha_bytes(content) == row['sha256']
        actual_git = subprocess.check_output(['git', 'show', m['source_commit'] + ':' + row['path']], cwd=ROOT)
        assert content == actual_git
    for row in admission['source_files']:
        assert entries[row['relative_path']]['sha256'] == row['sha256']
    github = read(SUPPORT / 'source-github-readback.json')
    github_rows = []
    for row in github:
        assert row['status'] == 'fulfilled' and not row['value'].get('isError')
        data = row['value']['structuredContent']
        content = data['content'].encode()
        assert sha_bytes(content) == entries[row['path']]['sha256']
        assert hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == data['sha']
        assert '/' + m['source_commit'] + '/' in data['display_url']
        github_rows.append({'path': row['path'], 'sha256': sha_bytes(content), 'display_url': data['display_url']})
    plan = read(COMPILED / 'plan.json')
    assert plan['source_python_sha256'] == {r['path']: r['sha256'] for r in m['files'] if r['path'].endswith('.py')}
    assert len(plan['source_python_sha256']) == 204
    scene = read(SUPPORT / 'initial-scene.json')
    shared = scene['config']
    old = {s['id']: s for s in shared['stages']}
    names = {'sportsa2-quantize', 'sportsa2-metadata', 'sportsa2-gpu-gate', 'sportsa2-train', 'sportsa2-eval-0', 'sportsa2-eval-3', 'sportsa2-eval-4', 'sportsa2-attention-0', 'sportsa2-attention-3', 'sportsa2-attention-4', 'sportsa2-stats'}
    paths = sorted((COMPILED / 'specs').glob('*.json'))
    assert {p.stem for p in paths} == names and len(paths) == 11
    old_capture = old['sportsa2-quantize']['request']['capture_version']
    capture = m['source_commit'] + '/' + m['snapshot_sha256']
    mapping = [(deployment['root'], shared['snapshot']['root']), (capture, old_capture)]
    mapping += [('ga-20261018-' + n + '-fd-recovery-v1', old[n]['request']['job_id']) for n in names]
    spec_rows = []
    for path in paths:
        value = read(path)
        assert value['capture_version'] == capture and value['workflow'] == 'Sports-A2'
        assert value['research_identity'] == 'Sports-A2-T5RKMeans-DualTable-NewTraining'
        assert value['seed'] == 42 and value['generation_k'] == 10 and value['test_selection_forbidden']
        assert plan_sha(value) in plan['template_plan_sha256']
        expected = read(ROOT / 'manifests/grid-a-launch-20261017/compiled/specs' / path.name)
        expected.pop('launch_review', None)
        actual = replace(value, mapping)
        actual.pop('launch_review', None)
        if path.stem == 'sportsa2-quantize':
            assert actual.pop('fd_engineering') == {'purpose': 'A2 DataLoader IPC/FD engineering adaptation', 'worker_hard_seconds': 3300}
            assert actual['prerequisite_capture_versions'] == {old['sportsa2-encode']['request']['job_id']: old_capture}
        else:
            assert actual['prerequisite_capture_versions'] == {}
        actual['prerequisite_capture_versions'] = expected['prerequisite_capture_versions']
        assert actual == expected, path.name
        spec_rows.append({'name': path.stem, 'phase': value['phase'], 'shuffle_n': value['shuffle_n'], 'plan_sha256': plan_sha(value), 'sha256': sha_bytes(path.read_bytes()), 'resolve': value.get('resolve')})
    assert len(set(plan['template_plan_sha256'])) == 11
    assert set(plan['template_plan_sha256']) == {r['plan_sha256'] for r in spec_rows}
    assert plan['only_fixed_seed'] == 42 and plan['only_recovery']
    assert plan['encode_reused'] == old['sportsa2-encode']['request']['job_id']
    assert plan['original_absolute_queue_deadline'] == shared['continuation']['original_absolute_queue_deadline']
    assert plan['workflows'] == {'Sports-A2': {'modes': sorted({r['phase'] for r in spec_rows}), 'results_review': 'NOT_PREAPPROVED'}}
    sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
    from transport import Transport
    transport = Transport('new-server')
    remote_identity = transport.identity()
    code = '''import pathlib,hashlib,json,resource,datetime
root=pathlib.Path(ROOT)
raw=(root/'DEPLOYMENT.json').read_bytes()
m=json.loads(raw)
rows=[]
for r in m['files']:
 p=root/r['path']
 assert not p.is_symlink()
 content=p.read_bytes()
 assert len(content)==r['bytes'] and hashlib.sha256(content).hexdigest()==r['sha256']
 if r['path'] in CRITICAL: rows.append(dict(r,mode=oct(p.stat().st_mode&0o777)))
plan=pathlib.Path(PLAN)
print(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),manifest_sha256=hashlib.sha256(raw).hexdigest(),source_commit=m['source_commit'],snapshot_sha256=m['snapshot_sha256'],all_files_verified=len(m['files']),critical_files=rows,plan_path=str(plan),plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),fresh_ssh_limits=list(resource.getrlimit(resource.RLIMIT_NOFILE)))))
'''.replace('ROOT', repr(deployment['root'])).replace('CRITICAL', repr([r['relative_path'] for r in admission['source_files']])).replace('PLAN', repr(REMOTE + '/plan.json'))
    remote = json.loads(transport.python(code, timeout=60))
    assert remote['all_files_verified'] == 262 and remote['manifest_sha256'] == deployment['manifest_sha256']
    assert remote['plan_sha256'] == sha_bytes((COMPILED / 'plan.json').read_bytes())
    assert remote['source_commit'] == m['source_commit'] and remote['snapshot_sha256'] == m['snapshot_sha256']
    assert all(r['mode'] == '0o444' for r in remote['critical_files'])
    result = {
        'reviewer': '/root/fd_repair_reviewer', 'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'IMMUTABLE_SOURCE_AND_ELEVEN_TEMPLATE_PLAN_PASS_CONCRETE_REQUEST_PENDING',
        'source_commit': m['source_commit'], 'snapshot': deployment['root'],
        'manifest_sha256': deployment['manifest_sha256'], 'all_262_local_files_Git_bytes_exact': True,
        'all_admitted_sources_same_bytes': True, 'source_python_count': 204,
        'github_fixed_commit_raw_contents_independently_checked': github_rows,
        'remote_identity': {k: remote_identity[k] for k in ['hostname', 'uid', 'user']},
        'independent_server_source_and_plan_read': remote,
        'eleven_templates_scientific_fields_equal_original_after_identity_relocation': spec_rows,
        'allowed_template_changes': ['fixed source/capture and job identities', 'new source path for unchanged training protocol', 'quantizer references exact old encode capture', 'remove unrelated historical prerequisite metadata overrides', 'scoped quantizer fd_engineering3300s'],
        'plan': {'path': REMOTE + '/plan.json', 'sha256': sha_bytes((COMPILED / 'plan.json').read_bytes())},
        'future_results_approved': False, 'concrete_requests_admitted': False,
        'scientific_result': 'SCIENTIFIC_RESULT_NOT_YET_AVAILABLE',
    }
    target = OUT / 'deployment-plan-independent-inspection.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(target), 'sha256': sha_bytes(target.read_bytes()), 'status': result['status']}))


if __name__ == '__main__':
    main()
