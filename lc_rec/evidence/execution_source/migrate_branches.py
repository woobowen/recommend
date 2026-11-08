#!/usr/bin/env python3
"""One-time explicitly authorized native GitHub branch renames. No force operations."""
import datetime, json, pathlib, subprocess, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence' / 'branch_migration.json'
REPO = 'repos/woobowen/recommend'
records = {'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'operations': []}
def save():
    OUT.write_text(json.dumps(records, indent=2) + '\n')
def api(path, *args):
    proc = subprocess.run(['gh', 'api', REPO + path, *args], text=True, capture_output=True, timeout=45)
    if proc.returncode:
        raise RuntimeError(f'GitHub {path}: exit {proc.returncode}: {proc.stderr}')
    return json.loads(proc.stdout)
def branches():
    return {v['name']: v for v in api('/branches?per_page=100')}
assert not OUT.exists(), 'Existing receipt: inspect before any retry'
before = branches()
records['before'] = before
records['repo_before'] = {k: api('')[k] for k in ('default_branch', 'visibility')}
records['open_prs'] = api('/pulls?state=open&per_page=100')
records['rulesets'] = api('/rulesets?per_page=100')
assert records['repo_before']['default_branch'] == 'main'
assert not records['open_prs'] and not records['rulesets']
mapping = [('wbw', 'wbw-grid'), ('wmh', 'wmh-grid'), ('yjc', 'yjc-grid')]
assert 'wbw-lc-rec' not in before
for old, new in mapping:
    assert old in before and new not in before and not before[old]['protected']
records['lc_base_sha'] = before['wbw']['commit']['sha']
save()
for old, new in mapping:
    current = branches()
    assert old in current and new not in current
    sha = current[old]['commit']['sha']
    assert sha == before[old]['commit']['sha'], 'Concurrent advance: stop without reset'
    op = {'old': old, 'new': new, 'sha_before': sha, 'request_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    records['operations'].append(op)
    save()
    try:
        op['rename_response'] = api('/branches/' + old + '/rename', '--method', 'POST', '-f', 'new_name=' + new)
    except Exception as exc:
        op['request_error'] = str(exc)
        save()
        raise
    for attempt in range(4):
        current = branches()
        op['readback'] = current
        # Collection reads check real refs, avoiding old-name redirect endpoints.
        refs = api('/git/matching-refs/heads/')
        op['actual_refs'] = {v['ref']: v['object']['sha'] for v in refs}
        if 'refs/heads/' + old not in op['actual_refs'] and op['actual_refs'].get('refs/heads/' + new) == sha:
            op['status'] = 'PASS'
            break
        if new in current and current[new]['commit']['sha'] != sha:
            raise RuntimeError('Concurrent or conflicting SHA; preserve state')
        time.sleep(2)
    else:
        op['status'] = 'BLOCKED_READBACK'
        save()
        raise RuntimeError('Bounded readback did not converge')
    save()
records['after_renames'] = branches()
records['finish_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save()
print(json.dumps({'lc_base_sha': records['lc_base_sha'], 'operations': [{k:v for k,v in op.items() if k in ('old','new','sha_before','status')} for op in records['operations']]}, indent=2))
