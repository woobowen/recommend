"""Content-bound CPU restoration invocation, preceding formal research launch."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport, sha, atomic
import gridctl

local = Path('/home/addaswsw/recommend-support/grid-a-launch-20261017')
repo = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
remote = '/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017'
t = Transport('new-server')
t.identity()
files = [repo / 'experiments/launch_20261017' / name for name in ('assets_prepare.py', 'a2_sources.json')]
identity = hashlib.sha256(''.join(sha(p) for p in files).encode()).hexdigest()
root = remote + '/preparation/' + identity[:16]
inputs = []
for path in files:
    receipt = t.put(path, root + '/' + path.name)
    inputs.append({k: receipt[k] for k in ('path', 'sha256')})
p = t.profile
job = 'ga-20261017-assets-prepare-v1'
request = {'schema': 1, 'job_id': job, 'host': p['hostname'], 'user': p['user'], 'uid': p['uid'],
    'namespace': p['namespace'], 'deployment_sha256': p['deployment_sha256'],
    'capture_version': 'nonresearch-asset-preparation/content-sha256/' + identity,
    'argv': ['/usr/bin/python3', '-I', '-S', '-B', root + '/assets_prepare.py', '--output-root', remote + '/assets'],
    'cwd': root, 'timeout_seconds': 21600, 'env': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
    'inputs': inputs}
path = local / (job + '-request.json')
atomic(path, request)
receipt = gridctl.submit(t, path)
atomic(local / (job + '-submission.json'), receipt)
print(json.dumps(receipt, indent=2))
