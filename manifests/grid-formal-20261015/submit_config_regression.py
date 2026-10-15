"""Submit the approved CPU-only serialization regression to the existing runner."""
import json
from pathlib import Path
import sys

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
sys.path[:0] = [str(REPO), '/home/addaswsw/recommend-support/sid-extensions-p0/scripts']
from transport import Transport
import gridctl
from experiments.formal_20261015.deploy import REMOTE, snapshot
from experiments.beauty_extensions.common import sha256, write_json


def main():
    t = Transport('new-server')
    t.identity()
    root, manifest, manifest_sha = snapshot(t)
    script = SUPPORT / 'config_roundtrip.py'
    published = REPO / 'manifests/grid-formal-20261015/config_roundtrip.py'
    assert script.read_bytes() == published.read_bytes()
    remote_script = REMOTE + '/regression/' + sha256(script) + '/config_roundtrip.py'
    t.put(script, remote_script)
    inputs = [{'path': remote_script, 'sha256': sha256(script)},
              {'path': root + '/DEPLOYMENT.json', 'sha256': manifest_sha}]
    argv = ['/data/bwwu/envs/grid-20261009-e61baf3/bin/python', '-B', remote_script, '--repo', root]
    for name in ('beauty-a-0', 'beauty-b-0', 'toys-mlp', 'sports-mlp'):
        path = REPO / 'experiments/formal_20261015/specs' / (name + '.json')
        remote = REMOTE + '/specs/' + sha256(path) + '.json'
        t.put(path, remote)
        argv += ['--spec', remote]
        inputs.append({'path': remote, 'sha256': sha256(path)})
        spec = json.loads(path.read_text())
        inputs.extend({key: spec[name][key] for key in ('path', 'sha256')}
                      for name in ('config', 'sid'))
    job = 'gf-20261015-config-roundtrip-r4'
    request = {'schema': 1, 'job_id': job, 'host': t.profile['hostname'], 'user': t.profile['user'],
        'uid': t.profile['uid'], 'namespace': t.profile['namespace'],
        'deployment_sha256': t.profile['deployment_sha256'],
        'capture_version': manifest['source_commit'] + '/' + manifest['snapshot_sha256'],
        'argv': argv, 'cwd': root, 'timeout_seconds': 600,
        'env': {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4', 'TF_CPP_MIN_LOG_LEVEL': '3',
            'TF_NUM_INTRAOP_THREADS': '2', 'TF_NUM_INTEROP_THREADS': '2',
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': root, 'HF_HUB_OFFLINE': '1',
            'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false'},
        'inputs': inputs}
    path = SUPPORT / (job + '-request.json')
    if path.exists():
        assert json.loads(path.read_text()) == request, 'Query original ID; request differs'
    write_json(path, request)
    receipt = gridctl.submit(t, path)
    write_json(SUPPORT / (job + '-submission.json'), receipt)
    print(json.dumps({'deployment': root, 'receipt': receipt}, ensure_ascii=False))


if __name__ == '__main__':
    main()
