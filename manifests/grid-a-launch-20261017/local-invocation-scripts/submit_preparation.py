"""Submit named immutable preparation requests; caller decides after real queries."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, '/home/addaswsw/recommend-worktrees/sid-extensions-p0')
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from experiments.launch_20261017.build import make_request
from transport import Transport, atomic
import gridctl

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--timeout', type=int, required=True)
parser.add_argument('names', nargs='+')
args = parser.parse_args()
index = json.loads(args.index.read_text())
t = Transport('new-server')
t.identity()
for name in args.names:
    assert name.endswith(('-metadata', '-cpu-gate'))
    t.put(Path(index['templates'][name]), index['template_remote_paths'][name])
    request = make_request(t, index, name, args.timeout)
    path = Path(index['local_root']) / (name + '-request.json')
    atomic(path, request)
    receipt = gridctl.submit(t, path)
    atomic(path.with_name(name + '-submission.json'), receipt)
    print(json.dumps({'job_id': receipt['job_id'], 'status': receipt['status'],
        'received_at': receipt['received_at']}), flush=True)
