"""Bind bounded, locked real GPU checks; no formal checkpoint is reused."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, '/home/addaswsw/recommend-worktrees/sid-extensions-p0')
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from experiments.launch_20261017.build import make_request, job, GPU0, JOB_ROOT
from experiments.launch_20261017.queue import GROUP_CAPS
from transport import Transport, atomic, sha

support = Path('/home/addaswsw/recommend-support/grid-a-launch-20261017')
parser=argparse.ArgumentParser()
parser.add_argument('--index',type=Path,required=True)
parser.add_argument('--only',nargs='+',default=['toys5300','toys5000','sportsa1'])
args=parser.parse_args()
index = json.loads(args.index.read_text())
t = Transport('new-server')
p = t.profile
remote_config = index['remote_root'] + '/gpu-admission-config.json'
config = {'schema': 1, 'ready': True, 'queue_id': 'ga-20261017-admission-config',
    'runner': {'python': p['python'], 'script': p['runner'], 'script_sha256': p['runner_sha256'],
        'job_root': p['job_root'], 'deployment_sha256': p['deployment_sha256']},
    'namespace_identity': {k: p['hostname'] if k == 'host' else p[k] for k in ('host', 'user', 'uid', 'namespace')},
    'snapshot': {'root': index['source_deployment']['root'], 'commit': index['source_deployment']['manifest']['source_commit']},
    'state_dir': index['remote_root'] + '/admission-state',
    'gpu_locks_dir': '/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/gpu-locks',
    'poll_seconds': 30, 'max_queue_seconds': 172700,
    'free_space': [{'path': '/home/bwwu', 'min_free_bytes': 20*1024**3}, {'path': '/data/bwwu', 'min_free_bytes': 20*1024**3}],
    'storage_roots': ['/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017'] +
        [JOB_ROOT + '/' + job(name + '-gpu-gate') for name in ('toys5300', 'toys5000', 'sportsa1')],
    'max_storage_bytes': 50*1024**3, 'gpu_budget_seconds': 143100,
    'budget_groups': GROUP_CAPS, 'blocked_workflows': {w:'Outside this isolated admission configuration; not a formal workflow cancellation.' for n,w in {'toys5300':'toys-a-5300','toys5000':'toys-a-5000','sportsa1':'sports-a1','sportsa2':'sports-a2'}.items() if n not in args.only}, 'stages': []}
for name, timeout in (('toys5300', 300), ('toys5000', 300), ('sportsa1', 900)):
    if name not in args.only:
        continue
    phase = name + '-gpu-gate'
    t.put(index['templates'][phase], index['template_remote_paths'][phase])
    workflow = {'toys5300': 'toys-a-5300', 'toys5000': 'toys-a-5000', 'sportsa1': 'sports-a1'}[name]
    status_path = JOB_ROOT + '/' + job(name + '-cpu-gate') + '/status.json'
    result_path = JOB_ROOT + '/' + job(name + '-cpu-gate') + '/output/RESULT.json'
    config['stages'].append({'id': phase, 'workflow': workflow, 'kind': 'admission',
        'depends_on': [], 'gpu_uuid': GPU0, 'budget_group': 'engineering',
        'request': make_request(t, index, phase, timeout, GPU0),
        'gates': [{'path': status_path, 'json_equals': {'status': 'SUCCEEDED', 'exit_code': 0, 'cleanup.verified': True}},
            {'path': result_path, 'json_equals': {'status': 'CPU_ADMISSION_PASS_GPU_PENDING' if name == 'sportsa1' else 'PASS'}}]})
local_config = Path(index['local_root']) / 'gpu-admission-config.json'
atomic(local_config, config)
t.put(local_config, remote_config)
command = [p['python'], '-I', '-S', '-B', index['source_deployment']['root'] + '/experiments/launch_20261017/queue.py',
    'render-requests', '--config', remote_config]
code = 'import subprocess;print(subprocess.check_output(' + repr(command) + ',text=True))'
requests = json.loads(t.python(code))
for req in requests:
    atomic(Path(index['local_root']) / (req['job_id'] + '-wrapped-request.json'), req)
atomic(Path(index['local_root']) / 'gpu-admission-render-receipt.json',
    {'remote_config': remote_config, 'config_sha256': sha(local_config), 'requests': requests})
print(json.dumps({'requests': len(requests), 'reserved_seconds': 1500, 'remote_config': remote_config}))
