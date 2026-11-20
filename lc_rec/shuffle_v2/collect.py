"""Retrieve only task-bound lightweight outputs with verification via Windows SCP."""
import argparse
import json
from pathlib import Path

from lc_rec.shuffle_v2 import control, core


def receive(connection, remote, local):
    meta = json.loads(control.remote_python(connection, f'''from pathlib import Path
import hashlib,json
p=Path({remote!r}); assert p.is_file() and not p.is_symlink() and p.stat().st_size < 20*1024**2
print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))'''))
    local.parent.mkdir(parents=True, exist_ok=True)
    return connection.get_verified(remote, local, meta)


def hardware(connection):
    return json.loads(control.remote_python(connection, '''import datetime,json,subprocess
api=['nvidia-smi']
def run(*args): return subprocess.check_output(api+list(args), text=True)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(dict(observed_utc=now,gpus=run('--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv'),
                        processes=run('--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'),
                        space=subprocess.check_output(['df','-hP','/home/bwwu','/data/bwwu'],text=True))))'''))


def collect(connection, output, complete):
    states = control.job_states(connection)
    captured = {'states': states, 'hardware': hardware(connection), 'collected': []}
    evidence = captured['collected']
    cpu_replay = control.stage_success(states, 'cpu-replay')
    fixtures = control.stage_success(states, 'cpu-fixtures')
    boundary_fixture = control.stage_success(states, 'cpu-boundary-fixture')
    normal = control.stage_success(states, 'cpu-eval-normal')
    single = control.stage_success(states, 'cpu-single-shard')
    eval_fixture = control.stage_success(states, 'cpu-evaluation-fixture')
    paths = {eval_fixture: ('summary.json',), single: ('audit.json',), cpu_replay: ('audit.json','assets.json','examples.json','samples.json'), fixtures: ('summary.json',),
             boundary_fixture: ('summary.json',),
             normal: ('summary.json','prediction_manifest.json','effective_config.json','independent_original.json', 'independent_extended.json',
                      'primary/metrics.json','primary/per_user_metrics.npz', 'primary/bootstrap_draw_means.npy','primary/bootstrap.json')}
    if complete:
        preflight = control.stage_success(states, 'gpu-preflight')
        paths[preflight] = ('summary.json', *(f'{mode}/predictions.rank{rank}.jsonl' for mode in core.MODES for rank in (0,)),
                            *(f'{mode}/rank{rank}.json' for mode in core.MODES for rank in (0,)),
                            *(f'{mode}/derived.rank{rank}.diff' for mode in core.MODES for rank in (0,)))
        for mode in core.MODES:
            gpu = control.stage_success(states, 'gpu-formal-' + mode)
            cpu = control.stage_success(states, 'cpu-eval-' + mode)
            paths[gpu] = ('summary.json', f'{mode}.request.json', f'{mode}.exit.json',
                          *(f'test/rank{rank}.json' for rank in (0,)),
                          *(f'test/derived.rank{rank}.diff' for rank in (0,)))
            paths[cpu] = ('summary.json','prediction_manifest.json','prediction_checks.json','effective_config.json',
                          'independent_original.json','independent_extended.json','primary/metrics.json','primary/per_user_metrics.npz',
                          *(f'evaluator-{r}.diff' for r in ('original','primary','extended')))
        paths[control.stage_success(states, 'group-series')] = ('summary.json',)
        paths[control.stage_success(states, 'cpu-summary')] = ('summary.json', 'metrics.csv', 'paired.csv', 'arrays.npz')
    for job, files in paths.items():
        for relative in files:
            evidence.append(receive(connection, f'{control.ROOT}/{job}/{relative}', output / job / relative))
    for full_id, state in states.items():
        leaves = ('request.json', 'accepted.json', 'status.json')
        if state.get('cleanup') is not None:
            leaves += ('cleanup.json',)
        for relative in leaves:
            evidence.append(receive(connection, f'{control.BASE}/jobs/{full_id}/{relative}', output / 'jobs' / full_id / relative))
    control.transport.atomic(output / 'collection.json', captured)
    return {'jobs': len(states), 'files': len(evidence), 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--hardware-only', action='store_true')
    args = parser.parse_args()
    connection = control.connect()
    if args.hardware_only:
        evidence = hardware(connection)
        control.transport.atomic(args.output, evidence)
    else:
        evidence = collect(connection, args.output, args.complete)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
