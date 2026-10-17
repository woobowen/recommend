"""Render only the four user-approved workflows; no metric-driven decisions."""
import argparse
import json
from pathlib import Path
import sys

from experiments.beauty_extensions.common import REPO, sha256, write_json
from .deploy import SUPPORT, REMOTE, TRANSPORT, request
from .run import template_plan_sha256

PYTHON = '/data/bwwu/envs/grid-20261009-e61baf3/bin/python'
JOB_ROOT = '/home/bwwu/grid-audits/sid-extensions-p0/jobs'
GPU0 = 'GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae'
GPU1 = 'GPU-59cac3bd-3645-668b-79c8-839e0d878893'
NAMES = {'toys5300': 'Toys-A-5300', 'toys5000': 'Toys-A-5000',
    'sportsa1': 'Sports-A1', 'sportsa2': 'Sports-A2'}


def job(name):
    return 'ga-20261017-' + name + ('-v2' if name == 'sportsa1-cpu-gate' else '-v1')


def build(deployment, t, reuse_index=None):
    short = deployment['manifest']['snapshot_sha256'][:16]
    root = SUPPORT / ('compiled-' + short)
    remote = REMOTE + '/compiled/' + short
    root.mkdir(parents=True, exist_ok=True)
    sources = json.loads((REPO / 'experiments/launch_20261017/a2_sources.json').read_text())
    by_path = {r['archive_path']: {'path': REMOTE + '/assets/' + r['archive_path'],
        'sha256': r['sha256']} for r in sources['original_assets']}
    config_beauty = by_path['A800_GRID/logs/train/runs/2025-11-03/02-47-09/.hydra/config.yaml']
    sid_toys = by_path['A800_GRID/logs/inference/runs/2025-09-17/21-57-54/pickle/merged_predictions_tensor.pt']
    capture = deployment['manifest']['source_commit'] + '/' + deployment['manifest']['snapshot_sha256']
    prior = {}
    if reuse_index is not None:
        old = reuse_index['source_deployment']['manifest']
        old_capture = old['source_commit'] + '/' + old['snapshot_sha256']
        prior = {job(name + '-metadata'): old_capture for name in ('toys5300', 'toys5000', 'sportsa1')}
    protocol = {'path': deployment['root'] + '/experiments/launch_20261017/training_protocol.json',
        'sha256': sha256(REPO / 'experiments/launch_20261017/training_protocol.json')}
    bases = {}
    for name, workflow in NAMES.items():
        domain = 'toys' if name.startswith('toys') else 'sports'
        spec = {'workflow': workflow, 'route': 'A', 'domain': domain, 'seed': 42,
            'generation_k': 10, 'shuffle_n': 0, 'formal_approved': True,
            'capture_version': capture, 'job_root': JOB_ROOT,
            'prerequisite_capture_versions': prior,
            'data_dir': '/data/bwwu/sid-extensions-assets/563dc50b1c7e/original_assets/data/amazon_data/' + domain,
            'config': config_beauty, 'training_protocol': protocol,
            'test_selection_forbidden': True, 'final_result_review': 'PENDING_FUTURE_CODEX'}
        if domain == 'toys':
            historical = '2025-10-29/23-42-04' if name == 'toys5300' else '2025-11-03/02-31-15'
            step = '005300' if name == 'toys5300' else '005000'
            prefix = 'A800_GRID/logs/train/runs/' + historical
            spec.update(config=by_path[prefix + '/.hydra/config.yaml'], sid=sid_toys,
                checkpoint=by_path[prefix + '/checkpoints/checkpoint_epoch=000_step=' + step + '.ckpt'],
                historical_uncertainty='Qwen/3584 configuration versus preserved 512-dimensional upstream producer is not uniquely resolved; new adaptation only.')
        elif name == 'sportsa1':
            spec.update(sid={'path': '/data/bwwu/sid-extensions-assets/563dc50b1c7e/experiments/sports_rkmeans_qwen_20261010_093035/semantic_ids.pt',
                'sha256': '89b2c2bbb94eda26ef0e147404fbff29e6ec8246fbdc754c5d885f313f2de293'},
                research_identity='Sports-A1-QwenSID-DualTable-NewTraining')
        else:
            spec.update(upstream_assets=REMOTE + '/assets',
                research_identity='Sports-A2-T5RKMeans-DualTable-NewTraining',
                resolve={'sid': {'job': job('sportsa2-quantize'), 'field': 'sid'}})
        bases[name] = spec
    templates, modes = {}, {}
    def add(name, mode, base, **extra):
        spec = json.loads(json.dumps(base))
        spec.update(extra)
        spec['phase'] = mode
        templates[name] = spec
        modes[name] = mode
    for name, base in bases.items():
        add(name + '-metadata', 'metadata', base)
        prepared = json.loads(json.dumps(base))
        prepared.setdefault('resolve', {})['metadata'] = job(name + '-metadata')
        if name.startswith('toys'):
            add(name + '-cpu-gate', 'admission-cpu', prepared)
            add(name + '-gpu-gate', 'admission-gpu', prepared)
        else:
            if name == 'sportsa1':
                add(name + '-cpu-gate', 'train-admission', prepared, admission_device='cpu')
            add(name + '-gpu-gate', 'train-admission', prepared, admission_device='gpu')
            add(name + '-train', 'train', prepared, hard_cap_hours=12)
        for n in (0, 3, 4):
            evaluating = json.loads(json.dumps(prepared))
            if name.startswith('sports'):
                evaluating['resolve']['checkpoint'] = {'job': job(name + '-train'), 'field': 'best_checkpoint'}
            add(name + '-eval-' + str(n), 'evaluate', evaluating, shuffle_n=n)
            observing = json.loads(json.dumps(evaluating))
            observing['resolve']['evaluation'] = job(name + '-eval-' + str(n))
            add(name + '-attention-' + str(n), 'attention', observing, shuffle_n=n)
        add(name + '-stats', 'stats', prepared,
            attention_jobs=[job(name + '-attention-' + str(n)) for n in (0, 3, 4)])
    upstream = json.loads(json.dumps(bases['sportsa2']))
    upstream.pop('resolve')
    add('sportsa2-encode', 'a2-encode', upstream, gpu_duty_cycle=0.2,
        engineering_schedule='Original batch8, sync and idle four times measured compute wall to reduce shared-GPU interference; all idle time charged.')
    add('sportsa2-quantize', 'a2-quantize', upstream,
        resolve={'embedding': {'job': job('sportsa2-encode'), 'field': 'embedding_asset'}})
    for name, spec in templates.items():
        write_json(root / 'specs' / (name + '.json'), spec)
    index = {'source_deployment': deployment, 'local_root': str(root), 'remote_root': remote,
        'modes': modes, 'templates': {name: str(root / 'specs' / (name + '.json')) for name in templates},
        'template_remote_paths': {name: remote + '/specs/' + name + '.json' for name in templates}}
    plan = {'workflows': {workflow: {'modes': sorted(set(modes.values())),
            'results_review': 'NOT_PREAPPROVED'} for workflow in NAMES.values()},
        'template_plan_sha256': [template_plan_sha256(spec) for spec in templates.values()],
        'hard_gpu_budget_seconds': 143100, 'only_fixed_seed': 42,
        'research_scope': 'two existing Toys checkpoints and two new Sports SID-source routes only',
        'source_python_sha256': {r['path']: r['sha256'] for r in deployment['manifest']['files'] if r['path'].endswith('.py')}}
    write_json(root / 'plan.json', plan)
    write_json(root / 'index.json', index)
    return index


def make_request(t, index, name, timeout, gpu=''):
    path = Path(index['templates'][name])
    remote = index['template_remote_paths'][name]
    spec = json.loads(path.read_text())
    inputs = [{'path': remote, 'sha256': sha256(path)}]
    for field in ('config', 'sid', 'checkpoint', 'training_protocol', 'launch_review'):
        if isinstance(spec.get(field), dict) and 'path' in spec[field]:
            inputs.append({k: spec[field][k] for k in ('path', 'sha256')})
    return request(t, index['source_deployment'], job(name),
        [PYTHON, '-B', '-m', 'experiments.launch_20261017.run', index['modes'][name], '--spec', remote],
        timeout, inputs, gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--reuse-preparation-index', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(TRANSPORT))
    from transport import Transport
    prior = json.loads(args.reuse_preparation_index.read_text()) if args.reuse_preparation_index else None
    value = build(json.loads(args.deployment.read_text()), Transport('new-server'), prior)
    print(json.dumps({'local_root': value['local_root'], 'templates': len(value['templates'])}))


if __name__ == '__main__':
    main()
