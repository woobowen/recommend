"""Deterministic stage bridges; reuse the audited author models and evaluator."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

from experiments.beauty_extensions.common import (REPO, bind_source, instantiate,
    load_config, sha256, verify_asset, write_json)


def record(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def template_plan_sha256(template):
    plan = {k: v for k, v in template.items() if k != 'launch_review'}
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def completed(job_id, spec):
    root = Path(spec['job_root']) / job_id
    request = json.loads((root / 'request.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    result = json.loads((root / 'output/RESULT.json').read_text())
    assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
    assert status['cleanup']['verified']
    expected_capture = spec.get('prerequisite_capture_versions', {}).get(job_id, spec['capture_version'])
    assert request['capture_version'] == expected_capture
    assert result['capture_version'] == expected_capture
    assert result['workflow'] == spec['workflow']
    return root, result


def resolve(template):
    spec = dict(template)
    links = spec.pop('resolve', {})
    for field in ('sid', 'checkpoint', 'embedding'):
        if field in links:
            link = links[field]
            _, result = completed(link['job'], spec)
            spec[field] = result[link['field']]
            verify_asset(spec[field])
            if field == 'checkpoint':
                assert result['training_complete'] and not result.get('incomplete', False)
                assert result['sid']['sha256'] == spec['sid']['sha256']
    if 'metadata' in links:
        root, result = completed(links['metadata'], spec)
        assert result['sid']['sha256'] == spec['sid']['sha256']
        for field, filename in (('test_index', 'test_target_index.jsonl'),
                ('evaluation_index', 'evaluation_target_index.jsonl'),
                ('attention_cohort', 'fixed_attention_samples.json')):
            spec[field] = record(root / 'output' / filename)
    if 'evaluation' in links:
        root, result = completed(links['evaluation'], spec)
        assert result['formal'] and result['test_complete'] and result['status'] == 'COMPLETE'
        assert result['shuffle_n'] == spec['shuffle_n']
        for field in ('sid', 'checkpoint', 'config'):
            assert result[field] == spec[field]
        spec['evaluation_result'] = record(root / 'output/RESULT.json')
        spec['evaluation_status'] = record(root / 'status.json')
    return spec


def admit(template, mode):
    deployment = json.loads((REPO / 'DEPLOYMENT.json').read_text())
    assert template['route'] == 'A' and template['seed'] == 42
    assert template['phase'] == mode
    assert template['generation_k'] == 10 and template['shuffle_n'] in (0, 3, 4)
    assert template['capture_version'] == deployment['source_commit'] + '/' + deployment['snapshot_sha256']
    for item in deployment['files']:
        assert sha256(REPO / item['path']) == item['sha256'], item['path']
    if mode not in ('metadata', 'admission-cpu', 'admission-gpu', 'train-admission'):
        verify_asset(template['launch_review'])
        review = json.loads(Path(template['launch_review']['path']).read_text())
        assert review['status'] == 'SOURCE_AND_LAUNCH_ADMITTED_CONDITIONALLY'
        assert review['future_results_approved'] is False
        python_hashes = {r['path']: r['sha256'] for r in deployment['files'] if r['path'].endswith('.py')}
        assert review['source_python_sha256'] == python_hashes
        verify_asset(review['plan'])
        plan = json.loads(Path(review['plan']['path']).read_text())
        assert template['workflow'] in plan['workflows']
        assert mode in plan['workflows'][template['workflow']]['modes']
        # Fixed template identity, including deterministic future-asset links.
        assert template_plan_sha256(template) in plan['template_plan_sha256']


def metadata(spec, output):
    from experiments.formal_20261015.run import metadata as original
    cfg = load_config(spec, output)
    value = original(spec, cfg, output)
    value.update({key: record(output / filename) for key, filename in (
        ('test_index', 'test_target_index.jsonl'),
        ('evaluation_index', 'evaluation_target_index.jsonl'),
        ('attention_cohort', 'fixed_attention_samples.json'))})
    return value


def model_admission(spec, output, gpu=False):
    import hydra
    import torch
    from experiments.beauty_extensions.author.checks import describe_author_model, check_original_shuffle_trace
    from experiments.beauty_extensions.common import make_loader, batch_to
    from experiments.beauty_extensions.cli import check_data, numeric_difference
    cfg, model, _ = instantiate(spec, output, checkpoint=True)
    check_data(spec, cfg, output)
    structure = describe_author_model(model)
    assert structure['num_hierarchies'] == 4 and not structure['shared_sid_table']
    assert len(model.decoder.decoder_mlp) == 4
    device = 'cuda:0' if gpu else 'cpu'
    model.to(device).eval()
    batch = batch_to(next(iter(make_loader(cfg, 'val', limit_rows=1))), device)
    from omegaconf import OmegaConf
    args = OmegaConf.to_container(cfg.model, resolve=True)
    args['_target_'] = 'src.models.modules.semantic_id.tiger_generation_model_shuffle.SemanticIDEncoderDecoder'
    original = hydra.utils.instantiate(args).to(device).eval()
    loaded = original.load_state_dict(model.state_dict(), strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    with torch.inference_mode():
        logits, loss = model.model_step(batch[0], batch[1])
        old_logits, old_loss = original.model_step(batch[0], batch[1])
    assert torch.isfinite(loss)
    difference = {'logits': numeric_difference(logits, old_logits), 'loss': numeric_difference(loss, old_loss)}
    assert all(v['allclose'] for v in difference.values())
    shuffle = check_original_shuffle_trace(model)
    if gpu:
        from experiments.beauty_extensions.observation import capture_batch
        # Validation sample, engineering only; formal fixed32 was fixed earlier.
        observer = {}
        del original
        torch.cuda.empty_cache()
        for n in (0, 3, 4):
            model.shuffle_embedding_on_test = n != 0
            model.hierarchies_to_shuffle = n or 3
            observer[str(n)] = capture_batch({**spec, 'shuffle_n': n}, model, batch,
                output / ('observer_probe_' + str(n)), engineering=True)
    else:
        observer = None
    return {'status': 'PASS', 'formal': False, 'strict_checkpoint': True,
        'structure': structure, 'original_normal_parity': difference, 'shuffle_original_parity': shuffle,
        'observer_checks': observer, 'finite_loss': float(loss),
        'gpu_peak_bytes': torch.cuda.max_memory_allocated() if gpu else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['metadata', 'admission-cpu', 'admission-gpu',
        'train-admission', 'train', 'evaluate', 'attention', 'stats', 'a2-encode', 'a2-quantize'])
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    os.environ['GRID_STAGE_TEMPLATE'] = str(args.spec)
    template = json.loads(args.spec.read_text())
    job = Path(os.environ['GRID_JOB_DIR'])
    output = job / 'output'
    output.mkdir(exist_ok=True)
    rank_zero = int(os.environ.get('LOCAL_RANK', '0')) == 0
    if rank_zero:
        assert not (output / 'invocation.json').exists()
    request = json.loads((job / 'request.json').read_text())
    assert request['capture_version'] == template['capture_version']
    assert request['cwd'] == str(REPO)
    assert {r['path']: r['sha256'] for r in request['inputs']}[str(args.spec)] == sha256(args.spec)
    admit(template, args.mode)
    spec = resolve(template)
    for field in ('config', 'sid', 'checkpoint'):
        if field in spec:
            verify_asset(spec[field])
    if rank_zero:
        write_json(output / 'invocation.json', {'argv': sys.argv, 'interpreter': sys.executable,
            'template': record(args.spec), 'effective_spec': spec,
            'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'code_identity': template['capture_version']})
    bind_source('A')
    if args.mode == 'metadata':
        value = metadata(spec, output)
    elif args.mode in ('admission-cpu', 'admission-gpu'):
        value = model_admission(spec, output, args.mode == 'admission-gpu')
    elif args.mode in ('train', 'train-admission'):
        from .train import run_training, admission
        if args.mode == 'train':
            value = run_training(spec, output, request, json.loads((job / 'status.json').read_text()))
        else:
            value = admission(spec, output)
    elif args.mode in ('a2-encode', 'a2-quantize'):
        from .a2_upstream import bind_sources, item_axis_gate, encode, quantize
        from types import SimpleNamespace
        upstream = SimpleNamespace(assets=Path(spec['upstream_assets']), data=Path(spec['data_dir']),
            output=output, embedding=Path(spec['embedding']['path']) if 'embedding' in spec else None,
            gpu_duty_cycle=spec.get('gpu_duty_cycle', 1.0), fd_engineering=spec.get('fd_engineering'))
        if upstream.fd_engineering:
            assert args.mode == 'a2-quantize' and spec['workflow'] == 'Sports-A2'
        sources = bind_sources(upstream.assets, output)
        item_axis_gate(sources, upstream.data, output)
        (encode if args.mode == 'a2-encode' else quantize)(upstream, sources)
        if not rank_zero:
            return
        value = json.loads((output / 'stage_result.json').read_text())
        asset = record(value['path'])
        value['embedding_asset' if args.mode == 'a2-encode' else 'sid'] = asset
        value['method_gate'] = 'MECHANICAL_IDENTITY_ONLY; RESULTS_PENDING_FUTURE_REVIEW'
    elif args.mode in ('evaluate', 'attention'):
        from experiments.formal_20261015.run import evaluate, attention
        if args.mode == 'evaluate':
            assert spec['test_index'] and spec['attention_cohort']
            value = evaluate(spec, load_config(spec, output), output)
        else:
            value = attention(spec, output)
    else:
        from .statistics import analyze
        values = []
        for job_id in template['attention_jobs']:
            root, result = completed(job_id, spec)
            assert result['samples'] == 32 and result['status'] == 'COMPLETE'
            values.append(analyze(job_id, root))
        value = {'status': 'DESCRIPTIVE_STATISTICS_COMPLETE', 'jobs': values,
            'scientific_invariance_claim': None, 'final_result_review': 'PENDING_FUTURE_CODEX'}
    if 'sid' in spec:
        value['sid'] = spec['sid']
    value.update({'workflow': spec['workflow'], 'config': spec['config'],
        'capture_version': template['capture_version'], 'future_result_review': 'NOT_PREAPPROVED'})
    write_json(output / 'RESULT.json', value)
    print(json.dumps({'workflow': spec['workflow'], 'mode': args.mode, 'status': value['status']}), flush=True)


if __name__ == '__main__':
    main()
