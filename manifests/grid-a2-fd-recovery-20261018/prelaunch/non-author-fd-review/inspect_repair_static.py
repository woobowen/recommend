"""Independent exact-diff and fixed-source review, without research imports."""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
SUPPORT = Path('/home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018')
OUT = Path(__file__).resolve().parent
PACKAGE = Path('/home/addaswsw/recommend-support/grid-a-launch-20261017/deliveries/a2-emfile-capture-20261017T175451Z/package')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after=''):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


def main():
    baseline = json.loads((OUT / 'baseline-inspection.json').read_text())
    original_preserved = []
    for row in baseline['local_vs_preserved_C2']:
        name = row['relative_path']
        if name.startswith('references/'):
            assert sha(ROOT / name) == row['local']['sha256']
            original_preserved.append({'path': name, 'sha256': sha(ROOT / name)})
    upstream_path = ROOT / 'experiments/launch_20261017/a2_upstream.py'
    text = upstream_path.read_text()
    text = replace_once(text, '''    fd_engineering = getattr(args, "fd_engineering", None)
    if fd_engineering:
        from experiments.a2_fd_recovery_20261018.adapter import configure_process, adapt_config, resource_callback
        configure_process(args.output)
''')
    text = replace_once(text, '''    fd_callbacks = []
    if fd_engineering:
        changes = adapt_config(cfg)
        write_json(args.output / ("fd-config-rank-" + os.environ.get("LOCAL_RANK", "0") + ".json"), changes)
        fd_callbacks = [resource_callback(args.output, tensor)]
''')
    text = replace_once(text, 'fd_engineering["worker_hard_seconds"] if fd_engineering else 3600', '3600')
    text = replace_once(text, 'PredictionWriter(args.output, "cluster_ids"), *fd_callbacks]', 'PredictionWriter(args.output, "cluster_ids")]')
    text = replace_once(text, '''    if fd_engineering and trainer.is_global_zero:
        payload = torch.load(args.output / "quantizer.ckpt", map_location="cpu", weights_only=False)
        assert payload["global_step"] == 30
        current = model.state_dict()
        assert current.keys() == payload["state_dict"].keys()
        assert all(torch.equal(value.detach().cpu(), payload["state_dict"][key]) for key, value in current.items())
        strict = model.load_state_dict(payload["state_dict"], strict=True)
        assert not strict.missing_keys and not strict.unexpected_keys
        write_json(args.output / "quantizer-strict-load.json", {"strict_load": True, "global_step": 30,
                   "state_exact_to_trained_model": True, "keys": len(current),
                   "initialized": [bool(q.is_initialized) for q in model.quantization_layer_list]})
''')
    assert text == (PACKAGE / 'a2-code/scientific-C2/experiments/launch_20261017/a2_upstream.py').read_text()
    run_path = ROOT / 'experiments/launch_20261017/run.py'
    text = run_path.read_text()
    text = replace_once(text, '''gpu_duty_cycle=spec.get('gpu_duty_cycle', 1.0), fd_engineering=spec.get('fd_engineering'))
        if upstream.fd_engineering:
            assert args.mode == 'a2-quantize' and spec['workflow'] == 'Sports-A2'
''', "gpu_duty_cycle=spec.get('gpu_duty_cycle', 1.0))\n")
    assert text == (PACKAGE / 'a2-code/scientific-C2/experiments/launch_20261017/run.py').read_text()
    candidate = json.loads((SUPPORT / 'cpu-candidate.json').read_text())
    candidate_checks = []
    for row in candidate['files']:
        path = ROOT / row['path']
        ast.parse(path.read_text(), filename=str(path))
        assert row['bytes'] == path.stat().st_size and row['sha256'] == sha(path)
        candidate_checks.append({**row, 'current_local_exact': True})
    adapter = ROOT / 'experiments/a2_fd_recovery_20261018/adapter.py'
    tree = ast.parse(adapter.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collate_items_contiguous')
    source = ast.get_source_segment(adapter.read_text(), node)
    assert 'result = collate_fn_items(batch, item_id_field, feature_to_input_name)' in source
    assert 'result.item_ids = torch.stack(result.item_ids, dim=0)' in source
    assert 'worker_init_fn' not in adapter.read_text()
    assert 'set_sharing_strategy' not in adapter.read_text()
    for name in ['random.', 'np.random', 'torch.rand', 'torch.seed', 'torch.manual_seed']:
        assert name not in adapter.read_text(), name
    result = {
        'reviewer': '/root/fd_repair_reviewer', 'at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'STATIC_REPAIR_REVIEW_PASS_RUNTIME_AND_REQUEST_REVIEW_PENDING',
        'scientific_result': 'SCIENTIFIC_RESULT_NOT_YET_AVAILABLE',
        'exact_upstream_difference_is_only_scoped_engineering': True,
        'exact_run_difference_is_only_scoped_engineering': True,
        'preserved_author_files_unchanged': original_preserved,
        'candidate_executed_files_current_local_exact': candidate_checks,
        'legacy_scoped_paths': [{'path': str(p), 'sha256': sha(p)} for p in [upstream_path, run_path]],
        'RNG_worker_init_and_sharing_strategy_mutations_absent_from_adapter': True,
        'remaining': ['read raw CPU equivalence/IPC outputs and runner terminal/cleanup',
                      'read complete new A2-only request/config/dependency/budget/deadline binding',
                      'verify immutable source publication/deployment binding before formal submission'],
        'no_dependencies_installed': True,
    }
    path = OUT / 'static-repair-inspection.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(path), 'sha256': sha(path), 'status': result['status']}))


if __name__ == '__main__':
    main()
