"""Frozen A0 metrics with V2-specific full prediction and prefill verification."""
import argparse
import difflib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from lc_rec.shuffle_v2 import core
from lc_rec.shuffle_v2.replay import CONFIG, dump, frozen

LC = CONFIG.parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def verify_predictions(test, mode, run, audit, single_shard, config):
    from torch.utils.data.distributed import DistributedSampler
    rows = [json.loads(line) for line in (audit.parent / 'inputs.jsonl').open()]
    receipt = json.loads(audit.read_text())
    assert receipt['status'] == 'PASS_CPU_V2_FULL' and receipt['protocol'] == core.PROTOCOL
    assert frozen.sha(audit.parent / 'inputs.jsonl') == receipt['manifest_sha256']
    index, catalog, users, _ = frozen.catalog_and_users(config['baseline']['data-root'])
    assert len(rows) == len(users) == config['users']
    single = json.loads(single_shard.read_text())
    assert single['status'] == 'PASS_SINGLE_SHARD_CPU' and single['manifest_sha256'] == receipt['manifest_sha256']
    evaluator = frozen.load_evaluator(config['baseline']['source-root'])
    symbol_ids = receipt['token_ids']
    allowed_by_position = [{symbol_ids[sid[pos]] for sid in index.values()} for pos in range(4)]
    eos_path = Path(config['baseline']['model']) / 'generation_config.json'
    eos = json.loads(eos_path.read_text())['eos_token_id']
    documents, shards = [], []
    unchanged = histories = candidates = 0
    for rank in (0,):
        report = json.loads((test / f'rank{rank}.json').read_text())
        assert report['status'] == 'PASS' and report['RUN_COMMIT'] == run and report['completed_users'] == config['users'] and report['world_size'] == 1
        assert report['adapter_files']['adapter_model.bin'] == config['baseline']['adapter_identity']
        assert report['assignment_sha256'] == single['assignment_sha256']
        assert {k: report['shuffle'][k] for k in core.mode_metadata(mode)} == core.mode_metadata(mode)
        assignment = list(DistributedSampler(range(len(users)), num_replicas=1, rank=rank,
                                             shuffle=True, seed=0, drop_last=True))
        assert report['assignment_sha256'] == core.canonical_sha256(assignment)
        path = test / f'predictions.rank{rank}.jsonl'
        with path.open('rb') as stream:
            from hashlib import sha256
            content_hash = sha256()
            for wanted, line in zip(assignment, stream, strict=True):
                content_hash.update(line)
                row = json.loads(line)
                assert row['row_idx'] == wanted and row['rank'] == rank
                check = rows[wanted]
                desired = check['conditions'][mode]
                ev = row['shuffle']
                assert {k: ev[k] for k in core.mode_metadata(mode)} == core.mode_metadata(mode)
                assert ev['row_idx'] == wanted and row['prompt_id'] == 0
                assert ev['cpu_manifest_sha256'] == receipt['manifest_sha256']
                assert ev['input_changed'] == desired['input_changed']
                assert ev['original_sha256'] == check['original_sha256']
                assert ev['attention_mask_sha256'] == check['attention_mask_sha256']
                assert ev['shuffled_sha256'] == desired['input_sha256'] == core.canonical_sha256(row['input_token_ids'])
                assert len(row['input_token_ids']) == row['input_nonpadding_tokens'] == check['encoded_length']
                assert ev['actual_prefill'] == {
                    'input_sha256': desired['input_sha256'], 'attention_mask_sha256': check['attention_mask_sha256'],
                    'position_ids_sha256': core.canonical_sha256(list(range(check['encoded_length']))),
                    'cache_length': 0, 'beam_expansion': 20,
                }
                assert ev['history_group_count'] == check['visible_history_length']
                assert row['target'] == check['target'] and row['user_id'] == check['user_id']
                frozen.audit_prediction(row, users[wanted], catalog, evaluator)
                for candidate in row['candidates']:
                    token_ids = candidate['generated_token_ids']
                    assert candidate['eos_position'] == 4 and token_ids[4] == eos
                    assert all(token_ids[i] in allowed for i, allowed in enumerate(allowed_by_position))
                    assert all(token in (eos, 0) for token in token_ids[5:])
                histories += ev['history_group_count']
                unchanged += not ev['input_changed']
                candidates += len(row['candidates'])
        documents.append({'rank': rank, 'path': str(path.resolve()), 'bytes': path.stat().st_size,
                          'sha256': content_hash.hexdigest()})
        source = test / f'derived.rank{rank}.py'
        shards.append({'rank': rank, 'rows': len(assignment), 'assignment_sha256': report['assignment_sha256'],
                       'derived_entry_sha256': frozen.sha(source), 'rank_report_sha256': frozen.sha(test / f'rank{rank}.json')})
    assert histories == config['groups'] and candidates == config['users'] * 20
    assert unchanged == receipt['modes'][mode]['unchanged_users']
    proof = {'status': 'PASS_V2_FULL_PREDICTIONS', **core.mode_metadata(mode),
             'rows': len(users), 'history_groups': histories, 'candidates': candidates,
             'unchanged_users': unchanged, 'cpu_manifest_sha256': receipt['manifest_sha256'],
             'shards': shards, 'model_boundary_checks': 'CPU hashes match every actual generate and first model prefill'}
    return documents, proof


def normal_predictions(test, config):
    identities = {asset['path']: asset['sha256'] for asset in config['files']}
    paths = [test / f'predictions.rank{rank}.jsonl' for rank in range(2)]
    manifests = []
    for rank, path in enumerate(paths):
        digest = frozen.sha(path)
        assert digest == identities[str(path)], path
        manifests.append({'rank': rank, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest})
    return manifests


def derive_evaluator(text, role):
    def replace(old, new, occurrences=1):
        nonlocal text
        assert text.count(old) == occurrences, (role, old)
        text = text.replace(old, new)
    if role == 'original':
        replace('for rank in (0, 1):', 'for rank in (0,):', 2)
        replace('N // 2', 'N', 3)
        replace('and state.get("assigned_users") == N, "Rank formal completion")',
                'and state.get("assigned_users") == N and state.get("world_size") == 1, "Rank formal completion")')
        replace('key in states[0] and key in states[1]\n              and states[0][key] == states[1][key]', 'key in states[0]')
        replace('all("generation" in state for state in states)\n              and states[0]["generation"] == states[1]["generation"]',
                '"generation" in states[0]')
        replace('len(claims) == 2', 'len(claims) == 1')
        replace('set(claims) == {0, 1}', 'set(claims) == {0}')
    elif role == 'primary':
        replace('for rank in (0, 1):', 'for rank in (0,):')
        replace('len(assignments)==N//2', 'len(assignments)==N')
    elif role == 'extended':
        replace('len(records) == 2, "Two frozen prediction shards"', 'len(records) == 1, "One frozen prediction shard"')
        replace('and rank in (0, 1)', 'and rank == 0')
        replace('set(predictions) == {0, 1}', 'set(predictions) == {0}')
        replace('len(stage_records) == 2', 'len(stage_records) == 1')
        replace('set(stage_manifests) == {0, 1}', 'set(stage_manifests) == {0}')
        replace('for rank in (0, 1):', 'for rank in (0,):')
        replace('N // 2', 'N', 2)
    else:
        raise ValueError(role)
    return text


def single_evaluator(out, path, role):
    text = path.read_text()
    derived = derive_evaluator(text, role)
    actual = out / f'evaluator-{role}.py'
    compile(derived, str(actual), 'exec')
    actual.write_text(derived)
    diff = out / f'evaluator-{role}.diff'
    diff.write_text(''.join(difflib.unified_diff(text.splitlines(True), derived.splitlines(True),
                                                 fromfile=str(path), tofile=actual.name)))
    return actual


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--mode', choices=('normal', *core.MODES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path)
    parser.add_argument('--cpu-audit', type=Path)
    parser.add_argument('--single-shard-audit', type=Path)
    parser.add_argument('--run-commit', required=True, help='Commit of the predictions')
    parser.add_argument('--evaluation-commit', required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    config = json.loads(CONFIG.read_text())
    data = Path(config['baseline']['data-root'])
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if args.mode == 'normal':
        test = Path(config['baseline']['normal_test_dir'])
        assert args.test_dir is None and args.run_commit == config['baseline']['run-commit']
        documents = normal_predictions(test, config)
    else:
        test = args.test_dir
        documents, proof = verify_predictions(test, args.mode, args.run_commit, args.cpu_audit, args.single_shard_audit, config)
        dump(out / 'prediction_checks.json', proof)
    manifest = {'schema': 'a1_frozen_prediction_manifest_v1', 'condition': args.mode, 'model_key': 'a0',
                'gpu_run_commit': args.run_commit, 'test_dir': str(test), 'predictions': documents}
    dump(out / 'prediction_manifest.json', manifest)
    original = LC / 'letter/review/formal_det_20261112/independent_metrics.py'
    stage_a = out / 'independent_original.json'
    evaluator = LC / 'a1_qwen3_06b/scripts/offline_metrics.py'
    official = LC / 'letter/review/final_audit_20261113/external/official_cce.py'
    independent = LC / 'a1_qwen3_06b/review/independent_extended.py'
    config_path = LC / 'letter/config/a0_evaluation_v1.json'
    original_paths = (original, evaluator, independent, official, config_path)
    if args.mode != 'normal':
        original = single_evaluator(out, original, 'original')
        evaluator = single_evaluator(out, evaluator, 'primary')
        independent = single_evaluator(out, independent, 'extended')
    protocol = json.loads(config_path.read_text())
    protocol['original_run_commit'] = args.run_commit
    dump(out / 'effective_config.json', protocol)
    subprocess.run([sys.executable, str(original), '--data-root', str(data), '--test-dir', str(test),
                    '--output', str(stage_a)], check=True)
    primary = module('frozen_primary', evaluator)
    primary.RUN = args.run_commit
    primary.PRED = [(r['bytes'], r['sha256']) for r in documents]
    from types import SimpleNamespace
    primary.evaluate(SimpleNamespace(output=out / 'primary', test_dir=test, config=out / 'effective_config.json',
                     official_cce=official, stage_a_report=stage_a, data_root=data, evaluation_commit=args.evaluation_commit))
    subprocess.run([sys.executable, str(independent), '--data-root', str(data), '--test-dir', str(test),
                    '--stage-a-report', str(stage_a), '--manifest', str(out / 'prediction_manifest.json'),
                    '--per-user', str(out / 'primary/per_user.csv'), '--official-cce', str(official),
                    '--official-sha256', protocol['official_cce_sha256'], '--output', str(out / 'independent_extended.json')], check=True)
    results = json.loads((out / 'primary/metrics.json').read_text())['metrics']
    corroborated = json.loads((out / 'independent_extended.json').read_text())['metrics']
    assert set(results) == set(corroborated) and len(results) == 16
    assert max(abs(v - corroborated[k]) for k, v in results.items()) <= 1e-13
    if args.mode == 'normal':
        import numpy as np
        with (np.load(out / 'primary/per_user_metrics.npz', allow_pickle=False) as new,
              np.load(Path(config['normal_primary']) / 'per_user_metrics.npz', allow_pickle=False) as old):
            assert set(new.files) == set(old.files)
            assert all(np.array_equal(new[key], old[key]) for key in new.files)
    dump(out / 'summary.json', {'status': 'PASS_V2_EVALUATION', 'prediction_run_commit': args.run_commit,
                               'evaluation_run_commit': args.evaluation_commit, 'condition': args.mode,
                               'metrics': results, 'gpu_used': False,
                               'evaluation_source_sha256': {p.name: frozen.sha(p) for p in (evaluator, original, independent, official, config_path)},
                               'frozen_evaluation_source_sha256': {p.name: frozen.sha(p) for p in original_paths}})


if __name__ == '__main__':
    main()
