"""Evaluate Beauty predictions with the original 16 metric definitions and CCE."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from lc_rec.shuffle_v2 import beauty, core
from lc_rec.shuffle_v2.replay import dump, frozen


def derive_metrics(text):
    """Keep the frozen metric functions and make their shard contract explicit."""
    def change(before, after):
        nonlocal text
        assert text.count(before) == 1, before
        text = text.replace(before, after)

    change('    for rank in (0, 1):\n        path = args.test_dir',
           '    for rank, (expected_count, expected_assignment) in enumerate(args.assignments):\n        path = args.test_dir')
    change('len(assignments)==(N+1-rank)//2', 'len(assignments)==expected_count')
    change('assignment_sha == RANK_ASSIGNMENT_SHA[rank]', 'assignment_sha == expected_assignment')
    change("                candidates = row['candidates']",
           "                args.check_row(row)\n                candidates = row['candidates']")
    return text


def validate_input(row, replay, mode, manifest_hash, token_ids, original_callback_hash):
    evidence = row['shuffle']
    expected = replay['conditions'][mode]
    assert all(evidence.get(k) == v for k, v in core.mode_metadata(mode).items())
    assert evidence['row_idx'] == row['row_idx'] == replay['row_idx']
    assert (row['user_id'], row['target']) == (replay['user_id'], replay['target'])
    assert evidence['cpu_manifest_sha256'] == manifest_hash
    assert (evidence['original_sha256'], evidence['attention_mask_sha256']) == (replay['original_sha256'], replay['attention_mask_sha256'])
    assert evidence['history_group_count'] == replay['visible_history_length']
    assert evidence['input_changed'] == expected['input_changed']
    ids = row['input_token_ids']
    assert len(ids) == row['input_nonpadding_tokens'] == replay['encoded_length']
    assert len([token for token in ids if token in token_ids]) == replay['visible_history_length'] * 4
    assert core.canonical_sha256(ids) == evidence['shuffled_sha256'] == expected['input_sha256']
    prefill = evidence['actual_prefill']
    assert prefill['input_sha256'] == evidence['shuffled_sha256'] and prefill['attention_mask_sha256'] == evidence['attention_mask_sha256']
    assert prefill['position_ids_sha256'] == core.canonical_sha256(list(range(len(ids))))
    assert (prefill['cache_length'], prefill['beam_expansion']) == (0, 20)
    assert row['original_callback_verified_all_tokens'] and original_callback_hash
    assert all(c['eos_position'] == 4 and c['generated_token_ids'][4] == 2 for c in row['candidates'])


def evaluate(args, config):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    base = config['baseline']
    normal = args.mode == 'normal'
    assert normal == (args.run_commit == base['run-commit'])
    test = args.test_dir or Path(base['normal_test_dir'])
    original_config = json.loads(Path(config['normal_evaluation_config']).read_text())
    original_config['original_run_commit'] = args.run_commit
    stage_a = args.stage_a or (Path(base['normal_stage_a']) if normal else test / 'metrics.json')
    recorded_axis = original_config['user_axis_sha256']
    assert recorded_axis == config['user_axis_sha256']
    original_metrics = Path(config['frozen_metrics'])
    source = beauty.frozen_dependency(config, original_metrics)
    text = original_metrics.read_text()
    derived = derive_metrics(text)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'offline_metrics.derived.py').open('x') as stream:
        stream.write(derived)
    with (args.output / 'offline_metrics.diff').open('x') as stream:
        stream.writelines(difflib.unified_diff(text.splitlines(True), derived.splitlines(True),
                                              fromfile='frozen/offline_metrics.py', tofile='V2/offline_metrics.py'))
    config_path = args.output / 'evaluation-config.json'
    dump(config_path, original_config)
    if normal:
        assignments = list(zip((11182, 11181), source.RANK_ASSIGNMENT_SHA, strict=True))

        def check_row(row):
            assert 'shuffle' not in row and row['original_callback_verified_all_tokens']
    else:
        cpu = json.loads(args.cpu_audit.read_text())
        single = json.loads(args.single_shard_audit.read_text())
        manifest = args.cpu_audit.with_name('inputs.jsonl')
        manifest_hash = frozen.sha(manifest)
        assert manifest_hash == cpu['manifest_sha256'] == single['manifest_sha256']
        assert cpu['status'] == 'PASS_CPU_V2_FULL' and cpu['dataset'] == config['dataset']
        assert single['status'] == 'PASS_SINGLE_SHARD_CPU' and single['users'] == cpu['users'] == config['users']
        assert cpu['user_axis_sha256'] == recorded_axis
        with manifest.open() as stream:
            rows = [json.loads(line) for line in stream]
        assert all(row['row_idx'] == i for i, row in enumerate(rows))
        assignments = [(config['users'], single['assignment_sha256'])]
        token_ids = set(cpu['token_ids'].values())
        report = json.loads((test / 'rank0.json').read_text())
        assert report['status'] == 'PASS' and report['completed_users'] == config['users']
        assert report['world_size'] == 1 and report['rank'] == report['local_rank'] == 0
        assert report['RUN_COMMIT'] == args.run_commit
        assert report['adapter_files'] == base['adapter_files'] and report['model'] == base['model_identity']
        assert report['allowed_callback_sha256'] == base['allowed_callback_sha256']
        assert report['shuffle']['cpu_manifest_sha256'] == manifest_hash
        assert report['beauty_normal_binding'] == base['run-commit']

        def check_row(row):
            validate_input(row, rows[row['row_idx']], args.mode, manifest_hash, token_ids, report['allowed_callback_sha256'])

    audited = SimpleNamespace(test_dir=test, output=args.output / 'primary', stage_a_report=stage_a,
                              config=config_path, run_commit=args.run_commit, evaluation_commit=args.evaluation_commit,
                              official_cce=Path(config['official_cce']), data_root=Path(base['data-root']),
                              assignments=assignments, check_row=check_row)
    exec(compile(derived, str(original_metrics), 'exec'), source.__dict__)
    result = source.evaluate(audited)
    summary = json.loads(result.read_text())
    if normal:
        import csv
        import numpy as np
        primary = Path(config['normal_primary'])
        historical_metrics = json.loads((primary / 'metrics.json').read_text())
        assert summary['metrics'] == historical_metrics['metrics']
        assert summary['sid_statistics'] == historical_metrics['sid_statistics']
        with np.load(primary / 'per_user_metrics.npz', allow_pickle=False) as old, np.load(result.with_name('per_user_metrics.npz'), allow_pickle=False) as new:
            assert set(old.files) == set(new.files)
            assert all(np.array_equal(old[k], new[k]) for k in old.files)
        with (primary / 'per_user.csv').open(newline='') as old, result.with_name('per_user.csv').open(newline='') as new:
            assert all(left == right for left, right in zip(csv.DictReader(old), csv.DictReader(new), strict=True))
        summary['frozen_normal_cpu_reproduction'] = {'matrix_exact': True, 'summary_exact': True, 'per_user_csv_exact': True,
                                                     'historical_primary': str(primary)}
    summary.update(status='PASS_V2_EVALUATION', condition=args.mode, candidate_count=config['users'] * 20,
                   shards=len(assignments), evaluator_derived_sha256=hashlib.sha256(derived.encode()).hexdigest())
    dump(args.output / 'summary.json', summary)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--config', type=Path, default=beauty.CONFIG)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('normal', *core.MODES), required=True)
    p.add_argument('--run-commit', required=True)
    p.add_argument('--evaluation-commit', required=True)
    p.add_argument('--test-dir', type=Path)
    p.add_argument('--stage-a', type=Path)
    p.add_argument('--cpu-audit', type=Path)
    p.add_argument('--single-shard-audit', type=Path)
    args = p.parse_args()
    evaluate(args, json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
