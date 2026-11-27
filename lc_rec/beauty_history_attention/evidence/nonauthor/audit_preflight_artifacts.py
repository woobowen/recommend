"""Independent, read-only audit of the bounded 7B outputs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

MODES = ('normal', 'shuffle2', 'shuffle3a', 'shuffle3b', 'shuffle4')
REPLAY = Path('/data/bwwu/lc-rec/letter-shuffle-v2-beauty/cpu-replay-r3')
ASSET = Path('/home/bwwu/grid-audits/sid-extensions-p0/transfers/letter-a0-beauty-author-8d0154e28de3/Beauty/Beauty.index.json')
JOBS = Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')

def load(path):
    return json.loads(path.read_text())

def lines(path):
    with path.open() as f:
        return list(map(json.loads, f))

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(',', ':'), sort_keys=True).encode()).hexdigest()

def check_train(folder):
    rank = load(folder / 'rank0.json')
    assert rank['mode'] == 'bidirectional' and rank['scope'] == 'PREFLIGHT'
    assert rank['status'] == 'BOUNDARY_PREFLIGHT_COMPLETE'
    assert rank['sample_flow']['max_steps'] == rank['global_step'] == 10
    assert rank['optimizer_parameter_count'] == 290037760 and len(rank['optimizer_parameter_names']) == 450
    assert rank['optimizer_membership_exact']
    assert rank['versions']['torch'] == '2.6.0' and rank['versions']['transformers'] == '4.31.0'
    assert rank['parameter_update_validation']['status'] == 'PASS'
    assert rank['parameter_update_validation']['all_frozen_tensor_bytes_unchanged']
    assert all(v > 0 for v in rank['parameter_update_validation']['changed_families'].values())
    events = lines(folder / 'events.rank0.jsonl')
    terminal = [e for e in events if e['event'] == 'epoch_end']
    assert [e['step'] for e in terminal] == [2, 5, 7, 10]
    micro = [e for e in events if e['event'] == 'micro_loss']
    assert len(micro) == 80
    for i, e in enumerate(micro, 1):
        assert math.isfinite(e['loss'])
        assert e['ds_global_step_before'] == (i-1)//8 and e['ds_global_step_after'] == i//8
        assert e['ds_accumulation_boundary'] == (i % 8 == 0)
        assert e['scheduler_last_epoch'] == (i-1)//8
    updates = [e for e in events if e['event'] == 'optimizer_step']
    assert [e['step'] for e in updates] == list(range(1, 11))
    assert all(e['grad_norm'] is not None and math.isfinite(e['grad_norm']) for e in updates)
    candidates = [(v['values']['eval_loss'], v['step']) for v in events if v['event'] == 'log' and 'eval_loss' in v.get('values',{}) and 'step' in v]
    best = load(folder / 'best_selection.json')
    assert best['export_matches_selected_checkpoint_tensors']
    assert best['scope'] == 'PREFLIGHT' and best['completed_training_step'] == 10
    assert best['mode'] == 'bidirectional' and best['run_commit'] == rank['run_commit']
    assert len(candidates) == 4
    assert min(candidates) == (best['value'], int(Path(best['checkpoint']).name.split('-')[-1]))
    restored = [e for e in events if e['event'] == 'partial_restore_exact']
    assert len(restored) == 1 and restored[0]['micro_steps'] == 20 and restored[0]['global_steps'] == 2
    return {'run_commit': rank['run_commit'], 'optimizer_updates': [e['step'] for e in updates],
            'epoch_boundary_updates': [e['step'] for e in terminal], 'gradient_norms': [e['grad_norm'] for e in updates],
            'checkpoint_selection': best, 'validation_losses': candidates, 'restored': restored, 'parameter_changes': rank['parameter_update_validation']['changed_families'],
            'elapsed_seconds': rank['elapsed_seconds'], 'peak_allocated_bytes': rank['peak_allocated_bytes'],
            'peak_reserved_bytes': rank['peak_reserved_bytes']}

def metric_oracle(ranked, target, catalog):
    target_slots = [i for i, sid in enumerate(ranked, 1) if sid == target]
    first = target_slots[0] if target_slots else math.inf
    result = {f'hit@{k}': float(first <= k) for k in (1,5,10,20)}
    for k in (5,10,20):
        result[f'mrr@{k}'] = 1./first if first <= k else 0.
        result[f'ndcg@{k}'] = sum(1./math.log2(i+1) for i in target_slots if i <= k)
    cursor = 0
    possible_target_item_ranks = []
    for sid in ranked:
        g = catalog.get(sid,1)
        if sid == target:
            possible_target_item_ranks = list(range(cursor+1,cursor+g+1))
            break
        cursor += g
    for k in (5,10,20):
        credit = [i for i in possible_target_item_ranks if i <= k]
        result[f'ItemHit@{k}'] = len(credit)/catalog.get(target,1)
        result[f'ItemNDCG@{k}'] = sum(1./math.log2(i+1) for i in credit)/catalog.get(target,1)
    return result

def check_generation(folder, metrics):
    report = load(folder / 'generation.json')
    assert report['status'] == 'PREFLIGHT_INTERFACE_PASS' and report['purpose'] == 'preflight'
    assert report['history_attention']['mode'] == 'bidirectional'
    generation = report['generation']
    assert generation['num_beams'] == generation['num_return_sequences'] == 20
    assert generation['use_cache'] and generation['early_stopping'] and not generation['do_sample']
    assert generation['max_new_tokens'] == 10
    assert report['strict_adapter_reload']['keys'] == 450 and report['strict_adapter_reload']['elements'] == 290037760
    names = Counter(check['loaded_dtype'] for check in report['strict_adapter_reload']['tensor_checks'].values())
    assert names == Counter({'torch.float32':448, 'torch.bfloat16':2})
    manifest = lines(REPLAY / 'inputs.jsonl')
    token_ids = load(REPLAY / 'audit.json')['token_ids']
    ids_to_token = {value: key for key, value in token_ids.items()}
    index = load(ASSET)
    sizes = Counter(''.join(tokens) for tokens in index.values())
    allowed_positions = [set(token_ids[t] for t in group) for group in zip(*index.values(), strict=True)]
    assert len(sizes) == 12088
    ids = report['preflight_assignment']
    normal = {}
    identity = Counter({m:0 for m in MODES})
    total = metric_checks = 0
    for mode in MODES:
        rows = lines(folder / mode / 'predictions.rank0.jsonl')
        assert [r['row_idx'] for r in rows] == ids
        sample = load(metrics / (mode + '.json'))['sample'] if metrics else None
        if sample is not None:
            assert [r['row_idx'] for r in sample] == ids
        for i, row in enumerate(rows):
            ref = manifest[row['row_idx']]
            assert (row['prompt_id'], row['rank']) == (0,0)
            assert row['target'] == ref['target'] == ''.join(index[row['target_item_id']])
            expected_hash = ref['original_sha256'] if mode == 'normal' else ref['conditions'][mode]['input_sha256']
            actual = row['input_token_ids']
            assert canonical_sha(actual) == expected_hash
            assert row['shuffle']['original_sha256'] == ref['original_sha256']
            pre = row['shuffle']['actual_prefill']
            assert pre['cache_length'] == 0 and pre['beam_expansion'] == 20 and pre['input_ids'] == actual
            assert pre['attention_mask'] == [1]*len(actual) and pre['position_ids'] == list(range(len(actual)))
            expected_history = [token in ids_to_token for token in actual]
            assert pre['history_sid_mask'] == expected_history
            assert sum(expected_history) == 4*ref['visible_history_length']
            assert len(pre['decodes']) == 4
            for offset, data in enumerate(pre['decodes']):
                assert data['cache_length'] == len(actual)+offset and data['query_length'] == 1
                assert data['position_ids'] == [len(actual)+offset]*20
            candidates = row['candidates']
            assert [c['generation_order'] for c in candidates] == list(range(20))
            assert all(math.isfinite(c['sequence_score']) for c in candidates)
            for c in candidates:
                suffix = c['generated_token_ids']
                assert c['eos_position'] == 4 and suffix[4] == 2
                assert all(suffix[k] in allowed_positions[k] for k in range(4))
                assert c['sid'] == ''.join(ids_to_token[t] for t in suffix[:4])
                assert c['sid'] == c['raw_decoded'].rsplit('Response:', 1)[1].strip().replace(' ', '')
            scores = [c['sequence_score'] for c in candidates]
            order = sorted(range(20), key=lambda k: (-scores[k],k))
            assert order == row['sorted_generation_order']
            ranked = [candidates[k]['sid'] for k in order]
            assert row['relevance'] == [int(sid == row['target']) for sid in ranked]
            if mode == 'normal':
                normal[row['row_idx']] = (actual,candidates,order)
            elif expected_hash == ref['original_sha256']:
                assert normal[row['row_idx']] == (actual,candidates,order)
                identity[mode] += 1
            if sample is not None:
                observed = sample[i]['metrics']
                recomputed = metric_oracle(ranked, row['target'], sizes)
                assert set(observed) == set(recomputed) and len(recomputed) == 16
                assert all(math.isclose(observed[k], recomputed[k], abs_tol=1e-14, rel_tol=0.) for k in observed)
                metric_checks += len(recomputed)
            total += len(candidates)
    return {'run_commit':report['RUN_COMMIT'], 'rows_each':ids, 'candidates':total, 'identity_rows':dict(identity),
            'independently_recomputed_metric_values':metric_checks, 'mode':report['history_attention']['mode'],
            'gpu_peak_reserved_bytes':report['gpu_peak_reserved_bytes'], 'elapsed_wall_seconds':report['elapsed_wall_seconds']}

p = argparse.ArgumentParser()
p.add_argument('--training', type=Path)
p.add_argument('--generation', type=Path)
p.add_argument('--metrics', type=Path)
p.add_argument('--jobs', nargs='+', default=[])
args = p.parse_args()
result = {}
if args.training:
    result['training'] = check_train(args.training)
if args.generation:
    result['generation'] = check_generation(args.generation, args.metrics)
if args.jobs:
    result['jobs'] = {}
    for job in args.jobs:
        status = load(JOBS/job/'status.json')
        cleaned = load(JOBS/job/'cleanup.json')
        assert status['worker_may_be_live'] is False and cleaned['last_observation']['verified_empty']
        result['jobs'][job] = {'status':status['status'], 'exit_code':status['exit_code'], 'started':status['started_at'],
                               'ended':status['ended_at'], 'cleanup_reason':cleaned['reason'], 'clean':cleaned['verified']}
print(json.dumps(result, indent=2, allow_nan=False))
