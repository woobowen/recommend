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
            'gpu_peak_reserved_bytes':report['gpu_peak_reserved_bytes'], 'elapsed_wall_seconds':report['prepublication_elapsed_seconds']}


def check_transaction(root, metrics_root):
    import hashlib
    import numpy as np

    spec_path = root / "generation-spec.json"
    spec = load(spec_path)
    final = root / "final"
    assignment = spec["assignment"]
    assert len(assignment) == len(set(assignment)) and len(assignment) <= 20
    assert spec["binding"]["purpose"] == "preflight" and spec["binding"]["backend"] == "eager"
    assert spec["binding"]["mode"] == "bidirectional" and spec["binding"]["plan_sha256"] is None
    final_report = load(final / "generation.json")
    assert final_report["attention_runtime"]["backend"] == "native_eager"
    assert final_report["attention_runtime"]["layer_count"] == 32
    assert final_report["history_attention"]["checkpoint_purpose"] == "preflight"
    assert final_report["history_attention"]["adapter_sha256"] == "a9ce5337e30d0ffe6917755455c18f6544ad9a73c4a29cf6e656e6872ccb3665"
    assert final_report["prediction_binding"] == spec["binding"]
    expected = [assignment[i:i+spec["chunk_users"]] for i in range(0,len(assignment),spec["chunk_users"])]
    assert len(expected) >= 2
    chunks_root = root / "chunks"
    chunks = sorted(chunks_root.iterdir())
    assert [x.name for x in chunks] == [f"{i:06d}" for i in range(len(expected))]
    assert final_report["reused_chunks"] == 1
    assert final_report["reused_users"] == len(expected[0])
    assert final_report["generated_this_attempt"] == len(assignment)-len(expected[0])
    attempts = [load(path) for path in (root / "attempts").glob("*/attempt.json")]
    assert len(attempts) == 2
    stopped = [a for a in attempts if a["status"] == "PREFLIGHT_STOPPED_AFTER_CHUNKS"]
    resumed = [a for a in attempts if a["status"] == "PREFLIGHT_INTERFACE_PASS"]
    assert len(stopped) == len(resumed) == 1
    assert stopped[0]["assigned_users"] == resumed[0]["assigned_users"] == len(assignment)
    assert stopped[0]["reused_users"] == 0 and resumed[0]["reused_users"] == len(expected[0])
    receipt = load(final / "completion.json")
    assert receipt["spec_sha256"] == hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert receipt["runtime_sha256"] == hashlib.sha256((root/"generation-runtime.json").read_bytes()).hexdigest()
    for path, identities in receipt["files"].items():
        raw = (final / path).read_bytes()
        assert len(raw) == identities["bytes"] and hashlib.sha256(raw).hexdigest() == identities["sha256"]
    for mode in MODES:
        joined = []
        for i, chunk in enumerate(chunks):
            meta = load(chunk / "chunk.json")
            assert meta["assignment"] == expected[i] and meta["ordinal"] == i
            raw = (chunk / (mode+".jsonl")).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == meta["files"][mode]["sha256"]
            assert [json.loads(x)["row_idx"] for x in raw.splitlines()] == expected[i]
            joined.append(raw)
        assert b"".join(joined) == (final / mode / "predictions.rank0.jsonl").read_bytes()
    primary = check_generation(final, metrics_root)
    metrics_receipt = load(metrics_root / "completion.json")
    for path, identities in metrics_receipt["files"].items():
        raw = (metrics_root/path).read_bytes()
        assert len(raw) == identities["bytes"] and hashlib.sha256(raw).hexdigest() == identities["sha256"]
    summary = load(metrics_root / "summary.json")
    assert summary["status"] == "PASS_PREFLIGHT_METRICS" and summary["users_per_condition"] == len(assignment)
    assert summary["generation_sha256"] == receipt["files"]["generation.json"]["sha256"]
    paired = load(metrics_root / "bounded-paired.json")
    assert paired["status"] == "PASS_BOUNDED_PAIRING" and paired["row_indices"] == assignment
    assert paired["seed"] == 20261113 and paired["bootstrap_replicates"] == 5000
    rng = np.random.Generator(np.random.PCG64(20261113))
    draws = rng.integers(0,len(assignment),size=(5000,len(assignment)),dtype=np.int64)
    assert hashlib.sha256(draws.astype("<i8",copy=False).tobytes(order="C")).hexdigest() == paired["draw_indices_sha256_little_endian"]
    weights = np.stack([np.bincount(row,minlength=len(assignment)) for row in draws])
    samples = {mode:load(metrics_root / (mode+".json"))["sample"] for mode in MODES}
    comparison_pairs = [("normal", m) for m in MODES[1:]] + [("shuffle3a","shuffle3b")]
    errors = []
    for left, right in comparison_pairs:
        values=paired["comparisons"][f"{right}-{left}"]
        assert len(values) == 16
        for item in values:
            metric=item["metric"]
            deltas=np.asarray([b["metrics"][metric]-a["metrics"][metric]
                 for a,b in zip(samples[left],samples[right],strict=True)])
            independent=(weights@deltas)/len(assignment)
            lo,hi=np.percentile(independent,[2.5,97.5],method="linear")
            errors.extend([abs(item["difference"]-float(deltas.mean())),
                           abs(item["interface_percentile_low"]-float(lo)),
                           abs(item["interface_percentile_high"]-float(hi))])
    assert len(errors) == 240 and max(errors) < 1e-13
    primary.update(chunks=expected, current_peak_allocated=final_report["gpu_peak_allocated_bytes"],
                   runtime_native_LLaMA_sha256=final_report["attention_runtime"]["native_source_sha256"],
                   first_attempt_metadata=stopped[0]["status"], resumed_row_count=len(assignment)-len(expected[0]),
                   verified_pairs_and_percentiles=len(errors), maximum_independent_paired_abs_error=max(errors))
    return primary

p = argparse.ArgumentParser()
p.add_argument("--generation", type=Path, required=True)
p.add_argument("--metrics", type=Path, required=True)
args=p.parse_args()
print(json.dumps(check_transaction(args.generation,args.metrics),indent=2,allow_nan=False))

