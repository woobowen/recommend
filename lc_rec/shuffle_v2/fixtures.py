"""Bounded CPU checks against the actual collator and the frozen statistical reference."""
import argparse
import json
import os
from pathlib import Path

from lc_rec.shuffle_v2 import core, reference
from lc_rec.shuffle_v2.evaluate import LC, module
from lc_rec.shuffle_v2.replay import CONFIG, dump, frozen_data
from lc_rec.shuffle_v2.statistics import paired_means


def tokenizer_fixtures(config):
    data, tokenizer, collate, token_ids = frozen_data(config)
    sid_ids = set(token_ids.values())
    row_idx = next(i for i in range(len(data)) if reference.relative(4, i, 0)[0] != reference.relative(4, i, 1)[0])
    real = data[row_idx]
    repeated_item = list(data.indices)[0]
    sid = ''.join(data.indices[repeated_item])
    group = [token_ids[token] for token in data.indices[repeated_item]]
    info = {**data.inter_data[row_idx], 'inters': data.his_sep.join((sid, sid))}
    text, label = data._get_text_data(info, data.prompts[0])
    assert label == real['labels']
    repeated = {'input_ids': text, 'labels': label}
    controls = [{'input_ids': ('word ' * 70) + text, 'labels': label}, repeated]
    padded, labels = collate(controls)
    pad_ids = padded['input_ids'][1].tolist()
    pad_mask = padded['attention_mask'][1].tolist()
    raw, _ = collate([repeated])
    raw_ids = raw['input_ids'][0].tolist()
    padding = len(pad_ids) - len(raw_ids)
    assert padding > 0 and pad_ids == [tokenizer.pad_token_id] * padding + raw_ids
    assert pad_mask == [0] * padding + [1] * len(raw_ids) and labels == [label, label]
    evidence = {}
    for mode in core.MODES:
        padded_result, padded_record = core.transform(pad_ids, pad_mask, [group, group], sid_ids, mode, row_idx)
        raw_result, raw_record = core.transform(raw_ids, [1] * len(raw_ids), [group, group], sid_ids, mode, row_idx)
        assert padded_result[:padding] == pad_ids[:padding] and padded_result[padding:] == raw_result
        for rp, rr in zip(padded_record['groups'], raw_record['groups'], strict=True):
            assert rp['positions'] == [p + padding for p in rr['positions']]
            assert rp['shuffled_ids'] == rr['shuffled_ids']
        evidence[mode] = {'relative_permutations': [r['relative_permutation'] for r in raw_record['groups']],
                          'untouched_padding_tokens': padding, 'target_sha256': core.canonical_sha256(label),
                          'padded_input_sha256': padded_record['shuffled_sha256']}
    assert len({tuple(p) for p in evidence['shuffle4']['relative_permutations']}) == 2
    return {'row_idx': row_idx, 'uses_real_frozen_tokenizer_and_collator': True,
            'fixture_contains_real_sid_twice': True, 'results': evidence}


def statistics_fixture():
    import numpy as np
    vals = np.arange(9 * 16, dtype=np.float64).reshape(9, 16)
    values = [np.zeros_like(vals), vals, -vals, vals / 10, vals * 2]
    draws, digest = paired_means(values, 9)
    original = module('statistics_reference', LC / 'letter/review/final_audit_20261113/statistics_verify.py')
    variance = []
    for expected, diff in zip(draws, values, strict=True):
        _, verified = original.bootstrap_columns(diff)
        variance.append(float(np.max(np.abs(expected - verified))))
    assert max(variance) <= 1e-12
    return {'labels': 'SYNTHETIC_METRIC_MATRICES', 'repetitions': 5000, 'draws_sha256': digest,
            'max_draw_deltas': variance, 'uses_paired_pcg64': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    dump(args.output / 'summary.json', {'status': 'PASS_CPU_FIXTURES', 'tokenizer': tokenizer_fixtures(config),
                                       'stats': statistics_fixture()})


if __name__ == '__main__':
    main()
