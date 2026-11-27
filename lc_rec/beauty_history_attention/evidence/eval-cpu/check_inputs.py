"""Read-only all-user input parity against fixed Beauty Shuffle V2 launch manifest."""

from collections import Counter
import hashlib
import json

from lc_rec.beauty_history_attention.evaluate import CONDITIONS, REPLAY_DIR, REPLAY_SHA, paired_inputs, preflight_rows, read_replay
from lc_rec.shuffle_v2 import beauty, core, replay

config = json.loads(beauty.CONFIG.read_text())
audit, rows, sampler = read_replay(REPLAY_DIR, config)
raw_data, tokenizer, collate, token_ids = replay.frozen_data(config)
assert token_ids == audit['token_ids']
sid_ids = set(token_ids.values())
input_streams = {mode: hashlib.sha256() for mode in CONDITIONS}
identity = Counter()
group_count = 0
for row in rows:
    i = row['row_idx']
    uid = row['user_id']
    seq = raw_data.inters[uid]
    batch, canonical = collate([raw_data[i]])
    assert canonical == [row['target']] == [''.join(raw_data.indices[str(seq[-1])])]
    groups = [[token_ids[sid] for sid in raw_data.indices[str(item)]] for item in seq[:-1][-20:]]
    original, mask = batch['input_ids'][0].tolist(), batch['attention_mask'][0].tolist()
    history, conditions = paired_inputs(original, mask, groups, sid_ids, row)
    assert [original[j] for j, active in enumerate(history) if active] == [x for group in groups for x in group]
    assert sum(history) == 4 * len(groups)
    group_count += len(groups)
    for mode, (ids, evidence) in conditions.items():
        assert all(before == after for active, before, after in zip(history, original, ids, strict=True) if not active)
        assert len(original) == len(ids) == len(history)
        identity[mode] += ids == original
        input_streams[mode].update(core.canonical_bytes([i, ids]) + b'\n')
    assert conditions['shuffle3a'][1]['permutation_ids'] == conditions['shuffle3b'][1]['permutation_ids']
assert group_count == config['groups'] == 159966
assert identity == {'normal': 22363, 'shuffle2': 640, 'shuffle3a': 6, 'shuffle3b': 6, 'shuffle4': 0}
print(json.dumps({'status': 'PASS_REAL_FROZEN_CPU_INPUT_PARITY', 'users': len(rows), 'history_groups': group_count,
                  'user_axis_sha256': audit['user_axis_sha256'],
                  'input_replay_sha256': REPLAY_SHA['inputs.jsonl'], 'whole_input_identity_rows': identity,
                  'computed_input_stream_sha256': {mode: stream.hexdigest() for mode, stream in input_streams.items()},
                  'outcome_blind_preflight_indices': preflight_rows(rows)}, indent=2))
