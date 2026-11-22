"""Independent, CPU-only check of the first six flushed formal G3 S2 rows."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import struct

GPU_COMMIT = '86660be5a62c8325caf74cf2810581850ca12803'
TRAIN_COMMIT = '21d495b0be5b606853ec7cec68a3eaa1b58acbd7'
MODEL_SHA = '3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915'
ADAPTER_SHA = 'd3a179b59ba7a7218d6e1767884fd1c5961fffb9b756ea5f8bc5522775db6a28'
CPU_AUDIT_SHA = '98cd89fcc3440d3ac506483ff9daf223560c6eeca943cb3f267aeb65adee0798'
CPU_MANIFEST_SHA = '2329944065d81f07c02041dbf5a62daf2609af7795e71804ed854db8faf58906'
ASSIGNMENT_SHA = '22ee58fe9f948f50e5cbc92160b6dc7df7e2f633cb9e8bf3121da5fb35bcb088'
NORMAL_SHA = 'a2247ec7016c4d88c08957fea6c0338515a124ca316245e9592ac260e8df730a'
TOKEN_MAP_SHA = 'dba960c3166b49d9098c713095273db3b6faa856ae2485ec8eda408ae66e82bc'
FROZEN_SOURCE_SHA = '9c7748ce7659a085d15128e060a1b72f6ac92b62a8642056810d0fa6df6a14e7'
DERIVED_SHA = 'f2d4fc05252dc0215553e0a35d0a5fcdecacd0b29323fb0ffb77e5133a23c80b'
INDEX_SHA = '420fc060caba3a22b0f03900dfe26abc67da8753f3bac2803f7f3c8703a05c22'
INTER_SHA = '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5'
PROTOCOL = 'GRID-LETTER-SHUFFLE-V2'
MODE = 'shuffle2'
BEAMS = 20
USERS = 22363
EXPECTED_KWARGS = {'max_new_tokens': 10, 'num_beams': BEAMS, 'num_return_sequences': BEAMS,
                   'output_scores': True, 'return_dict_in_generate': True, 'early_stopping': True}
EXPECTED_NATIVE = {'do_sample': False, 'use_cache': True, 'eos_token_id': 2, 'pad_token_id': 0}


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_checked(path, expected=None):
    raw = Path(path).read_bytes()
    if expected is not None:
        assert sha256(raw).hexdigest() == expected, path
    return json.loads(raw)


def review_rank(rank, binding, expected_uuid):
    assert rank['RUN_COMMIT'] == binding['V2_RUN_COMMIT'] == GPU_COMMIT
    assert rank['MODEL_TRAINING_COMMIT'] == binding['MODEL_TRAINING_COMMIT'] == TRAIN_COMMIT
    assert rank['scope'] == binding['scope'] == binding['adapter_scope'] == 'FORMAL'
    assert rank['preflight_only'] is False and 'preflight_manifest' not in rank
    assert rank['status'] == 'RUNNING' and rank['conditions'] == [MODE] and rank['condition'] == MODE
    assert (rank['rank'], rank['local_rank'], rank['world_size']) == (0, 0, 1)
    assert rank['model_identity_sha256'] == MODEL_SHA
    assert rank['adapter_identity']['scope'] == 'FORMAL'
    assert rank['adapter_identity']['run_commit'] == TRAIN_COMMIT
    assert rank['adapter_identity']['checkpoint_source']['global_step'] == 6160
    assert rank['adapter_identity']['adapter_sha256'] == ADAPTER_SHA
    assert rank['entry_sha256'] == FROZEN_SOURCE_SHA == binding['frozen_source_sha256']
    assert binding['derived_sha256'] == DERIVED_SHA
    assert (binding['protocol'], binding['identity_policy'], binding['mode'], binding['seed'], binding['active_positions']) == (
        PROTOCOL, 'allow_identity', MODE, 42, [2, 3])
    assert not any(x.startswith('--preflight') or x == '--verify-normal-off' for x in binding['effective_argv'])
    assert binding['cpu_audit_sha256'] == CPU_AUDIT_SHA
    assert rank['v2_cpu']['manifest_sha256'] == binding['cpu_manifest_sha256'] == CPU_MANIFEST_SHA
    tokenizer = rank['tokenizer_identity']
    assert (tokenizer['vocab_size'], tokenizer['new_token_count'], tokenizer['token_to_id_sha256']) == (32786, 786, TOKEN_MAP_SHA)
    assert rank['assigned_users'] == USERS
    assignment = rank['assigned_row_indices']
    assert sorted(assignment) == list(range(USERS))
    assert rank['assignment_sha256'] == canonical(assignment) == ASSIGNMENT_SHA
    env = rank['v2_model']['rank_environment']
    assert (env['CUDA_VISIBLE_DEVICES'], env['WORLD_SIZE'], env['RANK'], env['LOCAL_RANK']) == (expected_uuid, '1', '0', '0')
    assert rank['gpu']['visible_devices'] == expected_uuid
    assert rank['gpu']['logical_device'] == 0 and rank['gpu']['logical_device_count'] == 1
    assert env['DS_ACCELERATOR'] == 'cuda' and env['TORCHINDUCTOR_COMPILE_THREADS'] == '1'
    assert env['CUDA_HOME'] == '/data/bwwu/lc-rec/offload-20261108/cuda-12.4'
    assert env['LD_LIBRARY_PATH'] == env['CUDA_HOME'] + '/lib64:/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages/torch/lib'
    runtime = rank['v2_model']
    assert runtime['all_parameters_on_logical_cuda_0'] and runtime['eval_mode'] and runtime['requires_grad_false']
    assert runtime['base_dtype'] == runtime['adapter_embedding_and_head_dtype'] == 'torch.bfloat16'
    assert runtime['lora_dtype'] == 'torch.float32'
    strict = rank['strict_adapter_reload']
    assert (strict['keys'], strict['lora_keys'], strict['elements']) == (450, 448, 288571392)
    checks = strict['tensor_checks']
    assert len(checks) == 450 and sum(math.prod(c['shape']) for c in checks.values()) == strict['elements']
    assert sum('lora_' in key for key in checks) == 448
    for name, c in checks.items():
        assert c['finite_and_exact_after_explicit_dtype_conversion']
        assert c['loaded_dtype'] == ('torch.float32' if 'lora_' in name else 'torch.bfloat16')
    return assignment


def s2_draw(row_idx, occurrence):
    # Both S2 slots form exactly two permutations. The rejection limit equals
    # 2**256, so every digest is accepted immediately.
    payload = json.dumps([PROTOCOL, 42, 2, row_idx, occurrence, 0], separators=(',', ':'))
    return int.from_bytes(sha256(payload.encode()).digest(), 'big') % 2


def review_row(row, cpu, interaction, index, vocab, slots, catalog):
    idx = row['row_idx']
    uid, seq = interaction
    expected_item = str(seq[-1])
    assert uid == row['user_id'] == cpu['user_id']
    assert expected_item == row['target_item_id'] == cpu['target_item_id']
    assert ''.join(index[expected_item]) == row['target'] == cpu['target']
    assert row['V2_RUN_COMMIT'] == GPU_COMMIT and row['model_identity_sha256'] == MODEL_SHA
    assert (row['rank'], row['world_size'], row['scope'], row['condition'], row['prompt_id']) == (0, 1, 'FORMAL', MODE, 0)
    raw, actual, mask = (row[k] for k in ('original_input_token_ids', 'input_token_ids', 'attention_mask'))
    assert row['preflight_extra_left_padding'] == 0 and mask == [1] * len(raw)
    assert len(actual) == cpu['encoded_length'] == row['input_nonpadding_tokens'] == len(raw)
    assert cpu['original_sha256'] == canonical(raw) and cpu['attention_mask_sha256'] == canonical(mask)
    assert row['v2_cpu_binding'] == {'manifest_sha256': CPU_MANIFEST_SHA, 'row_sha256': canonical(cpu),
        'original_sha256': canonical(raw), 'attention_mask_sha256': canonical(mask),
        'input_sha256': canonical(actual), 'mode': MODE, 'row_idx': idx}
    proof = row['shuffle']
    assert (proof['protocol'], proof['identity_policy'], proof['mode'], proof['seed'], proof['active_positions']) == (
        PROTOCOL, 'allow_identity', MODE, 42, [2, 3])
    assert proof['row_idx'] == idx and proof['original_input_token_ids'] == raw and proof['attention_mask'] == mask
    evidence = proof['evidence']
    expected = raw.copy()
    history = [str(value) for value in seq[:-1][-20:]]
    locations = [i for i, token in enumerate(raw) if token in slots['all']]
    assert len(locations) == 4 * len(history) and evidence['history_group_count'] == len(history)
    ids = []
    for occurrence, (item, fixed, group) in enumerate(zip(history, cpu['groups'], evidence['groups'], strict=True)):
        positions = locations[occurrence*4:occurrence*4 + 4]
        assert positions == list(range(positions[0], positions[0] + 4))
        tokens = [vocab[symbol] for symbol in index[item]]
        assert fixed == [item, positions, tokens]
        assert tokens == [raw[i] for i in positions]
        pick = s2_draw(idx, occurrence)
        permutation = [0, 1, 2, 3] if pick == 0 else [0, 1, 3, 2]
        reordered = [tokens[i] for i in permutation]
        assert group['positions'] == positions and group['original_ids'] == tokens
        assert group['relative_permutation'] == ([0, 1] if pick == 0 else [1, 0])
        assert group['permutation_id'] == pick and group['rejection_counter'] == 0
        assert group['permutation'] == permutation and group['shuffled_ids'] == reordered
        for i, token in zip(positions, reordered, strict=True):
            expected[i] = token
        ids.append(pick)
    assert expected == actual
    assert cpu['conditions'][MODE] == {'input_sha256': canonical(actual), 'input_changed': raw != actual, 'permutation_ids': ids}
    assert evidence['original_sha256'] == canonical(raw) and evidence['shuffled_sha256'] == canonical(actual)
    assert evidence['attention_mask_sha256'] == canonical(mask) and evidence['input_changed'] == (raw != actual)
    prefill = row['actual_model_prefill']
    position_ids = list(range(len(actual)))
    assert prefill['beam_expansion'] == BEAMS and prefill['initial_cache_length'] == 0
    assert prefill['effective_use_cache'] is True
    for kind, values, field in (('input', actual, 'input_ids'), ('mask', mask, 'attention_mask'),
                                 ('position', position_ids, 'position_ids')):
        assert prefill[field] == values
        assert prefill[f'all_beam_{kind}_sha256'] == canonical([values] * BEAMS)
    lookup = prefill['embedding_lookup']
    assert lookup['shape'] == [BEAMS, len(actual), 4096] and lookup['dtype'] == 'torch.bfloat16'
    assert lookup['all_beams_exact'] and row['lookup_equivalence_exact'] and lookup['input_sha256'] == canonical(actual)
    assert len(bytes.fromhex(lookup['first_beam_bf16_sha256'])) == 32
    gen = row['effective_generation']
    assert gen['native_score_dtype'] == 'torch.float32'
    assert gen['kwargs'] == EXPECTED_KWARGS and gen['effective'] == {**EXPECTED_KWARGS, **EXPECTED_NATIVE}
    assert all(gen['generation_config'][key] == value for key, value in EXPECTED_NATIVE.items())
    assert row['original_callback_verified_all_tokens'] and len(row['candidates']) == BEAMS
    scores, sid_values = [], []
    for i, candidate in enumerate(row['candidates']):
        suffix = candidate['generated_token_ids']
        assert candidate['generation_order'] == i and candidate['eos_position'] == 4
        assert suffix[4] == 2 and all(t in (0, 2) for t in suffix[5:])
        assert all(t in slots[slot] for slot, t in enumerate(suffix[:4]))
        sid = ''.join(slots['symbol'][t] for t in suffix[:4])
        assert candidate['sid'] == sid == candidate['raw_decoded'].rsplit('Response:', 1)[-1].strip().replace(' ', '')
        score = candidate['sequence_score']
        assert math.isfinite(score) and struct.unpack('!f', struct.pack('!f', score))[0] == score
        scores.append(score)
        sid_values.append(sid)
    order = sorted(range(BEAMS), key=lambda i: (-scores[i], i))
    assert row['sorted_generation_order'] == order
    assert row['relevance'] == [int(sid_values[i] == row['target']) for i in order]
    return {'row_idx': idx, 'history_items': len(history), 'tokens': len(raw),
            'changed': actual != raw, 'out_of_catalog': sum(sid not in catalog for sid in sid_values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('rows','derived','cpu-audit','cpu-manifest','index','inter'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--rank', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--capture', type=Path,
                        help='Observer receipt containing embedded rank and binding JSON.')
    parser.add_argument('--uuid', required=True)
    args = parser.parse_args()
    capture = read_checked(args.capture) if args.capture else None
    if capture is not None:
        assert args.rank is None and args.binding is None
        rank, binding = capture['rank'], capture['binding']
    else:
        assert args.rank is not None and args.binding is not None
        rank, binding = read_checked(args.rank), read_checked(args.binding)
    assigned = review_rank(rank, binding, args.uuid)
    assert sha256(args.derived.read_bytes()).hexdigest() == binding['derived_sha256']
    raw_rows = args.rows.read_bytes()
    assert raw_rows.endswith(b'\n') and raw_rows.count(b'\n') == 6
    if capture is not None:
        assert len(raw_rows) == capture['end_offset_exclusive']
        assert sha256(raw_rows).hexdigest() == capture['prediction_sha256']
    rows = [json.loads(line) for line in raw_rows.splitlines()]
    assert [row['row_idx'] for row in rows] == assigned[:6]
    audit = read_checked(args.cpu_audit, CPU_AUDIT_SHA)
    assert audit['manifest_sha256'] == CPU_MANIFEST_SHA and audit['assignment_sha256'] == ASSIGNMENT_SHA
    assert audit['normal_sha256'] == NORMAL_SHA and audit['identity_policy'] == 'allow_identity'
    assert audit['sids'] == audit['items'] == 12101 and audit['collisions'] == 0
    vocab = audit['token_ids']
    reverse = {number: symbol for symbol, number in vocab.items()}
    assert len(reverse) == len(vocab) == 786 and canonical(vocab) == TOKEN_MAP_SHA
    index = read_checked(args.index, INDEX_SHA)
    inter = list(read_checked(args.inter, INTER_SHA).items())
    assert len(inter) == USERS and len(index) == 12101
    slots = {n:{vocab[code[n]] for code in index.values()} for n in range(4)}
    slots.update(all=set(reverse), symbol=reverse)
    catalog = {''.join(code) for code in index.values()}
    assert len(catalog) == 12101
    cpu, hasher, source_rows = {}, sha256(), 0
    wanted = set(assigned[:6])
    with args.cpu_manifest.open('rb') as file:
        for pos, line in enumerate(file):
            hasher.update(line)
            source_rows += 1
            if pos in wanted:
                cpu[pos] = json.loads(line)
                assert cpu[pos]['row_idx'] == pos
    assert source_rows == USERS and hasher.hexdigest() == CPU_MANIFEST_SHA
    results = [review_row(row, cpu[row['row_idx']], inter[row['row_idx']], index, vocab, slots, catalog)
               for row in rows]
    print(json.dumps({'status':'PASS_FORMAL_FIRST_SIX_NON_AUTHOR','captured_review_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'six physical file-prefix rows and persisted runtime boundary guards, CPU-only',
        'row_prefix_sha256':sha256(raw_rows).hexdigest(), 'row_prefix_bytes':len(raw_rows),
        'cpu_manifest_sha256':CPU_MANIFEST_SHA, 'assignment_sha256':ASSIGNMENT_SHA,
        'expected_full_user_count':USERS, 'candidate_rows_checked':6 * BEAMS, 'gpu_uuid':args.uuid,
        'rows':results, 'source_version':GPU_COMMIT, 'model_identity_sha256':MODEL_SHA,
        'rank_snapshot_completed_users':rank['completed_users']}, indent=2))


if __name__ == '__main__':
    main()
