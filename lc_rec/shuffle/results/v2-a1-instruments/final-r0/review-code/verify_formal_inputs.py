"""Read and independently reconstruct every formal A1 Shuffle V2 prediction row."""
from argparse import ArgumentParser
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import struct

from independent_inputs import (
    CATALOG, DATA_SHA256, EOS, MODES, NORMAL_SHARD_SHA256, ORIGINAL_CALLBACK, ORIGINAL_ENTRY,
    ORIGINAL_RUN, PROTOCOL, VOCAB_SHA256, digest, json_bytes, open_tokenizer, prompt0, relative_permutation,
    sha_file, verify_candidates, verify_derived,
)

USERS = 24772
TOTAL_GROUPS = 172929
GPU_COMMIT = 'c487f7e7176bc223445d3deecfab87e0bcc30d1a'
CARD = 'GPU-59cac3bd-3645-668b-79c8-839e0d878893'
PLAN_HASH = 'f37fbe7ac295480a181daf2fed098810bd5b2d3c9499e19a7eec67d160603c7b'
CHECKPOINT_HASH = 'fe96ea74fc3c7253b0c3687239266b0f073e1a51ef9158e7d97bf30b4527bd03'
MODEL_REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
MODEL_HASH = 'cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba'
TOKENIZER_HASH = 'a65b6872ff61bcfba1e617d065654eba58d678e876ada9547a4c2cc78ba619c4'
DERIVED_HASH = '9fe9dde11d60e706bcec6f32a066c40d693726bdef3220ca6fca1c42fdbba11e'
MANIFEST_HASH = '93e7a63e53715e639b7d6aea088b4d68d1309658e7a9a8b04cddfbbc0d99d31c'
ASSIGNMENT_HASH = 'fc56a36be7d27e20783eb0e8f07430a35919dadd13db8661ad34c3085cb216ec'
GENERATION = {'num_beams':20, 'num_return_sequences':20, 'max_new_tokens':10,
              'early_stopping':True, 'do_sample':False, 'use_cache':True, 'output_scores':True,
              'return_dict_in_generate':True, 'eos_token_id':EOS, 'pad_token_id':EOS}
NATURAL_COUNTS = {'shuffle2':714, 'shuffle3a':6, 'shuffle3b':6, 'shuffle4':0}
MODE_PREFILL_FIELDS = ('input_sha256','attention_mask_sha256','position_ids_sha256','cache_length',
                       'beam_expansion','all_beams_ids_mask_positions_exact')
CANDIDATE_FIELDS = ('generated_token_ids','sid','generation_order','raw_decoded','eos_position','sequence_score')


def read(path):
    return json.loads(Path(path).read_bytes())


class Shard:
    def __init__(self, stream):
        self.stream = stream
        self.sha = sha256()
        self.line = 0

    def next(self):
        raw = self.stream.readline()
        self.line += 1
        assert raw.endswith(b'\n'), (self.stream.name, self.line, 'missing complete JSONL record')
        self.sha.update(raw)
        return json.loads(raw)

    def end(self):
        assert self.stream.readline() == b'', self.stream.name


def load_assets(data, tokenizer_path, selection_path, cpu):
    for name, expected in DATA_SHA256.items():
        assert sha_file(data/f'Instruments.{name}.json') == expected, name
    index = read(data/'Instruments.index.json')
    item = read(data/'Instruments.item.json')
    inter = list(read(data/'Instruments.inter.json').items())
    catalog = Counter(''.join(values) for values in index.values())
    assert len(inter) == USERS and len(index) == len(item) == 9922 and set(index) == set(item)
    assert len(catalog) == 9897 and sum(c > 1 for c in catalog.values()) == 23
    assert all(len(value) == 4 for value in index.values())
    assert sha_file(selection_path) == '0d1d24fae109f5acf3b8105e5e5f970164a92b17a82f7d9b340b600ad3d40bd0'
    selection = read(selection_path)
    assert (selection['run_commit'],selection['loaded_best_checkpoint_step'],selection['adapter_sha256']) == (
        ORIGINAL_RUN,8240,CHECKPOINT_HASH)
    assert selection['checkpoint'].endswith('/checkpoint-8240')
    audit = read(cpu/'audit.json')
    assert audit['users'] == USERS and audit['protocol'] == PROTOCOL and audit['status'] == 'PASS_CPU_V2_FULL'
    single = read(cpu/'single-shard.json')
    assert single['status'] == 'PASS_SINGLE_SHARD_CPU' and single['manifest_sha256'] == MANIFEST_HASH
    assert audit['manifest_sha256'] == MANIFEST_HASH and sha_file(cpu/'inputs.jsonl') == MANIFEST_HASH
    manifest = []
    with (cpu/'inputs.jsonl').open('rb') as f:
        for number,line in enumerate(f):
            assert line.endswith(b'\n')
            entry=json.loads(line)
            assert entry['row_idx'] == number
            manifest.append(entry)
    assert len(manifest) == USERS
    expected_ranks = single['normal_rank_assignments']
    assert len(expected_ranks) == 2 and all(len(x) == USERS // 2 for x in expected_ranks)
    assignment = [index_ for pair in zip(*expected_ranks,strict=True) for index_ in pair]
    assert assignment == single['assignment'] and set(assignment) == set(range(USERS))
    assert digest(assignment) == single['assignment_sha256'] == ASSIGNMENT_HASH
    assert sha_file(tokenizer_path) == TOKENIZER_HASH
    tokenizer = open_tokenizer(tokenizer_path)
    vocab = tokenizer.get_vocab()
    vocab_hash = sha256(json.dumps(vocab,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert vocab_hash == audit['tokenizer_vocab_sha256'] == VOCAB_SHA256 and len(vocab) == 152587
    tokens = audit['token_ids']
    assert len(tokens) == len(set(tokens.values())) == 918
    assert set(tokens) == {token for group in index.values() for token in group}
    assert all(tokenizer.encode(s,add_special_tokens=False).ids == [i] for s,i in tokens.items())
    permitted = [{tokens[value[slot]]:value[slot] for value in index.values()} for slot in range(4)]
    assert [len(level) for level in permitted] == [150,256,256,256]
    return audit, single, assignment, manifest, index, inter, catalog, tokenizer, tokens, permitted


def verify_rank_receipts(evidence, normal, original_source, audit, single):
    norm_ranks = [read(normal/f'rank{k}.json') for k in range(2)]
    matching = ('entry_sha256','allowed_callback_sha256','adapter_files','data_identity',
                'author_sources','model','strict_adapter_reload','tokenizer','versions')
    for k, rank in enumerate(norm_ranks):
        assert rank['status'] == 'PASS' and rank['RUN_COMMIT'] == ORIGINAL_RUN
        assert (rank['rank'],rank['local_rank'],rank['world_size']) == (k,k,2)
        assert rank['entry_sha256'] == ORIGINAL_ENTRY and rank['allowed_callback_sha256'] == ORIGINAL_CALLBACK
        assert digest(single['normal_rank_assignments'][k]) == rank['assignment_sha256']
        for field in matching:
            assert rank[field] == norm_ranks[0][field], field
    found = {}
    for mode, active in MODES.items():
        gpu = evidence/f'formal-gpu-{mode}-r0'
        test = gpu/'test'
        rank = read(test/'rank0.json')
        job = read(gpu/'summary.json')
        launch = read(gpu/f'{mode}.request.json')
        done = read(gpu/f'{mode}.exit.json')
        verify_derived(original_source,test/'derived.rank0.py',test/'derived.rank0.diff')
        assert sha_file(test/'derived.rank0.py') == DERIVED_HASH
        assert rank['status'] == 'PASS' and rank['RUN_COMMIT'] == GPU_COMMIT
        assert (rank['rank'],rank['local_rank'],rank['world_size']) == (0,0,1)
        assert rank['preflight_only'] is False
        assert rank['assigned_users'] == rank['completed_users'] == USERS
        assert rank['assignment_sha256'] == rank['formal_assignment_sha256'] == ASSIGNMENT_HASH
        assert rank['gpu']['visible_devices'] == CARD and rank['gpu']['name'] == 'NVIDIA GeForce RTX 4090'
        assert rank['model']['revision'] == MODEL_REVISION and rank['model']['weight_sha256'] == MODEL_HASH
        assert rank['adapter_files']['adapter_model.bin']['sha256'] == CHECKPOINT_HASH
        for field in matching:
            assert rank[field] == norm_ranks[0][field], field
        strict = rank['strict_adapter_reload']
        assert (strict['keys'],strict['lora_keys']) == (394,392) and len(strict['tensor_checks']) == 394
        assert all(tensor['saved_dtype'] == tensor['loaded_dtype'] == 'torch.bfloat16' and
                   tensor['finite_exact_and_same_bf16_dtype'] for tensor in strict['tensor_checks'].values())
        shared = strict['sharing']
        assert shared['same_parameter_object'] and shared['same_storage']
        assert shared['input_dtype'] == shared['output_dtype'] == 'torch.bfloat16'
        assert shared['input_shape'] == shared['output_shape'] == [152587,1024]
        tok = rank['tokenizer']
        assert tok['expanded_tokenizer_size'] == 152587 and tok['added_sid_tokens'] == 918
        assert tok['bos_token_id'] is None and tok['eos_token_id'] == tok['pad_token_id'] == EOS
        assert tok['padding_side'] == 'left' and not tok['encodes_automatic_bos']
        generation = rank['generation']
        assert generation['filter_items'] is False and generation['per_rank_batch_size'] == 1
        assert generation['prompt_id'] == 0 and generation['oom_beam_reduction'] is False
        shuffle = rank['shuffle']
        assert (shuffle['protocol'],shuffle['identity_policy'],shuffle['mode'],shuffle['seed'],shuffle['active_positions']) == (
            PROTOCOL,'allow_identity',mode,42,list(active))
        assert shuffle['actual_model_module'] == 'Qwen3Model' and shuffle['sdpa'] and not shuffle['quantized']
        assert shuffle['cpu_manifest_sha256'] == audit['manifest_sha256']
        assert shuffle['generation_scalar_overrides'] == GENERATION
        assert all(shuffle['effective_generation_config'][k] == v for k,v in GENERATION.items())
        assert shuffle['effective_generation_config']['cache_implementation'] is None
        assert shuffle['generate_calls'] == USERS and shuffle['parameter_versions_unchanged'] is True
        assert shuffle['parameter_sha256_before'] is shuffle['parameter_sha256_after'] is None
        assert launch['run_commit'] == job['run_commit'] == GPU_COMMIT
        assert launch['device']['uuid'] == job['device']['uuid'] == CARD
        assert launch['device']['visible_device'] == launch['device']['world_size'] - 1 == 0
        assert '--preflight' not in launch['argv'] and '--nproc_per_node=1' in launch['argv']
        assert launch['argv'][launch['argv'].index('--mode')+1] == mode
        assert job['status'] == 'PASS' and job['job_id'] == f'a1-instruments-v2-formal-gpu-{mode}-r0'
        assert job['stages'][mode]['users'] == USERS and job['stages'][mode]['exit_code'] == done['exit_code'] == 0
        assert job['identity_discrepancies'] == []
        found[mode] = {'pid':rank['pid'],'model_load_seconds':rank['model_load_and_strict_verification_seconds'],
                       'generation_seconds':rank['elapsed_generation_seconds'],
                       'output_rows':rank['completed_users'],'expected_weight_versions_unchanged':True,
                       'rank_receipt_sha256':sha_file(test/'rank0.json'),
                       'derived_sha256':sha_file(test/'derived.rank0.py'),
                       'derived_diff_sha256':sha_file(test/'derived.rank0.diff')}
    return found


def check_identity(row, frozen, mode, formal_line, normal_line, normal_rank):
    assert len(row['candidates']) == len(frozen['candidates']) == 20
    counts = Counter()
    diff = 0.0
    detail = []
    for beam, (a,b) in enumerate(zip(row['candidates'],frozen['candidates'],strict=True)):
        keys = [k for k in CANDIDATE_FIELDS if k != 'sequence_score' and a[k] != b[k]]
        if struct.pack('>f',a['sequence_score']) != struct.pack('>f',b['sequence_score']):
            keys.append('sequence_score')
        if keys:
            counts.update(keys)
            detail.append({'beam':beam,'changed_fields':keys})
        diff = max(diff,abs(a['sequence_score']-b['sequence_score']))
    order_exact = row['sorted_generation_order'] == frozen['sorted_generation_order']
    assert row['input_token_ids'] == frozen['input_token_ids']
    return {'mode':mode,'row_idx':row['row_idx'],'formal_line':formal_line,'normal_rank':normal_rank,
            'normal_line':normal_line,'exact':not counts and order_exact,'field_differences':dict(counts),
            'max_abs_sequence_score_delta':diff,'sorted_order_exact':order_exact,'beam_differences':detail}


def check_row(row, identity, expected, mask, position_hash, prompt_separator, independent, permitted):
    user_idx, uid, target_item, sid = identity
    assert (row['row_idx'],row['user_id'],row['target_item_id'],row['target']) == identity
    assert row['rank'] == row['prompt_id'] == 0 and row['input_nonpadding_tokens'] == len(expected)
    assert row['input_token_ids'] == expected and row['original_callback_verified_all_tokens']
    shuffle = row['shuffle']
    assert all(shuffle[k] == v for k,v in independent.items()), (user_idx,independent['mode'])
    assert shuffle['eval_mode'] is True and shuffle['grad_enabled'] is False
    assert shuffle['sequence_score_dtype'] == 'torch.float32'
    prefill = shuffle['actual_prefill']
    assert tuple(prefill[k] for k in MODE_PREFILL_FIELDS) == (
        independent['shuffled_sha256'],mask,position_hash,0,20,True)
    callback = shuffle['callback']
    assert callback['original_validation'] and callback['bound_prefix_sha256'] == independent['shuffled_sha256']
    assert callback['separator_ids'] == prompt_separator and expected[-len(prompt_separator):] == prompt_separator
    assert callback['allowed_counts'] == [150,256,256,256,1]
    assert len(callback['generation_position_calls']) == 5 and all(n > 0 for n in callback['generation_position_calls'])
    verify_candidates(row,permitted)


def full_check(evidence, normal, data, tokenizer_path, selection, original_source):
    cpu = evidence/'cpu-replay-r0'
    audit,single,axis,manifest,idx,inter,catalog,tok,tokens,permitted = load_assets(data,tokenizer_path,selection,cpu)
    rank_receipts = verify_rank_receipts(evidence,normal,original_source,audit,single)
    sid_ids = set(tokens.values())
    reconstructed = {mode:[None]*USERS for mode in MODES}
    count = {mode:{'permutations':Counter(),'unchanged_groups':0,'unchanged_users':0,
                   'moved_ids':[0]*4,'moved_slots':[0]*4,'out_of_catalog_candidates':0,'duplicate_candidates':0}
             for mode in MODES}
    candidates, identity = {mode:0 for mode in MODES}, []
    history_groups_count = 0
    with ExitStack() as resources:
        norm = [Shard(resources.enter_context((normal/f'predictions.rank{k}.jsonl').open('rb'))) for k in range(2)]
        formal = {mode:Shard(resources.enter_context(
            (evidence/f'formal-gpu-{mode}-r0'/'test'/'predictions.rank0.jsonl').open('rb'))) for mode in MODES}
        for sequence_position, user_idx in enumerate(axis):
            try:
                normal_rank = sequence_position % 2
                normal_line = sequence_position // 2 + 1
                old = norm[normal_rank].next()
                uid, history = inter[user_idx]
                target_item = str(history[-1])
                target = ''.join(idx[target_item])
                user = (user_idx,uid,target_item,target)
                receipt = manifest[user_idx]
                for actual in old,receipt:
                    assert (actual['row_idx'],actual['user_id'],actual['target_item_id'],actual['target']) == user
                assert (old['rank'],old['prompt_id']) == (normal_rank,0)
                groups = [idx[str(item)] for item in history[:-1][-20:]]
                prompt = prompt0(groups)
                encoding = tok.encode(prompt,add_special_tokens=False)
                raw = encoding.ids
                assert raw == old['input_token_ids'] and len(raw) == old['input_nonpadding_tokens'] == receipt['encoded_length']
                assert encoding.attention_mask == [1]*len(raw)
                original = digest(raw)
                mask = digest(encoding.attention_mask)
                position_hash = digest(list(range(len(raw))))
                assert original == receipt['original_sha256'] and mask == receipt['attention_mask_sha256']
                separator = [token for token,(_,hi) in zip(raw,encoding.offsets,strict=True) if hi > len(prompt)-len('Response:')]
                assert separator
                slots = [offset for offset,token in enumerate(raw) if token in sid_ids]
                assert len(slots) == 4*len(groups) and len(groups) == receipt['visible_history_length']
                history_groups_count += len(groups)
                group_frames = []
                for occurrence, group in enumerate(groups):
                    positions = slots[occurrence*4:occurrence*4+4]
                    assert positions == list(range(positions[0],positions[0]+4))
                    source = [tokens[symbol] for symbol in group]
                    assert [raw[p] for p in positions] == source
                    draws = {number:relative_permutation(number,user_idx,occurrence) for number in (2,3,4)}
                    group_frames.append((positions,source,draws))
                assert receipt['conditions']['shuffle3a']['permutation_ids'] == receipt['conditions']['shuffle3b']['permutation_ids']
                for mode,active in MODES.items():
                    pred = formal[mode].next()
                    expected = list(raw)
                    selected, counter = [], count[mode]
                    for positions,source,draws in group_frames:
                        choice, perm, _ = draws[len(active)]
                        selected.append(choice)
                        counter['permutations'][choice] += 1
                        counter['unchanged_groups'] += choice == 0
                        for dest, previous in zip(active,perm,strict=True):
                            old_slot = active[previous]
                            expected[positions[dest]] = source[old_slot]
                            counter['moved_slots'][dest] += dest != old_slot
                            counter['moved_ids'][dest] += source[dest] != source[old_slot]
                    transformed = digest(expected)
                    changed = raw != expected
                    condition = receipt['conditions'][mode]
                    assert (condition['permutation_ids'],condition['input_sha256'],condition['input_changed']) == (
                        selected,transformed,changed)
                    independent = {'protocol':PROTOCOL,'identity_policy':'allow_identity','mode':mode,'seed':42,
                                   'active_positions':list(active),'row_idx':user_idx,'input_changed':changed,
                                   'original_sha256':original,'shuffled_sha256':transformed,'attention_mask_sha256':mask,
                                   'cpu_manifest_sha256':MANIFEST_HASH,'history_group_count':len(groups)}
                    check_row(pred,user,expected,mask,position_hash,separator,independent,permitted)
                    candidates[mode] += len(pred['candidates'])
                    names = [c['sid'] for c in pred['candidates']]
                    counter['out_of_catalog_candidates'] += sum(value not in catalog for value in names)
                    counter['duplicate_candidates'] += len(names)-len(set(names))
                    reconstructed[mode][user_idx] = json_bytes([user_idx,expected])+b'\n'
                    if not changed:
                        counter['unchanged_users'] += 1
                        verify_candidates(old,permitted)
                        identity.append(check_identity(pred,old,mode,sequence_position+1,normal_line,normal_rank))
            except Exception as exc:
                raise RuntimeError(f'full audit failed at formal line {sequence_position+1}, row_idx {user_idx}, mode {locals().get("mode")}') from exc
        for rank, stream in enumerate(norm):
            stream.end()
            assert stream.line == USERS//2 and stream.sha.hexdigest() == NORMAL_SHARD_SHA256[rank]
        for mode, stream in formal.items():
            stream.end()
            assert stream.line == USERS and candidates[mode] == 20*USERS
            report = audit['modes'][mode]
            observed = count[mode]
            m = len(MODES[mode])
            assert [observed['permutations'][j] for j in range(len(CATALOG[m]))] == report['permutation_counts']
            assert observed['unchanged_groups'] == report['identity_permutations'] == report['token_unchanged_groups']
            assert observed['unchanged_users'] == report['unchanged_users'] == NATURAL_COUNTS[mode]
            assert observed['moved_slots'] == report['permuted_slots_by_position']
            assert observed['moved_ids'] == report['moved_ids_by_position']
            assert sum(observed['moved_ids']) == report['actual_changed_token_count']
            semantic = sha256()
            for line in reconstructed[mode]:
                semantic.update(line)
            assert semantic.hexdigest() == audit['input_stream_sha256'][mode]
            manifest_path = evidence/f'formal-cpu-{mode}-r0'/'prediction_manifest.json'
            entry, = read(manifest_path)
            assert (entry['rows'],entry['rank'],entry['run_commit']) == (USERS,0,GPU_COMMIT)
            assert entry['sha256'] == stream.sha.hexdigest()
            assert entry['bytes'] == Path(stream.stream.name).stat().st_size
            assert entry['rank_report_sha256'] == rank_receipts[mode]['rank_receipt_sha256']
            observed.update({'prediction_sha256':stream.sha.hexdigest(),'input_stream_sha256':semantic.hexdigest(),
                             'candidates':candidates[mode],'users':USERS})
    assert history_groups_count == TOTAL_GROUPS and len(identity) == sum(NATURAL_COUNTS.values())
    deltas = {mode:{'compared':NATURAL_COUNTS[mode],
                    'exact':sum(r['mode']==mode and r['exact'] for r in identity),
                    'max_abs_sequence_score_delta':max((r['max_abs_sequence_score_delta'] for r in identity if r['mode']==mode),default=None)}
              for mode in MODES}
    failures = [row for row in identity if not row['exact']]
    return {'status':'PASS_A_FULL_FORMAL_INPUTS_AND_NATURAL_IDENTITY' if not failures else 'NATURAL_IDENTITY_DISCREPANCIES',
            'users_per_mode':USERS,'checked_formal_user_conditions':USERS*len(MODES),
            'checked_formal_candidates':20*USERS*len(MODES),'checked_visible_item_conditions':len(MODES)*TOTAL_GROUPS,
            'frozen_original_shard_sha256':list(NORMAL_SHARD_SHA256),'cpu_manifest_sha256':MANIFEST_HASH,
            'assignment_sha256':ASSIGNMENT_HASH,'tokenizer_vocab_sha256':VOCAB_SHA256,
            'runtime_commit':GPU_COMMIT,'rank_receipts':rank_receipts,'mode_checks':count,
            'natural_identity':deltas,'natural_identity_discrepancies':failures,'all_natural_identity_rows':identity}


def main():
    args = ArgumentParser()
    for option in ('evidence','normal','data','tokenizer','selection','original-source'):
        args.add_argument('--'+option,required=True,type=Path)
    a = args.parse_args()
    result = full_check(a.evidence,a.normal,a.data,a.tokenizer,a.selection,a.original_source)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    if result['natural_identity_discrepancies']:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
