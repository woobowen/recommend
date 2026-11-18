#!/usr/bin/env python3
"""Real G3 Dataset/collator/callback checks and isolated production-gate fixtures.

Synthetic corrupted checkpoint/prediction fixtures test guards, never model execution.
No model weights or CUDA context are loaded by this program.
"""
import argparse
from collections import Counter
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'data-root', 'model', 'config', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    check(not args.output.exists(), 'preserve prior CPU receipt')
    args.output.mkdir(parents=True)
    a0 = args.repo / 'lc_rec/a0_beauty'
    vendor = args.repo / 'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'
    sys.path.insert(0, str(a0))
    sys.path.insert(0, str(vendor))
    import torch
    import numpy as np
    import transformers
    import data as data_module
    import collator as collator_module
    import utils as utils_module
    from transformers import LlamaTokenizer
    import g3_identity as gi
    import formal_train
    import formal_evaluate as fe
    import offline_metrics as om
    import paired_beauty as pb
    import single_gpu_training as single_training
    import training_rows as historical_rows
    from preflight_users import validate as validate_users
    from uneven_sampler import assignment, validate_assignments
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text())
    check(cfg['experiment_id'] == gi.EXPERIMENT_ID and cfg['only_train_response'] is True,
          'production G3 config identity/labels')
    gi.verify_execution_profile({'world_size':cfg['world_size'],'per_device_train_batch_size':cfg['per_device_batch_size'],
        'gradient_accumulation_steps':cfg['gradient_accumulation_steps'],'global_batch':cfg['global_batch']})
    asset_rows = gi.verify_assets(args.data_root)
    root = args.data_root / 'Beauty'
    index = json.loads((root / 'Beauty.index.json').read_text())
    inter = json.loads((root / 'Beauty.inter.json').read_text())
    items = json.loads((root / 'Beauty.item.json').read_text())
    check(set(index) == set(items), 'item/index full axis')
    check(all(str(item) in index for sequence in inter.values() for item in sequence), 'all interaction references')
    axis = list(inter)
    check(digest(axis) == '7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f', 'user axis')
    utils_module.set_seed(42)
    cfg['data_path'] = str(args.data_root)
    options = SimpleNamespace(**cfg)
    train, valid = utils_module.load_datasets(options)
    test = data_module.SeqRecDataset(options, mode='test', sample_num=-1)
    test.set_prompt(0)
    check([len(train), len(valid), len(test)] == [131413, 44726, 22363], 'derived split sizes')
    sid = {item: ''.join(tokens) for item, tokens in index.items()}
    cursor = 0
    for i, (uid, sequence) in enumerate(inter.items()):
        mapped = [sid[str(item)] for item in sequence]
        for target_position in range(1, len(sequence)-2):
            actual = train.datasets[0].inter_data[cursor]
            check(actual == {'item': mapped[target_position], 'inters': ', '.join(mapped[max(0, target_position-20):target_position])}, 'train split/history/target')
            cursor += 1
        check(valid.inter_data[i] == {'item': mapped[-2], 'inters': ', '.join(mapped[:-2][-20:])}, 'validation split/history/target')
        check(test.inter_data[i] == {'item': mapped[-1], 'inters': ', '.join(mapped[:-1][-20:])}, 'test split/history/target')
        check(test[i]['labels'] == mapped[-1], 'test text target')
    check(cursor == len(train), 'all train rows consumed')
    tokenizer = LlamaTokenizer.from_pretrained(args.model, model_max_length=2048,
                                              padding_side='right', local_files_only=True)
    tokenizer.pad_token_id = 0
    check(tokenizer.add_tokens(train.datasets[0].get_new_tokens()) == gi.NEW_TOKEN_COUNT, 'derived new SID count')
    token_identity = gi.tokenizer_identity(tokenizer, index)
    saved_tokenizer = args.output / 'tokenizer'
    tokenizer.save_pretrained(saved_tokenizer)
    reloaded = LlamaTokenizer.from_pretrained(saved_tokenizer, local_files_only=True)
    reloaded.pad_token_id = 0
    check(reloaded.get_vocab() == tokenizer.get_vocab() and gi.tokenizer_identity(reloaded, index) == token_identity,
          'full tokenizer save/reload mapping')
    lengths = [len(sequence[:-1][-20:]) for sequence in inter.values()]
    chosen = [min(range(len(axis)), key=lambda i: (lengths[i], i)), max(range(len(axis)), key=lambda i: (lengths[i], -i))]
    repeat = [i for i, sequence in enumerate(inter.values()) if len(set(sequence[:-1][-20:])) < lengths[i]]
    if repeat:
        chosen.append(repeat[0])
    fourth = {zero: next(i for i, sequence in enumerate(inter.values())
                         if any((index[str(item)][3] == '<d_0>') == zero for item in sequence[:-1][-20:]))
              for zero in (True, False)}
    chosen = list(dict.fromkeys(chosen + list(fourth.values())))
    collator = collator_module.Collator(options, tokenizer)
    batch_rows = [train[0], train[len(train)-1], valid[0], valid[len(valid)-1]]
    # Real short/long/repeated histories using a real train/validation prompt template.
    for i in chosen:
        real = test.inter_data[i]
        left, right = valid._get_text_data(real, valid.prompts[0])
        batch_rows.append({'input_ids': left, 'labels': right})
    synthetic_repeated = {'item': sid[str(inter[axis[0]][-1])], 'inters': ', '.join([sid[str(inter[axis[0]][0])]] * 2)}
    left, right = valid._get_text_data(synthetic_repeated, valid.prompts[0])
    batch_rows.append({'input_ids': left, 'labels': right})
    batch = collator(batch_rows)
    labels_checked = 0
    for i, row in enumerate(batch_rows):
        full_ids = tokenizer(row['labels'] + tokenizer.eos_token)['input_ids']
        prefix_ids = tokenizer(row['input_ids'])['input_ids']
        check(len(full_ids) <= 2048, 'real audit row truncation boundary')
        check(full_ids[:len(prefix_ids)] == prefix_ids, 'response prefix token boundary')
        expected_input = full_ids + [tokenizer.pad_token_id] * (batch['input_ids'].shape[1]-len(full_ids))
        expected_mask = [1] * len(full_ids) + [0] * (len(expected_input)-len(full_ids))
        expected_labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):] + [-100] * (len(expected_input)-len(full_ids))
        check(batch['input_ids'][i].tolist() == expected_input, 'collator input position')
        check(batch['attention_mask'][i].tolist() == expected_mask, 'collator attention position')
        check(batch['labels'][i].tolist() == expected_labels, 'response-only every label position')
        check(full_ids[-1] == tokenizer.eos_token_id and expected_labels[len(full_ids)-1] == tokenizer.eos_token_id, 'supervised EOS')
        check(len(full_ids)-len(prefix_ids) == 5, 'four complete SID tokens and EOS supervision')
        labels_checked += len(expected_labels)
    check((batch['attention_mask'] == 0).any().item(), 'real collator padding covered')
    test_collator = collator_module.TestCollator(options, tokenizer)
    test_batch, targets = test_collator([test[i] for i in chosen])
    check(tokenizer.padding_side == 'left' and (test_batch['attention_mask'] == 0).any().item(), 'actual left inference padding')
    check(targets == [sid[str(inter[axis[i]][-1])] for i in chosen], 'test collator labels unchanged')
    callback = test.get_prefix_allowed_tokens_fn(tokenizer)
    callback_receipts = []
    for i in chosen:
        prefix = tokenizer(test[i]['input_ids'])['input_ids']
        target_ids = tokenizer(test[i]['labels'], add_special_tokens=False)['input_ids']
        for position in range(5):
            allowed = set(callback(0, torch.tensor(prefix + target_ids[:position], dtype=torch.long)))
            expected = {tokenizer.encode(row[position], add_special_tokens=False)[0] for row in index.values()} if position < 4 else {tokenizer.eos_token_id}
            check(allowed == expected, 'actual callback position/EOS')
            callback_receipts.append({'row_idx': i, 'position': position, 'allowed_count': len(allowed), 'allowed_sha256': digest(sorted(allowed))})
    official = om.module(args.repo / 'lc_rec/letter/review/final_audit_20261113/external/official_cce.py')
    metric_tests = om.selftest(official)
    catalog_sizes = Counter(sid.values())
    target = next(iter(catalog_sizes))
    unique = [s for s in catalog_sizes if s != target][:19]
    sids = unique[:3] + [target] + unique[3:]
    ordinary, _ = om.ranking(sids, target)
    corrected = om.cce(sids, target, catalog_sizes)
    check(all(ordinary[f'hit@{k}'] == corrected[f'ItemHit@{k}'] and ordinary[f'ndcg@{k}'] == corrected[f'ItemNDCG@{k}'] for k in (5,10,20)), 'zero-collision once-target degeneration')
    repeated, _ = om.ranking([target, target] + unique[:18], target)
    first_only = om.cce([target, target] + unique[:18], target, catalog_sizes)
    check(repeated['ndcg@5'] > first_only['ItemNDCG@5'], 'zero collision retains duplicate-hit distinction')
    invalid = next(''.join((a[0], b[1], a[2], a[3])) for a in index.values() for b in list(index.values())[:50]
                   if ''.join((a[0], b[1], a[2], a[3])) not in catalog_sizes)
    check(invalid not in catalog_sizes, 'position-legal invalid complete SID exists')
    difference = np.zeros((32, 16), dtype=np.float64)
    draws, low, high = pb.paired_bootstrap(np, difference, 'PREFLIGHT')
    independent = pb.module(args.repo / 'lc_rec/letter/review/final_audit_20261113/statistics_verify.py')
    intervals, reference = independent.bootstrap_columns(difference)
    check(np.array_equal(draws, reference) and np.array_equal(np.array([low, high]), intervals), 'preflight paired same frozen algorithm')
    # Exact sample serialization and native single-process DataLoader behavior.
    full_schedule=single_training.execution_schedule(len(train))
    short_schedule=single_training.execution_schedule(len(train),preflight=True)
    check(full_schedule['per_epoch_microbatches']==16428 and full_schedule['full_epoch_serial_rows']==131424 and
          full_schedule['tail_repeat_rows']==11 and full_schedule['epoch_steps']==[2053,4107,6160,8214], 'single GPU full schedule')
    check(short_schedule['per_epoch_microbatches']==20 and short_schedule['epoch_steps']==[2,5,7,10] and
          short_schedule['max_steps']==10, 'single GPU short schedule')
    reference_generator=torch.Generator().manual_seed(42)
    reference_state=reference_generator.get_state().clone()
    old_shards=historical_rows.reconstruct_native_shards(len(train),reference_state,8214)
    expected_stream=[row for start in range(0,len(old_shards[0]),8) for shard in old_shards for row in shard[start:start+8]]
    serial_stream=single_training.reconstruct_serial_epoch(len(train),reference_state)
    short_stream=single_training.reconstruct_serial_epoch(len(train),reference_state,microbatches=20)
    check(serial_stream==expected_stream and short_stream==expected_stream[:160], 'single GPU exact interleaved historical sample stream')
    check(torch.equal(reference_generator.get_state(),reference_state), 'CPU reconstruction must not mutate live generator')
    historical_path=a0/'evidence/e2e-20261116-r0/continuous/epoch1_rows.json'
    check(gi.sha(historical_path)=='a5453647ebf95c1726d400c5127b276687986b24caabdcaf09969c19f68b21a6', 'fixed historical sample-flow evidence')
    historical=json.loads(historical_path.read_text())['ranks']
    historical_expected=[row for start in range(0,80,8) for shard in historical for row in shard[start:start+8]]
    historical_generator=torch.Generator()
    historical_actual=single_training.reconstruct_serial_epoch(len(train),historical_generator.get_state(),microbatches=20)
    check(historical_actual==historical_expected, 'single GPU stream agrees with actual historical loader rows')
    stream_audit=single_training.audit_serial_epoch(serial_stream,len(train))
    short_audit=single_training.audit_serial_epoch(short_stream,len(train),preflight=True)
    check(stream_audit['repeat_rows']==11 and stream_audit['distinct_rows']==len(train), 'historical tail completion preserved')
    from accelerate import Accelerator
    from accelerate.data_loader import DataLoaderShard
    Accelerator(cpu=True,gradient_accumulation_steps=8)
    live_generator=torch.Generator().manual_seed(42)
    serial_sampler=single_training.build_serial_sampler(range(len(train)),generator=live_generator)
    native_loader=DataLoaderShard(range(len(train)),batch_size=8,sampler=serial_sampler,
        rng_types=['generator'],synchronized_generator=live_generator,drop_last=False)
    actual_stream=[int(row) for batch in native_loader for row in batch.tolist()]
    check(actual_stream==expected_stream and len(native_loader)==16428, 'actual single-process DataLoaderShard sample stream')
    inference_rows=assignment(len(test),0,1)
    validate_assignments([inference_rows],len(test))
    check(len(inference_rows)==len(set(inference_rows))==22363, 'single GPU inference never pads users')
    import inspect
    from deepspeed.runtime.engine import DeepSpeedEngine
    scale_method=DeepSpeedEngine._scale_loss_by_gas
    check('prescaled_loss' in inspect.signature(scale_method).parameters, 'installed DeepSpeed loss scaling signature')
    engine_fixture=SimpleNamespace(gradient_accumulation_steps=lambda:8)
    gradients=[];scaled_gradients=[]
    for micro in range(8):
        parameter=torch.tensor([.25,-.5],dtype=torch.float64,requires_grad=True)
        features=torch.arange(8,dtype=torch.float64)
        # A synthetic eight-example mean tests scaling, not the model loss or a token-global objective.
        mean_loss=((parameter[0]*features+parameter[1]-(micro+1))**2).mean()
        raw_gradient=torch.autograd.grad(mean_loss,parameter,retain_graph=True)[0]
        scaled=scale_method(engine_fixture,mean_loss)
        check(torch.equal(scaled,mean_loss/8), 'actual DeepSpeed GAS8 loss scaling')
        gradients.append(raw_gradient);scaled_gradients.append(torch.autograd.grad(scaled,parameter)[0])
    old_weighting=(torch.stack(gradients[0::2]).sum(0)/4+torch.stack(gradients[1::2]).sum(0)/4)/2
    new_weighting=torch.stack(scaled_gradients).sum(0)
    check(torch.equal(old_weighting,new_weighting), 'historical microbatch mean weights preserved exactly in synthetic FP64 fixture')
    single_gpu_checks={'full_schedule':full_schedule,'short_schedule':short_schedule,
        'full_stream':{k:v for k,v in stream_audit.items() if k!='historical_logical_rank_rows'},
        'short_stream':{k:v for k,v in short_audit.items() if k!='historical_logical_rank_rows'},
        'historical_interleaving_exact':True,'actual_DataLoaderShard_stream_exact':True,
        'actual_historical_first160_rows_exact':True,'historical_default_generator_seed':historical_generator.initial_seed(),
        'historical_raw_evidence':gi.source_identity([historical_path])[0],
        'reconstruction_generator_unchanged':True,'single_rank_inference_users':len(inference_rows),
        'inference_assignment_sha256':digest(inference_rows),
        'loss_scaling':{'scope':'SYNTHETIC_MICROBATCH_WEIGHTING_ONLY','native_method':str(inspect.signature(scale_method)),
            'source':gi.source_identity([inspect.getsourcefile(scale_method)])[0],
            'method_sha256':hashlib.sha256(inspect.getsource(scale_method).encode()).hexdigest(),
            'single_GPU_divisor':8,'historical_divisor_per_rank':4,'historical_rank_average_divisor':2,
            'synthetic_FP64_gradient_exact':True,'global_token_mean_equivalence_claimed':False,
            'single_dual_GPU_numerical_trajectory_equivalence_claimed':False}}
    negative = []
    def rejects(name, operation, expected):
        try:
            operation()
        except (AssertionError, ValueError, FileNotFoundError, KeyError) as error:
            check(any(value in str(error) for value in (expected if isinstance(expected,tuple) else (expected,))), name + ': wrong guard ' + repr(error))
            negative.append({'case': name, 'guard': type(error).__name__, 'message': str(error), 'synthetic_fixture': True})
        else:
            raise AssertionError(name + ': corruption accepted')
    rejects('single_GPU_wrong_world',lambda:gi.verify_execution_profile({**gi.EXECUTION_PROFILE,'world_size':2}),'single GPU execution profile mismatch')
    rejects('single_GPU_wrong_accumulation',lambda:gi.verify_execution_profile({**gi.EXECUTION_PROFILE,'gradient_accumulation_steps':4}),'single GPU execution profile mismatch')
    rejects('serial_training_missing_row',lambda:single_training.audit_serial_epoch(serial_stream[:-1],len(train)),'row count/range')
    duplicate_stream=serial_stream.copy()
    stream_counts=Counter(serial_stream);victim=next(i for i,row in enumerate(serial_stream) if stream_counts[row]==1)
    duplicate_stream[victim]=serial_stream[victim+1]
    rejects('serial_training_wrong_padding_or_duplicate',lambda:single_training.audit_serial_epoch(duplicate_stream,len(train)),'missing rows or wrong tail repeat policy')
    rejects('short_training_duplicate_row',lambda:single_training.audit_serial_epoch([short_stream[1]]+short_stream[1:],len(train),preflight=True),'distinct native sampled rows')
    rejects('inference_duplicate_user_padding',lambda:validate_assignments([inference_rows+[inference_rows[0]]],len(test)),'assignment has missing, duplicate, or out-of-range rows')
    sorted_tokens = sorted({t for row in index.values() for t in row})
    class WrongMapping:
        def __len__(self): return len(tokenizer)
        all_special_ids = tokenizer.all_special_ids
        def encode(self, token, **kwargs):
            if token in sorted_tokens[:2]: token = sorted_tokens[1] if token == sorted_tokens[0] else sorted_tokens[0]
            return tokenizer.encode(token, **kwargs)
        def get_vocab(self): return tokenizer.get_vocab()
    rejects('same_count_wrong_token_to_ID', lambda: gi.tokenizer_identity(WrongMapping(), index), 'token to ID mapping mismatch')
    old = LlamaTokenizer.from_pretrained(args.model, local_files_only=True)
    rejects('old_base_tokenizer', lambda: gi.tokenizer_identity(old, index), 'tokenizer vocabulary')
    fixture = args.output / 'SYNTHETIC_CPU_INTEGRATION_ONLY'
    fixture.mkdir()
    # Corrupt copied small assets only. Primary fixed data remain untouched.
    wrong_data = fixture / 'wrong-data' / 'Beauty'; wrong_data.mkdir(parents=True)
    for kind in gi.ASSETS:
        (wrong_data / f'Beauty.{kind}.json').symlink_to((root / f'Beauty.{kind}.json').resolve())
    (wrong_data / 'Beauty.g3.upstream.json').write_bytes((root / 'Beauty.g3.upstream.json').read_bytes())
    (wrong_data / 'Beauty.index.json').unlink()
    (wrong_data / 'Beauty.index.json').write_text('{}')
    rejects('old_or_wrong_index', lambda: gi.verify_assets(wrong_data.parent), 'G3 data asset identity')
    (wrong_data / 'Beauty.index.json').unlink()
    (wrong_data / 'Beauty.index.json').symlink_to((root / 'Beauty.index.json').resolve())
    upstream = json.loads((wrong_data / 'Beauty.g3.upstream.json').read_text())
    bad_bridge = fixture / 'wrong_bridge.csv'; bad_bridge.write_text('wrong bridge\n')
    upstream['bridge']['path'] = str(bad_bridge)
    save(wrong_data / 'Beauty.g3.upstream.json', upstream)
    rejects('wrong_bridge_after_valid_data', lambda: gi.verify_assets(wrong_data.parent), 'upstream bridge identity')
    wrong_model = fixture / gi.MODEL_REVISION; wrong_model.mkdir()
    for name in gi.MODEL_FILES:
        (wrong_model/name).write_bytes(b'SYNTHETIC_WRONG_SAME_MODEL_SOURCE')
    rejects('same_revision_wrong_model_source', lambda: gi.verify_base_model(wrong_model), 'model content identity')
    import shutil
    wrong_tokenizer = fixture/'same_count_saved_wrong_tokenizer'
    shutil.copytree(saved_tokenizer,wrong_tokenizer)
    token_map=json.loads((wrong_tokenizer/'added_tokens.json').read_text())
    left,right=sorted(token_map)[:2]
    token_map[left],token_map[right]=token_map[right],token_map[left]
    save(wrong_tokenizer/'added_tokens.json',token_map)
    def wrong_saved_mapping():
        gi.tokenizer_identity(LlamaTokenizer.from_pretrained(wrong_tokenizer,local_files_only=True),index)
    rejects('saved_tokenizer_wrong_mapping',wrong_saved_mapping,('token to ID mapping mismatch','Non-consecutive added token','Wrong index found'))
    cp = fixture / 'checkpoint-2'; cp.mkdir()
    commit = 'a'*40; config_sha = 'b'*64
    for member in formal_train.checkpoint_required_files(2):
        member_path = cp/member; member_path.parent.mkdir(parents=True, exist_ok=True)
        member_path.write_bytes(b'SYNTHETIC_STATE_GUARD_FIXTURE')
    (cp/'latest').write_text('global_step2\n'); save(cp/'trainer_state.json', {'global_step': 2})
    member_names = ['adapter_model.bin','adapter_config.json','tokenizer.model','tokenizer_config.json','added_tokens.json','special_tokens_map.json']
    adapter_meta = {'experiment_id': gi.EXPERIMENT_ID, 'scope':'PREFLIGHT', 'assets':gi.ASSETS,
                    'upstream':{'g3':gi.G3_SHA,'bridge':gi.BRIDGE_SHA}, 'run_commit':commit,
                    'config_sha256':config_sha, 'tokenizer_identity':token_identity,
                    'execution_topology':gi.EXECUTION_TOPOLOGY,'execution_profile':gi.EXECUTION_PROFILE,
                    'base_model_identity':{'revision':gi.MODEL_REVISION,'files':{n:{'sha256':h} for n,h in gi.MODEL_FILES.items()}},
                    'adapter_sha256':gi.sha(cp/'adapter_model.bin'), 'files':gi.source_identity([cp/n for n in member_names])}
    save(cp/'g3_identity.json', adapter_meta)
    manifest = {'run_commit':commit,'config_sha256':config_sha,'experiment_id':gi.EXPERIMENT_ID,
                'scope':'PREFLIGHT','assets':gi.ASSETS,'collective_ranks':1,'world_size':1,'global_step':2,
                'execution_topology':gi.EXECUTION_TOPOLOGY,'execution_profile':gi.EXECUTION_PROFILE,
                'files':[{'path':str(p.relative_to(cp)),'bytes':p.stat().st_size,'sha256':gi.sha(p)} for p in sorted(cp.rglob('*')) if p.is_file()]}
    save(cp/'complete.json', manifest)
    formal_train.verify_checkpoint(cp, commit, config_sha)
    gi.verify_adapter_identity(cp,scope='PREFLIGHT',run_commit=commit,config_sha256=config_sha)
    rejects('adapter_wrong_source_commit',lambda:gi.verify_adapter_identity(cp,scope='PREFLIGHT',run_commit='c'*40,config_sha256=config_sha),'G3 adapter source commit')
    rejects('adapter_wrong_effective_config',lambda:gi.verify_adapter_identity(cp,scope='PREFLIGHT',run_commit=commit,config_sha256='c'*64),'G3 adapter effective config mismatch')
    rejects('wrong_checkpoint_source', lambda: formal_train.verify_checkpoint(cp, 'c'*40, config_sha), 'checkpoint source commit mismatch')
    rejects('wrong_checkpoint_config', lambda: formal_train.verify_checkpoint(cp, commit, 'c'*64), 'checkpoint effective config mismatch')
    incomplete = copy.deepcopy(manifest); incomplete['files'] = [r for r in incomplete['files'] if r['path']!='rng_state.pth']
    save(cp/'complete.json', incomplete)
    rejects('missing_rank_member', lambda: formal_train.verify_checkpoint(cp, commit, config_sha), 'checkpoint missing required rank/member')
    incomplete=copy.deepcopy(manifest);incomplete['collective_ranks']=2
    save(cp/'complete.json',incomplete)
    rejects('missing_collective_rank',lambda:formal_train.verify_checkpoint(cp,commit,config_sha),'checkpoint collective rank completion mismatch')
    for case, changes, message in [
        ('checkpoint_wrong_world_size',{'world_size':2},'checkpoint world size mismatch'),
        ('checkpoint_wrong_execution_topology',{'execution_topology':'DUAL_GPU'},'checkpoint execution topology mismatch'),
        ('checkpoint_old_accumulation_profile',{'execution_profile':{**gi.EXECUTION_PROFILE,'gradient_accumulation_steps':4}},'single GPU execution profile mismatch')]:
        mixed=copy.deepcopy(manifest);mixed.update(changes);save(cp/'complete.json',mixed)
        rejects(case,lambda:formal_train.verify_checkpoint(cp,commit,config_sha),message)
    foreign=cp/'loader_state_1.pth';foreign.write_bytes(b'SYNTHETIC_FOREIGN_DUAL_RANK_MEMBER')
    mixed=copy.deepcopy(manifest);mixed['files'].append({'path':foreign.name,'bytes':foreign.stat().st_size,'sha256':gi.sha(foreign)})
    save(cp/'complete.json',mixed)
    rejects('checkpoint_foreign_rank1_member',lambda:formal_train.verify_checkpoint(cp,commit,config_sha),'checkpoint foreign rank member in single GPU execution')
    save(cp/'complete.json', manifest)
    foreign.unlink()
    (cp/'complete.json').rename(cp/'complete.incomplete.json')
    rejects('checkpoint_not_collectively_completed', lambda: formal_train.verify_checkpoint(cp, commit, config_sha), 'complete.json')
    (cp/'complete.incomplete.json').rename(cp/'complete.json')
    wrong_adapter = copy.deepcopy(adapter_meta); wrong_adapter['tokenizer_identity']['token_to_id_sha256']='0'*64
    save(cp/'g3_identity.json', wrong_adapter)
    rejects('old_adapter_tokenizer_mapping', lambda: gi.verify_adapter_identity(cp, scope='PREFLIGHT'), 'adapter token mapping')
    mixed_adapter=copy.deepcopy(adapter_meta);mixed_adapter['execution_profile']['gradient_accumulation_steps']=4
    save(cp/'g3_identity.json',mixed_adapter)
    rejects('adapter_old_accumulation_profile',lambda:gi.verify_adapter_identity(cp,scope='PREFLIGHT'),'single GPU execution profile mismatch')
    save(cp/'g3_identity.json', adapter_meta)
    original = (cp/'adapter_model.bin').read_bytes()
    (cp/'adapter_model.bin').write_bytes(b'X'+original[1:])
    rejects('same_shape_or_size_wrong_weights', lambda: gi.verify_adapter_identity(cp, scope='PREFLIGHT'), 'adapter member identity')
    (cp/'adapter_model.bin').write_bytes(original)
    (cp/'tokenizer.model').rename(cp/'missing-tokenizer.model')
    rejects('adapter_missing_tokenizer',lambda:gi.verify_adapter_identity(cp,scope='PREFLIGHT'),'adapter member identity')
    (cp/'missing-tokenizer.model').rename(cp/'tokenizer.model')
    rejects('preflight_adapter_as_formal', lambda: gi.verify_adapter_identity(cp, scope='FORMAL'), 'adapter scope mismatch')
    rejects('no_formal_authorization', lambda: gi.verify_launch_authorization(None,None,commit,config_sha), 'formal launch authorization required')
    old_auth=fixture/'old-authorization.json';save(old_auth,{'experiment_id':'old-letter','formal_launch_authorized':True})
    rejects('old_experiment_authorization',lambda:gi.verify_launch_authorization(old_auth,gi.sha(old_auth),commit,config_sha),'different experiment')
    # Build internally coherent fixtures from real target/user identities; corrupt one semantic field at a time.
    _, catalog, users, _ = fe.catalog_and_users(args.data_root)
    evaluator = fe.load_evaluator(vendor)
    selected = list(range(32)); paths = []
    for rank in range(1):
        path=fixture/f'predictions.rank{rank}.jsonl';paths.append(path)
        rows=[]
        for i in selected:
            candidate_sids=[users[i]['target']]+[s for s in catalog if s!=users[i]['target']][:19]
            candidates=[{'generation_order':j,'raw_decoded':'Response: '+s,'sid':s,'sequence_score':float(-j)} for j,s in enumerate(candidate_sids)]
            raw_ids=tokenizer(test[i]['input_ids'])['input_ids']
            raw_mask=[1]*len(raw_ids)
            rows.append({'original_input_token_ids':raw_ids,'input_token_ids':raw_ids,'attention_mask':raw_mask,
                         'actual_model_prefill':{'input_ids':raw_ids,'attention_mask':raw_mask},**users[i],'rank':rank,'prompt_id':0,'scope':'PREFLIGHT','condition':'normal','model_identity_sha256':'synthetic-model',
                         'candidates':candidates,'sorted_generation_order':list(range(20)),'relevance':[1]+[0]*19,'generation_seconds':0.0})
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        save(fixture/f'rank{rank}.json',{'status':'PASS','rank':rank,'world_size':1,
              'execution_topology':gi.EXECUTION_TOPOLOGY,'execution_profile':gi.EXECUTION_PROFILE,'RUN_COMMIT':commit,'scope':'PREFLIGHT','preflight_only':True,
              'condition':'normal','model_identity_sha256':'synthetic-model','assigned_row_indices':selected,
              'assignment_sha256':hashlib.sha256(json.dumps(selected,separators=(',',':')).encode()).hexdigest(),'completed_users':32,'elapsed_generation_seconds':0,
              'adapter_files':{'adapter_model.bin':{'sha256':'synthetic-model'}},'data_identity':asset_rows,
              'author_sources':{},'generation':{'num_beams':20},'allowed_callback_sha256':'synthetic-callback'})
    def aggregate():
        return fe.aggregate_predictions(paths,users,catalog,evaluator,run_commit=commit,scope='PREFLIGHT',expected_indices=selected,condition='normal')
    check(aggregate()['actual_test_users']==32,'positive prediction fixture')
    rank_path=fixture/'rank0.json';rank_original=json.loads(rank_path.read_text())
    for case,changes,message in [
        ('prediction_old_dual_world',{'world_size':2},'Single-GPU prediction topology mismatch'),
        ('prediction_old_accumulation_profile',{'execution_profile':{**gi.EXECUTION_PROFILE,'gradient_accumulation_steps':4}},'single GPU execution profile mismatch')]:
        bad=copy.deepcopy(rank_original);bad.update(changes);save(rank_path,bad)
        rejects(case,aggregate,message)
    save(rank_path,rank_original)
    rejects('prediction_extra_rank_shard',lambda:fe.aggregate_predictions(paths+paths,users,catalog,evaluator,run_commit=commit,scope='PREFLIGHT',expected_indices=selected,condition='normal'),'exactly one rank shard')
    pristine=paths[0].read_text();rows=[json.loads(line) for line in pristine.splitlines()]
    def corrupt_prediction(case, mutate, expected):
        changed=copy.deepcopy(rows);mutate(changed)
        paths[0].write_text(''.join(json.dumps(row)+'\n' for row in changed))
        rejects(case,aggregate,expected)
        paths[0].write_text(pristine)
    corrupt_prediction('prediction_missing_user',lambda r:r.pop(),'shard length')
    corrupt_prediction('prediction_duplicate_user',lambda r:r.append(r[0]),'Duplicate/out-of-range')
    corrupt_prediction('prediction_wrong_target',lambda r:r[0].update(target='wrong-SID'),'identity/target mismatch')
    corrupt_prediction('prediction_wrong_model',lambda r:r[0].update(model_identity_sha256='wrong-model'),'model/scope/Shuffle binding')
    corrupt_prediction('prediction_wrong_shuffle',lambda r:r[0].update(condition='s4'),'model/scope/Shuffle binding')
    corrupt_prediction('prediction_preflight_as_formal',lambda r:r[0].update(scope='FORMAL'),'model/scope/Shuffle binding')
    rejects('formal_subset_population',lambda:fe.aggregate_predictions(paths,users,catalog,evaluator,run_commit=commit,scope='FORMAL',expected_indices=selected),'full fixed 22363-user axis')
    imported = [Path(m.__file__).resolve() for m in (data_module,collator_module,utils_module,gi,formal_train,fe,om,pb,single_training,historical_rows)]
    imported += [vendor/'prompt.py',vendor/'evaluate.py',Path(__file__).resolve(),args.config.resolve(),
                 Path(inspect.getsourcefile(scale_method)),Path(inspect.getsourcefile(DataLoaderShard))]
    result={'status':'PASS','scope':'PREFLIGHT_CPU_VALIDATION','experiment_id':gi.EXPERIMENT_ID,
            'world_size':1,'execution_topology':gi.EXECUTION_TOPOLOGY,'execution_profile':gi.EXECUTION_PROFILE,
            'dual_gpu_requirement':'NOT_APPLICABLE_THIS_RUN',
            'argv':sys.argv,'cwd':os.getcwd(),'python':sys.executable,'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__},
            'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'assets':asset_rows,'config':gi.source_identity([args.config])[0],
            'actual_import_files':gi.source_identity(imported),'dataset_rows':[len(train),len(valid),len(test)],'all_train_rows_verified':cursor,
            'all_validation_test_targets_verified':len(inter),'tokenizer':token_identity,'saved_tokenizer_files':gi.source_identity([p for p in saved_tokenizer.iterdir() if p.is_file()]),
            'boundary_rows':[{ 'row_idx':i,'user_id':axis[i],'history_length':lengths[i],
                'repeated_history':len(set(inter[axis[i]][:-1][-20:]))<lengths[i],
                'fourth_values':[index[str(item)][3] for item in inter[axis[i]][:-1][-20:]]} for i in chosen],
            'real_repeat_history_users':len(repeat),'synthetic_repeated_history_fixture':{'scope':'SYNTHETIC_INPUT_BOUNDARY_ONLY','historical_items':2,'same_actual_SID_repeated':True},
            'label_positions_checked':labels_checked,
            'collator_shapes':{k:list(v.shape) for k,v in batch.items()},'callback_invocations':callback_receipts,
            'metrics':{'inherited':metric_tests,'zero_collision_once_target_equal':True,'duplicate_target_ndcg_distinct':True,
                       'position_legal_invalid_sid':invalid,'preflight_pairing_equal_independent':True},
            'single_gpu_checks':single_gpu_checks,'negative_cases':negative,'negative_fixture_scope':'SYNTHETIC_CPU_INTEGRATION_ONLY; no model weights executed',
            'gpu_used':False,'exit_code':0,'elapsed_seconds':time.monotonic()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    save(args.output/'result.json',result)
    print(json.dumps({'status':'PASS','negative_cases':len(negative),'label_positions_checked':labels_checked,'result':str(args.output/'result.json')}),flush=True)


if __name__=='__main__':
    main()
