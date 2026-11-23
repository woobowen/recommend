"""Independent, CPU-only full G3 Shuffle V2 original-prediction audit."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import permutations
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

GPU_COMMIT = '86660be5a62c8325caf74cf2810581850ca12803'
TRAIN_COMMIT = '21d495b0be5b606853ec7cec68a3eaa1b58acbd7'
MODEL_SHA = '3315622866cd97833741608ca21d82b72a10184ab8bff4dc688892146d0bf915'
ADAPTER_SHA = 'd3a179b59ba7a7218d6e1767884fd1c5961fffb9b756ea5f8bc5522775db6a28'
CPU_AUDIT_SHA = '98cd89fcc3440d3ac506483ff9daf223560c6eeca943cb3f267aeb65adee0798'
CPU_MANIFEST_SHA = '2329944065d81f07c02041dbf5a62daf2609af7795e71804ed854db8faf58906'
ASSIGNMENT_SHA = '22ee58fe9f948f50e5cbc92160b6dc7df7e2f633cb9e8bf3121da5fb35bcb088'
NORMAL_SHA = 'a2247ec7016c4d88c08957fea6c0338515a124ca316245e9592ac260e8df730a'
NORMAL_RANK_SHA = '82a026533a9ea619d38ba4f3baa632eee9270860b2d3e43030873b9626a94e1f'
TOKEN_MAP_SHA = 'dba960c3166b49d9098c713095273db3b6faa856ae2485ec8eda408ae66e82bc'
FROZEN_SOURCE_SHA = '9c7748ce7659a085d15128e060a1b72f6ac92b62a8642056810d0fa6df6a14e7'
DERIVED_SHA = 'f2d4fc05252dc0215553e0a35d0a5fcdecacd0b29323fb0ffb77e5133a23c80b'
INDEX_SHA = '420fc060caba3a22b0f03900dfe26abc67da8753f3bac2803f7f3c8703a05c22'
INTER_SHA = '8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5'
ITEM_SHA = '74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330'
PLAN_SHA = 'cdf5754c1eeb6c22836f0b4a5914c1fb126f0d4bc6ff6c3139ef09446e5b04e2'
PROTOCOL = 'GRID-LETTER-SHUFFLE-V2'
MODES = {'shuffle2':(2,3),'shuffle3a':(0,1,2),'shuffle3b':(1,2,3),'shuffle4':(0,1,2,3)}
CATA = {width:tuple(permutations(range(width))) for width in (2,3,4)}
LIMITS = {width:2**256//len(catalog)*len(catalog) for width,catalog in CATA.items()}
BEAMS = 20
USERS = 22363
EXPECTED_KWARGS = {'max_new_tokens':10,'num_beams':BEAMS,'num_return_sequences':BEAMS,
                   'output_scores':True,'return_dict_in_generate':True,'early_stopping':True}
EXPECTED_NATIVE = {'do_sample':False,'use_cache':True,'eos_token_id':2,'pad_token_id':0}
SERVER_RUN_ROOT = Path('/data/bwwu/lc-rec/g3-beauty-v2')
SERVER_DATA_ROOT = Path('/home/bwwu/grid-audits/sid-extensions-p0/g3-production-20261118/data')
SERVER_PLAN = Path('/home/bwwu/grid-audits/sid-extensions-p0/requests/g3-beauty-v2-group-formal-r0.plan.json')
SERVER_NORMAL = Path('/data/bwwu/lc-rec/g3-formal-20261118/21d495b0be5b/attempts/normal/a0/test')
AXIS_FIELDS = ('row_idx','user_id','target_item_id','target')
ORIG_FIELDS = ('model','model_identity_sha256','adapter_files','tokenizer','tokenizer_identity','allowed_callback_sha256',
               'execution_topology','execution_profile','experiment_id','data_identity','versions','strict_adapter_reload',
               'adapter_identity','installed_model_sources')


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest_obj(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_file(path):
    with path.open('rb') as file:
        h=hashlib.sha256()
        for block in iter(lambda:file.read(1024**2),b''):
            h.update(block)
        return h.hexdigest()


def read(path, expected=None):
    raw=path.read_bytes()
    if expected is not None:
        assert hashlib.sha256(raw).hexdigest()==expected,path
    return json.loads(raw)


def signature(s):
    return (s.st_ino,s.st_dev,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def raw_rows(path, expected_sha, expected_size):
    if not path.exists() and path.suffix!='.gz':
        path=path.with_name(path.name+'.gz')
    before=signature(path.stat())
    opener=gzip.open if path.suffix=='.gz' else open
    h=hashlib.sha256()
    size=0
    with opener(path,'rb') as file:
        assert signature(os.fstat(file.fileno()))==before,path
        for raw in file:
            assert raw.endswith(b'\n'),path
            h.update(raw)
            size+=len(raw)
            yield json.loads(raw)
        assert signature(os.fstat(file.fileno()))==before,path
    assert signature(path.stat())==before and size==expected_size and h.hexdigest()==expected_sha,path


def draw(width,row,occurrence):
    catalog=CATA[width]
    revision=0
    while True:
        n=int.from_bytes(hashlib.sha256(canonical_bytes([PROTOCOL,42,width,row,occurrence,revision])).digest(),'big')
        if n<LIMITS[width]:
            local=n%len(catalog)
            return local,catalog[local],revision
        revision+=1


def metadata(mode):
    return {'protocol':PROTOCOL,'identity_policy':'allow_identity','mode':mode,'seed':42,'active_positions':list(MODES[mode])}


def load_frozen(a):
    audit=read(a.cpu_audit,CPU_AUDIT_SHA)
    assert (audit['status'],audit['users'],audit['items'],audit['sids'],audit['collisions'],audit['groups'])==(
        'PASS_G3_CPU_V2_FULL',USERS,12101,12101,0,159966)
    assert audit['manifest_sha256']==CPU_MANIFEST_SHA and audit['normal_sha256']==NORMAL_SHA
    vocab=audit['token_ids']
    assert digest_obj(vocab)==TOKEN_MAP_SHA and len(vocab)==786
    assert len(set(vocab.values()))==786
    index=read(a.data_root/'Beauty/Beauty.index.json',INDEX_SHA)
    inter=list(read(a.data_root/'Beauty/Beauty.inter.json',INTER_SHA).items())
    assert digest_file(a.data_root/'Beauty/Beauty.item.json')==ITEM_SHA
    assert len(inter)==USERS and len(index)==12101 and {len(v) for v in index.values()}=={4}
    catalog={''.join(code) for code in index.values()}
    slots=tuple({vocab[code[n]] for code in index.values()} for n in range(4))
    assert len(catalog)==12101 and [len(v) for v in slots]==[256,256,256,18]
    ids={number:symbol for symbol,number in vocab.items()}
    normal_rank=read(a.normal_rank,NORMAL_RANK_SHA)
    assignment=normal_rank['assigned_row_indices']
    assert sorted(assignment)==list(range(USERS))
    assert digest_obj(assignment)==normal_rank['assignment_sha256']==audit['assignment_sha256']==ASSIGNMENT_SHA
    assert normal_rank['model_identity_sha256']==MODEL_SHA
    cpus=[]
    h=hashlib.sha256()
    with a.cpu_manifest.open('rb') as file:
        before=signature(os.fstat(file.fileno()))
        for i,line in enumerate(file):
            h.update(line)
            row=json.loads(line)
            user,history=inter[i]
            assert (row['row_idx'],row['user_id'],row['target_item_id'],row['target'])==(
                    i,user,str(history[-1]),''.join(index[str(history[-1])]))
            assert [g[0] for g in row['groups']]==list(map(str,history[:-1][-20:]))
            for item, positions, tokens in row['groups']:
                assert tokens==[vocab[symbol] for symbol in index[item]]
                assert positions==list(range(positions[0],positions[0]+4))
            assert row['conditions']['shuffle3a']['permutation_ids']==row['conditions']['shuffle3b']['permutation_ids']
            cpus.append(row)
        assert signature(os.fstat(file.fileno()))==before
    assert signature(a.cpu_manifest.stat())==before
    assert len(cpus)==USERS and h.hexdigest()==CPU_MANIFEST_SHA
    assert sum(len(x['groups']) for x in cpus)==audit['groups']
    return audit,ids,catalog,slots,normal_rank,assignment,cpus


def check_rank(a, mode, expected, frozen_rank, card, assignment):
    folder=a.root/'predictions'/mode if a.root else a.run_root/f'gpu-formal-{mode}-r0'
    rank=read(folder/'rank0.json')
    binding=read(folder/'v2-binding.rank0.json')
    stage=read(folder/'metrics.json')
    assert all(rank[field]==frozen_rank[field] for field in ORIG_FIELDS),mode
    assert rank['scope']==binding['scope']==binding['adapter_scope']==stage['scope']=='FORMAL'
    assert rank['RUN_COMMIT']==stage['RUN_COMMIT']==binding['V2_RUN_COMMIT']==GPU_COMMIT
    assert rank['MODEL_TRAINING_COMMIT']==binding['MODEL_TRAINING_COMMIT']==TRAIN_COMMIT
    assert rank['entry_sha256']==binding['frozen_source_sha256']==FROZEN_SOURCE_SHA
    assert rank['preflight_only'] is False and 'preflight_manifest' not in rank
    assert rank['normal_off_same_process_exact']==[]
    assert (rank['rank'],rank['local_rank'],rank['world_size'])==(0,0,1)
    assert rank['model_identity_sha256']==stage['model_identity_sha256']==MODEL_SHA
    assert (rank['status'],stage['status'],rank['conditions'],rank['condition'],stage['condition'])==(
        'PASS','PASS',[mode],mode,mode)
    assert rank['completed_users']==rank['assigned_users']==stage['actual_test_users']==USERS
    inspected=rank['assigned_row_indices']
    assert inspected==assignment and rank['assignment_sha256']==digest_obj(inspected)==ASSIGNMENT_SHA
    assert stage['all_row_indices_exactly_once'] and stage['test_complete']
    assert rank['tokenizer_identity']['vocab_size']==32786 and rank['tokenizer_identity']['new_token_count']==786
    assert rank['adapter_identity']['checkpoint_source']['global_step']==6160
    assert rank['adapter_identity']['adapter_sha256']==stage['adapter_model_sha256']==ADAPTER_SHA
    assert all(binding[k]==v for k,v in metadata(mode).items())
    assert binding['cpu_audit_sha256']==CPU_AUDIT_SHA
    assert binding['cpu_manifest_sha256']==rank['v2_cpu']['manifest_sha256']==CPU_MANIFEST_SHA
    assert binding['derived_sha256']==digest_file(a.derived or folder/'derived.rank0.py')==DERIVED_SHA
    assert not any(arg.startswith('--preflight') or arg=='--verify-normal-off' for arg in binding['effective_argv'])
    assert rank['generation']==frozen_rank['generation']
    runtime=rank['v2_model']
    assert runtime['all_parameters_on_logical_cuda_0'] and runtime['eval_mode'] and runtime['requires_grad_false']
    assert runtime['parameter_identity_storage_and_versions_unchanged_each_row']
    assert runtime['base_dtype']==runtime['adapter_embedding_and_head_dtype']=='torch.bfloat16'
    assert runtime['lora_dtype']=='torch.float32'
    env=runtime['rank_environment']
    assert (env['CUDA_VISIBLE_DEVICES'],env['WORLD_SIZE'],env['RANK'],env['LOCAL_RANK'])==(card,'1','0','0')
    assert rank['gpu']['visible_devices']==card
    assert rank['gpu']['logical_device']==0 and rank['gpu']['logical_device_count']==1
    assert env['DS_ACCELERATOR']=='cuda' and env['TORCHINDUCTOR_COMPILE_THREADS']=='1'
    assert env['CUDA_HOME']==expected['env']['CUDA_HOME'] and env['LD_LIBRARY_PATH']==expected['env']['LD_LIBRARY_PATH']
    load=rank['strict_adapter_reload']
    assert (load['keys'],load['lora_keys'],load['elements'])==(450,448,288571392)
    checks=load['tensor_checks']
    assert len(checks)==450 and sum(math.prod(c['shape']) for c in checks.values())==load['elements']
    assert sum('lora_' in key for key in checks)==448
    for name, c in checks.items():
        assert c['finite_and_exact_after_explicit_dtype_conversion']
        assert c['loaded_dtype']==('torch.float32' if 'lora_' in name else 'torch.bfloat16')
    receipt,=stage['prediction_files']
    assert receipt['rank']==0 and receipt['users']==USERS
    return folder,rank,stage,receipt


def verify_candidates(row, ids, slots, catalog):
    assert row['original_callback_verified_all_tokens'] and len(row['candidates'])==BEAMS
    vals,scores=[],[]
    outside=0
    for n,c in enumerate(row['candidates']):
        suffix=c['generated_token_ids']
        assert c['generation_order']==n and c['eos_position']==4
        assert suffix[4]==2 and all(token in (0,2) for token in suffix[5:])
        assert all(token in slots[pos] for pos,token in enumerate(suffix[:4]))
        sid=''.join(ids[token] for token in suffix[:4])
        assert c['sid']==sid==c['raw_decoded'].rsplit('Response:',1)[-1].strip().replace(' ','')
        score=c['sequence_score']
        assert math.isfinite(score) and struct.unpack('!f',struct.pack('!f',score))[0]==score
        vals.append(sid)
        scores.append(score)
        outside+=sid not in catalog
    order=sorted(range(BEAMS),key=lambda i:(-scores[i],i))
    assert row['sorted_generation_order']==order
    assert row['relevance']==[int(vals[i]==row['target']) for i in order]
    return outside,BEAMS-len(set(vals))


def verify_row(row,cpu,mode,token_set):
    idx=cpu['row_idx']
    assert all(row[k]==cpu[k] for k in AXIS_FIELDS)
    assert row['V2_RUN_COMMIT']==GPU_COMMIT and row['model_identity_sha256']==MODEL_SHA
    assert (row['rank'],row['world_size'],row['scope'],row['condition'],row['prompt_id'])==(0,1,'FORMAL',mode,0)
    assert math.isfinite(row['generation_seconds']) and row['generation_seconds']>=0
    raw,actual,mask=(row[k] for k in ('original_input_token_ids','input_token_ids','attention_mask'))
    assert row['preflight_extra_left_padding']==0 and mask==[1]*len(raw)
    assert len(actual)==cpu['encoded_length']==row['input_nonpadding_tokens']==len(raw)
    rh,ah,mh=(digest_obj(values) for values in (raw,actual,mask))
    assert rh==cpu['original_sha256'] and mh==cpu['attention_mask_sha256']
    assert row['v2_cpu_binding']=={'manifest_sha256':CPU_MANIFEST_SHA,'row_sha256':digest_obj(cpu),
         'original_sha256':rh,'attention_mask_sha256':mh,'input_sha256':ah,'mode':mode,'row_idx':idx}
    proof=row['shuffle']
    assert all(proof[k]==v for k,v in metadata(mode).items())
    assert proof['row_idx']==idx and proof['original_input_token_ids']==raw and proof['attention_mask']==mask
    evidence=proof['evidence']
    assert evidence['history_group_count']==len(cpu['groups'])
    locations=[i for i,token in enumerate(raw) if token in token_set]
    assert len(locations)==4*len(cpu['groups'])
    active=MODES[mode]
    expected=raw.copy()
    ids, moved, shuffled_groups = [],[0]*4,0
    for occ, ((_,positions,tokens),group) in enumerate(zip(cpu['groups'],evidence['groups'],strict=True)):
        assert positions==locations[occ*4:occ*4+4]
        assert tokens==[raw[i] for i in positions]
        pick,relative,revision=draw(len(active),idx,occ)
        full=list(range(4))
        for target,source in zip(active,relative,strict=True):
            full[target]=active[source]
        shuffled=[tokens[i] for i in full]
        assert group['positions']==positions and group['original_ids']==tokens
        assert group['visible_history_item_index']==occ and group['start']==positions[0]
        assert group['relative_permutation']==list(relative) and group['permutation_id']==pick
        assert group['rejection_counter']==revision and group['permutation']==full
        assert group['shuffled_ids']==shuffled and group['tokens_changed']==(tokens!=shuffled)
        assert group['nonidentity']==(pick!=0)
        for pos,token in zip(positions,shuffled,strict=True):
            expected[pos]=token
        ids.append(pick)
        shuffled_groups+=int(shuffled!=tokens)
        for slot,(old,new) in enumerate(zip(tokens,shuffled,strict=True)):
            moved[slot]+=old!=new
    assert expected==actual
    assert cpu['conditions'][mode]=={'input_sha256':ah,'input_changed':raw!=actual,'permutation_ids':ids}
    assert (evidence['original_sha256'],evidence['shuffled_sha256'],evidence['attention_mask_sha256'],
            evidence['input_changed'])==(rh,ah,mh,raw!=actual)
    prefill=row['actual_model_prefill']
    assert prefill['beam_expansion']==BEAMS and prefill['initial_cache_length']==0
    assert prefill['effective_use_cache'] is True
    for kind,values,key in (('input',actual,'input_ids'),('mask',mask,'attention_mask'),
                            ('position',list(range(len(actual))),'position_ids')):
        assert prefill[key]==values and prefill[f'all_beam_{kind}_sha256']==digest_obj([values]*BEAMS)
    lookup=prefill['embedding_lookup']
    assert lookup['shape']==[BEAMS,len(actual),4096] and lookup['dtype']=='torch.bfloat16'
    assert lookup['all_beams_exact'] and row['lookup_equivalence_exact'] and lookup['input_sha256']==ah
    assert len(bytes.fromhex(lookup['first_beam_bf16_sha256']))==32
    gen=row['effective_generation']
    assert gen['native_score_dtype']=='torch.float32' and gen['kwargs']==EXPECTED_KWARGS
    assert gen['effective']=={**EXPECTED_KWARGS,**EXPECTED_NATIVE}
    assert all(gen['generation_config'][k]==v for k,v in EXPECTED_NATIVE.items())
    return ids,moved,shuffled_groups,raw==actual


def frozen_normal(a, assignment, cpus):
    natural={i for i,cpu in enumerate(cpus) if any(not cpu['conditions'][m]['input_changed'] for m in MODES)}
    retained={}
    for idx,row in zip(assignment,raw_rows(a.normal_predictions,NORMAL_SHA,400029854),strict=True):
        cpu=cpus[idx]
        assert all(row[k]==cpu[k] for k in AXIS_FIELDS)
        assert row['condition']=='normal' and row['scope']=='FORMAL' and row['rank']==0
        assert row['model_identity_sha256']==MODEL_SHA
        assert row['input_token_ids']==row['original_input_token_ids']
        assert digest_obj(row['input_token_ids'])==cpu['original_sha256']
        assert digest_obj(row['attention_mask'])==cpu['attention_mask_sha256']
        if idx in natural:
            retained[idx]=row
    assert set(retained)==natural
    return retained


def compare_identity(mode,row,baseline):
    assert row['input_token_ids']==baseline['input_token_ids'] and row['attention_mask']==baseline['attention_mask']
    old,new=baseline['candidates'],row['candidates']
    sans_scores=('generation_order','generated_token_ids','eos_position','raw_decoded','sid')
    same_content=all(all(x[k]==y[k] for k in sans_scores) for x,y in zip(old,new,strict=True))
    same_score=all(x['sequence_score']==y['sequence_score'] for x,y in zip(old,new,strict=True))
    delta=max(abs(x['sequence_score']-y['sequence_score']) for x,y in zip(old,new,strict=True))
    same_order=baseline['sorted_generation_order']==row['sorted_generation_order']
    return {'condition':mode,'row_idx':row['row_idx'],'candidate_content_exact':same_content,
            'scores_exact':same_score,'sorted_order_exact':same_order,'all_candidates_exact':old==new,
            'max_abs_score_delta_vs_normal':delta}


def terminal_readback(a, plan):
    if a.runner is None:
        return None
    assert digest_file(a.runner)==plan['runner_sha256']
    jobs=a.job_root or Path(plan['job_root'])
    names=[plan['parent']['name'],*(v['name'] for m in MODES for v in plan['modes'][m].values()),plan['paired']['name']]
    result={}
    for name in names:
        job_path=jobs/name
        raw=subprocess.check_output(['/usr/bin/python3','-I','-S','-B',str(a.runner),'query',str(job_path)],text=True,timeout=40)
        row=json.loads(raw)
        request=read(job_path/'request.json')
        accepted=read(job_path/'accepted.json')
        assert request['job_id']==row['job_id']==name
        assert row['status']=='SUCCEEDED' and row['exit_code']==0 and row['cleanup_verified']
        assert not any(row[k] for k in ('supervisor_alive','worker_alive','workgroup_live'))
        verify=row['workgroup_observation']
        assert verify['scan_complete'] and verify['verified_empty']
        assert accepted['runner_sha256']==plan['runner_sha256']
        canon=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        assert canon==accepted['request_sha256']==row['request_sha256']
        if '-gpu-formal-' in name:
            assert request['capture_version']==GPU_COMMIT and request['env']['CUDA_VISIBLE_DEVICES']==plan['card']
            assert request['argv'][request['argv'].index('--scope')+1]=='FORMAL'
        else:
            assert request['capture_version']==plan['evaluation_and_controller_commit']
            assert request['env']['CUDA_VISIBLE_DEVICES']==''
        result[name]={k:row[k] for k in ('status','exit_code','started_at','ended_at','cleanup_verified','request_sha256')}
        result[name]['workgroup_evidence']=verify
    for one,two in zip(names[1:],names[2:]):
        assert datetime.fromisoformat(result[one]['ended_at'])<=datetime.fromisoformat(result[two]['started_at'])
    return result


def mode_audit(a, plan, mode, data, baseline, identity_output):
    audit,tokens,catalog,slots,nrank,assignment,cpus=data
    expected=plan['modes'][mode]['gpu']['request']
    folder,rank,stage,receipt=check_rank(a,mode,expected,nrank,plan['card'],assignment)
    out=Counter()
    movements=[0]*4
    stream=hashlib.sha256()
    relative_stream=hashlib.sha256()
    global_inputs=[None]*USERS
    identities=[]
    freq=Counter()
    token_set=set(tokens)
    for idx,row in zip(assignment,raw_rows(folder/'predictions.rank0.jsonl',receipt['sha256'],receipt['bytes']),strict=True):
        cpu=cpus[idx]
        try:
            ids,moved,changed,identity=verify_row(row,cpu,mode,token_set)
            assert (row['execution_topology'],row['execution_profile'])==(rank['execution_topology'],rank['execution_profile'])
            invalid,duplicate=verify_candidates(row,tokens,slots,catalog)
        except (AssertionError,KeyError,TypeError,ValueError,IndexError) as exc:
            raise ValueError(f'{mode}: raw global row_idx {idx}: {exc}') from exc
        freq.update(ids)
        out.update(users=1,history_groups=len(ids),changed_history_groups=changed,invalid=invalid,duplicate=duplicate,
                   users_invalid=bool(invalid),users_duplicate=bool(duplicate))
        for i,n in enumerate(moved): movements[i]+=n
        relative_stream.update(canonical_bytes([idx,ids])+b'\n')
        line=canonical_bytes([idx,row['input_token_ids']])+b'\n'
        stream.update(line)
        global_inputs[idx]=line
        if identity:
            check=compare_identity(mode,row,baseline[idx])
            identities.append(check)
            identity_output.write(json.dumps(check,sort_keys=True,allow_nan=False)+'\n')
    assert out['users']==USERS and out['history_groups']==159966
    counts=[freq[i] for i in range(len(CATA[len(MODES[mode])]))]
    frozen=audit['modes'][mode]
    assert counts==frozen['permutation_counts'] and movements==frozen['moved_ids_by_slot']
    assert out['history_groups']-out['changed_history_groups']==frozen['identity_groups']
    assert len(identities)==frozen['identity_users']
    # Expected audit input digest is in global row order, unlike raw sampler order.
    full_stream=hashlib.sha256()
    for line in global_inputs:
        full_stream.update(line)
    assert full_stream.hexdigest()==frozen['inputs_sha256']
    anomalies=stage['anomalies']
    assert anomalies['candidate_count']==USERS*BEAMS
    assert out['invalid']==anomalies['invalid_complete_sid_candidates']
    assert out['duplicate']==anomalies['duplicate_candidate_occurrences']
    assert out['users_invalid']==anomalies['users_with_invalid_sid']
    assert out['users_duplicate']==anomalies['users_with_duplicate_sid']
    assert anomalies['targets_with_colliding_sid']==anomalies['colliding_sid_candidate_occurrences']==0
    eval_path=(a.eval_root/mode/'summary.json' if a.eval_root
               else a.run_root/f'cpu-formal-eval-{mode}-r0/summary.json' if a.root is None else None)
    if eval_path:
        eval_report=read(eval_path)
        assert eval_report['status']=='PASS_G3_V2_EVALUATION' and eval_report['scope']=='FORMAL'
        assert (eval_report['prediction_sha256'],eval_report['prediction_bytes'],eval_report['users'])==(
                 receipt['sha256'],receipt['bytes'],USERS)
        assert eval_report['candidate_count']==USERS*BEAMS
    all_exact=sum(x['all_candidates_exact'] for x in identities)
    return {'prediction_sha256':receipt['sha256'],'prediction_bytes':receipt['bytes'],'rows':out['users'],
            'candidates':USERS*BEAMS,'rank_report_sha256':digest_file(folder/'rank0.json'),
            'gpu_stage_sha256':digest_file(folder/'metrics.json'),'raw_assignment_sha256':ASSIGNMENT_SHA,
            'gpu_uuid':plan['card'],'history_groups':out['history_groups'],'num_permutations':len(counts),
            'permutation_counts':counts,'moved_ids_by_slot':movements,'natural_identity_rows':len(identities),
            'natural_rows_candidate_content_exact':sum(x['candidate_content_exact'] for x in identities),
            'natural_rows_scores_exact':sum(x['scores_exact'] for x in identities),
            'natural_rows_all_candidates_exact':all_exact,'natural_rows_sorted_order_exact':sum(x['sorted_order_exact'] for x in identities),
            'natural_rows_max_abs_score_delta_vs_normal':max((x['max_abs_score_delta_vs_normal'] for x in identities),default=None),
            'out_of_catalog_candidates':out['invalid'],'duplicate_candidates':out['duplicate'],
            'observed_sampler_order_decoded_input_stream_sha256':stream.hexdigest(),
            'independently_verified_global_order_decoded_input_stream_sha256':full_stream.hexdigest(),
            'relative_permutation_stream_sha256':relative_stream.hexdigest(),
            'rank_model_unchanged_guard_pass':True,'all_actual_saved_prefill_witnesses_pass':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    scheme=parser.add_mutually_exclusive_group()
    scheme.add_argument('--root',type=Path,help='Extracted ZIP root using predictions/, inputs/, data/.')
    scheme.add_argument('--run-root',type=Path,help='Original server result root. Default is G3 Beauty V2.')
    for name in ('data-root','cpu-audit','cpu-manifest','normal-predictions','normal-rank','plan','eval-root','derived'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--runner',type=Path,help='Optional live read-only runner query of the ten formal terminal records.')
    parser.add_argument('--job-root',type=Path)
    args=parser.parse_args()
    assert not os.environ.get('CUDA_VISIBLE_DEVICES'), 'Audit must run CPU-only'
    if args.root:
        default_data=args.root/'data'
        default_cpu=args.root/'inputs/cpu'
        default_normal=args.root/'predictions/normal'
        assert args.plan is not None, 'Pass the packaged formal plan path.'
    else:
        args.run_root=args.run_root or SERVER_RUN_ROOT
        default_data=SERVER_DATA_ROOT
        default_cpu=args.run_root/'cpu-replay-r2'
        default_normal=SERVER_NORMAL
        args.plan=args.plan or SERVER_PLAN
    args.data_root=args.data_root or default_data
    args.cpu_audit=args.cpu_audit or default_cpu/'audit.json'
    args.cpu_manifest=args.cpu_manifest or default_cpu/'inputs.jsonl'
    args.normal_predictions=args.normal_predictions or default_normal/'predictions.rank0.jsonl'
    args.normal_rank=args.normal_rank or default_normal/'rank0.json'
    start=time.monotonic()
    plan=read(args.plan,PLAN_SHA)
    assert list(plan['modes'])==list(MODES)
    data=load_frozen(args)
    terminal=terminal_readback(args,plan)
    normal=frozen_normal(args,data[5],data[6])
    args.output.mkdir(parents=True,exist_ok=False)
    measured={}
    with (args.output/'natural-identity.jsonl').open('x') as fixture:
        for mode in MODES:
            measured[mode]=mode_audit(args,plan,mode,data,normal,fixture)
            print(json.dumps({'mode':mode,'rows':measured[mode]['rows'],'elapsed':time.monotonic()-start}),flush=True)
    assert measured['shuffle3a']['relative_permutation_stream_sha256']==measured['shuffle3b']['relative_permutation_stream_sha256']
    result={'status':'PASS_NON_AUTHOR_FULL_FORMAL_GENERATION','checked_wall_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds':time.monotonic()-start,'GPU_RUN_COMMIT':GPU_COMMIT,'MODEL_TRAINING_COMMIT':TRAIN_COMMIT,
            'cpu_audit_sha256':CPU_AUDIT_SHA,'cpu_manifest_sha256':CPU_MANIFEST_SHA,
            'complete_predictions_checked':sum(m['rows'] for m in measured.values()),
            'complete_candidates_checked':sum(m['candidates'] for m in measured.values()),
            'independent_occurrence_checks':sum(m['history_groups'] for m in measured.values()),
            'S3a_S3b_relative_sharing_exact':True,'historical_normal_source_sha256':NORMAL_SHA,
            'historical_normal_raw_inputs_checked':USERS,
            'each_formal_input_matched_cpu_oracle':True,'modes':measured,'terminal_readback':terminal,
            'natural_identity_jsonl_sha256':digest_file(args.output/'natural-identity.jsonl'),
            'scope':'independent verification of saved raw outputs, runtime witnesses and optional real terminal state; CPU only'}
    with (args.output/'summary.json').open('x') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
