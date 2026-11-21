
from pathlib import Path
import sys, hashlib, itertools, json, math, os, subprocess, datetime

h = lambda value: hashlib.sha256(json.dumps(value, separators=(',',':'), ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
bsha = lambda raw: hashlib.sha256(raw).hexdigest()
read = lambda p: json.loads(p.read_bytes())
ref = '1182dfd914c20bd3dc11786587eae6879fba6b9d'
runner_hash = 'cbb963fab68c34cc977653afc46f260e7ede962cb4a8f105458331a0e0e19a75'
target = 'letter-beauty-v2-group-series-' + sys.argv[1]
child = 'letter-beauty-v2-gpu-formal-shuffle2-' + sys.argv[1]
base = Path('/home/bwwu/grid-audits/sid-extensions-p0')
root = Path('/data/bwwu/lc-rec/letter-shuffle-v2-beauty')
parent_dir = base/'jobs'/target
child_dir = base/'jobs'/child
parent_req = read(parent_dir/'request.json')
child_req = read(child_dir/'request.json')
assert parent_req['capture_version'] == child_req['capture_version'] == ref
assert parent_req['argv'][3] == 'lc_rec.shuffle_v2.series'
assert child_req['argv'][3] == 'lc_rec.shuffle_v2.generate'
assert '--preflight-users-per-rank' not in child_req['argv']
assert parent_req['env']['CUDA_VISIBLE_DEVICES'] == ''
assert child_req['env']['CUDA_VISIBLE_DEVICES'] in {
  'GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae','GPU-59cac3bd-3645-668b-79c8-839e0d878893'}
assert child_req['env']['TORCHINDUCTOR_COMPILE_THREADS'] == '1'
plan_path = Path(parent_req['argv'][parent_req['argv'].index('--plan')+1])
plan = read(plan_path)
assert any(v['path']==str(plan_path) and v['sha256']==bsha(plan_path.read_bytes()) for v in parent_req['inputs'])
assert list(plan['modes']) == ['shuffle2','shuffle3a','shuffle3b','shuffle4']
assert plan['modes']['shuffle2']['gpu'] == child_req
assert plan['runner_sha256'] == bsha(Path(plan['runner']).read_bytes()) == runner_hash
for folder, req in [(parent_dir,parent_req),(child_dir,child_req)]:
    accepted = read(folder/'accepted.json')
    assert h(req) == accepted['request_sha256']
    assert accepted['runner_sha256'] == runner_hash

rank_dir = root/('gpu-formal-shuffle2-'+sys.argv[1])/'test'
config_path=Path(child_req['argv'][child_req['argv'].index('--config')+1])
cfg=read(config_path)
assert all(child_req['env'][k] == v for k,v in cfg['gpu_environment'].items())
release = Path(child_req['cwd'])
manifest = read(release/'RUN_VERSION.json')
assert manifest['RUN_COMMIT'] == ref
relative = {v['path']:v for v in manifest['files']}
part_names = ['generate.py','series.py','runtime.py','beauty.py','beauty_evaluate.py','statistics.py','a0_beauty_letter.json']
for filename in part_names:
    p = 'lc_rec/shuffle_v2/'+filename
    assert bsha((release/p).read_bytes()) == relative[p]['sha256']

rank=read(rank_dir/'rank0.json')
binding=read(rank_dir/'v2-binding.rank0.json')
assert rank['rank']==rank['local_rank']==0 and rank['world_size']==1 and rank['assigned_users']==22363
assert rank['RUN_COMMIT'] == binding['V2_RUN_COMMIT'] == ref
assert not rank.get('preflight_only',False) and not rank['shuffle']['capture_lookup']
assert {'--preflight-users-per-rank','--preflight-manifest'}.isdisjoint(binding['effective_argv'])
assert rank['cuda_environment']['actual_rank_environment'] == cfg['gpu_environment']
assert rank['gpu']['visible_devices'] == child_req['env']['CUDA_VISIBLE_DEVICES']
assert rank['allowed_callback_sha256'] == cfg['baseline']['allowed_callback_sha256']
assert rank['generation']['per_rank_batch_size']==1 and rank['generation']['prompt_id']==0
assert rank['strict_adapter_reload']['keys'] == 450
assert rank['adapter_files']['adapter_model.bin']['sha256'] == 'dd9f3d2983af7ce85ace99534bd23df9362d9cd91ac39d0b356b219cb790b1c1'
assert rank['tokenizer']['vocab_size']==32965 and len(rank['tokenizer']['added_tokens'])==965
assert rank['data_identity']==read(Path(cfg['baseline']['normal_test_dir'])/'rank0.json')['data_identity']
expected_config={'num_beams':20, 'num_return_sequences':20, 'max_new_tokens':10,'early_stopping':True,
                 'do_sample':False,'use_cache':True,'eos_token_id':2,'pad_token_id':0}
cfg_seen=rank['shuffle']['effective_generation_config']
assert all(cfg_seen[k]==v for k,v in expected_config.items())
assert bsha((rank_dir/'derived.rank0.py').read_bytes()) == binding['derived_sha256']

def prefix_records(path, count):
    out=[]
    with path.open('rb') as stream:
        for _ in range(count):
            start=stream.tell()
            raw=stream.readline()
            if not raw.endswith(b'\n'):
                break
            out.append((json.loads(raw), start, stream.tell(), bsha(raw)))
    return out
pred_path=rank_dir/'predictions.rank0.jsonl'
n=6
pred=prefix_records(pred_path,n)
assert len(pred)==n
original_dir=Path(cfg['baseline']['normal_test_dir'])
originals = [prefix_records(original_dir/f'predictions.rank{i}.jsonl',(n+1)//2) for i in (0,1)]
auditdir = root/'cpu-replay-r3'
audit=read(auditdir/'audit.json')
single=read(auditdir/'single-shard.json')
assert rank['assignment_sha256'] == single['assignment_sha256']
assert single['users'] == cfg['users'] == 22363
assert audit['manifest_sha256'] == single['manifest_sha256'] == rank['shuffle']['cpu_manifest_sha256']
catalog=read(Path(cfg['baseline']['data-root'])/'Beauty'/'Beauty.index.json')
inter=read(Path(cfg['baseline']['data-root'])/'Beauty'/'Beauty.inter.json')
lookup=audit['token_ids']
valid=[{lookup[sid[i]]:sid[i] for sid in catalog.values()} for i in range(4)]
vocab=set(lookup.values())
rows={p[0]['row_idx'] for p in pred}
replay={}
with (auditdir/'inputs.jsonl').open('rb') as stream:
    for i,line in enumerate(stream):
        if i in rows:
            replay[i]=json.loads(line)
            assert i == replay[i]['row_idx']
        if len(replay)==n:
            break

records=[]
for sequence,(row,start,end,sha) in enumerate(pred):
    old,_,_,old_sha=originals[sequence%2][sequence//2]
    j=row['row_idx']
    uid=row['user_id']
    target=str(inter[uid][-1])
    it=inter[uid][:-1][-20:]
    expected=replay[j]
    proof=row['shuffle']
    assert (row['row_idx'],uid,row['target'])==(old['row_idx'],old['user_id'],old['target'])
    assert row['target_item_id']==target and row['target']==''.join(catalog[target])
    assert row['prompt_id']==row['rank']==0
    assert proof['protocol']=='GRID-LETTER-SHUFFLE-V2' and proof['identity_policy']=='allow_identity'
    assert proof['mode']=='shuffle2' and proof['seed']==42 and proof['active_positions']==[2,3]
    assert proof['history_group_count']==expected['visible_history_length']==len(it) and proof['row_idx']==j
    ids=old['input_token_ids']
    offsets=[k for k,v in enumerate(ids) if v in vocab]
    assert len(offsets)==len(it)*4 and len(ids)==row['input_nonpadding_tokens']==expected['encoded_length']
    edited=ids.copy()
    choices=[]
    for occurrence,item in enumerate(it):
        group=offsets[occurrence*4:(occurrence+1)*4]
        assert group==list(range(group[0],group[0]+4))
        assert [ids[g] for g in group]==[lookup[v] for v in catalog[str(item)]]
        counter=0
        # Both S2 catalog members make 2 divide 2^256 exactly.
        choice=int.from_bytes(hashlib.sha256(json.dumps(['GRID-LETTER-SHUFFLE-V2',42,2,j,occurrence,counter],separators=(',',':')).encode()).digest(),'big')%2
        choices.append(choice)
        if choice:
            a,b=group[2:4]
            edited[a],edited[b]=edited[b],edited[a]
    assert choices==expected['conditions']['shuffle2']['permutation_ids']
    assert edited==row['input_token_ids']
    assert proof['input_changed']==expected['conditions']['shuffle2']['input_changed']==(edited!=ids)
    assert proof['original_sha256']==expected['original_sha256']==h(ids)
    assert proof['shuffled_sha256']==expected['conditions']['shuffle2']['input_sha256']==h(edited)
    assert proof['cpu_manifest_sha256']==audit['manifest_sha256']
    prefill=proof['actual_prefill']
    assert prefill['input_sha256']==h(edited)
    assert prefill['attention_mask_sha256']==proof['attention_mask_sha256']==expected['attention_mask_sha256']==h([1]*len(edited))
    assert prefill['position_ids_sha256']==h(list(range(len(edited))))
    assert prefill['cache_length']==0 and prefill['beam_expansion']==20
    assert row['original_callback_verified_all_tokens']
    candidates=row['candidates']
    assert len(candidates)==20 and [c['generation_order'] for c in candidates]==list(range(20))
    for cand in candidates:
        tok=cand['generated_token_ids']
        assert cand['eos_position']==4 and tok[4]==2 and set(tok[5:])<={0,2}
        assert cand['sid']==''.join(valid[i][tok[i]] for i in range(4))
        assert cand['sid']==cand['raw_decoded'].split('Response:')[-1].strip().replace(' ','')
        assert math.isfinite(cand['sequence_score'])
    ranks = sorted(range(20), key=lambda i:candidates[i]['sequence_score'],reverse=True)
    assert ranks==row['sorted_generation_order']
    assert [int(candidates[i]['sid']==row['target']) for i in ranks]==row['relevance']
    records.append({'sequence_one_based':sequence+1,'row_idx':j,'file_byte_range_half_open':[start,end],
                    'raw_record_sha256':sha,'archived_normal_record_sha256':old_sha,
                    'permutation_ids':choices,'input_token_count':len(ids),'input_changed':edited!=ids,
                    'changed_positions':sum(a!=b for a,b in zip(edited,ids)),
                    'candidate_count':len(candidates),'model_boundary_proof':prefill})
print(json.dumps({'status':'PASS_NONAUTHOR_FORMAL_FIRST_SAMPLES','reviewer':'/root/formal_review',
                  'scope':'First six complete formal S2 lines only; independently reconstructed SHA-256 permutation from frozen Normal prompt, Beauty history and token table; no full-condition metrics.',
                  'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'parent_job':target,'child_job':child,'run_commit':ref,'runner_sha256':runner_hash,
                  'parent_request_canonical_sha256':h(parent_req),'child_request_canonical_sha256':h(child_req),
                  'remote_plan_sha256':bsha(plan_path.read_bytes()),'formal_assignment_sha256':rank['assignment_sha256'],
                  'assigned_users':rank['assigned_users'],'mode':'shuffle2','world_size':1,'local_rank':0,
                  'gpu_uuid':rank['gpu']['visible_devices'],'effective_generation_settings':expected_config,
                  'source':str(pred_path),'sample_count':len(records),'records':records},indent=2))

