"""Independent read-only checks against terminal server files, no model imports."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import permutations
from pathlib import Path
import json
import math
import re
import subprocess
import time

started = time.monotonic()
run_version = 'e2f54ebd2d7709821de1071693f91644f5ed9a30'
controller_version = '4f5758ffb578f3a8d3d06a54262f59d3f63db370'
base = Path('/home/bwwu/grid-audits/sid-extensions-p0')
root = Path('/data/bwwu/lc-rec/letter-shuffle-v2-20261120')
runner = base/'scripts/runner-a1-96h-1fefc09a15c1cb73.py'
runner_digest = '1fefc09a15c1cb73878417f9eb50d0fe9e8fa1b2f60cc70b27f7cc4ed9cb0947'
prefix = 'letter-a0-20261112-v2-20261120-'
protocol = 'GRID-LETTER-SHUFFLE-V2'
modes = {'shuffle2':(2,3),'shuffle3a':(0,1,2),'shuffle3b':(1,2,3),'shuffle4':(0,1,2,3)}
condition_dirs = base/'releases/letter-shuffle-v2'


def load(path):
    return json.loads(path.read_bytes())


def sha_file(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value, ascii=True):
    return json.dumps(value, ensure_ascii=ascii, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def signature(value, ascii=True):
    return sha256(canonical(value, ascii=ascii)).hexdigest()


config = load(condition_dirs/run_version/'lc_rec/shuffle_v2/a0_instruments.json')
assert sha_file(runner) == runner_digest
release_digests = {}
for release in (run_version, controller_version):
    folder = condition_dirs/release
    manifest = load(folder/'RUN_VERSION.json')
    assert manifest['RUN_COMMIT'] == release and manifest['release'] == str(folder)
    assert manifest['runner_sha256'] == runner_digest
    d = {entry['path']:sha_file(folder/entry['path']) for entry in manifest['files']}
    assert d == {entry['path']:entry['sha256'] for entry in manifest['files']}
    release_digests[release] = d

job_root = base/'jobs'
job_names = {p.name.removeprefix(prefix) for p in job_root.glob(prefix+'*') if (p/'status.json').is_file()}
expected_names = {'gpu-preflight-r0','group-series-r0','cpu-boundary-fixture-r0','cpu-eval-normal-r0',
                  'cpu-evaluation-fixture-r0','cpu-fixtures-r0','cpu-replay-r0','cpu-replay-r1',
                  'cpu-single-shard-r1','cpu-summary-r0'}
expected_names.update('gpu-formal-'+mode+'-r0' for mode in modes)
expected_names.update('cpu-eval-'+mode+'-r0' for mode in modes)
assert job_names == expected_names, (job_names, expected_names)
requests, statuses = {}, {}
for name in sorted(job_names):
    folder = job_root/(prefix+name)
    request = load(folder/'request.json')
    accepted = load(folder/'accepted.json')
    stored = load(folder/'status.json')
    seen = json.loads(subprocess.check_output(['/usr/bin/python3','-I','-S','-B',str(runner),'query',str(folder)], timeout=30))
    cleanup = load(folder/'cleanup.json')
    assert stored['job_id'] == request['job_id'] == seen['job_id'] == prefix+name
    assert signature(request, False) == accepted['request_sha256'] == seen['request_sha256']
    assert stored['status'] == seen['status'] and stored['ended_at'] == seen['ended_at']
    expected_status = 'WORKERS_TERMINATED' if name == 'cpu-replay-r0' else 'SUCCEEDED'
    assert stored['status'] == expected_status and stored['exit_code'] == seen['exit_code'] == 0
    assert cleanup['verified'] and seen['cleanup_verified'] and cleanup['last_observation']['verified_empty']
    assert not seen['worker_alive'] and not seen['workgroup_live'] and not seen['supervisor_alive']
    assert not seen['workgroup_observation']['owned_live']
    wall = (datetime.fromisoformat(stored['ended_at'])-datetime.fromisoformat(stored['started_at'])).total_seconds()
    assert wall > 0
    requests[name] = request
    statuses[name] = {'state':stored['status'],'exit_code':stored['exit_code'],
                      'start':stored['started_at'],'end':stored['ended_at'],'wall_seconds':wall,
                      'timeout_seconds':request['timeout_seconds'],'commit':request['capture_version'],
                      'request_sha256':seen['request_sha256'],'cleanup_reason':cleanup['reason'],
                      'cleanup_events':cleanup['events'],'fresh_query_no_workers':True}

series = load(root/'group-series-r0/summary.json')
assert series['status'] == 'PASS_V2_GROUP_COMPUTE' and series['formal_run_commit'] == run_version
plan = load(base/'requests'/(prefix+'group-series-r0.plan.json'))
assert plan['formal_run_commit'] == run_version
expected_order = [name for mode in modes for name in (f'gpu-formal-{mode}-r0', f'cpu-eval-{mode}-r0')] + ['cpu-summary-r0']
assert [s['job_id'] for s in series['stages']] == [prefix+n for n in expected_order]
assert all(statuses[a]['end'] < statuses[b]['start'] for a,b in zip(expected_order,expected_order[1:]))
assert requests['group-series-r0']['capture_version'] == controller_version
assert requests['cpu-summary-r0'] == plan['paired']
assert all(requests[f'gpu-formal-{m}-r0'] == plan['modes'][m]['gpu'] and requests[f'cpu-eval-{m}-r0'] == plan['modes'][m]['cpu'] for m in modes)
assert all(requests[k]['capture_version'] == run_version for k in expected_order)
assert all(not requests[k]['env']['CUDA_VISIBLE_DEVICES'] for k in requests if k.startswith('cpu-'))

audit_root = root/'cpu-replay-r1'
audit = load(audit_root/'audit.json')
shard = load(root/'cpu-single-shard-r1/audit.json')
assert audit['status'] == 'PASS_CPU_V2_FULL' and audit['protocol'] == protocol
assert shard['status'] == 'PASS_SINGLE_SHARD_CPU'
assert sha_file(audit_root/'inputs.jsonl') == audit['manifest_sha256'] == shard['manifest_sha256']
manifest = {r['row_idx']:r for r in map(json.loads,(audit_root/'inputs.jsonl').open())}
assert len(manifest) == 24772
assets = {p['path']:p['sha256'] for p in config['files']}
data_path = Path(config['baseline']['data-root'])/'Instruments'
dataset = {}
for key in ('index','inter','item'):
    p = data_path/f'Instruments.{key}.json'
    assert sha_file(p) == assets[str(p)]
    dataset[key] = load(p)
index = dataset['index']
assert set(index) == set(dataset['item'])
assert len(index) == 9922 and len(dataset['inter']) == 24772
uid_axis = list(dataset['inter'])
assert signature(uid_axis,False) == audit['user_axis_sha256']
sets = [set(audit['token_ids'][code[pos]] for code in index.values()) for pos in range(4)]
vocab = set.union(*sets)
rev_vocab = {value:key for key,value in audit['token_ids'].items()}
known_candidates = Counter(''.join(v) for v in index.values())
assert len(known_candidates) == 9897 and sum(1 for n in known_candidates.values() if n > 1) == 23


def check_row(row):
    pos = row['row_idx']
    uid = uid_axis[pos]
    all_items = dataset['inter'][uid]
    target_item = str(all_items[-1])
    assert row['user_id'] == uid and row['target_item_id'] == target_item and row['target'] == ''.join(index[target_item])
    assert row['prompt_id'] == 0 and row['original_callback_verified_all_tokens'] is True
    beans = row['candidates']
    assert len(beans) == 20 and [t['generation_order'] for t in beans] == list(range(20))
    scores, sids = [], []
    for candidate in beans:
        tokens = candidate['generated_token_ids']
        assert len(tokens) >= 5 and tokens[4] == 2 and all(v in (0,2) for v in tokens[5:])
        assert candidate['eos_position'] == 4
        assert all(v in allowed for v,allowed in zip(tokens[:4],sets,strict=True))
        sid = ''.join(rev_vocab[x] for x in tokens[:4])
        assert candidate['sid'] == sid
        assert math.isfinite(candidate['sequence_score'])
        scores.append(candidate['sequence_score']); sids.append(sid)
    order = sorted(range(20), key=lambda i:-scores[i])
    assert row['sorted_generation_order'] == order
    assert row['relevance'] == [int(sids[i] == row['target']) for i in order]
    invalid = len([v for v in sids if v not in known_candidates])
    duplicated = len(sids) - len(set(sids))
    colliding = len([v for v in sids if known_candidates[v] > 1])
    return invalid,duplicated,colliding


normal, normal_order, normal_digests = {}, [], []
normal_pred = Path(config['baseline']['normal_test_dir'])
for rank in range(2):
    pred = normal_pred/f'predictions.rank{rank}.jsonl'
    content = sha256()
    order = []
    with pred.open('rb') as stream:
        for line in stream:
            content.update(line)
            row = json.loads(line)
            check_row(row)
            row_idx = row['row_idx']
            assert row['rank'] == rank and row_idx not in normal
            ids = row['input_token_ids']
            assert signature(ids) == manifest[row_idx]['original_sha256']
            history = dataset['inter'][row['user_id']][:-1][-20:]
            assert len(history) == manifest[row_idx]['visible_history_length']
            expected_tokens = [[audit['token_ids'][s] for s in index[str(i)]] for i in history]
            occurrences = [offset for offset,value in enumerate(ids) if value in vocab]
            assert len(occurrences) == len(history)*4
            starts = occurrences[::4]
            assert occurrences == [start+j for start in starts for j in range(4)]
            assert [ids[start:start+4] for start in starts] == expected_tokens
            normal[row_idx] = (ids,starts)
            order.append(row_idx)
    assert len(order) == 12386
    assert content.hexdigest() == assets[str(pred)]
    normal_digests.append(content.hexdigest())
    normal_order.append(order)
assert set(normal) == set(range(24772))
axis = [x for pair in zip(*normal_order,strict=True) for x in pair]
assert signature(axis) == shard['assignment_sha256']

formal = {}
catalogs = {m:tuple(permutations(range(m))) for m in (2,3,4)}
for mode, active in modes.items():
    job_name = f'gpu-formal-{mode}-r0'
    test = root/job_name/'test'
    rank = load(test/'rank0.json')
    evaluation = root/f'cpu-eval-{mode}-r0'
    check = load(evaluation/'prediction_checks.json')
    saved = load(evaluation/'prediction_manifest.json')['predictions'][0]
    summary = load(evaluation/'summary.json')
    assert rank['world_size'] == 1 and rank['rank'] == rank['local_rank'] == 0
    assert rank['RUN_COMMIT'] == run_version and rank['status'] == 'PASS'
    assert rank['completed_users'] == rank['assigned_users'] == 24772
    assert rank['gpu']['visible_devices'] == 'GPU-59cac3bd-3645-668b-79c8-839e0d878893'
    assert rank['gpu']['visible_devices'] == requests[job_name]['env']['CUDA_VISIBLE_DEVICES']
    assert rank['generation']['num_return_sequences'] == rank['generation']['num_beams'] == 20
    assert rank['generation']['per_rank_batch_size'] == 1 and rank['generation']['filter_items'] is False
    assert rank['adapter_files']['adapter_model.bin'] == config['baseline']['adapter_identity']
    assert rank['generation']['max_new_tokens'] == 10 and rank['generation']['oom_beam_reduction'] is False
    assert rank['assignment_sha256'] == shard['assignment_sha256']
    assert check['status'] == 'PASS_V2_FULL_PREDICTIONS' and summary['status'] == 'PASS_V2_EVALUATION'
    assert summary['prediction_run_commit'] == summary['evaluation_run_commit'] == run_version
    assert check['shards'][0]['derived_entry_sha256'] == sha_file(test/'derived.rank0.py')
    assert check['shards'][0]['rank_report_sha256'] == sha_file(test/'rank0.json')
    assert not (test/'predictions.rank1.jsonl').exists()
    source = test/'predictions.rank0.jsonl'
    assert saved['path'] == str(source) and saved['bytes'] == source.stat().st_size
    hash_raw = sha256()
    counts = [0]*len(catalogs[len(active)])
    moved = [0]*4
    identity_users = 0
    exceptions = [0,0,0]
    n = len(counts)
    cutoff = (2**256//n)*n
    with source.open('rb') as stream:
        for want,line in zip(axis,stream,strict=True):
            hash_raw.update(line)
            row = json.loads(line)
            pos = row['row_idx']
            assert pos == want and row['rank'] == 0
            row_checks = check_row(row)
            for j,value in enumerate(row_checks):
                exceptions[j] += value
            original,starts = normal[pos]
            result = original.copy()
            relative_ids = []
            for occurrence,start in enumerate(starts):
                counter = 0
                while True:
                    hashed = int.from_bytes(sha256(canonical([protocol,42,len(active),pos,occurrence,counter])).digest(),'big')
                    if hashed < cutoff:
                        break
                    counter += 1
                i = hashed % n
                relative_ids.append(i)
                counts[i] += 1
                relative = catalogs[len(active)][i]
                for dest,original_pos in enumerate(relative):
                    q,p=active[dest],active[original_pos]
                    result[start+q] = original[start+p]
                    moved[q] += original[start+q] != original[start+p]
            expect = manifest[pos]['conditions'][mode]
            assert relative_ids == expect['permutation_ids']
            if mode in ('shuffle3a','shuffle3b'):
                assert relative_ids == manifest[pos]['conditions']['shuffle3b' if mode=='shuffle3a' else 'shuffle3a']['permutation_ids']
            assert result == row['input_token_ids'] and len(result) == row['input_nonpadding_tokens']
            changed = result != original
            identity_users += not changed
            inputs = signature(result)
            mask = signature([1]*len(result))
            pos_ids = signature(list(range(len(result))))
            ev = row['shuffle']
            assert ev == {'actual_prefill':{'attention_mask_sha256':mask,'beam_expansion':20,'cache_length':0,'input_sha256':inputs,'position_ids_sha256':pos_ids},
                          'active_positions':list(active), 'attention_mask_sha256':mask,'cpu_manifest_sha256':audit['manifest_sha256'],
                          'history_group_count':len(starts),'identity_policy':'allow_identity','input_changed':changed,'mode':mode,
                          'original_sha256':manifest[pos]['original_sha256'],'protocol':protocol,'row_idx':pos,'seed':42,'shuffled_sha256':inputs}
            assert mask == manifest[pos]['attention_mask_sha256'] and inputs == expect['input_sha256'] and changed == expect['input_changed']
    assert hash_raw.hexdigest() == saved['sha256']
    assert counts == audit['modes'][mode]['permutation_counts'] and moved == audit['modes'][mode]['moved_ids_by_position']
    assert sum(counts) == check['history_groups'] == 172929 and check['rows'] == 24772 and check['candidates'] == 495440
    assert identity_users == audit['modes'][mode]['unchanged_users'] == check['unchanged_users']
    out = load(test/'metrics.json')['anomalies']
    assert exceptions == [out['invalid_complete_sid_candidates'],out['duplicate_candidate_occurrences'],out['colliding_sid_candidate_occurrences']]
    log_path = root/job_name/(mode+'.log')
    suspicious = [line.strip() for line in log_path.read_text(errors='replace').splitlines()
                  if re.search(r'traceback|out of memory|\b(?:error|timed out)\b',line,re.I)]
    formal[mode] = {'raw_sha256':hash_raw.hexdigest(),'raw_bytes':source.stat().st_size,'rows':len(axis),'candidates':495440,
                    'groups':sum(counts),'identity_groups':counts[0],'identity_users':identity_users,'permutation_counts':counts,
                    'moved_ids_by_position':moved, 'candidate_anomalies':exceptions, 'gpu_peak_reserved_bytes':rank['gpu_peak_reserved_bytes'],
                    'gpu_peak_allocated_bytes':rank['gpu_peak_allocated_bytes'],'suspicious_log_lines':suspicious}

statistic = load(root/'cpu-summary-r0/summary.json')
assert statistic['status'] == 'PASS_V2_PAIRED' and statistic['users'] == 24772
assert statistic['user_axis_sha256'] == audit['user_axis_sha256']
formal_gpu = sum(s['wall_seconds'] for name,s in statuses.items() if name.startswith('gpu-formal-'))
preflight_gpu = statuses['gpu-preflight-r0']['wall_seconds']
cpu = sum(s['wall_seconds'] for name,s in statuses.items() if name.startswith('cpu-'))
assert formal_gpu < config['budget']['formal_total_wall_seconds'] == config['budget']['formal_total_device_seconds']
assert all(s['wall_seconds'] < config['budget']['formal_attempt_wall_seconds'] for name,s in statuses.items() if name.startswith('gpu-formal-'))
assert cpu < config['budget']['cpu_total_wall_seconds']
assert preflight_gpu < config['budget']['preflight_device_seconds']
gpus = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
known_pids = {load(job_root/(prefix+name)/'status.json').get('worker',{}).get('pid') for name in statuses}
assert all(int(row.split(',')[1]) not in known_pids for row in gpus.splitlines() if row.strip())
print(json.dumps({'status':'PASS_INDEPENDENT_ORIGINAL_RUN_REVIEW','reviewer':'/root/final_run_review',
                  'completed_utc':datetime.now(timezone.utc).isoformat(),'audit_wall_seconds':time.monotonic()-started,
                  'run':run_version,'controller':controller_version,'normal_version':config['baseline']['run-commit'],
                  'normal_digests':normal_digests,'release_digests':release_digests,'runner_sha256':runner_digest,
                  'statuses':statuses,'formal':formal,'formal_gpu_seconds':formal_gpu,'cpu_seconds':cpu,'preflight_gpu_seconds':preflight_gpu,
                  'original_paired_summary_sha256':sha_file(root/'cpu-summary-r0/summary.json'),
                  'partition_sha256':signature(axis),'cpu_replay_sha256':audit['manifest_sha256'],'gpus_at_review':gpus},indent=2))
