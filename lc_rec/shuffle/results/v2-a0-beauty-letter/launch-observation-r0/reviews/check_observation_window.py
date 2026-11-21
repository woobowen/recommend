"""Check the captured seven-point Beauty formal window without contacting the server."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median
import sys
from zoneinfo import ZoneInfo

source = Path(sys.argv[1])
review = Path(sys.argv[2])
destination = Path(sys.argv[3]) if len(sys.argv) > 3 else None
read = lambda path: json.loads(path.read_bytes())
ref = '1182dfd914c20bd3dc11786587eae6879fba6b9d'
expected_assignment = '22ee58fe9f948f50e5cbc92160b6dc7df7e2f633cb9e8bf3121da5fb35bcb088'
expected_gpu = 'GPU-59cac3bd-3645-668b-79c8-839e0d878893'
parent = 'letter-beauty-v2-group-series-r0'
child = 'letter-beauty-v2-gpu-formal-shuffle2-r0'
max_users = 22363
err = re.compile(r'out.of.memory|traceback \(most recent call last\)|childfailederror|failed_gate_or_child|cuda error|assertionerror', re.I)
launch_review = read(review / 'launch-nonauthor.json')
sample_review = read(review / 'first-samples-nonauthor.json')
assert launch_review['status'] == 'PASS_NONAUTHOR_FORMAL_LAUNCH'
assert sample_review['status'] == 'PASS_NONAUTHOR_FORMAL_FIRST_SAMPLES'
assert sample_review['assigned_users'] == max_users and sample_review['gpu_uuid'] == expected_gpu
assert sample_review['formal_assignment_sha256'] == expected_assignment
assert sample_review['sample_count'] == 6 and sample_review['records'][0]['row_idx'] == 8755
assert all(row['model_boundary_proof']['beam_expansion'] == 20 for row in sample_review['records'])
pworker = launch_review['receipts']['parent']['worker']
cworker = launch_review['receipts']['child']['worker']
t0 = read(source / 't00.json')
confirmation = next(r for r in map(json.loads, (source / 'launch-confirmations.jsonl').read_text().splitlines())
                    if r['observed_utc'] == t0['observed_utc'])
assert confirmation['jobs'] == t0['jobs']
assert t0['rank_reports'][child] == confirmation['jobs'][child]['progress']
assert not t0['outputs']
estimates = []
observations = []
prior_t = None
prior_bytes = prior_lines = prior_end = 0
start = datetime.fromisoformat(launch_review['receipts']['child']['started_at'])
for minute in range(0,31,5):
    path=source/f't{minute:02}.json'
    obs=read(path)
    observed=datetime.fromisoformat(obs['observed_utc'])
    assert obs['parent_id']==parent
    s=obs['jobs']
    assert s[parent]['status']==s[child]['status']=='RUNNING'
    assert s[parent]['worker']==pworker and s[child]['worker']==cworker
    assert s[parent]['exit_code'] is None and s[child]['exit_code'] is None
    if minute:
        assert s[parent]['supervisor_alive'] and s[child]['supervisor_alive']
        assert not obs['unexpected_active_formal']
        assert obs['controller_summary']['status']=='RUNNING'
        assert obs['controller_summary']['formal_run_commit']==ref
        assert [v['job_id'] for v in obs['controller_summary']['stages']]==[child]
        assert obs['disk_free_bytes'] > 20 * 1024**3
    else:
        assert s[parent]['worker_pid_identity_alive'] and s[child]['worker_pid_identity_alive']
        assert '/data' in obs['disk_df']
    lines=rank_users=byte_size=end=0
    rank=obs['rank_reports'][child]
    assert rank['status']=='RUNNING' and rank['error'] is None
    devices=[gpu for gpu in obs['gpu_compute'] if gpu['owned_formal_job_id']]
    assert len(devices)==1
    dev=devices[0]
    assert (dev['uuid'], dev['owned_formal_job_id'], dev['pid'])==(expected_gpu,child,rank['pid'])
    gpu=next(g for g in obs['gpu_records'] if g['uuid']==expected_gpu)
    assert gpu['physical_index']==1 and 0<=gpu['utilization_percent']<=100
    assert 0<gpu['memory_used_mib']<=gpu['memory_total_mib']
    if minute:
        assert rank['RUN_COMMIT']==ref and rank['assignment_sha256']==expected_assignment
        assert (rank['rank'],rank['local_rank'],rank['world_size'],rank['assigned_users'])==(0,0,1,max_users)
        rank_users=rank['completed_users']
        active_rank=next(m for m in s[child]['live_members'] if m['identity']['pid']==rank['pid'])
        assert active_rank['identity']['start_ticks'] and active_rank['identity']['boot_id']==obs['boot_id']
        env=active_rank['effective_env']
        assert (env['CUDA_VISIBLE_DEVICES'],env['WORLD_SIZE'],env['RANK'],env['LOCAL_RANK'])==(expected_gpu,'1','0','0')
        assert env['CUDA_HOME'] and env['LD_LIBRARY_PATH'] and env['TORCHINDUCTOR_COMPILE_THREADS']=='1'
        assert s[child]['request_sha256']==sample_review['child_request_canonical_sha256']
        assert s[parent]['request_sha256']==sample_review['parent_request_canonical_sha256']
        o=obs['outputs'][child]
        assert o['source']==sample_review['source']
        inode=o['inode']
        if minute>5: assert inode==prior_inode
        prior_inode=inode
        lines=o['complete_lines']; byte_size=o['snapshot_size']; end=o['complete_end_byte']
        assert o['trailing_bytes_outside_complete_rows']==byte_size-end
        assert o['counted_from_byte']==prior_end
        assert max_users>=lines>=rank_users>0 and lines>prior_lines
        assert byte_size>prior_bytes and end>prior_end
        assert o['candidate_count_by_enforced_20_per_row']==lines*20
        for entry in obs['log_tails'].values():
            content='\n'.join(entry) if isinstance(entry,list) else json.dumps(entry)
            assert err.search(content) is None
    else:
        assert rank['completed_users']==0
    spend=obs['cumulative_budget_spend']
    assert abs(spend['formal_device_seconds']-(observed-start).total_seconds())<=2
    assert abs(spend['preflight_device_seconds']-648.991733)<1e-5
    assert abs(spend['cpu_worker_seconds']-571.62509)<1e-5
    interval=(observed-prior_t).total_seconds() if prior_t else None
    if interval:
        assert 240 <= interval <= 380
    if minute>5:
        estimates.append((lines-prior_lines)/interval)
    observations.append({'point':f'T+{minute}' if minute else 'T0','observed_utc':observed.isoformat(),
                         'observed_new_york':observed.astimezone(ZoneInfo('America/New_York')).isoformat(),
                         'interval_seconds':interval,'rank_completed_users':rank_users,'complete_prediction_lines':lines,
                         'complete_candidates_by_enforced_contract':20*lines,'new_prediction_lines':lines-prior_lines,
                         'prediction_snapshot_size':byte_size,'new_prediction_bytes':byte_size-prior_bytes,
                         'gpu_used_mib':gpu['memory_used_mib'],'gpu_utilization_percent':gpu['utilization_percent'],
                         'child_formal_device_seconds':spend['formal_device_seconds'],'source_sha256':sha256(path.read_bytes()).hexdigest()})
    prior_t,prior_bytes,prior_lines,prior_end=observed,byte_size,lines,end
span=(prior_t-datetime.fromisoformat(observations[0]['observed_utc'])).total_seconds()
assert span>=1800
assert datetime.fromisoformat(observations[0]['observed_utc']) < datetime.fromisoformat(sample_review['observed_utc']) < prior_t
steady=median(estimates)
receipt={'status':'PASS_NONAUTHOR_FORMAL_OBSERVATION_WINDOW','reviewer':'/root/formal_review',
         'scope':'Seven saved live observations, original launch confirmation, independent launch and first-six sample readbacks. Full-condition metrics remain pending.',
         'parent_id':parent,'child_id':child,'run_commit':ref,'gpu_uuid':expected_gpu,
         'monitoring_seconds':span,'samples':observations,
         'last_complete_users':prior_lines,'last_complete_candidates':prior_lines*20,
         'last_remaining_s2_users':max_users-prior_lines,
         'observed_median_post_load_users_per_second':steady,
         's2_remaining_wall_projection_seconds':(max_users-prior_lines)/steady}
if destination:
    with destination.open('x') as out:
        json.dump(receipt,out,indent=2)
        out.write('\n')
print(json.dumps(receipt,indent=2))
