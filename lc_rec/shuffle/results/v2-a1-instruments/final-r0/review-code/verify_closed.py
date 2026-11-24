"""Verify the closed r0 run, accepted request lineage, source snapshots and budget."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

PREFIX = 'a1-instruments-v2-'
OUTPUT = '/data/bwwu/lc-rec/a1-instruments-v2/'
PLAN_SHA = 'f37fbe7ac295480a181daf2fed098810bd5b2d3c9499e19a7eec67d160603c7b'
GPU_COMMIT = 'c487f7e7176bc223445d3deecfab87e0bcc30d1a'
CPU_COMMIT = '3cc1110097c4764ebfd93bb38ee2913c8afb6958'
MODES = ('shuffle2','shuffle3a','shuffle3b','shuffle4')
FAILURE = re.compile(r'Traceback \(most recent|(?:Runtime|Assertion|Value|Timeout|Import|OSError|Memory)Error:|CUDA out of memory|ChildFailedError|I/O error|No space left', re.I)
NY = ZoneInfo('America/New_York')


def read(p):
    return json.loads(p.read_bytes())


def canon(obj):
    return sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def verify(raw):
    evidence = raw/'evidence'
    c = raw/'control'
    plan = read(c/'formal-plan.json')
    approval = read(c/'authorization.json')
    approval_sha = digest(c/'authorization.json')
    assert plan['formal_submitted'] is False and digest(c/'formal-plan.json') == PLAN_SHA
    assert (plan['preflight_gpu_commit'],plan['evaluation_and_controller_commit']) == (GPU_COMMIT,CPU_COMMIT)
    assert approval['plan_sha256'] == PLAN_SHA and approval['gpu_commit'] == GPU_COMMIT
    assert approval['web_review_pass'] and approval['user_formal_launch_authorized'] and approval['attempt'] == 'r0'
    q = read(c/'job-queries.json')
    inventory = read(c/'inventory.json')
    expected_ids = [plan['parent']['job_id'], *(phase['request']['job_id'] for phase in plan['phases'])]
    assert [Path(path).name for path in inventory['jobs']] == expected_ids
    assert sorted(name for name in q if name.startswith(PREFIX+'formal-')) == sorted(expected_ids)
    assert {Path(path).name for path in inventory['request_paths']} == {
        *(name+'.json' for name in expected_ids), Path(plan['server_plan_path']).name, Path(plan['approval_path']).name}
    assert {Path(path).name for path in inventory['outputs']} == {
        Path(req['argv'][req['argv'].index('--output')+1]).name
        for req in [plan['parent'],*(phase['request'] for phase in plan['phases'])]}
    release_receipt = read(c/'releases.json')
    sourcefiles = {}
    for sha, files in release_receipt['releases'].items():
        origin = raw/'sources/deployments'
        deployment = read(origin/(sha+'.deployment.json'))
        assert deployment['RUN_COMMIT'] == sha
        assert [{k:member[k] for k in ('path','sha256')} for member in files] == plan['release_inputs'][sha]
        for row in files:
            rpath = row['path'].removeprefix(deployment['release']+'/')
            candidate = origin/sha/rpath
            assert digest(candidate)==row['sha256'] and candidate.stat().st_size==row['bytes']
            sourcefiles[row['path']] = row['sha256']
    assert digest(raw/'sources/runner/runner.py') == plan['runner_sha256'] == release_receipt['runner_sha256']
    assert digest(raw/'sources/runner/deployment.json') == release_receipt['runner_deployment_sha256'] == plan['parent']['deployment_sha256']
    matching_normal = read(c/'frozen-normal-readback.json')
    for path, item in matching_normal['members'].items():
        raw_path = raw/path
        assert raw_path.stat().st_size == item['bytes'] and digest(raw_path) == item['sha256']
    assert digest(evidence/'cpu-replay-r0/assets.json') == matching_normal['asset_manifest_sha256']
    parent = read(evidence/'formal-parent-r0/summary.json')
    assert parent['status']=='PASS_A1_INSTRUMENTS_V2_FORMAL' and parent['used_gpu_device_seconds_before']==0
    assert [phase['job_id'] for phase in parent['phases']] == expected_ids[1:]
    assert all(phase['status']=='PASS' for phase in parent['phases'])
    sha_by_path = sourcefiles.copy()
    sha_by_path[plan['server_plan_path']] = PLAN_SHA
    sha_by_path[plan['approval_path']] = approval_sha

    def bound(p):
        if p not in sha_by_path:
            assert p.startswith(OUTPUT)
            sha_by_path[p] = digest(evidence/p.removeprefix(OUTPUT))
        return sha_by_path[p]

    def check(req, phase):
        named = req['job_id']
        short = named.removeprefix(PREFIX)
        directory = raw/'jobs'/short
        saved = read(directory/'request.json')
        accepted = read(directory/'accepted.json')
        attempt = read(directory/'attempt.json')
        status = read(directory/'status.json')
        cleanup = read(directory/'cleanup.json')
        query = q[named]
        assert saved == read(raw/'requests'/(short+'.json'))
        assert query['request_sha256'] == accepted['request_sha256'] == attempt['request_sha256'] == canon(saved)
        assert accepted['runner_sha256'] == plan['runner_sha256'] and accepted['runner_path'] == plan['runner']
        assert query['status'] == status['status'] == 'SUCCEEDED' and query['exit_code'] == status['exit_code'] == 0
        assert query['cleanup_verified'] and not any(query[k] for k in ('worker_alive','supervisor_alive','workgroup_live'))
        assert cleanup == query['cleanup'] == status['cleanup'] and cleanup['verified'] and cleanup['last_observation']['verified_empty']
        assert cleanup['reason'] == 'NORMAL_EXIT_CHECK' and not cleanup['events']
        assert query['worker']['argv'] == status['worker']['argv'] == saved['argv']
        copy = dict(req)
        copy['inputs'] = plan['release_inputs'][req['capture_version']] + req['inputs']
        if phase is None:
            copy['argv'] = [approval_sha if s.startswith('AWAITING_WEB_REVIEW_') else s for s in req['argv']]
            copy['inputs'] += [{'path':plan['server_plan_path'],'sha256':PLAN_SHA}]
        copy['inputs'] += [{'path':plan['approval_path'],'sha256':approval_sha}]
        if phase:
            for path in phase['predecessor_members']:
                copy['inputs'].append({'path':path,'sha256':bound(path)})
        for item in copy['inputs']:
            assert bound(item['path']) == item['sha256']
        assert copy == saved
        start, end = (datetime.fromisoformat(status[s]) for s in ('started_at','ended_at'))
        assert datetime.fromisoformat(accepted['accepted_at']) <= datetime.fromisoformat(attempt['claimed_at']) <= start < end
        for logfile in ('supervisor.log','work.log'):
            assert not FAILURE.search((directory/logfile).read_text())
        return {'job_id':named,'kind':phase['kind'] if phase else 'parent','mode':phase['mode'] if phase else 'parent',
                'started_utc':start.isoformat(),'ended_utc':end.isoformat(),'ended_ny':end.astimezone(NY).isoformat(),
                'attempt':'r0','status':status['status'],'exit_code':0,'cleanup_verified':True,
                'accepted_utc':accepted['accepted_at'],'request_sha256':canon(saved),'capture_version':saved['capture_version'],
                'worker_pid':query['worker']['pid'],'time_limit_seconds':saved['timeout_seconds'],
                'wall_seconds':(end-start).total_seconds()}

    closure = check(plan['parent'],None)
    stages = []
    closed = {'job_id':None,'ended_utc':closure['started_utc']}
    for phase, parent_record in zip(plan['phases'],parent['phases'],strict=True):
        stage = check(phase['request'],phase)
        assert phase['depends_on'] == closed['job_id']
        assert datetime.fromisoformat(closed['ended_utc']) <= datetime.fromisoformat(stage['accepted_utc'])
        ps = parent_record['runner_state']
        for k in ('status','exit_code','started_at','ended_at','cleanup_verified','request_sha256'):
            assert ps[k]==q[stage['job_id']][k]
        output = evidence / Path(phase['request']['argv'][phase['request']['argv'].index('--output')+1]).name
        staged = read(output/'summary.json')
        mode = phase['mode']
        kind = phase['kind']
        if kind == 'gpu':
            assert staged['status']=='PASS' and staged['run_commit']==GPU_COMMIT and staged['job_id']==stage['job_id']
            assert not staged['identity_discrepancies']
            assert phase['request']['env']['CUDA_VISIBLE_DEVICES']==plan['card']
            assert staged['device']['uuid']==plan['card'] and staged['device']['visible_device']==0
            report = read(output/'test/rank0.json')
            assert report['RUN_COMMIT']==GPU_COMMIT and report['status']=='PASS'
            assert report['completed_users']==report['assigned_users']==24772 and report['preflight_only'] is False
            assert (report['rank'],report['world_size'],report['local_rank'])==(0,1,0)
            assert staged['stages'][mode]['users']==24772 and read(output/(mode+'.exit.json'))['exit_code']==0
            log = (output/(mode+'.log')).read_text()
            assert not FAILURE.search(log)
            samples = [json.loads(line) for line in (output/(mode+'.resources.jsonl')).read_text().splitlines()]
            assert all(r['gpu'].split(',')[0]==plan['card'] for r in samples)
            assert min(r['host_mem_available_bytes'] for r in samples) >= 40*1024**3
            assert min(r['disk_free_bytes'] for r in samples) >= 20*1024**3
            stage.update(users=24772,candidates=495440,logical_gpu=staged['device']['visible_device'],
                         physical_gpu=staged['device']['physical_index'],resource_samples=len(samples),
                         model_load_s=report['model_load_and_strict_verification_seconds'],
                         generation_s=report['elapsed_generation_seconds'],peak_gpu_allocated_bytes=report['gpu_peak_allocated_bytes'],
                         peak_gpu_reserved_bytes=report['gpu_peak_reserved_bytes'],
                         output_path=str(phase['request']['argv'][phase['request']['argv'].index('--output')+1]))
        else:
            assert phase['request']['env']['CUDA_VISIBLE_DEVICES']==''
            assert staged['preflight'] is False and staged['users']==24772
            if kind == 'cpu':
                assert staged['status']=='PASS_V2_EVALUATION' and staged['candidate_anomalies']['candidate_count']==495440
                stage.update(users=24772,candidates=495440)
            else:
                assert staged['status']=='PASS_V2_PAIRED' and sum(map(len, staged['comparisons'].values()))==80
                assert digest(output/'arrays.npz')==staged['arrays_file_sha256']
                stage.update(users=24772,paired_records=80)
        stages.append(stage)
        closed = stage
    assert datetime.fromisoformat(closure['ended_utc']) >= datetime.fromisoformat(closed['ended_utc'])
    phases = [(stage['mode'], stage['kind']) for stage in stages]
    assert phases == [(mode,kind) for mode in MODES for kind in ('gpu','cpu')]+[('paired','paired')]
    gpu = sum(row['wall_seconds'] for row in stages if row['kind']=='gpu')
    cpu = sum(row['wall_seconds'] for row in stages if row['kind'] in ('cpu','paired'))
    def preparation(kind):
        relevant = {name:row for name,row in q.items() if name.startswith(PREFIX+kind)}
        assert all(row['cleanup_verified'] and row['ended_at'] for row in relevant.values())
        return sum((datetime.fromisoformat(row['ended_at'])-datetime.fromisoformat(row['started_at'])).total_seconds()
                   for row in relevant.values() if row['started_at'])
    before_gpu, before_cpu = preparation('gpu-preflight-'), preparation('cpu-')
    assert closure['wall_seconds'] <= closure['time_limit_seconds']
    assert gpu <= 129600 and max(r['wall_seconds'] for r in stages if r['kind']=='gpu') <= 28800
    assert before_gpu <= 10800 and cpu+before_cpu <= 28800
    return {'status':'PASS_ORIGINAL_R0_CONTROL_CHAIN','observed_utc':inventory['observed_utc'],
            'host':inventory['host'],'plan_sha256':PLAN_SHA,'authorization_sha256':approval_sha,
            'verified_export_members':len(sourcefiles), 'gpu_uuid':plan['card'],
            'parent':closure,'children':stages,'formal_retry_count':0,
            'budgets':{'formal_gpu_device_seconds':gpu,'formal_gpu_remaining_seconds':129600-gpu,
                       'formal_eval_cpu_worker_seconds':cpu,'preflight_gpu_device_seconds':before_gpu,
                       'preflight_gpu_remaining_seconds':10800-before_gpu,'preparation_cpu_worker_seconds':before_cpu,
                       'cpu_worker_total_seconds':cpu+before_cpu,'cpu_worker_remaining_seconds':28800-cpu-before_cpu,
                       'parent_wall_seconds':closure['wall_seconds'],'parent_limit_seconds':closure['time_limit_seconds']},
            'normal_sha256':{k:v['sha256'] for k,v in matching_normal['members'].items()},
            'output_available_bytes_at_observation':inventory['output_available_bytes']}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('raw',type=Path)
    args=ap.parse_args()
    print(json.dumps(verify(args.raw),indent=2,ensure_ascii=False,allow_nan=False))
