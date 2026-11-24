"""Independently bind terminal jobs to reviewed requests, byte identities and budget."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from independent_inputs import MODES, sha_file

PREFIX = 'a1-instruments-v2-'
ROOT = Path('/data/bwwu/lc-rec/a1-instruments-v2')
GPU = 'c487f7e7176bc223445d3deecfab87e0bcc30d1a'
CPU = '3cc1110097c4764ebfd93bb38ee2913c8afb6958'
REVIEW = '18347ef5914a6ca9ec638f31823fed719e56e4e8'
WEB = 'a31bf0ecee6774ef8abc439f581ec2f4752ada80'
PLAN_REL = 'lc_rec/shuffle/results/v2-a1-instruments/formal-plan.json'
README_REL = 'lc_rec/shuffle/results/v2-a1-instruments/README.md'
PLAN_SHA = 'f37fbe7ac295480a181daf2fed098810bd5b2d3c9499e19a7eec67d160603c7b'
PLACEHOLDER = 'AWAITING_WEB_REVIEW_AND_SEPARATE_USER_AUTHORIZATION'
CARD = 'GPU-59cac3bd-3645-668b-79c8-839e0d878893'
CLOCK = datetime.fromisoformat


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def flag(argv,name):
    return argv[argv.index(name)+1]


def identify_sources(root, control, plan, git_worktree):
    releases = read(control/'releases.json')
    packed = root/'sources'/'deployments'
    assert releases['host'] == 'zwlab-01'
    assert releases['runner_sha256'] == plan['runner_sha256']
    assert releases['runner_deployment_sha256'] == plan['parent']['deployment_sha256']
    assert sha_file(root/'sources'/'runner'/'runner.py') == releases['runner_sha256']
    assert sha_file(root/'sources'/'runner'/'deployment.json') == releases['runner_deployment_sha256']
    assert set(releases['releases']) == {GPU,CPU}
    assert set(plan['release_inputs']) == {GPU,CPU}
    known, counts = {}, {}
    for commit, listing in releases['releases'].items():
        receipt = read(packed/f'{commit}.deployment.json')
        export = Path(receipt['release'])
        assert receipt['RUN_COMMIT'] == export.name == commit
        assert {name:row['sha256'] for name,row in ((x['path'],x) for x in listing)} == {
            row['path']:row['sha256'] for row in plan['release_inputs'][commit]}
        assert all(member['path'].startswith(str(export)+'/') for member in listing)
        assert {str(export/row['path']):(row['sha256'],row['bytes']) for row in receipt['files']} == {
            row['path']:(row['sha256'],row['bytes']) for row in listing}
        counts[commit] = len(listing)
        for member in listing:
            remote = Path(member['path'])
            relative = remote.relative_to(export)
            raw = (packed/commit/relative).read_bytes()
            assert len(raw) == member['bytes'] and sha256(raw).hexdigest() == member['sha256'], remote
            if git_worktree:
                original = subprocess.check_output(['git','-C',str(git_worktree),'show',f'{commit}:{relative}'])
                assert raw == original, relative
            known[str(remote)] = member['sha256']
    return known, counts


def verify_frozen(root, control, pointer):
    normal = read(control/'frozen-normal-readback.json')
    assert sha_file(root/'evidence'/'cpu-replay-r0'/'assets.json') == normal['asset_manifest_sha256']
    frozen = {}
    for label, source in normal['members'].items():
        path = root/label
        if label.endswith('.jsonl'):
            rank = int(path.name.split('rank')[1][0])
            actual = pointer['frozen_original_shard_sha256'][rank]
        else:
            actual = sha_file(path)
        assert actual == source['sha256'] and path.stat().st_size == source['bytes']
        frozen[source['server_path']] = actual
    return frozen


def verify_terminal(job, captures, request, plan):
    raw_status = read(job/'status.json')
    terminal = read(job/'cleanup.json')
    accepted = read(job/'accepted.json')
    attempt = read(job/'attempt.json')
    assert raw_status['job_id'] == request['job_id'] == captures['job_id']
    assert raw_status['status'] == captures['status'] == 'SUCCEEDED'
    assert raw_status['exit_code'] == captures['exit_code'] == 0
    for k,v in raw_status.items():
        assert captures[k] == v, (job,k)
    assert accepted['runner_path'] == plan['runner'] and accepted['runner_sha256'] == plan['runner_sha256']
    assert attempt['request_sha256'] == accepted['request_sha256'] == captures['request_sha256'] == canonical(request)
    moments = [CLOCK(x) for x in (accepted['accepted_at'], raw_status['received_at'],attempt['claimed_at'],
                                 raw_status['started_at'],terminal['started_at'],terminal['ended_at'],raw_status['ended_at'])]
    assert moments == sorted(moments)
    assert raw_status['worker']['argv'] == request['argv'] and raw_status['worker']['uid'] == request['uid']
    assert raw_status['cleanup'] == terminal and terminal['verified'] is True
    assert not terminal['events'] and terminal['reason'] == 'NORMAL_EXIT_CHECK'
    assert captures['cleanup_verified'] and raw_status['worker_may_be_live'] is False
    assert not any(captures[k] for k in ('supervisor_alive','worker_alive','workgroup_live')), job
    for observed in (terminal['initial_observation'],terminal['last_observation'],captures['workgroup_observation']):
        assert observed['verified_empty'] and observed['scan_complete'] and not observed['boot_changed']
        assert not any(observed[k] for k in ('owned_live','unconfirmed_live','unsupported_live','scan_errors','zombies'))
    return {'request_sha256':captures['request_sha256'],'accepted_utc':accepted['accepted_at'],
            'attempt_claimed_utc':attempt['claimed_at'],'started_utc':raw_status['started_at'],
            'ended_utc':raw_status['ended_at'],'worker_pid':raw_status['worker']['pid'],
            'elapsed_seconds':(moments[-1]-moments[3]).total_seconds(),
            'cleanup_reason':terminal['reason'],'exit_code':captures['exit_code']}


def verify(root, verified_inputs, git_worktree):
    control = root/'control'
    plan_file = control/'formal-plan.json'
    plan = read(plan_file)
    assert sha_file(plan_file) == PLAN_SHA
    assert (plan['preflight_gpu_commit'],plan['evaluation_and_controller_commit'],plan['card']) == (GPU,CPU,CARD)
    assert plan['users_per_mode'] == 24772 and plan['candidates_per_mode'] == 495440
    assert plan['formal_submitted'] is False and plan['formal_gpu_device_seconds_delta'] == 0
    if git_worktree:
        audited = subprocess.check_output(['git','-C',str(git_worktree),'show',f'{REVIEW}:{PLAN_REL}'])
        review = subprocess.check_output(['git','-C',str(git_worktree),'show',f'{WEB}:{README_REL}'])
        assert audited == plan_file.read_bytes() and b'FOURTH_GROUP_PREFLIGHT_COMPLETE_WEB_REVIEW_PASS' in review
    approval_file = control/'authorization.json'
    authorization = read(approval_file)
    assert authorization['plan_sha256'] == PLAN_SHA and authorization['experiment'] == plan['experiment']
    assert authorization['web_review_pass'] is True and authorization['user_formal_launch_authorized'] is True
    assert (authorization['gpu_commit'],authorization['controller_commit'],authorization['attempt']) == (GPU,CPU,'r0')
    assert authorization['web_review_commit'] == WEB and authorization['gpu_uuid'] == CARD
    assert authorization['scope'] == ['S2','S3a','S3b','S4','four CPU evaluations','five paired comparisons']
    assert authorization['observation_minutes'] == 30
    auth_sha = sha_file(approval_file)
    approval = {'path':plan['approval_path'],'sha256':auth_sha}
    known, release_counts = identify_sources(root,control,plan,git_worktree)
    known.update(verify_frozen(root,control,verified_inputs))
    known[plan['server_plan_path']] = PLAN_SHA
    known[plan['approval_path']] = auth_sha
    for mode,row in verified_inputs['mode_checks'].items():
        known[str(ROOT/f'formal-gpu-{mode}-r0'/'test'/'predictions.rank0.jsonl')] = row['prediction_sha256']
    inventory = read(control/'inventory.json')
    query = read(control/'job-queries.json')
    summary = read(root/'evidence/formal-parent-r0/summary.json')
    assert summary['status'] == 'PASS_A1_INSTRUMENTS_V2_FORMAL' and summary['used_gpu_device_seconds_before'] == 0
    phases = plan['phases']
    assert [(p['kind'],p['mode']) for p in phases] == [pair for m in MODES for pair in [('gpu',m),('cpu',m)]] + [('paired','paired')]
    terminal = summary['phases']
    all_ids = [plan['parent']['job_id'],*[p['request']['job_id'] for p in phases]]
    assert len(set(all_ids)) == 10 and all(name.endswith('-r0') for name in all_ids)
    assert set(all_ids) == {Path(path).name for path in inventory['jobs']}
    assert set(all_ids) == {job for job in query if job.startswith(PREFIX+'formal-')}
    assert set(inventory['outputs']) == {flag(r['argv'],'--output') for r in [plan['parent'],*[p['request'] for p in phases]]}
    expected_requests = {str(Path(plan['requests_root'])/(job+'.json')) for job in all_ids}
    assert expected_requests | {plan['approval_path'],plan['server_plan_path']} == set(inventory['request_paths'])
    assert len(terminal) == len(phases)
    order, budgets = [], {'gpu':0.0,'cpu':0.0,'paired':0.0}
    parent = deepcopy(plan['parent'])
    assert parent['capture_version'] == CPU and parent['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert parent['timeout_seconds'] == 136800
    parent['argv'][parent['argv'].index(PLACEHOLDER)] = auth_sha
    parent['inputs'] = [*plan['release_inputs'][CPU],*parent['inputs'],
                        {'path':plan['server_plan_path'],'sha256':PLAN_SHA},approval]

    def hash_input(remote):
        if remote not in known:
            relative = Path(remote).relative_to(ROOT)
            known[remote] = sha_file(root/'evidence'/relative)
        return known[remote]

    def check(request):
        short = request['job_id'].removeprefix(PREFIX)
        job = root/'jobs'/short
        assert request == read(job/'request.json') == read(root/'requests'/f'{short}.json')
        assert all(hash_input(row['path']) == row['sha256'] for row in request['inputs']), short
        return verify_terminal(job,query[request['job_id']],request,plan)

    parent_record = check(parent)
    assert CLOCK(authorization['recorded_at_utc']) < CLOCK(parent_record['accepted_utc'])
    assert parent_record['elapsed_seconds'] <= parent['timeout_seconds']
    seen = None
    previous = CLOCK(parent_record['started_utc'])
    for expected, completed in zip(phases,terminal,strict=True):
        mode,kind,req = expected['mode'],expected['kind'],expected['request']
        name = req['job_id']
        assert expected['depends_on'] == seen and completed['job_id'] == name
        assert completed['status'] == 'PASS' and completed['request'] == str(Path(plan['requests_root'])/(name+'.json'))
        assert req['capture_version'] == (GPU if kind == 'gpu' else CPU)
        assert req['env']['CUDA_VISIBLE_DEVICES'] == (CARD if kind == 'gpu' else '')
        assert req['timeout_seconds'] == {'gpu':28790,'cpu':1800,'paired':4500}[kind]
        prior = [{'path':path,'sha256':hash_input(path)} for path in expected['predecessor_members']]
        req = {**req, 'inputs':[*plan['release_inputs'][req['capture_version']],*req['inputs'],approval,*prior]}
        receipt = check(req)
        assert completed['runner_state']['request_sha256'] == receipt['request_sha256']
        assert completed['runner_state']['status'] == 'SUCCEEDED' and completed['runner_state']['cleanup_verified']
        assert previous <= CLOCK(receipt['accepted_utc'])
        previous = CLOCK(receipt['ended_utc'])
        assert receipt['elapsed_seconds'] <= req['timeout_seconds']
        budgets[kind] += receipt['elapsed_seconds']
        order.append({'id':name,'stage':kind,'mode':mode,'timeout_seconds':req['timeout_seconds'],**receipt})
        seen = name
    assert previous <= CLOCK(parent_record['ended_utc'])
    assert plan['future_budget'] == {'gpu_seconds_per_mode':28800,'gpu_device_seconds':129600,
                                    'gpu_seconds_reserved_this_plan':115200,'parent_wall_timeout_seconds':136800}
    assert budgets['gpu'] <= plan['future_budget']['gpu_device_seconds']
    preparation = {'gpu_preflight_seconds':0.0,'cpu_seconds':0.0}
    for name, receipt in query.items():
        if 'formal-' in name:
            continue
        assert receipt['cleanup_verified'] and not any(receipt[k] for k in ('worker_alive','supervisor_alive','workgroup_live'))
        expected_status = 'TIMED_OUT' if name == plan['timeout_fixture'] else 'SUCCEEDED'
        expected_exit = -15 if expected_status == 'TIMED_OUT' else 0
        assert (receipt['status'],receipt['exit_code']) == (expected_status,expected_exit)
        key = 'gpu_preflight_seconds' if name.startswith(PREFIX+'gpu-preflight-') else 'cpu_seconds'
        preparation[key] += (CLOCK(receipt['ended_at'])-CLOCK(receipt['started_at'])).total_seconds()
    return {'status':'PASS_A_FIXED_DEPLOYMENTS_AUTHORIZATION_TERMINAL_CHAIN_AND_CLEANUP',
            'release_members':release_counts,'compared_fixed_git_objects':bool(git_worktree),
            'plan_sha256':PLAN_SHA,'authorization_sha256':auth_sha,'parent':parent_record,
            'formal_stages':order,'job_count':len(all_ids),'formal_attempts':['r0'],
            'budget_seconds':budgets,'preparation_budget_seconds':preparation,
            'bound_source_and_input_identities':len(known),'server_inventory_utc':inventory['observed_utc']}


def main():
    parser = ArgumentParser()
    parser.add_argument('raw',type=Path)
    parser.add_argument('verified_inputs',type=Path)
    parser.add_argument('--git-worktree',type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.raw,read(args.verified_inputs),args.git_worktree),
                     indent=2,sort_keys=True,allow_nan=False))

if __name__ == '__main__':
    main()
