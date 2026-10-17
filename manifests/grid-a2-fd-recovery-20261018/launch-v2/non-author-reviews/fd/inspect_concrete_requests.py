"""Independent full concrete-request audit and current read-only server checks.

No submit/supervise/export, source edits, GPU computation, or remote writes.
The accepted runner's query/verify routines and existing lock probes are used.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
BASE = Path('/home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018')
COMPILED = BASE / 'compiled-c31d2d3eaaafbd18'
OUT = Path(__file__).resolve().parent
PYTHON = '/data/bwwu/envs/grid-20261009-e61baf3/bin/python'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def replace(value, mapping):
    if isinstance(value, str):
        for a, b in mapping:
            value = value.replace(a, b)
        return value
    if isinstance(value, list):
        return [replace(x, mapping) for x in value]
    if isinstance(value, dict):
        return {k: replace(v, mapping) for k, v in value.items()}
    return value


def main():
    source = read(BASE / 'source-deployment.json')
    deployment = read(BASE / 'queue-deployment.json')
    cfg = read(COMPILED / 'queue-config.json')
    plan = read(COMPILED / 'plan.json')
    final_review = read(OUT / 'launch-review.json')
    supervisor = read(COMPILED / 'supervisor-request.json')
    frozen = read(COMPILED / 'frozen-evidence.json')
    requests = read(COMPILED / 'wrapped-requests.json')
    scene = read(BASE / 'initial-scene.json')
    shared = scene['config']
    root = source['root']
    remote = deployment['remote_root']
    capture = source['manifest']['source_commit'] + '/' + source['manifest']['snapshot_sha256']
    config_sha = sha(COMPILED / 'queue-config.json')
    assert config_sha == deployment['config_sha256'] == '45fa8698ab605332b992ae37ce097acf67328777bf06900116304b789fafe3ac'
    assert final_review['status'] == 'SOURCE_AND_LAUNCH_ADMITTED_CONDITIONALLY'
    assert final_review['fd_engineering_status'] == 'FD_ENGINEERING_EQUIVALENCE_PASS'
    assert final_review['future_results_approved'] is False
    assert final_review['source_python_sha256'] == plan['source_python_sha256']
    assert cfg['snapshot'] == dict(root=root, commit=source['manifest']['source_commit'])
    assert cfg['runner'] == shared['runner'] and cfg['gpu_locks_dir'] == shared['gpu_locks_dir']
    assert cfg['approved_gpu_uuids'] == shared['approved_gpu_uuids']
    for key in ['free_space', 'max_storage_bytes', 'prior_gpu_request_usage', 'namespace_identity']:
        assert cfg[key] == shared[key]
    assert set(shared['storage_roots']) <= set(cfg['storage_roots'])
    assert cfg['recovery']['frozen_evidence'] == frozen and len({r['path'] for r in frozen}) == len(frozen)
    assert cfg['recovery']['terminal_controller_ids'] == ['ga-20261017-queue-v1', 'ga-20261018-shared-queue-v1']
    assert cfg['recovery']['original_failure_job'] == 'ga-20261017-sportsa2-quantize-v1'
    absolute = shared['continuation']['original_absolute_queue_deadline']
    assert cfg['recovery']['original_absolute_queue_deadline'] == absolute
    assert cfg['gpu_budget_seconds'] == shared['gpu_budget_seconds'] - scene['accounting']['used_gpu_request_seconds']
    assert cfg['budget_groups'] == {k: v - scene['accounting']['groups'][k] for k, v in shared['budget_groups'].items()}
    assert cfg['budget_groups']['sports-a2-rkmeans'] == 6897.950524
    assert cfg['recovery']['prior_gpu_request_seconds'] == 58.650573
    expected_dependencies = {
        'sportsa2-quantize': [], 'sportsa2-metadata': ['sportsa2-quantize'],
        'sportsa2-gpu-gate': ['sportsa2-metadata'], 'sportsa2-train': ['sportsa2-gpu-gate'],
        **{'sportsa2-eval-' + str(n): ['sportsa2-train'] for n in [0, 3, 4]},
        **{'sportsa2-attention-' + str(n): ['sportsa2-eval-0', 'sportsa2-eval-3', 'sportsa2-eval-4'] for n in [0, 3, 4]},
        'sportsa2-stats': ['sportsa2-attention-0', 'sportsa2-attention-3', 'sportsa2-attention-4'],
    }
    stages = {s['id']: s for s in cfg['stages']}
    assert set(stages) == set(expected_dependencies) and len(cfg['stages']) == len(requests) == 11
    assert all(s['workflow'] == 'sports-a2' for s in stages.values())
    originals = {s['id']: s for s in shared['stages']}
    changes = [(shared['snapshot']['root'], root)]
    changes += [(originals[n]['request']['job_id'], stages[n]['request']['job_id']) for n in stages]
    wrapped = {r['job_id']: r for r in requests}
    assert len(wrapped) == 11
    groups = {g: 0 for g in cfg['budget_groups']}
    rows = []
    all_inputs = {deployment['remote_config']: config_sha, root + '/DEPLOYMENT.json': source['manifest_sha256'], remote + '/plan.json': sha(COMPILED / 'plan.json')}
    for row in frozen:
        all_inputs[row['path']] = row['sha256']
    for name, stage in stages.items():
        old = originals[name]
        spec_path = COMPILED / 'specs' / (name + '.json')
        spec = read(spec_path)
        assert hashlib.sha256(canonical({k: v for k, v in spec.items() if k != 'launch_review'})).hexdigest() in plan['template_plan_sha256']
        assert spec['launch_review'] == dict(path=remote + '/launch-review.json', sha256=sha(OUT / 'launch-review.json'))
        assert stage['depends_on'] == expected_dependencies[name]
        for key in ['workflow', 'kind', 'budget_group', 'gpu_uuids', 'gpu_candidates', 'resource_policy', 'priority']:
            assert stage.get(key) == old.get(key), (name, key)
        request = stage['request']
        assert request['job_id'] == 'ga-20261018-' + name + '-fd-recovery-v1'
        assert request['capture_version'] == capture and request['cwd'] == root
        assert request['argv'] == [PYTHON, '-B', '-m', 'experiments.launch_20261017.run', spec['phase'], '--spec', remote + '/specs/' + name + '.json']
        expected_env = copy.deepcopy(old['request']['env'])
        expected_env['PYTHONPATH'] = root
        assert request['env'] == expected_env
        expected_wall = 3420 if name == 'sportsa2-quantize' else old['request']['timeout_seconds']
        assert request['timeout_seconds'] == expected_wall <= old['request']['timeout_seconds']
        assert {k: v for k, v in request.items() if k not in ['job_id', 'capture_version', 'cwd', 'argv', 'env', 'inputs', 'timeout_seconds']} == {k: v for k, v in old['request'].items() if k not in ['job_id', 'capture_version', 'cwd', 'argv', 'env', 'inputs', 'timeout_seconds']}
        input_map = {r['path']: r['sha256'] for r in request['inputs']}
        assert len(input_map) == len(request['inputs'])
        expected_inputs = {root + '/DEPLOYMENT.json': source['manifest_sha256'], remote + '/specs/' + name + '.json': sha(spec_path), remote + '/launch-review.json': sha(OUT / 'launch-review.json')}
        for field in ['config', 'sid', 'checkpoint', 'training_protocol']:
            if isinstance(spec.get(field), dict) and 'path' in spec[field]:
                expected_inputs[spec[field]['path']] = spec[field]['sha256']
        if name == 'sportsa2-quantize':
            expected_inputs[scene['embedding']['path']] = scene['embedding']['sha256']
        assert input_map == expected_inputs
        all_inputs.update(input_map)
        for gate in replace(old['gates'], changes):
            assert gate in stage['gates'], (name, 'original gate missing', gate)
        for dep in stage['depends_on']:
            jobroot = cfg['runner']['job_root'] + '/' + stages[dep]['request']['job_id']
            assert dict(path=jobroot + '/status.json', json_equals={'status': 'SUCCEEDED', 'exit_code': 0, 'cleanup.verified': True}) in stage['gates']
            assert dict(path=jobroot + '/output/RESULT.json', json_equals={'capture_version': capture, 'workflow': 'Sports-A2'}) in stage['gates']
        if name == 'sportsa2-metadata':
            path = cfg['runner']['job_root'] + '/' + stages['sportsa2-quantize']['request']['job_id'] + '/output/quantizer-strict-load.json'
            assert dict(path=path, json_equals={'strict_load': True, 'state_exact_to_trained_model': True, 'global_step': 30}) in stage['gates']
        gpus = stage.get('gpu_uuids', [])
        if gpus:
            groups[stage['budget_group']] += expected_wall * len(gpus)
        actual = wrapped[request['job_id']]
        expected = copy.deepcopy(request)
        expected['argv'] = [PYTHON, '-u', root + '/experiments/a2_fd_recovery_20261018/controller.py', 'run-stage', '--config', deployment['remote_config'], '--stage', name]
        expected['env']['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        queue_script = root + '/experiments/launch_20261017/queue.py'
        controller_script = root + '/experiments/a2_fd_recovery_20261018/controller.py'
        expected['inputs'] += [dict(path=queue_script, sha256=sha(REPO / 'experiments/launch_20261017/queue.py')), dict(path=deployment['remote_config'], sha256=config_sha), dict(path=controller_script, sha256=sha(REPO / 'experiments/a2_fd_recovery_20261018/controller.py'))]
        expected['queue_stage'] = dict(id=name, workflow='sports-a2', config_sha256=config_sha, command=request['argv'], budget_group=stage.get('budget_group'), gpu_uuids=gpus, source_commit=source['manifest']['source_commit'])
        assert actual == expected, (name, 'wrapped immutable request mismatch')
        all_inputs.update({r['path']: r['sha256'] for r in actual['inputs']})
        rows.append({'stage': name, 'job_id': request['job_id'], 'kind': stage['kind'], 'wall_seconds': expected_wall, 'gpu_uuids': gpus, 'GPU_candidates': stage.get('gpu_candidates', []), 'depends_on': stage['depends_on'], 'spec_sha256': sha(spec_path), 'wrapped_request_sha256': hashlib.sha256(canonical(actual)).hexdigest(), 'expected_argv_env_inputs_scope_checked': True})
    assert sum(groups.values()) == 59940
    assert groups['sports-a2-rkmeans'] == 6840 and groups['sports-a2-train'] == 43200
    assert groups['sports-a2-eval-attention'] == 9000 and groups['engineering'] == 900
    assert all(groups[g] <= cfg['budget_groups'][g] for g in groups)
    assert supervisor['job_id'] == cfg['queue_id'] == 'ga-20261018-a2-fd-queue-v1'
    assert supervisor['argv'] == [PYTHON, '-B', '-m', 'experiments.a2_fd_recovery_20261018.controller', 'supervise', '--config', deployment['remote_config']]
    assert supervisor['cwd'] == root and supervisor['capture_version'] == capture
    assert supervisor['env']['CUDA_VISIBLE_DEVICES'] == '' and supervisor['env']['PYTHONPATH'] == root
    outer_inputs = {r['path']: r['sha256'] for r in supervisor['inputs']}
    assert outer_inputs == {root + '/DEPLOYMENT.json': source['manifest_sha256'], deployment['remote_config']: config_sha, deployment['outer_runner']: deployment['outer_runner_sha256'], deployment['outer_runner'] + '.deployment.json': cfg['runner']['deployment_sha256'], remote + '/launch-review.json': sha(OUT / 'launch-review.json')}
    all_inputs.update(outer_inputs)
    assert not ({'GRID_JOB_ID', 'GRID_JOB_DIR'} & supervisor['env'].keys())
    sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
    from transport import Transport
    t = Transport('new-server')
    t.identity()
    code = '''import pathlib,json,hashlib,runpy,datetime,sys,os,fcntl,traceback
inputs=INPUTS
checks=[]
for name,expected in inputs.items():
 p=pathlib.Path(name);actual=hashlib.sha256(p.read_bytes()).hexdigest()
 assert actual==expected,name
 checks.append(dict(path=name,bytes=p.stat().st_size,sha256=actual))
cfg=json.loads(pathlib.Path(CONFIG).read_text())
sys.path.insert(0,cfg['snapshot']['root'])
from experiments.a2_fd_recovery_20261018 import controller
from experiments.launch_20261017 import queue
summary=queue.validate(cfg,CONFIG)
historical=controller.lineage(cfg,live=True)
runner=runpy.run_path(OUTER,run_name='independent_request_audit')
runner['verify'](SUPERVISOR)
query=[]
for job in ['ga-20261017-queue-v1','ga-20261018-shared-queue-v1','ga-20261017-sportsa2-encode-v1','ga-20261017-sportsa2-quantize-v1','ga-20261018-a2-fd-cpu-equivalence-v1']:
 v=runner['query'](pathlib.Path(cfg['runner']['job_root'])/job)
 query.append(dict(job_id=job,**{k:v.get(k) for k in ['status','exit_code','supervisor_alive','worker_alive','workgroup_live','worker_may_be_live','cleanup_verified','request_sha256']}))
 assert v['cleanup_verified'] and not any(v.get(k) for k in ['supervisor_alive','worker_alive','workgroup_live','worker_may_be_live'])
conflicts=[str(pathlib.Path(cfg['runner']['job_root'])/s['request']['job_id']) for s in cfg['stages'] if (pathlib.Path(cfg['runner']['job_root'])/s['request']['job_id']).exists()]
assert not conflicts and not (pathlib.Path(cfg['runner']['job_root'])/cfg['queue_id']).exists()
qstage=next(s for s in cfg['stages'] if s['id']=='sportsa2-quantize')
resource_ok,samples=queue.prelaunch_samples(qstage)
disk=queue.disk_snapshot(cfg)
locks=[]
try:
 for device in sorted(qstage['gpu_uuids']):
  path=pathlib.Path(cfg['gpu_locks_dir'])/(device+'.lock')
  fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC);locks.append(fd)
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 locks_available=True
finally:
 for fd in locks:os.close(fd)
deadline=datetime.datetime.fromisoformat(cfg['recovery']['original_absolute_queue_deadline'])
remaining=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
assert remaining>qstage['request']['timeout_seconds']+60
print(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only_no_submission=True,input_hash_checks=checks,source_commit=cfg['snapshot']['commit'],summary=summary,historical_accounting=historical,selected_old_job_queries=query,new_ID_conflicts=conflicts,supervisor_request_accepted_by_readonly_verify=True,quantizer_prelaunch_resource_ok=resource_ok,quantizer_resource_samples=samples,disk=disk,existing_project_GPU_locks_available=locks_available,deadline=deadline.isoformat(),deadline_remaining_seconds=remaining)))
'''.replace('INPUTS', repr(all_inputs)).replace('CONFIG', repr(deployment['remote_config'])).replace('OUTER', repr(deployment['outer_runner'])).replace('SUPERVISOR', repr(supervisor))
    remote_result = json.loads(t.python(code, timeout=60))
    target = OUT / 'concrete-requests-independent-inspection.json'
    result = {'reviewer': '/root/fd_repair_reviewer', 'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'CONCRETE_REQUESTS_AND_CURRENT_LIVE_ADMISSION_PASS' if remote_result['quantizer_prelaunch_resource_ok'] and remote_result['disk']['ok'] else 'CONCRETE_REQUESTS_PASS_RESOURCE_WAIT_REQUIRED', 'queue_config_sha256': config_sha, 'supervisor_request_file_sha256': sha(COMPILED / 'supervisor-request.json'), 'supervisor_canonical_request_sha256': hashlib.sha256(canonical(supervisor)).hexdigest(), 'source_commit': source['manifest']['source_commit'], 'snapshot': root, 'launch_review_sha256': sha(OUT / 'launch-review.json'), 'stage_request_checks': rows, 'GPU_reserved_seconds': sum(groups.values()), 'reserved_by_group': groups, 'remote_independent_check': remote_result, 'scientific_results_preapproved': False}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(target), 'sha256': sha(target), 'status': result['status'], 'supervisor_wall_seconds': supervisor['timeout_seconds'], 'resource_ok': remote_result['quantizer_prelaunch_resource_ok'], 'disk_ok': remote_result['disk']['ok']}))


if __name__ == '__main__':
    main()
