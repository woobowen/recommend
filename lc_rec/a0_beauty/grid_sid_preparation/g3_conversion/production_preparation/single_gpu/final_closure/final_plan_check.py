#!/usr/bin/env python3
"""CPU-only closure: prepare and validate the real plan without launching it."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace


RUN = '21d495b0be5b606853ec7cec68a3eaa1b58acbd7'
CONFIG = '8e4c68c45b200d81624f25dd1c9bc204b5324f5da98e46ede6c68bc82a834853'
OLD_DRAFT = '44e6cc1a03f232f1d62a81ae0bde8fa1162ffd18830497bea369390c285efcbe'
STAGE_CAPS = dict(train=108000, verify_best_adapter=900, normal=43200,
                  normal_eval=900, s3=21600, s3_eval=900, s4=21600,
                  s4_eval=900, paired=1200, collect=600)
GROUP_CAPS = dict(train_normal=151200, s3=21600, s4=21600,
                  cpu=5400, total=199800, cleanup_reserve=120)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(plan, registry, identity, digest, read):
    root = Path(plan['root'])
    old = root / 'preparation-plan.json'
    check(old.stat().st_size == 214758 and identity(old)['sha256'] == OLD_DRAFT,
          'historical draft identity')
    key = digest(dict(dataset='Beauty', assets=plan['asset_sha256'],
                      config=plan['config_sha256'], model_revision=Path(plan['model']).name,
                      seed=42, protocol='GRID-LETTER-SHUFFLE-V1', conditions=['normal','s3','s4'],
                      series=plan['experiment_series']))
    binding = [str(p) for p in (registry / (key + '.json'), registry / (key + '.lock')) if p.exists()]
    known_prior = []
    for record in sorted(registry.glob('*.json')):
        registered = read(record)
        check(registered.get('task_id') != plan['task_id'] and registered.get('root') != str(root),
              'existing registry reference to G3')
        old_plan = read(Path(registered['root']) / 'plan.json')
        if old_plan.get('experiment_series') == 'beauty-a0-seed42-original':
            known_prior.append({'registration': identity(record),
                                'series': old_plan['experiment_series']})
    jobs = sorted(p.name for p in Path(plan['job_root']).glob(
        'letter-a0-beauty-formal-' + plan['task_id'] + '-*'))
    formal_processes = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(str(root) in arg or plan['task_id'] in arg for arg in args) and any(
                Path(arg).name in ('production_control.py', 'production_pipeline.py',
                                   'formal_train.py', 'formal_evaluate.py') for arg in args):
            formal_processes.append(proc.name)
    children = sorted(p.name for p in root.iterdir())
    check(set(children) <= {'preparation-plan.json', 'plan.json'} and
          not binding and not jobs and not formal_processes and known_prior,
          'unexpected formal files, registry, job or process')
    return dict(utc=datetime.now(timezone.utc).isoformat(), root_children=children,
                scoped_job_ids=jobs, registry_key=key, registry_entries=binding,
                prior_series=known_prior, formal_processes=formal_processes, old_draft=identity(old),
                current_plan=identity(root/'plan.json') if 'plan.json' in children else None)


def budget_checks(control, plan, fixture, read, write):
    """Exercise production methods with a stopped request factory and fake statuses."""
    stamp = datetime.now(timezone.utc)
    stage = 's3'
    attempts = [dict(stage=stage, job_id='synthetic-failed', request_sha256='synthetic',
                     charged_seconds=500, status='FAILED')]
    complete = {s: {'status': 'COMPLETE'} for s in control.STAGES[:4]}
    state = dict(status='FAILED', stages={**complete, stage: {'attempt_index': 0}}, attempts=attempts)
    status = dict(request_sha256='synthetic', status='FAILED', exit_code=1,
                  started_at=(stamp-timedelta(seconds=600)).isoformat(), ended_at=stamp.isoformat(),
                  cleanup_verified=True, worker_alive=False, workgroup_live=False)
    safe_plan = {**plan, 'evidence_scope': control.SYNTHETIC, 'root': str(fixture)}
    state_path = fixture / 'counter-test-state.json'
    write(state_path, state)
    calls = []

    class StopBeforeSubmission(Exception):
        pass

    def stop_at_factory(active_plan, actual_stage, attempt, timeout, *_):
        calls.append(dict(stage=actual_stage, retry_attempt=attempt, timeout_seconds=timeout))
        raise StopBeforeSubmission()

    fake = SimpleNamespace(plan=safe_plan, root=fixture, state_path=state_path,
                           path=fixture/'never-submitted.json', factory=stop_at_factory,
                           runner=SimpleNamespace(query=lambda job: status))
    fake.charge = lambda ledger: control.Control.charge(fake, ledger)
    try:
        control.Control.step(fake)
    except ValueError as exc:
        check('failed stage requires explicit retry' in str(exc), str(exc))
    else:
        raise AssertionError('automatic retry permitted')
    check(not calls and read(state_path)['attempts'][0]['charged_seconds'] == 600,
          'failed attempt not charged')
    # A later shorter report cannot refund time already charged.
    status['started_at'] = (stamp-timedelta(seconds=550)).isoformat()
    try:
        control.Control.step(fake, retry=stage)
    except StopBeforeSubmission:
        pass
    else:
        raise AssertionError('synthetic factory did not stop')
    check(calls == [dict(stage='s3', retry_attempt=1, timeout_seconds=20880)] and
          read(state_path)['attempts'][0]['charged_seconds'] == 600, 'retry budget or monotonicity')
    return dict(scope=control.SYNTHETIC, explicit_retry_required=True,
                failed_charge_seconds=600, shorter_report_cannot_refund=True,
                actual_factory_boundary=calls[0], cleanup_reserve_seconds=120)


def main():
    parser = argparse.ArgumentParser()
    for name in ('release', 'old-draft', 'materials', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('DS_ACCELERATOR') == 'cpu',
          'checker must be CPU-only')
    scripts = args.release / 'lc_rec/a0_beauty'
    sys.path.insert(0, str(scripts))
    from formal_control import digest, durable_write, read
    from production_identity import identity, STEPS
    import production_control as control
    from formal_train import verify_web_review
    from g3_identity import verify_base_model, verify_launch_authorization
    from formal_evaluate import catalog_and_users, BEAMS
    from production_pipeline import compute_argv, training_root

    old = read(args.old_draft)
    check(args.old_draft == Path(old['root']) / 'preparation-plan.json' and args.release == Path(old['release']),
          'old draft location')
    before = snapshot(old, control.EXPERIMENT_REGISTRY, identity, digest, read)
    path = Path(old['root']) / 'plan.json'
    materials = args.materials
    approved = materials / 'approved_budget.json'
    approval = materials / 'budget_approval.json'
    review_path = materials / 'external_technical_review.json'
    check(read(approved) == dict(stage_seconds=STAGE_CAPS, budget=GROUP_CAPS), 'approved budget')
    check(read(approval)['approved_budget_identity'] == identity(approved), 'user budget provenance')
    deployed = read(args.release / 'RUN_VERSION.json')
    check(deployed['run_commit'] == old['run_commit'] == RUN and len(deployed['files']) == 539,
          'frozen runtime export')
    # Extra pinned inputs bind the current budget and its source without
    # changing the frozen export or asking the external reviewer to approve them.
    deployment = {**deployed, 'files': [*deployed['files'], identity(approved), identity(approval)]}
    dep_path = args.output / 'preparation-deployment.json'
    durable_write(dep_path, deployment)
    result = {'status':'RUNNING', 'before':before, 'commands':[], 'negative_checks':[]}
    receipt_path = args.output / 'dry_run.json'

    def cmd(label, argv, exit_expected=0, contains=None, env=None):
        run = subprocess.run(argv, text=True, capture_output=True, stdin=subprocess.DEVNULL,
                             check=False, env=env, timeout=270)
        result['commands'].append(dict(label=label, argv=argv, exit_code=run.returncode,
                                       stdout=run.stdout, stderr=run.stderr))
        durable_write(receipt_path, result)
        check(run.returncode == exit_expected and
              (not contains or contains in run.stderr), label + ' unexpected return/result')
        return run.stdout

    def reject(name, action, text):
        try:
            action()
        except (ValueError, PermissionError) as error:
            check(text in str(error), name + ' reached wrong guard: ' + str(error))
            result['negative_checks'].append(dict(name=name, status='REJECTED', error=str(error)))
        else:
            raise AssertionError(name + ' was accepted')

    space = lambda location: shutil.disk_usage(location)._asdict()
    result['resource_observation'] = dict(utc=datetime.now(timezone.utc).isoformat(),
        disk=space(old['root']), support_disk=space(args.output), gpu_query_only=subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader'],text=True))
    argv = [sys.executable, str(scripts/'prepare_production_plan.py'), '--submission-plan',
            '--deployment', str(dep_path), '--budget', str(approved), '--prelaunch-review', str(review_path)]
    for key in ('root','data_root','source_root','model','python','runner','precision_approval','task_id'):
        argv += ['--'+key.replace('_','-'), old[key]]
    argv += ['--gpu-uuids', *old['gpu_uuids']]
    cmd('production plan generator', argv)
    plan = control.validate_plan(read(path))
    plan_identity = identity(path)
    canonical = digest(plan)
    check(plan['scope']=='FORMAL_SUBMIT' and plan['config_sha256']==CONFIG and
          plan['run_commit']==plan['prediction_run_commit']==RUN, 'actual final plan identity')
    check(plan['inputs']==deployment['files'] and plan['stage_seconds']==STAGE_CAPS and
          plan['budget']==GROUP_CAPS, 'actual approved budget and input bindings')
    cpu_caps = sum(v for k, v in STAGE_CAPS.items() if control.group(k)=='cpu')
    check(cpu_caps==GROUP_CAPS['cpu']==5400 and
          sum(GROUP_CAPS[g] for g in ('train_normal','s3','s4','cpu'))==199800, 'group arithmetic')
    docs = control.approval_documents(plan, require_pass=True)
    training_sha = identity(scripts/'formal_train.py')['sha256']
    def training_review(p):
        return verify_web_review(str(p), identity(p)['sha256'], RUN, CONFIG, training_sha)
    check(training_review(review_path)==docs['prelaunch_review'] and
          docs['prelaunch_review']['formal_launch_authorized'] is False, 'positive review identity')

    cli = cmd('actual controller CLI prepare', [sys.executable, str(scripts/'production_control.py'),
                                               'prepare','--plan',str(path)])
    check(json.loads(cli)==dict(status='PREPARED_NOT_SUBMITTED', plan_sha256=canonical),
          'CLI canonical digest')
    reject('missing authorization through controller function',
           lambda: control.authorize(plan, None, None), 'separate user launch authorization')
    reject('budget approval cannot launch',
           lambda: control.authorize(plan, str(approval), identity(approval)['sha256']),
           'authorization belongs to a different experiment')
    fixture = args.output / control.SYNTHETIC
    fixture.mkdir()
    historical = fixture / 'old-experiment-invalid.json'
    durable_write(historical, dict(experiment_id='beauty-a0-seed42-original', scope=control.SYNTHETIC))
    reject('old experiment through controller', lambda: control.authorize(
        plan,str(historical),identity(historical)['sha256']), 'authorization belongs to a different experiment')
    reject('old experiment through direct training guard', lambda: verify_launch_authorization(
        str(historical),identity(historical)['sha256'],RUN,CONFIG), 'authorization belongs to a different experiment')
    for name, changes, control_error, train_error in (
        ('wrong review kind', {'review_kind':'INTERNAL_NONAUTHOR'},
         'G3 web second review evidence required','external webpage second review required'),
        ('wrong evidence SHA', {'evidence_files':[{
            **docs['prelaunch_review']['evidence_files'][0], 'sha256':'0'*64}]},
         'artifact hash mismatch','web review evidence member identity mismatch'),
    ):
        bad = fixture / (name.replace(' ','-')+'.json')
        durable_write(bad, {**docs['prelaunch_review'], **changes, 'test_fixture':control.SYNTHETIC})
        invalid = deepcopy(plan)
        invalid['prelaunch_review'] = str(bad)
        invalid['approval_files']['prelaunch_review'] = identity(bad)
        reject(name+' through controller', lambda: control.approval_documents(invalid,True),control_error)
        reject(name+' through direct training', lambda: training_review(bad),train_error)

    output = Path(plan['root'])/'attempts'/'train'/'a0'
    cmd('production worker CLI missing authorization',
        [sys.executable,str(scripts/'production_pipeline.py'),'--plan',str(path),'--stage','train',
         '--output',str(output)],1,'separate user launch authorization and user-pinned digest required')
    _, catalog, users, assets = catalog_and_users(plan['data_root'])
    check(len(catalog)==12101 and len(users)==22363 and BEAMS==20, 'full formal scope')
    runner = runpy.run_path(plan['runner'], run_name='readonly_formal_request_validation')
    gpu = ('train','normal','s3','s4')
    commands, requests = {}, {}
    for stage in control.STAGES:
        cap = min(STAGE_CAPS[stage],GROUP_CAPS[control.group(stage)],GROUP_CAPS['total'])-GROUP_CAPS['cleanup_reserve']
        req, out = control.request(plan,stage,0,cap,path)
        runner['verify'](req)  # validates request only; no submit, no controller
        check(req['env']['CUDA_VISIBLE_DEVICES']==('1' if stage in gpu else '') and
              '--authorization' not in req['argv'], 'request scope/environment')
        requests[stage] = {k:req[k] for k in ('job_id','argv','env','timeout_seconds')}
        requests[stage].update(input_count=len(req['inputs']), request_template_sha256=digest(req))
        if stage in gpu:
            commands[stage] = compute_argv(plan, stage, Path(out))
            check('--nproc_per_node=1' in commands[stage] and '--resume' not in commands[stage]
                  and not any('preflight' in a.lower() for a in commands[stage]), 'single-rank formal argv')
            if stage != 'train':
                check(str(training_root(plan)/'best_adapter') in commands[stage], 'formal best adapter dependency')
    check('--verify-normal-off' in commands['normal'] and
          all('--verify-normal-off' not in commands[s] for s in ('s3','s4')), 'normal OFF check')
    train_env = {**os.environ,'RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'1'}
    train = commands['train']
    check(train[-4::2] == ['--formal-authorization','--formal-authorization-sha256'],
          'expected training authorization placeholders')
    cmd('direct train CLI missing authorization', train[:1]+train[6:-4],
        1, 'new G3 formal launch authorization required', env=train_env)
    budget_test = budget_checks(control,plan,fixture,read,durable_write)
    after = snapshot(plan, control.EXPERIMENT_REGISTRY, identity, digest, read)
    check(after['root_children']==['plan.json','preparation-plan.json'] and
          before['old_draft']==after['old_draft'] and after['current_plan']==plan_identity and
          before['current_plan'] in (None,plan_identity),
          'plan isolation and immutability')
    result.update(status='PASS', scope='FORMAL_PLAN_CPU_DRY_RUN_ONLY', after=after,
        argv=sys.argv, cwd=os.getcwd(), interpreter=sys.executable, run_commit=RUN,
        plan=plan_identity, canonical_digest=canonical, config=identity(plan['config']),
        plan_input_count=len(plan['inputs']), budget=plan['budget'], stage_seconds=plan['stage_seconds'],
        cpu_stage_caps_total_seconds=cpu_caps, cpu_group_headroom_seconds=GROUP_CAPS['cpu']-cpu_caps,
        retry_fixture=budget_test, execution_profile=plan['execution_profile'],
        formal_epoch_steps=STEPS, users=len(users), catalog_size=len(catalog), beams=BEAMS,
        assets=assets, base_model=verify_base_model(plan['model']), review=identity(review_path),
        stage_order=control.STAGES, request_templates=requests, compute_argv=commands,
        formal_experiment_started=False, formal_launch_authorized=False, formal_jobs_submitted=0,
        imports=[identity(sys.modules[n].__file__) for n in ('production_control','production_pipeline',
                  'formal_control','formal_train','formal_evaluate','g3_identity','production_identity')],
        checker=identity(__file__), max_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        max_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    durable_write(receipt_path,result)
    print(json.dumps({k:result[k] for k in ('status','plan','canonical_digest','formal_jobs_submitted')}))


if __name__ == '__main__':
    main()
