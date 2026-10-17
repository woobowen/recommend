#!/usr/bin/env python3
"""Read-only final request binding and independent recompilation audit."""
from pathlib import Path
import copy,datetime,hashlib,json,sys,subprocess
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;SUPPORT=HERE.parents[1]
REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0');sys.path.insert(0,str(REPO))
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
from experiments.a2_fd_recovery_20261018 import build
from experiments.launch_20261017 import queue
from experiments.launch_20261017.run import template_plan_sha256
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
dep=read(SUPPORT/'source-deployment.json');qd=read(SUPPORT/'queue-deployment.json');local=Path(qd['local_root']);config=read(local/'queue-config.json');frozen=read(local/'frozen-evidence.json');wrapped=read(local/'wrapped-requests.json');supervisor=read(local/'supervisor-request.json');scene=read(SUPPORT/'initial-scene.json')
old_deployment=read(HERE/'actual-spec-recompile/spec-index.json')['source_deployment']
old_files={r['path']:r['sha256'] for r in old_deployment['manifest']['files']}
new_files={r['path']:r['sha256'] for r in dep['manifest']['files']}
assert set(old_files)==set(new_files)
source_changes=sorted(p for p in old_files if old_files[p]!=new_files[p])
assert source_changes==['experiments/launch_20261017/queue_runner.py']
for rel,digest in new_files.items():
 b=subprocess.run(['git','show',dep['manifest']['source_commit']+':'+rel],cwd=REPO,check=True,capture_output=True).stdout
 assert hashlib.sha256(b).hexdigest()==digest
patch=subprocess.run(['git','diff',old_deployment['manifest']['source_commit'],dep['manifest']['source_commit'],'--','experiments/launch_20261017/queue_runner.py'],cwd=REPO,check=True,capture_output=True).stdout
(HERE/'outer_runner_source_diff-v2.patch').write_bytes(patch)
expected_config=sys.argv[sys.argv.index('--expected-config-sha')+1]
assert sha(local/'queue-config.json')==qd['config_sha256']==expected_config
expected=[{'path':qd['remote_config'],'sha256':sha(local/'queue-config.json'),'local':str(local/'queue-config.json')},
 {'path':qd['remote_root']+'/plan.json','sha256':sha(local/'plan.json'),'local':str(local/'plan.json')},
 {'path':dep['root']+'/DEPLOYMENT.json','sha256':dep['manifest_sha256']},
 {'path':qd['outer_runner'],'sha256':qd['outer_runner_sha256']},
 {'path':qd['outer_runner']+'.deployment.json','sha256':config['runner']['deployment_sha256']},
 {'path':config['runner']['script'],'sha256':config['runner']['script_sha256']}]
expected.extend({'path':qd['remote_root']+'/specs/'+p.name,'sha256':sha(p),'local':str(p)} for p in sorted((local/'specs').glob('*.json')))
expected.extend(frozen)
code='''from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,time
expected=EXPECTED
cfg_path=Path(CONFIG)
cfg=json.loads(cfg_path.read_text());out={'files':[],'locks':{},'new_job_conflicts':[],'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
seen=set()
for r in expected:
 p=Path(r['path'])
 if str(p) in seen:continue
 seen.add(str(p));assert p.is_file() and not p.is_symlink(),str(p)
 b=p.read_bytes();actual=hashlib.sha256(b).hexdigest();assert actual==r['sha256'],str(p)
 if 'bytes' in r:assert len(b)==r['bytes']
 entry={'path':str(p),'bytes':len(b),'sha256':actual}
 if p.suffix=='.json':entry['json']=json.loads(b)
 out['files'].append(entry)
locks=[Path(cfg['gpu_locks_dir'])/(g+'.lock') for g in cfg['approved_gpu_uuids']]
for r in cfg['recovery']['frozen_evidence']:
 p=Path(r['path'])
 if p.name=='queue-status.json':locks.append(p.parent/'supervisor.lock')
for p in locks:
 assert p.is_file() and not p.is_symlink(),str(p)
 with p.open('r+') as h:
  fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);out['locks'][str(p)]='AVAILABLE';fcntl.flock(h,fcntl.LOCK_UN)
for jid in [cfg['queue_id']]+[s['request']['job_id'] for s in cfg['stages']]:
 if (Path(cfg['runner']['job_root'])/jid).exists():out['new_job_conflicts'].append(jid)
assert not out['new_job_conflicts']
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=cfg['snapshot']['root'];env['PYTHONDONTWRITEBYTECODE']='1'
python=cfg['stages'][0]['request']['argv'][0]
for mode in ['validate','render-requests']:
 p=subprocess.run([python,'-B','-m','experiments.a2_fd_recovery_20261018.controller',mode,'--config',str(cfg_path)],cwd=cfg['snapshot']['root'],env=env,capture_output=True,text=True,timeout=50)
 assert p.returncode==0,(mode,p.returncode,p.stdout,p.stderr)
 out[mode]=json.loads(p.stdout)
outer=Path(OUTER_RUNNER)
module_spec=importlib.util.spec_from_file_location('read_only_outer_admission',outer)
module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
module.verify(SUPERVISOR)
out['outer_verify']={'status':'PASS','function':'original deployed outer runner verify(request), no submit','runner_sha256':hashlib.sha256(outer.read_bytes()).hexdigest()}
out['GPU_or_model_calls']=0;out['real_submit_calls']=0
print(json.dumps(out))
'''.replace('EXPECTED',repr(expected)).replace('CONFIG',repr(qd['remote_config'])).replace('OUTER_RUNNER',repr(qd['outer_runner'])).replace('SUPERVISOR',repr(supervisor))
(HERE/'final_remote_query-v2.py').write_text(code)
if '--reuse-remote' in sys.argv:remote=read(HERE/'final_remote_binding-v2.json')
else:
 remote=json.loads(Transport('new-server').python(code,timeout=60));dump(HERE/'final_remote_binding-v2.json',remote)
by={r['path']:r for r in remote['files']}
assert by[qd['remote_config']]['json']==config and remote['render-requests']==wrapped
assert remote['validate']==read(local/'queue-validation-server.json')==qd['validation']
assert config['recovery']['frozen_evidence']==frozen
review_path=qd['remote_root']+'/launch-review.json';review=by[review_path]['json'];dump(HERE/'final_bound_launch_review-v2.json',review)
assert review['status']=='SOURCE_AND_LAUNCH_ADMITTED_CONDITIONALLY' and review['fd_engineering_status']=='FD_ENGINEERING_EQUIVALENCE_PASS' and review['future_results_approved'] is False
plan=read(local/'plan.json');assert review['plan']=={'path':qd['remote_root']+'/plan.json','sha256':sha(local/'plan.json'),'bytes':(local/'plan.json').stat().st_size} or (review['plan']['path']==qd['remote_root']+'/plan.json' and review['plan']['sha256']==sha(local/'plan.json'))
assert review['source_python_sha256']==plan['source_python_sha256']=={r['path']:r['sha256'] for r in dep['manifest']['files'] if r['path'].endswith('.py')}

# Independent recompilation uses the real frozen inputs and actual review bytes.
work=HERE/'final-real-recompile-v2';work.mkdir(exist_ok=True)
scene['shared_config_path']=config['recovery']['shared_config_path']
templates,replan=build.specs(dep,scene,work,qd['remote_root']);assert replan==plan
review_file=work/'launch-review.json';dump(review_file,review)
assert sha(review_file)==next(r['sha256'] for r in frozen if r['path']==review_path)
regenerated=build.compile_queue(dep,scene,work,qd['remote_root'],review_file,frozen)
# Recompiling later legitimately decreases this clock-derived admission horizon.
clock_horizon=regenerated['max_queue_seconds'];regenerated['max_queue_seconds']=config['max_queue_seconds']
assert sorted(regenerated['storage_roots'])==sorted(config['storage_roots'])
regenerated['storage_roots']=config['storage_roots']
assert regenerated==config
assert 0<clock_horizon<=config['max_queue_seconds']
specs=[]
for name in templates:
 p=local/'specs'/(name+'.json');q=work/'specs'/(name+'.json')
 assert p.read_bytes()==q.read_bytes()
 value=read(p);assert template_plan_sha256(value) in plan['template_plan_sha256']
 assert value['launch_review']['path']==review_path and value['launch_review']['sha256']==sha(review_file)
 specs.append({'stage':name,'bytes':p.stat().st_size,'sha256':sha(p),'template_identity':template_plan_sha256(value)})

# The actual wrappers must each exactly bind this config, snapshot and sole stage command.
by_stage={s['id']:s for s in config['stages']};assert len(by_stage)==len(wrapped)==11
for request in wrapped:
 stage=by_stage[request['queue_stage']['id']];inputs={x['path']:x['sha256'] for x in request['inputs']}
 assert request['job_id']==stage['request']['job_id'] and request['capture_version']==stage['request']['capture_version']
 assert request['queue_stage']['command']==stage['request']['argv']
 assert request['argv']==[stage['request']['argv'][0],'-u',dep['root']+'/experiments/a2_fd_recovery_20261018/controller.py','run-stage','--config',qd['remote_config'],'--stage',stage['id']]
 assert inputs[qd['remote_config']]==qd['config_sha256'] and inputs[qd['remote_root']+'/specs/'+stage['id']+'.json']==sha(local/'specs'/(stage['id']+'.json'))
 assert request['env']['CUDA_VISIBLE_DEVICES']==','.join(queue.gpus(stage))
assert supervisor['job_id']==config['queue_id'] and supervisor['cwd']==dep['root'] and supervisor['capture_version']==wrapped[0]['capture_version']
assert supervisor['argv']==[wrapped[0]['argv'][0],'-B','-m','experiments.a2_fd_recovery_20261018.controller','supervise','--config',qd['remote_config']]
assert supervisor['env']['CUDA_VISIBLE_DEVICES']=='' and supervisor['timeout_seconds']<=config['max_queue_seconds']
for x in supervisor['inputs']:assert by[x['path']]['sha256']==x['sha256']
old_config=by[config['recovery']['shared_config_path']]['json'];old_state=by[config['recovery']['shared_state_path']]['json']
historical=queue.accounting(old_config,old_state);assert historical==scene['accounting']
assert config['gpu_budget_seconds']==old_config['gpu_budget_seconds']-historical['used_gpu_request_seconds']
assert all(config['budget_groups'][g]==old_config['budget_groups'][g]-historical['groups'][g] for g in config['budget_groups'])
assert config['runner']==old_config['runner'] and config['gpu_locks_dir']==old_config['gpu_locks_dir'] and config['free_space']==old_config['free_space']
assert config['recovery']['original_absolute_queue_deadline']==old_config['continuation']['original_absolute_queue_deadline']=='2026-10-19T05:40:30.012007+00:00'
failure=by[config['runner']['job_root']+'/ga-20261017-sportsa2-quantize-v1/status.json']['json'];assert failure['status']=='INTERRUPTED' and failure['exit_code']==-9 and failure['cleanup']['verified']
assert by[config['runner']['job_root']+'/ga-20261017-sportsa2-encode-v1/status.json']['json']['status']=='SUCCEEDED'
assert old_state['status']=='TERMINAL_PENDING_RESULT_REVIEW' and not any(s['status']=='SUBMITTING' for s in old_state['stages'].values())
assert all(v=='AVAILABLE' for v in remote['locks'].values())
assert sha(HERE.parent/'sports_a1/REPORT.md')=='2d797794dfc44ca723dc1df0a3547a2a5b6b59705b0545995c19b809efaa5d72'
outer_wall_admissible=bool(remote['outer_verify']['status']=='PASS')
out={'status':'A2_ONLY_FINAL_SCHEDULER_ADMISSION_PASS' if outer_wall_admissible else 'PARTIAL_OUTER_SUPERVISOR_WALL_NOT_ADMITTED','reviewer':'/root/sports_a1_result_audit','scope':'exact immutable A2-only scheduler requests before submission; independent nonauthor, not scientific result approval','scientific_result_status':'SCIENTIFIC_RESULT_NOT_YET_AVAILABLE','source_commit':dep['manifest']['source_commit'],'snapshot_root':dep['root'],'snapshot_sha256':dep['manifest']['snapshot_sha256'],'config':{'path':qd['remote_config'],'sha256':qd['config_sha256'],'bytes':(local/'queue-config.json').stat().st_size},'launch_review':{'path':review_path,'sha256':sha(review_file),'fd_status':review['fd_engineering_status'],'future_results_approved':False},'independent_recompilation':'all config fields exact after normalizing shrinking clock-derived max_queue_seconds and set-derived storage_root enumeration order; all 11 final spec bytes exact','specs':specs,'frozen_files_verified':len(frozen),'all_frozen_source_status_and_review_hashes_verified':True,'source_revision_from_c31':{'changed_files':source_changes,'all_scientific_and_stage_controller_bytes_unchanged':True,'all_manifest_git_bytes_verified':len(new_files),'outer_diff_sha256':sha(HERE/'outer_runner_source_diff-v2.patch')},'wrapped_requests_exact_to_server_render':True,'supervisor_request_verified':True,'outer_supervisor':{'request_sha256':sha(local/'supervisor-request.json'),'wall_seconds':supervisor['timeout_seconds'],'runner_verified_allowance':'precise new CPU-controller long-wall whitelist','wall_admissible':outer_wall_admissible},'live_server_validate':remote['validate'],'historical_accounting':historical,'prior_seconds':config['recovery']['prior_gpu_request_seconds'],'preserved_absolute_deadline':config['recovery']['original_absolute_queue_deadline'],'locks':remote['locks'],'new_job_conflicts':remote['new_job_conflicts'],'captured_at_utc':remote['captured_at_utc'],'submissions_by_this_reviewer':0,'GPU_or_model_calls':0,'A1_report_unchanged':True,'limitations':['GPU admission and early effective quantizer/FD/step health require actual startup checks.','This approves the scheduler binding only; final quantization, training, testing and attention results require independent result review.'],'evidence':[{'path':str(HERE/f),'sha256':sha(HERE/f)} for f in ['scheduler_cpu_review.json','actual_specs_review-v2.json','actual_server_source_binding-v2.json','final_remote_binding-v2.json']]}
out.update(formal_submission_admitted=outer_wall_admissible,
           future_results_approved=False,
           config_sha256=qd['config_sha256'],
           outer_request_file_sha256=sha(local/'supervisor-request.json'))
dump(HERE/'final-request-review-v2.json',out);print(json.dumps(out,indent=2))
