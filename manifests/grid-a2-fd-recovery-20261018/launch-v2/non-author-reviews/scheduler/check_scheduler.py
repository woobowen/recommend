#!/usr/bin/env python3
"""Bounded local scheduler compilation/admission tests; all runs are fixtures."""
from pathlib import Path
import copy,datetime,hashlib,importlib,importlib.util,json,sys,tempfile
from unittest import mock
sys.dont_write_bytecode=True
REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO))
from experiments.a2_fd_recovery_20261018 import build
from experiments.launch_20261017 import queue,run

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def record(p):return dict(path=str(p),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def fail(name,call):
    try:call()
    except (ValueError,AssertionError,RuntimeError,FileNotFoundError,KeyError) as error:
        checks.append({'name':name,'status':'EXPECTED_REJECTION','exception':str(error)});return
    raise AssertionError('unexpectedly accepted: '+name)
def diff(a,b,p=''):
    if type(a)!=type(b):return [(p,a,b)]
    if isinstance(a,dict):
        out=[]
        for k in a.keys()|b.keys():
            if k not in a or k not in b:out.append((p+'.'+k,a.get(k),b.get(k)))
            else:out.extend(diff(a[k],b[k],p+'.'+k))
        return out
    if a!=b:return [(p,a,b)]
    return []

scene_original=load('/home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018/initial-scene.json')
scene=copy.deepcopy(scene_original)
work=Path(tempfile.mkdtemp(prefix='fixture-',dir=OUT));snap=work/'snapshot';compiled=work/'compiled';jobs=work/'jobs';history_dir=work/'historical-state'
checks=[]
assert scene_original['new_id_conflicts']==[] and scene_original['project_processes']==[] and scene_original['old_completed_jobs_not_succeeded']==[]
assert all(x=='AVAILABLE' for x in scene_original['locks'].values())
shared_state=copy.deepcopy(scene_original['states']['ga-20261018-shared-queue-v1'])
assert shared_state['status']=='TERMINAL_PENDING_RESULT_REVIEW' and not any(x['status']=='SUBMITTING' for x in shared_state['stages'].values())
history_calc=queue.accounting(scene_original['config'],shared_state)
assert history_calc==scene_original['accounting']
prior=scene_original['config']['prior_gpu_request_usage']['seconds']
assert abs(history_calc['used_gpu_request_seconds']+prior-41896.507458)<1e-6
assert scene_original['config']['continuation']['original_absolute_queue_deadline']=='2026-10-19T05:40:30.012007+00:00'
old_start=datetime.datetime.fromisoformat(scene_original['states']['ga-20261017-queue-v1']['started_at'])
assert old_start+datetime.timedelta(seconds=scene_original['old_config']['max_queue_seconds'])==datetime.datetime.fromisoformat(scene_original['config']['continuation']['original_absolute_queue_deadline'])
checks.append({'name':'actual_history_accounting_and_original_deadline','status':'PASS','historical_stage_seconds':history_calc['used_gpu_request_seconds'],'prior_seconds':prior,'original_remaining_seconds':143100-history_calc['used_gpu_request_seconds']-prior})

# Remap control paths to an isolated local fixture; science and original history stay intact.
scene['config']['runner']['job_root']=str(jobs)
scene['config']['state_dir']=str(history_dir)
scene['shared_config_path']=str(work/'shared-config.json')
dump(Path(scene['shared_config_path']),scene['config'])
dump(history_dir/'queue-status.json',shared_state)
controller_src=REPO/'experiments/a2_fd_recovery_20261018/controller.py'
controller_copy=snap/'experiments/a2_fd_recovery_20261018/controller.py';controller_copy.parent.mkdir(parents=True,exist_ok=True);controller_copy.write_bytes(controller_src.read_bytes())
spec=importlib.util.spec_from_file_location('independent_fixture_controller',controller_copy)
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
source_files=[{'path':str(p.relative_to(REPO)),'sha256':sha(p),'bytes':p.stat().st_size} for p in [controller_src,REPO/'experiments/a2_fd_recovery_20261018/build.py',REPO/'experiments/a2_fd_recovery_20261018/adapter.py',REPO/'experiments/launch_20261017/run.py',REPO/'experiments/launch_20261017/a2_upstream.py',REPO/'experiments/launch_20261017/training_protocol.json']]
manifest={'source_commit':'a'*40,'snapshot_sha256':'b'*64,'files':source_files}
dump(snap/'DEPLOYMENT.json',manifest)
deployment={'root':str(snap),'manifest':manifest,'manifest_sha256':sha(snap/'DEPLOYMENT.json')}
compiled.mkdir()
templates,plan=build.specs(deployment,scene,compiled,str(compiled))
assert len(templates)==11 and set(templates)==controller.STAGES
old_by={x['id']:x for x in scene_original['config']['stages']}
capture='a'*40+'/'+'b'*64
original_capture=old_by['sportsa2-quantize']['request']['capture_version']
mapping=[(scene_original['config']['snapshot']['root'],str(snap)),(original_capture,capture)]+[(old_by[x]['request']['job_id'],build.job(x)) for x in controller.STAGES]
template_diffs={}
for name,actual in templates.items():
    original=load(REPO/'manifests/grid-a-launch-20261017/compiled/specs'/(name+'.json'))
    original.pop('launch_review',None)
    normalized=build.replace(original,mapping);normalized['capture_version']=capture;normalized['prerequisite_capture_versions']={}
    if name=='sportsa2-quantize':
        normalized['prerequisite_capture_versions']={old_by['sportsa2-encode']['request']['job_id']:original_capture}
        normalized['fd_engineering']={'purpose':'A2 DataLoader IPC/FD engineering adaptation','worker_hard_seconds':3300}
    assert actual==normalized,name
    template_diffs[name]=diff(original,actual)
    assert actual['seed']==42 and actual['generation_k']==10 and actual['research_identity']=='Sports-A2-T5RKMeans-DualTable-NewTraining'
    assert set(actual['prerequisite_capture_versions'])==({'ga-20261017-sportsa2-encode-v1'} if name=='sportsa2-quantize' else set())
    assert 'ga-20261017-sportsa2-quantize-v1' not in json.dumps(actual)
    for field in ['config','training_protocol']:
        assert actual[field]['sha256']==original[field]['sha256']
    if name!='sportsa2-quantize':assert actual['resolve']['sid']['job']==build.job('sportsa2-quantize')
    if name.startswith('sportsa2-eval') or name.startswith('sportsa2-attention'):
        assert actual['resolve']['checkpoint']==dict(job=build.job('sportsa2-train'),field='best_checkpoint')
    if name=='sportsa2-stats':assert actual['attention_jobs']==[build.job('sportsa2-attention-'+str(n)) for n in [0,3,4]]
checks.append({'name':'eleven_unique_templates_science_unchanged_only_encode_legacy','status':'PASS'})

review={'status':'SOURCE_AND_LAUNCH_ADMITTED_CONDITIONALLY','fd_engineering_status':'FD_ENGINEERING_EQUIVALENCE_PASS','future_results_approved':False,'source_python_sha256':plan['source_python_sha256'],'plan':record(compiled/'plan.json')}
dump(compiled/'launch-review.json',review)
frozen=[record(scene['shared_config_path']),record(history_dir/'queue-status.json')]
config=build.compile_queue(deployment,scene,compiled,str(compiled),compiled/'launch-review.json',frozen)
summary=queue.validate(config,(compiled/'queue-config.json').resolve())
assert summary['stages']==11 and summary['reserved_gpu_seconds']==59940
assert summary['groups']['sports-a2-rkmeans']==6840 and summary['groups']['engineering']==900 and summary['groups']['sports-a2-train']==43200 and summary['groups']['sports-a2-eval-attention']==9000
assert summary['sports_a2_reserved_cpu_seconds']==2400
assert config['gpu_budget_seconds']==101203.143115
assert abs(config['budget_groups']['sports-a2-rkmeans']-6897.950524)<1e-6
assert all(x['workflow']=='sports-a2' for x in config['stages']) and not any(x['id']=='sportsa2-encode' for x in config['stages'])
assert len({x['request']['job_id'] for x in config['stages']})==11
assert set(config['blocked_workflows'])==queue.WORKFLOWS-{'sports-a2'}
assert config['runner']==scene['config']['runner'] and config['gpu_locks_dir']==scene_original['config']['gpu_locks_dir']
assert config['free_space']==scene_original['config']['free_space'] and config['max_storage_bytes']==53687091200
for stage in config['stages']:
    old=old_by[stage['id']]
    for key in ['kind','workflow','budget_group','gpu_uuids','gpu_candidates','resource_policy']:assert stage.get(key)==old.get(key)
    assert set(stage['depends_on'])==set(old['depends_on'])-{'sportsa2-encode'}
    assert stage['request']['timeout_seconds']==(3420 if stage['id']=='sportsa2-quantize' else old['request']['timeout_seconds'])
    for dep in stage['depends_on']:
        gates=stage['gates'];root=str(jobs/build.job(dep))
        assert any(x['path']==root+'/status.json' and x['json_equals']=={'status':'SUCCEEDED','exit_code':0,'cleanup.verified':True} for x in gates)
        assert any(x['path']==root+'/output/RESULT.json' and x['json_equals'].get('capture_version')==capture for x in gates)
assert controller.lineage(config,live=False)==history_calc
checks.append({'name':'compile_resource_reservation_and_lineage','status':'PASS','reservation':summary,'quantizer_remaining_buffer_seconds':config['budget_groups']['sports-a2-rkmeans']-6840})

# Exercise pure lineage refusal paths without submitting or touching real paths.
for name,mut in [
 ('group_budget_reset',lambda c:c['budget_groups'].__setitem__('sports-a2-rkmeans',7200)),
 ('global_budget_reset',lambda c:c.__setitem__('gpu_budget_seconds',143041)),
 ('deadline_reset',lambda c:c['recovery'].__setitem__('original_absolute_queue_deadline','2026-10-20T05:40:30.012007+00:00')),
 ('old_job_reuse',lambda c:c['stages'][0]['request'].__setitem__('job_id',old_by[c['stages'][0]['id']]['request']['job_id'])),
 ('GPU_policy_change',lambda c:c['stages'][0].__setitem__('resource_policy',{})),
 ('stage_scope_change',lambda c:c['stages'].pop()),
 ('runner_change',lambda c:c['runner'].__setitem__('script','/fake/runner.py')),
 ('omitted_old_storage',lambda c:c.__setitem__('storage_roots',[])),
]:
    candidate=copy.deepcopy(config);mut(candidate);fail(name,lambda c=candidate:controller.lineage(c,live=False))

# Historical live check is mocked strictly from captured terminal evidence.
received={}
for x in scene['config']['stages']:
    r=shared_state['stages'][x['id']].get('runner_status')
    if r:
        d=jobs/x['request']['job_id'];dump(d/'accepted.json',{});received[d.name]=r
received.update(scene_original['runner_statuses'])
orig_runner_call=queue.runner_call
queue.runner_call=lambda c,m,p:copy.deepcopy(received[Path(p).name])
controller.lineage(config,live=True)
checks.append({'name':'live_lineage_terminal_fixture','status':'PASS','real_runner_calls':0})
occupied=jobs/build.job('sportsa2-quantize');occupied.mkdir(parents=True)
fail('new_id_already_exists',lambda:controller.lineage(config,live=True))
# Keep the collision fixture as evidence and use live=False for subsequent tests.
old_state_hash=sha(history_dir/'queue-status.json')
badstate=copy.deepcopy(shared_state);next(iter(badstate['stages'].values()))['status']='SUBMITTING';dump(history_dir/'queue-status.json',badstate)
badconfig=copy.deepcopy(config);badconfig['recovery']['frozen_evidence']=[record(scene['shared_config_path']),record(history_dir/'queue-status.json')]
fail('unresolved_historical_SUBMITTING',lambda:controller.lineage(badconfig,live=False))
dump(history_dir/'queue-status.json',shared_state);assert sha(history_dir/'queue-status.json')==old_state_hash
queue.runner_call=orig_runner_call

# Resolve actual new links against local fake success artifacts, never loading a model.
asset=work/'fixture-asset.bin';asset.write_bytes(b'fixture only, not a model or SID\n');asset_record=record(asset)
fake_jobroot=work/'resolve-jobs';fixture_specs={}
for name,t in templates.items():
    t=copy.deepcopy(t);t['job_root']=str(fake_jobroot);fixture_specs[name]=t
    jid=build.job(name);d=fake_jobroot/jid
    dump(d/'request.json',{'capture_version':capture});dump(d/'status.json',{'status':'SUCCEEDED','exit_code':0,'cleanup':{'verified':True}})
    res={'capture_version':capture,'workflow':'Sports-A2','sid':asset_record,'checkpoint':asset_record,'best_checkpoint':asset_record,'training_complete':True,'formal':True,'test_complete':True,'status':'COMPLETE','config':t['config'],'shuffle_n':t['shuffle_n']}
    dump(d/'output/RESULT.json',res)
    if name=='sportsa2-metadata':
        for f in ['test_target_index.jsonl','evaluation_target_index.jsonl','fixed_attention_samples.json']:dump(d/'output'/f,{'fixture':True})
enc=fake_jobroot/'ga-20261017-sportsa2-encode-v1';dump(enc/'request.json',{'capture_version':original_capture});dump(enc/'status.json',{'status':'SUCCEEDED','exit_code':0,'cleanup':{'verified':True}});dump(enc/'output/RESULT.json',{'capture_version':original_capture,'workflow':'Sports-A2','embedding_asset':asset_record})
resolved={}
for name,t in fixture_specs.items():
    value=run.resolve(t);resolved[name]=value
    if name=='sportsa2-quantize':assert value['embedding']==asset_record
    else:assert value['sid']==asset_record
    if name.startswith('sportsa2-eval') or name.startswith('sportsa2-attention'):assert value['checkpoint']==asset_record
checks.append({'name':'dynamic_encode_new_sid_best_metadata_evaluation_resolution','status':'PASS','resolved_stages':len(resolved),'model_or_GPU_calls':0})
bad=copy.deepcopy(fixture_specs['sportsa2-train']);bad['resolve']['sid']['job']='ga-20261017-sportsa2-quantize-v1'
fail('old_failed_quantizer_link',lambda:run.resolve(bad))
training=fake_jobroot/build.job('sportsa2-train')/'output/RESULT.json';res=load(training);res['sid']={'path':str(asset),'sha256':'0'*64};dump(training,res)
fail('best_checkpoint_sid_mismatch',lambda:run.resolve(fixture_specs['sportsa2-eval-0']))

# Shared admission preserves peak+margin and both UUIDs; foreign presence is allowed.
quant=next(s for s in config['stages'] if s['id']=='sportsa2-quantize')
cards={g:{'free_mib':8192,'used_mib':16000,'utilization':40,'processes':[{'pid':99999999,'used_mib':16000}]} for g in quant['gpu_uuids']}
assert queue.resource_ok(quant,cards)
low=copy.deepcopy(cards);low[quant['gpu_uuids'][0]]['free_mib']=8191;assert not queue.resource_ok(quant,low)
busy=copy.deepcopy(cards);busy[quant['gpu_uuids'][0]]['utilization']=51;assert not queue.resource_ok(quant,busy)
missing=copy.deepcopy(cards);missing.pop(quant['gpu_uuids'][0]);assert not queue.resource_ok(quant,missing)
checks.append({'name':'shared_GPU_peer_presence_allowed_peak_margin_both_UUIDs_enforced','status':'PASS','foreign_peer_mib_each':16000,'required_free_mib_each':8192})

# Install the real wrapper and show a received failed attempt never gets resubmitted.
controller.install(config,compiled/'queue-config.json',live=False)
failed_request=queue.wrapped_request(config,compiled/'queue-config.json',quant)
failed_job=jobs/failed_request['job_id'];dump(failed_job/'request.json',failed_request);dump(failed_job/'accepted.json',{'fixture':True})
failure={'status':'FAILED','exit_code':1,'cleanup_verified':True,'started_at':'2026-10-17T18:00:00+00:00','ended_at':'2026-10-17T18:00:01+00:00','request_sha256':hashlib.sha256(queue.canonical(failed_request)).hexdigest()}
calls=[]
def fixture_runner(c,mode,path):
    calls.append((mode,str(path)))
    assert mode=='query' and Path(path)==failed_job,'fixture forbids every new submission'
    return copy.deepcopy(failure)
with mock.patch.object(queue,'runner_call',fixture_runner),mock.patch.object(queue,'gpu_snapshot',lambda:{}),mock.patch.object(queue,'disk_snapshot',lambda c:{'ok':True}):
    assert queue.supervise(config,compiled/'queue-config.json')==0
    fail('terminal_controller_automatic_restart',lambda:queue.supervise(config,compiled/'queue-config.json'))
state=load(Path(config['state_dir'])/'queue-status.json')
assert state['status']=='TERMINAL_PENDING_RESULT_REVIEW' and state['stages']['sportsa2-quantize']['status']=='BLOCKED_NO_RETRY'
assert sum(v['status']=='BLOCKED_DEPENDENCY' for v in state['stages'].values())==10
assert calls==[('query',str(failed_job))]
checks.append({'name':'existing_failed_quantizer_no_resubmit_all_10_dependents_blocked','status':'PASS','fixture_runner_queries':len(calls),'fixture_submits':0})

# The launch gate must fit the complete unchanged stage wall before the old deadline.
empty_gates=copy.deepcopy(quant);empty_gates['gates']=[]
assert queue.check_gates(empty_gates)==[]
absolute=datetime.datetime.fromisoformat(config['recovery']['original_absolute_queue_deadline'])
too_late=copy.deepcopy(empty_gates);too_late['request']['timeout_seconds']=int((absolute-datetime.datetime.now(datetime.timezone.utc)).total_seconds())+120
fail('new_submission_cannot_fit_original_absolute_deadline',lambda:queue.check_gates(too_late))
class NearDeadline(datetime.datetime):
    @classmethod
    def now(cls,tz=None):return absolute-datetime.timedelta(seconds=30)
with mock.patch.object(controller.dt,'datetime',NearDeadline):assert queue.queue_should_stop(config,0)
checks.append({'name':'old_absolute_deadline_60_second_reserve','status':'PASS','deadline':absolute.isoformat()})

out={'status':'A2_ONLY_SCHEDULER_SOURCE_CPU_REVIEW_PASS_CONDITIONAL_DEPLOYMENT','reviewer':'/root/sports_a1_result_audit','fixture_path':str(work),'fixture_only':True,'real_runner_submissions':0,'GPU_or_model_calls':0,'checks':checks,'template_differences':template_diffs,'reservation':summary,'source_files':source_files,'scene_sha256':sha('/home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018/initial-scene.json'),'pending':['real fixed commit/deployment manifest and uploaded spec/config/review hashes','real pre-submit controller validate with original runner on new-server','new controller acceptance independent of SSH and active quantizer FD/step health']}
dump(OUT/'scheduler_cpu_review.json',out)
print(json.dumps(out,indent=2))
