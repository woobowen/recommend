"""One CPU-only, read-only A2 training terminal audit; no load before success."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from pathlib import Path
import datetime,hashlib,json,math,sys
JOB=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261018-sportsa2-train-fd-recovery-v1')
SNAP=Path('/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/22286966cd544773')
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
status=read(JOB/'status.json')
if status['status'] in ['ACCEPTED','RUNNING']:
 print(json.dumps({'status':'A2_TRAINING_NOT_TERMINAL_NO_FINAL_REVIEW','runner_status':status['status'],'GPU_or_forward_calls':0,'CPU_model_instantiations':0}));sys.exit(0)
if not (status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']):
 print(json.dumps({'status':'A2_TRAINING_TERMINAL_INCOMPLETE_OR_FAILED','runner_status':status['status'],'exit_code':status['exit_code'],'training_result':read(JOB/'output/training_result.json') if (JOB/'output/training_result.json').is_file() else None,'GPU_or_forward_calls':0,'CPU_model_instantiations':0}));sys.exit(0)

out={'reviewer':'/root/sports_a1_result_audit','job_id':JOB.name,'source_commit':'85b77f5e2a5b497fdc1f620649ba488fc783a245','GPU_or_forward_calls':0,'CPU_model_instantiations':0,'filesystem_mutations':0}
request=read(JOB/'request.json');inv=read(JOB/'output/invocation.json');spec=inv['effective_spec'];result=read(JOB/'output/training_result.json');used=read(JOB/'output/used_config.json')
dep=read(SNAP/'DEPLOYMENT.json')
assert dep['source_commit']==out['source_commit'] and request['cwd']==str(SNAP)
assert request['capture_version']==inv['code_identity']==dep['source_commit']+'/'+dep['snapshot_sha256']
assert request['timeout_seconds']==43200 and len(request['queue_stage']['gpu_uuids'])==1
assert spec['seed']==42 and not spec.get('checkpoint') and not spec.get('resume_checkpoint') and spec['shuffle_n']==0
assert result['training_complete'] and result['fresh_initialization'] and result['full_parameter_training'] and result['independent_sid_tables']
assert result['status']=='TRAINING_COMPLETE_EVALUATION_PENDING' and not result['test_complete']
assert result['sid']==spec['sid'] and digest(spec['sid']['path'])==spec['sid']['sha256']=='06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0'
assert used['optim']=={'optimizer':{'_target_':'torch.optim.Adam','_partial_':True,'lr':0.001,'weight_decay':0.0001},'scheduler':None}
assert used['trainer']['precision']=='32-true' and used['trainer']['accumulate_grad_batches']==16 and used['trainer']['val_check_interval']==1600
assert (used['trainer']['max_steps'],used['trainer']['max_epochs'])==(320000,10)
assert not used['model']['should_check_prefix'] and used['model']['top_k_for_generation']==10 and not used['model']['shuffle_embedding_on_test']

# Stream every optimizer event once; never store the full stream in memory.
updates=JOB/'output/optimizer_updates.jsonl';assert updates.stat().st_size<=64*1024**2
count=windows=0;loss_min=math.inf;loss_max=-math.inf;grads={}
with updates.open() as f:
 for line in f:
  r=json.loads(line);count+=1
  assert count<=320000 and r['optimizer_update']==count and r['cumulative_microbatches']==count*16
  assert len(r['expanded_windows'])==len(r['returned_training_losses'])==16 and all(0<n<=512 for n in r['expanded_windows'])
  windows+=sum(r['expanded_windows']);assert windows==r['cumulative_expanded_windows']
  opt=r['optimizer'];assert opt['step_before']==count-1 and opt['all_available_gradients_finite'] and opt['optimizer_covers_every_trainable_parameter']
  for loss in r['returned_training_losses']:
   assert math.isfinite(loss);loss_min=min(loss_min,loss);loss_max=max(loss_max,loss)
  for key,value in opt['watched_gradient_norms'].items():
   assert value is not None and math.isfinite(value) and value>0
   bounds=grads.setdefault(key,[value,value]);bounds[0]=min(bounds[0],value);bounds[1]=max(bounds[1],value)
assert count==result['global_step'] and count*16==result['actual_microbatches'] and windows==result['expanded_windows']
vp=JOB/'output/validations.jsonl';assert vp.stat().st_size<=10*1024**2
vals=[json.loads(x) for x in vp.read_text().splitlines()];assert vals and len(vals)==result['completed_full_validations']
best_score=-math.inf;best_step=None;wait=0
for i,v in enumerate(vals,1):
 assert v['validation_number']==i and v['global_step']==100*i and v['targets']==35598 and v['batches']==4450 and v['exact_key_target_multiset']
 assert v['expected_index']['sha256']==spec['evaluation_index']['sha256'] and all(math.isfinite(x) for x in v['metrics'].values())
 score=v['metrics']['val/recall@5']
 if score>best_score:best_score=score;best_step=v['global_step'];wait=0
 else:wait+=1
assert digest(spec['evaluation_index']['path'])==spec['evaluation_index']['sha256']
assert best_score==result['best_checkpoint']['score']
termination=result['termination'];assert termination in ['patience10_complete_validation_early_stop','configured_max_steps','configured_max_epochs']
if termination=='patience10_complete_validation_early_stop':assert wait==result['early_stopping']['wait_count']==10 and vals[-1]['global_step']==count
elif termination=='configured_max_steps':assert count==320000
else:assert result['current_epoch']>=10

# Check all selected original artifacts against the immutable terminal manifest.
artifacts={r['path']:r for r in status['artifacts']};selected=['output/RESULT.json','output/training_result.json','output/used_config.json','output/effective_config.yaml','output/structure.json','output/actual_train_loader.json','output/training_protocol.json','output/optimizer_updates.jsonl','output/validations.jsonl','output/best_checkpoint_load.json']
for asset in [result['best_checkpoint'],result['final_recovery']['checkpoint']]:
 p=Path(asset['path']);assert p.is_file() and p.stat().st_size==asset['bytes'] and digest(p)==asset['sha256'];selected.append(str(p.relative_to(JOB)))
checked=[]
for name in selected:
 p=JOB/name;r=artifacts[name];assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'];checked.append(r)
begin=datetime.datetime.fromisoformat(status['started_at']);end=datetime.datetime.fromisoformat(status['ended_at']);wall=(end-begin).total_seconds()
assert 0<wall<=43200 and end<=datetime.datetime.fromisoformat('2026-10-19T05:40:30.012007+00:00')

# Load only after terminal-success and stream/resource checks have passed.
sys.path.insert(0,str(SNAP))
from experiments.beauty_extensions.common import bind_source,load_checkpoint
bind_source('A')
import src.utils.custom_hydra_resolvers
import torch,hydra
from omegaconf import OmegaConf
torch.set_num_threads(2)
sid=torch.load(spec['sid']['path'],map_location='cpu',weights_only=True);assert tuple(sid.shape)==(4,18357) and sid.dtype==torch.int64
cfg=OmegaConf.load(JOB/'output/effective_config.yaml');assert cfg.seed==42 and cfg.ckpt_path is None and cfg.train is True and cfg.test is False
model=hydra.utils.instantiate(cfg.model);out['CPU_model_instantiations']=1
assert sum(p.numel() for p in model.parameters())==13259136 and all(p.requires_grad for p in model.parameters())
state=load_checkpoint(result['best_checkpoint']['path']);assert state['global_step']==best_step
loaded=model.load_state_dict(state['state_dict'],strict=True);actual=model.state_dict()
assert not loaded.missing_keys and not loaded.unexpected_keys and set(actual)==set(state['state_dict'])
assert all(torch.equal(v.detach().cpu(),state['state_dict'][k].detach().cpu()) for k,v in actual.items())
assert all(torch.isfinite(v).all() for v in actual.values() if v.is_floating_point())
assert torch.equal(state['hyper_parameters']['codebooks'].cpu(),sid)
enc,dec=[model.get_embedding_table(k).weight for k in ['encoder','decoder']]
assert tuple(enc.shape)==tuple(dec.shape)==(1024,128) and enc.data_ptr()!=dec.data_ptr() and not torch.equal(enc,dec)
def optimizer_check(state,step):
 assert len(state['optimizer_states'])==1
 opt=state['optimizer_states'][0];steps=[int(x['step'].item()) for x in opt['state'].values()]
 assert len(steps)==128 and min(steps)==max(steps)==step
 assert all(g['lr']==0.001 and g['weight_decay']==0.0001 for g in opt['param_groups'])
 return {'parameter_states':len(steps),'all_parameter_steps':step}
best_opt=optimizer_check(state,best_step)
best_details={'checkpoint':result['best_checkpoint'],'global_step':best_step,'keys_shapes_values_exact':True,'missing_keys':[],'unexpected_keys':[],'state_tensors':len(actual),'all_finite':True,'embedded_sid_exact':True,'independent_tables':True,'optimizer':best_opt}
del model,actual,state,enc,dec
state=load_checkpoint(result['final_recovery']['checkpoint']['path']);assert state['global_step']==count
assert torch.equal(state['hyper_parameters']['codebooks'].cpu(),sid)
assert all(torch.isfinite(x).all() for x in state['state_dict'].values() if x.is_floating_point())
final_opt=optimizer_check(state,count)
callbacks={}
for name,v in state['callbacks'].items():
 callbacks[name]={k:(x.item() if torch.is_tensor(x) and x.numel()==1 else str(x) if torch.is_tensor(x) else x) for k,x in v.items() if k in ['wait_count','patience','best_score','best_model_score','best_model_path','stopped_epoch']}
 if name.startswith('EarlyStopping') and termination=='patience10_complete_validation_early_stop':assert v['wait_count']==v['patience']==10
 if name.startswith('ModelCheckpoint'):assert v['best_model_path']==result['best_checkpoint']['path'] and float(v['best_model_score'])==best_score
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();live=[]
for ident in [status['supervisor'],status['worker']]+status.get('worker_group',{}).get('known_members',[]):
 try:
  text=(Path('/proc')/str(ident['pid'])/'stat').read_text();fields=text[text.rfind(')')+2:].split()
  if ident['boot_id']==boot and str(ident['start_ticks'])==fields[19] and fields[0]!='Z':live.append(ident['pid'])
 except FileNotFoundError:pass
assert not live
assert 0<result['gpu_peak_allocated_bytes']<=result['gpu_peak_reserved_bytes']<=24*1024**3
out.update(status='A2_TRAINING_NONAUTHOR_REVIEW_PASS',termination=termination,final_step=count,best_step=best_step,best_validation_recall5=best_score,full_validations=len(vals),full_validation_targets_each=35598,full_validation_exact_multisets=True,actual_microbatches=count*16,expanded_windows=windows,loss_range=[loss_min,loss_max],gradient_ranges=grads,best=best_details,final_checkpoint=result['final_recovery']['checkpoint'],final_optimizer=final_opt,final_callbacks=callbacks,original_artifacts_verified=checked,wall_GPU_request_seconds=wall,training_group_cap_seconds=43200,runner_status=status['status'],exit_code=0,cleanup_verified=True,sid=spec['sid'],no_automatic_resume=True)
out.update(original_live_processes=live,gpu_peak_allocated_bytes=result['gpu_peak_allocated_bytes'],gpu_peak_reserved_bytes=result['gpu_peak_reserved_bytes'])
print(json.dumps(out))
