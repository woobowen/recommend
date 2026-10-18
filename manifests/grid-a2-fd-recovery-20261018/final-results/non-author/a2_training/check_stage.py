#!/usr/bin/env python3
"""Check bounded observed A2 training evidence, without claiming final success."""
from pathlib import Path
import hashlib,json,math,subprocess
P=Path(__file__).resolve().parent
O=Path(json.loads((P/'latest_observation_path.json').read_text())['path'])
A1=P.parent/'sports_a1/raw/ga-20261017-sportsa1-train-v1/output'
REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def compare(a,b,path=''):
 if type(a)!=type(b):return [(path,a,b)]
 if isinstance(a,dict):
  assert set(a)==set(b)
  return [d for k in a for d in compare(a[k],b[k],path+'.'+k)]
 if isinstance(a,list):
  assert len(a)==len(b)
  return [d for i,(x,y) in enumerate(zip(a,b)) for d in compare(x,y,path+'['+str(i)+']')]
 return [(path,a,b)] if a!=b else []
obs=read(O/'observation.json');inv=read(O/'output/invocation.json');request=read(O/'request.json');spec=inv['effective_spec'];status=read(O/'status.json')
assert obs['source_commit']=='85b77f5e2a5b497fdc1f620649ba488fc783a245'
assert inv['code_identity']==request['capture_version']==obs['source_commit']+'/'+obs['snapshot_sha256']
assert request['cwd']=='/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/22286966cd544773'
assert spec['workflow']=='Sports-A2' and spec['seed']==42 and spec['research_identity']=='Sports-A2-T5RKMeans-DualTable-NewTraining'
assert not spec.get('checkpoint') and not spec.get('resume_checkpoint') and spec['shuffle_n']==0
assert request['timeout_seconds']==43200 and len(request['queue_stage']['gpu_uuids'])==1
assert spec['sid']==obs['quantizer_result']['sid'] and spec['sid']['sha256']==obs['training_sid']['sha256']=='06cfa6ba0f64d6c68e0e94c4111edd51d086796e8cdb474ac3fcd424a4a2adf0'
assert obs['quantizer_status']['status']=='SUCCEEDED' and obs['quantizer_status']['exit_code']==0 and obs['quantizer_status']['cleanup']['verified']
for f in obs['source_files']:
 b=subprocess.run(['git','show',obs['source_commit']+':'+f['path']],cwd=REPO,check=True,capture_output=True).stdout
 assert hashlib.sha256(b).hexdigest()==f['sha256']
used=read(O/'output/used_config.json');old_used=read(A1/'used_config.json')
diff=compare(old_used,used)
expected_paths={'.model.codebooks._args_[0].file_path','.trainer.default_root_dir','.logger.csv.save_dir','.callbacks.model_checkpoint.dirpath'}|{'.datamodule.'+s+'_dataloader_config.dataset_config.semantic_id_map.sequence_data._args_[0].file_path' for s in ['train','val','test']}
assert {x[0] for x in diff}==expected_paths
for path,before,after in diff:
 if path.endswith('file_path'):assert after==spec['sid']['path']
 else:assert after.startswith('/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261018-sportsa2-train-fd-recovery-v1/output')
structure=read(O/'output/structure.json');assert structure==read(A1/'structure.json')
assert structure['parameters_total']==structure['parameters_trainable']==13259136 and structure['shared_sid_table'] is False
loader=read(O/'output/actual_train_loader.json');assert loader==read(A1/'actual_train_loader.json')
assert loader['fresh_optimizer_state_empty'] and loader['raw_microbatch']==16 and loader['collate_window_cap']==512
protocol=read(O/'output/training_protocol.json');old_protocol=read(A1/'training_protocol.json')
assert protocol['protocol']==old_protocol['protocol'] and protocol['source_config']==old_protocol['source_config']
assert protocol['all_parameters_fresh_random'] and protocol['no_test_execution_in_this_stage'] and protocol['selected_sid']==spec['sid']
assert protocol['asset']['sha256']==spec['training_protocol']['sha256']=='9d3823c2ada022f479fb3b8ac4e36bc2cf09267281db5b15edc636bec0a45c7e'
updates=rows(O/'output/optimizer_updates.jsonl');assert updates
windows=0;losses=[];gradients={}
for i,u in enumerate(updates,1):
 assert u['optimizer_update']==i and u['cumulative_microbatches']==i*16
 assert len(u['expanded_windows'])==len(u['returned_training_losses'])==16 and all(0<n<=512 for n in u['expanded_windows'])
 windows+=sum(u['expanded_windows']);assert windows==u['cumulative_expanded_windows']
 assert u['optimizer']['step_before']==i-1 and u['optimizer']['all_available_gradients_finite'] and u['optimizer']['optimizer_covers_every_trainable_parameter']
 losses.extend(u['returned_training_losses'])
 for k,v in u['optimizer']['watched_gradient_norms'].items():
  assert math.isfinite(v) and v>0;gradients.setdefault(k,[]).append(v)
assert all(math.isfinite(v) for v in losses)
validations=rows(O/'output/validations.jsonl')
for i,v in enumerate(validations,1):
 assert v['validation_number']==i and v['global_step']==100*i and v['targets']==35598 and v['batches']==4450 and v['exact_key_target_multiset']
 assert v['expected_index']['sha256']==spec['evaluation_index']['sha256']
 assert all(math.isfinite(x) for x in v['metrics'].values())
deadline=read(O/'output/deadline.json');assert deadline['allowance_seconds']==43200 and deadline['safety_reserve_seconds']==180 and deadline['training_termination_condition'] is False
out={'status':'A2_TRAINING_STAGE_EVIDENCE_CHECKED_RUNNING','final_training_pass':False,'reviewer':'/root/sports_a1_result_audit','captured_at_utc':obs['captured_at_utc'],'observation_directory':str(O),'atomic_snapshot':False,'job_id':request['job_id'],'runner_status':status['status'],'source_commit':obs['source_commit'],'snapshot_sha256':obs['snapshot_sha256'],'training_sid':obs['training_sid'],'effective_config_A1_comparison':'identical except chosen SID path in model+three loaders and output directories','all_effective_config_differences':diff,'from_scratch_evidence':['frozen _fresh_model instantiate with seed42 and fit ckpt_path=None; no checkpoint/resume field in actual spec','actual fresh_optimizer_state_empty=true','all parameters trainable and separate encoder/decoder SID tables','monotone actual Adam updates starting at1'], 'architecture':{k:v for k,v in structure.items() if k!='state_dict_shapes'},'observed_optimizer_updates':len(updates),'actual_microbatches_in_prefix':len(updates)*16,'expanded_windows_in_prefix':windows,'finite_returned_loss_range':[min(losses),max(losses)],'watched_gradient_ranges':{k:[min(v),max(v)] for k,v in gradients.items()},'completed_full_validations_observed':len(validations),'first_full_validation':validations[0] if validations else 'PENDING','latest_progress':read(O/'output/progress.json'),'deadline':deadline,'GPU_or_model_calls':0,'submit_or_stop_calls':0,'pending_terminal_checks':['full optimizer/validation streams through actual final step','authorized termination, best validation selection and patience count','best and final checkpoint direct CPU strict load/finite/state/embedded SID/Adam/callback checks','terminal exit/cleanup, artifact manifest hashes and actual resource budget'],'limitations':['From-scratch proof uses frozen code, actual spec, original empty optimizer state and real updates; no full initial random tensor dump exists.','This read is a bounded non-atomic training observation; no final PASS or exact resume claim.']}
(P/'stage_review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['all_effective_config_differences','architecture','deadline']},indent=2))
