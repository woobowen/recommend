from pathlib import Path
from collections import Counter
from urllib.parse import unquote,urlsplit
import datetime as dt,hashlib,json,re,sys
repo=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
support=Path('/home/addaswsw/recommend-support/grid-formal-20261015')
jobs=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server')
sys.path.insert(0,str(repo))
from experiments.formal_20261015.deploy import source_records
from experiments.beauty_extensions.authorization import canonical_plan
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def evidence(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
records=source_records()
code={r['path']:r['sha256'] for r in records if r['path'].endswith('.py')}
assert len(code)==171
source_review=load(support/'source-formal-admission.json')
assert code['experiments/formal_20261015/train.py']==source_review['scope']['reviewed_entry_sha256']
resources=load(support/'source-resource-review.json')
oldpath=Path('/home/addaswsw/recommend-support/wmh-a800-audit-20261010/analysis/dataset-check-v2/files.jsonl')
oldcounts={(r['domain'],r['split'],Path(r['path']).name):r['records'] for r in map(json.loads,oldpath.read_text().splitlines())}
reg=jobs/'gf-20261015-config-roundtrip-r4'/'retrieved'
rs=load(reg/'status.json')
assert rs['status']=='SUCCEEDED' and rs['exit_code']==0 and rs['cleanup']['verified']
outdir=support/'mlp-admission'
outdir.mkdir(exist_ok=True)
for domain,nrows,nfiles,U,ci,raw,acc,cadence,cap,workers in [('toys',19412,152,255,2,32,48,4800,512,8),('sports',35598,279,256,3,16,16,1600,128,0)]:
 sp=repo/'experiments/formal_20261015/specs'/f'{domain}-mlp.json';spec=load(sp)
 assert spec['domain']==domain and spec['route']=='MLP' and spec['formal_approved'] is True
 assert spec['formal_training_allowed'] and spec['formal_protocol_user_approved']
 assert spec['approved_protocol']=={'raw_microbatch':raw,'accumulate_grad_batches':acc,'val_check_interval':cadence,'collate_window_cap':cap}
 assert spec['seed']==42 and spec['generation_k']==10 and spec['shuffle_n']==0 and spec['hard_cap_hours']==12
 assert not any(spec.get(k) for k in ['checkpoint','resume_checkpoint','engineering_overlay','preflight_rows'])
 jid=f'gf-20261015-{domain}-metadata-v1';p=jobs/jid/'retrieved';o=p/'output'
 status=load(p/'status.json');request=load(p/'request.json');accepted=load(p/'accepted.json')
 inv=load(o/'invocation.json');result=load(o/'RESULT.json');dep=inv['code_identity']
 assert status['job_id']==jid and status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
 assert status['worker_may_be_live'] is False and status['cleanup']['last_observation']['verified_empty']
 assert load(p/'cleanup.json')==status['cleanup']
 assert digest(request)==accepted['request_sha256']
 assert request['env']['CUDA_VISIBLE_DEVICES']==inv['physical_gpu']=='' and inv['phase']=='metadata'
 assert request['uid']==1005 and request['host']=='zwlab-01' and result['status']=='PASS' and result['model_loaded'] is False
 assert {r['path']:r['sha256'] for r in dep['files'] if r['path'].endswith('.py')}==code
 assert request['capture_version']==dep['source_commit']+'/'+dep['snapshot_sha256']
 assert request['cwd'].endswith('/snapshots/'+dep['snapshot_sha256'][:16])
 assert hashlib.sha256(json.dumps({'source_commit':dep['source_commit'],'files':dep['files']},sort_keys=True).encode()).hexdigest()==dep['snapshot_sha256']
 inp={r['path']:r['sha256'] for r in request['inputs']}
 assert inp[request['argv'][-1]]==inv['spec_sha256']
 assert inp[request['cwd']+'/DEPLOYMENT.json']==hashlib.sha256((json.dumps(dep,indent=2)+'\n').encode()).hexdigest()
 assert {k:v for k,v in spec.items() if k not in ['evaluation_index','test_index','attention_cohort','nonauthor_gate']}==inv['spec']
 for k in ['config','sid']:assert spec[k]==inv['spec'][k] and inp[spec[k]['path']]==spec[k]['sha256']
 exported=load(p/'export-manifest.json')['files']
 assert len(exported)==len({r['path'] for r in exported})
 for r in exported:
  f=p/r['path'];assert not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts
  assert f.is_file() and not f.is_symlink() and f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
 ec=load(o/'evaluation_coverage.json');tc=load(o/'test_coverage.json');cohort=load(o/'fixed_attention_samples.json')
 assert result['evaluation']==ec and result['test']==tc
 index_reports={};allrows={};keysets={}
 for label,split,cov,filename in [('evaluation_index','evaluation',ec,'evaluation_target_index.jsonl'),('test_index','testing',tc,'test_target_index.jsonl')]:
  f=o/filename;asset=spec[label]
  assert asset['path']==f'/home/bwwu/grid-audits/sid-extensions-p0/jobs/{jid}/output/{filename}'
  assert asset['sha256']==sha(f) and asset['bytes']==f.stat().st_size
  assert cov['rows']==cov['unique_users']==nrows and cov['sid']==spec['sid'] and cov['model_execution'] is False
  lc=cov['loader'];order=lc['file_order']
  assert lc['num_workers']==workers and lc['batch_size_per_device']==8 and lc['drop_last'] is False
  assert lc['sequence_length']==120 and lc['assign_all_files_per_worker'] is False
  assert len(order)==len(set(order))==nfiles
  expected={i:oldcounts[(domain,split,Path(unquote(urlsplit(uri).path)).name)] for i,uri in enumerate(order)}
  assert sum(expected.values())==nrows
  assert all(unquote(urlsplit(uri).path).startswith(spec['data_dir']+'/'+split+'/') for uri in order)
  rows=[json.loads(line) for line in f.read_text().splitlines()]
  counts=Counter();pairs=Counter();keys=[];last=0
  assert len(rows)==nrows
  for i,r in enumerate(rows):
   key=r['sample_key'];target=r['target'];fi=r['source_file_index']
   assert re.fullmatch('[0-9a-f]{24}',key) and len(target)==4 and all(type(v)is int and 0<=v<U for v in target)
   assert type(fi)is int and 0<=fi<nfiles and fi in (last,last+1)
   assert r['row_index']==counts[fi] and (i!=0 or fi==0)
   last=fi;counts[fi]+=1;pairs[(key,tuple(target))]+=1;keys.append(key)
   if split=='testing':assert r['global_row_index']==i and 2<=r['raw_item_count']<=21
  assert dict(counts)==expected and len(set(keys))==len(pairs)==nrows
  if split=='testing':
   assert cov['key_set_sha256']==digest(sorted(set(keys))) and cov['duplicate_user_rows']==0
   strings=Counter({canonical([k,list(t)]).decode():v for (k,t),v in pairs.items()})
   assert cov['key_target_multiset_sha256']==digest(sorted(strings.items()))
   assert {r['source_file_index']:r['rows'] for r in cov['rows_per_file']}==expected
   assert cov['target_index_sha256']==asset['sha256'] and cov['target_index']==asset['path']
   assert cov['original_label_targets_verified'] and cov['complete_target_hidden_groups_verified']
  else:
   assert cov['target_index']=={k:asset[k] for k in ['path','sha256']}
   assert cov['original_NextKTokenMasking_and_hidden_target_verified']
  keysets[split]=set(keys);allrows[split]=rows
  index_reports[split]={'asset':asset,'rows':len(rows),'unique_user_keys':len(set(keys)),'unique_key_target_pairs':len(pairs),'files':nfiles,'per_file_contiguous_row_positions_and_counts_exact':True,'targets_H4_integer_range':True,'independent_key_target_multiset_sha256':digest(sorted([[k,list(t),v] for (k,t),v in pairs.items()]))}
 assert keysets['evaluation']==keysets['testing']
 ca=spec['attention_cohort'];cf=o/'fixed_attention_samples.json'
 assert ca['path']==f'/home/bwwu/grid-audits/sid-extensions-p0/jobs/{jid}/output/fixed_attention_samples.json'
 assert ca['sha256']==sha(cf) and ca['bytes']==cf.stat().st_size
 assert result['cohort']=={k:ca[k] for k in ['path','sha256']}
 assert cohort['selected_users']==cohort['metadata_rows_scanned']==32 and cohort['model_outputs_inspected'] is False
 assert cohort['sid']==spec['sid'] and cohort['loader']==ec['loader'] and cohort['user_id'] is None and cohort['raw_user_ids_exported'] is False
 assert len(set(cohort['sample_keys']))==len(cohort['sample_keys'])==32
 assert cohort['sample_keys']==[r['sample_key'] for r in allrows['evaluation'][:32]]
 assert cohort['targets']==[r['target'] for r in allrows['evaluation'][:32]]
 assert len(cohort['samples'])==len(cohort['input_ids'])==len(cohort['attention_mask'])==32
 for i,s in enumerate(cohort['samples']):
  row=allrows['evaluation'][i];ids=cohort['input_ids'][i];mask=cohort['attention_mask'][i];hist=s['visible_history_item_ids']
  assert s['sample_key']==row['sample_key'] and s['target']==row['target'] and s['source_file_index']==row['source_file_index'] and s['row_index']==row['row_index'] and s['global_row_index']==i
  assert s['target_hidden'] and s['input_dtype']==s['target_dtype']=='torch.int64'
  valid=sum(v>=0 for v in hist)
  assert len(hist)==30 and len(ids)==len(mask)==120 and valid==s['raw_item_count']-1 and 1<=valid<=20
  assert all(0<=v<(11924 if domain=='toys' else 18357) for v in hist[:valid]) and hist[valid:]==[-1]*(30-valid)
  assert mask==[1]*(4*valid)+[0]*(120-4*valid)
  assert all(type(v)is int and 0<=v<U for v in ids[:4*valid]) and ids[4*valid:]==[-1]*(120-4*valid)
 case=reg/'output'/f'config-{ci:02d}';rr=load(case/'RESULT.json')
 before=load(case/'used_config_before.json');used=load(case/'used_config_after_reload.json');protocol=load(case/'domain_protocol.json')
 assert rr['status']=='PASS' and rr['domain']==domain and rr['spec_sha256']==inv['spec_sha256']
 assert rr['model_constructions']==rr['GPU_model_calls']==0 and before==used
 assert rr['whole_template_roundtrip_exact'] and rr['actual_used_subtrees_resolve_exact']
 assert rr['run_py_sha256']==code['experiments/formal_20261015/run.py'] and rr['used_config_sha256']==hashlib.sha256(json.dumps(used,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
 assert chr(36)+chr(123) not in json.dumps(used) and '???' not in json.dumps(used)
 trainer=used['trainer']
 assert trainer['accumulate_grad_batches']==acc and trainer['val_check_interval']==cadence and trainer['max_steps']==320000 and trainer['max_epochs']==10
 assert trainer['precision']=='32-true' and trainer['devices']==1 and trainer['strategy']=='auto' and not any(k.startswith('limit_') for k in trainer)
 assert used['optim']['optimizer']['_target_']=='torch.optim.Adam' and used['optim']['optimizer']['lr']==.001 and used['optim']['optimizer']['weight_decay']==.0001 and used['optim']['scheduler'] is None
 assert used['callbacks']['model_checkpoint']['monitor']=='val/recall@5' and used['callbacks']['model_checkpoint']['mode']=='max'
 assert used['callbacks']['early_stopping']['patience']==10 and used['callbacks']['early_stopping']['min_delta']==0
 assert protocol['raw_microbatch']==raw and protocol['collate_window_cap']==cap and protocol['accumulate_grad_batches']==acc and protocol['validation_interval_microbatches']==cadence and protocol['hard_cap_hours_including_full_test']==12
 assert all(v['num_workers']==workers for v in protocol['source_worker_protocol_preserved'].values())
 for k in ['config','sid']:assert {q:rr['assets'][k][q] for q in ['path','sha256']}==spec[k]
 scenario=resources['conditional_cost_scenarios'][domain]
 assert resources['full_domain_counts'][domain]['evaluation']['rows']==resources['full_domain_counts'][domain]['testing']['rows']==nrows
 assert scenario['with_explicit_engineering_factor_and_startup_hours']<12
 plan=canonical_plan(spec);plan_sha=digest(plan)
 report={'schema':'grid-formal-independent-MLP-admission-v1','status':'PREFLIGHT_GATE_PASS','reviewer':'/root/source_conditions_review','domain':domain,'stage_id':spec['stage'],'reviewed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'plan_sha256':plan_sha,'canonical_plan_excluded_fields':['nonauthor_gate','formal_approved'],'plan':plan,'reviewed_spec':evidence(sp),'source_python_sha256':code,'source_python_count':171,'source_python_map_sha256':digest(code),'assets':{k:spec[k] for k in ['config','sid']},'metadata_execution':{'job_id':jid,'status':'SUCCEEDED','exit_code':0,'cleanup_verified':True,'worker_may_be_live':False,'CUDA_VISIBLE_DEVICES':'','model_loaded':False,'request_fingerprint_sha256':accepted['request_sha256'],'request_capture_version':request['capture_version'],'source_commit':dep['source_commit'],'snapshot_sha256':dep['snapshot_sha256'],'deployment_manifest_sha256':inp[request['cwd']+'/DEPLOYMENT.json'],'all171Python_exact_to_admitted_current_source':True,'exported_members_hash_verified':len(exported),'prototype_protocol_exact_before_only_new_output_refs':True,'started_at':status['started_at'],'ended_at':status['ended_at']},'full_indexes':index_reports,'evaluation_and_test_user_key_sets_exact':True,'cohort':{'asset':ca,'users':32,'unique_keys':32,'exact_complete_evaluation_index_prefix':True,'selected_before_model_outputs':True,'input_shape':[32,120],'mask_shape':[32,120],'target_shape':[32,4],'complete_item_groups_and_target_hiding_verified':True,'visible_item_axis_and_padding_verified':True,'original_metadata_item_to_SID_checks':'Actual metadata execution invokes original NextKTokenMasking and selected-SID last-item comparison; source hashes exact. Reviewer does not rerun raw TFRecord decode/SID deserialization.'},'CPU_approved_training_configuration':{'job_id':'gf-20261015-config-roundtrip-r4','case':ci,'actual_used_subtrees_save_reload_exact':True,'raw_microbatch':raw,'accumulate_grad_batches':acc,'val_microbatches':cadence,'collate_window_cap':cap,'workers_train_val_test':workers,'scratch_full_parameter_training':True,'engineering_checkpoint_reuse':False,'optimizer':'Adam lr.001/wd.0001/no scheduler','seed':42,'K':10,'prefix_constraint':False,'selection':'full val/recall@5 max; patience10/min_delta0','max_steps':320000,'max_epochs':10,'precision':'32-true','hard_resource_cap_total_seconds_including_test':43200},'metadata_config_boundary':'Metadata reads val/test using the source base template, train=False. Its unused Toys trainacc16/val1600 fields do not supersede the admitted train.configure overlay48/4800, actually executed by the successful CPU roundtrip case.','research_identity':spec['research_identity'],'inherited_engineering':'Fixed93567ea7f61e970c448838443595242d0b9beef0 passed real2+1updates/domain, gradients, strict restore and bounded evaluator; no preflight rerun. Source train.py remains the separately nonauthor-reviewed af7ec5159e0fbe9f7330d1172b81356920a1c626db9e15a6b339e6b074c79a1e.','provenance_limits_retained':'Toys upper-stage factory/path contradiction and unique historical producer gap remain disclosed; Sports is an approved new Qwen/RK+original itemMLP experiment, not an invented historical Sports reproduction.','resource_admission':{'actual_full_counts_match_prior_resource_scenarios':True,'reference_scenario_only_updates':3600,'reference_scenario_full_validations':36,'reference_scenario_complete_tests':1,'linear_scenario_hours':scenario['linear_scenario_hours'],'conservative_planning_scenario_hours':scenario['with_explicit_engineering_factor_and_startup_hours'],'assumptions':'No training speedup; largest observed Beauty full validation scaled by domain counts; explicit25percent planning overhead and600s startup/cleanup, not statistical bounds.','obviously_exceeds12h_in_reference_scenario':False,'completion_by12h_guaranteed':False,'3600_is_not_stop_setting':True,'source_max_steps320000_is_not_predicted_endpoint':True,'hard_cap_behavior':'Runner-start12h total includes all full validations, strict best/test/recompute and saving;180s checkpoint reserve; exhaustionINCOMPLETE, no autoextension or reduction of full coverage.','prelaunch_snapshot':evidence(support/'domain-prelaunch-resources.json'),'launch_refresh':'Controller/deployer must refresh actual GPU/disk availability at submission.'},'budget':'Inherit original3/3 correction budget; separate single user-approved configuration-save exception passed real4case CPU regression. No reset or new repair granted.','scope':'Independent technical preflight admission only; not a user authorization token or claim of formal training completion. Root separately binds user token, immutable commit/deployment and starts approved worker.','reviewer_actions':{'models_or_GPU_runs':0,'workers_submitted':0,'raw_TFRecord_rescans':0,'repository_or_asset_edits':0,'user_token_signed':False},'evidence':[evidence(p/n) for n in ['request.json','status.json','accepted.json','cleanup.json','export-manifest.json','output/invocation.json','output/RESULT.json','output/evaluation_coverage.json','output/test_coverage.json','output/fixed_attention_samples.json','output/evaluation_target_index.jsonl','output/test_target_index.jsonl']]+[evidence(case/n) for n in ['RESULT.json','domain_protocol.json','used_config_before.json','used_config_after_reload.json']]+[evidence(support/'source-formal-admission.json'),evidence(support/'source-resource-review.json')]}
 assert canonical_plan(load(sp))==plan
 path=outdir/f'{domain}-gate-review.json'
 assert not path.exists(),'Preserve existing independent admission identity'
 with path.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 saved=load(path);assert saved['plan_sha256']==digest(canonical_plan(load(sp))) and saved['source_python_sha256']==code
 print(json.dumps({'domain':domain,'status':'PREFLIGHT_GATE_PASS','path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'plan_sha256':plan_sha,'python_files':171,'rows_per_split':nrows,'cohort':32}),flush=True)

