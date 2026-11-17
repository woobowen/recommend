"""Role checks for documentation metadata, against frozen recorded evidence."""
import hashlib,json

def validate_beauty(row,state,plan,train_request,config_bytes,primary,summary,supp_request):
 def require(ok,msg):
  if not ok:raise ValueError(msg)
 cond=row['condition']; require(cond in ('normal','s3','s4'),'condition')
 require(row['run']['commit']==plan['run_commit']==plan['prediction_run_commit']==primary['run_commit']==primary['evaluation_commit'],'run commit mismatch')
 require(row['config']['sha256']==plan['config_sha256']==hashlib.sha256(config_bytes).hexdigest(),'config SHA mismatch')
 require(row['metrics']==primary['metrics']==summary['conditions'][cond]['metrics'],'primary metrics mismatch')
 require(row['predictions']==primary['prediction_files'],'prediction identity mismatch')
 require(row['run']['evaluation']['evaluation_config_recorded']==primary['config'] and row['run']['evaluation']['evaluation_script_recorded']==primary['script'],'evaluation config or script mismatch')
 require(primary['assets']==summary['audit']['data_assets'],'data asset identity mismatch')
 require(row['assets_recorded']['best_checkpoint']==summary['best_selection'],'best adapter selection mismatch')
 for key,role,stage in [('training_run','training','train'),('generation','generation',cond),('evaluation','evaluation',cond+'_eval')]:
  node=row[key] if key=='training_run' else row['run'][key]
  matches=[a for a in state['attempts'] if a['stage']==stage];require(len(matches)==1,'stage missing '+stage); ref=matches[0]
  require(node['role']==role and node['stage']==stage,'role/stage mismatch '+stage)
  for f in ('job_id','attempt','status'):require(node[f]==ref[f],f+' mismatch '+stage)
  for f in ('request_path','request_sha256'):require(node['recorded_'+f]==ref[f],f+' mismatch '+stage)
  require(node['worker_argv']==ref['runner']['worker']['argv'],'worker argv mismatch '+stage)
  require(node['state']['selector']=='attempts[stage='+stage+']' and node['state']['commit']=='12eb8b176b156f3be7d7533ee8f443db02441303' and node['state']['path'].endswith('/evidence/original/control/state.json'),'selector mismatch '+stage)
  require(state['stages'][stage]['attempt_index']==state['attempts'].index(ref) and state['stages'][stage]['status']=='COMPLETE','state completion mismatch '+stage)
  if role!='training':
   require(node['request'] is None,'unarchived request claimed present '+stage)
   require(node['request_availability']=='bytes not archived; path and sha recorded in state, not recomputed','unarchived request falsely verified '+stage)
  else:
   require(node['request'] is not None and node['request'].get('role')=='training' and node['request']['path']=='lc_rec/a0_beauty/evidence/formal-launch-20261116/request.json','training request role')
   require(train_request['beauty_stage']=='train' and train_request['job_id']==ref['job_id'] and train_request['capture_version']==plan['run_commit'],'training request identity')
   require(hashlib.sha256(json.dumps(train_request,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==ref['request_sha256'],'training canonical SHA mismatch')
 if cond=='normal':require(row['pairing']['role']=='no_shuffle_pair','normal pair mismatch')
 if cond=='s3':require(row['pairing']['role']=='original_s3_paired' and row['pairing']['result']['path'].endswith('/original/s3/paired_comparison.json'),'S3 pair mismatch')
 if cond=='s4':require(row['pairing']['role']=='s4_supplemental_pair' and row['pairing']['commit']==supp_request['capture_version'] and row['pairing']['job_id']==supp_request['job_id'],'S4 supplement mismatch')
 return True
