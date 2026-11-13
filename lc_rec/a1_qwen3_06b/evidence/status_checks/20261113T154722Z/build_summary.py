"""Derive this one-time status report from captured evidence only; no SSH."""
from pathlib import Path
import datetime as dt,json,math,statistics
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
x=read('server_crosscheck.raw.json');obs=read('control_observation.raw.json');q=read('runner_observe.raw.json')
RUN='0e3e2c6f5c1a76152ea0228390d94ffc1ff75ba5'
assert q['status']=='RUNNING' and all(q[k] for k in ['worker_alive','supervisor_alive','workgroup_live'])
matches=[]
for saved in [q['supervisor'],q['worker'],*q['worker_group']['known_members']]:
    current=x['processes'][str(saved['pid'])]
    match=current['alive'] and all(str(current[k])==str(saved[k]) for k in ['pid','uid','start_ticks','boot_id'])
    assert match;matches.append({'pid':saved['pid'],'saved_identity_matches_current_proc':match})
assert all(r['run_commit']==RUN and r['status']=='TRAINING' for r in x['rank_reports'].values())
a=x['events']['0'];b=x['events']['1'];s=a['latest_optimizer_step']
assert s['step']==b['latest_optimizer_step']['step'] and s['epoch']==b['latest_optimizer_step']['epoch']
last=[v for v in a['recent_train_logs'] if 'loss' in v.get('values',{})][-1]
assert all(math.isfinite(float(v)) for v in [last['values']['loss'],last['values']['grad_norm'],s['grad_norm'],*s['lr']])
boundary=(s['step']//2060)*2060
steps=[r for r in a['recent_optimizer_steps'] if r['step']>boundary][-21:]
assert len(steps)==21
seconds=steps[-1]['elapsed_seconds']-steps[0]['elapsed_seconds'];updates=steps[-1]['step']-steps[0]['step'];seconds_per_update=seconds/updates
val=a['validation'];assert all(v['records']==49544 for v in val)
assert len(val)==len(b['validation']) and all(v['step']==w['step'] and v['metrics']['eval_loss']==w['metrics']['eval_loss'] and v['records']==w['records'] for v,w in zip(val,b['validation']))
assert all(c['status']=='PASS_SEAL_CONTRACT' for c in x['checkpoint_checks'])
cp=max(x['checkpoint_checks'],key=lambda c:c['seal']['global_step']);state=cp['trainer_state']
assert state['best_model_checkpoint']==cp['path'] and state['best_metric']==val[-1]['metrics']['eval_loss']
finished=dt.datetime.fromisoformat(x['finished_utc']);elapsed=(finished-dt.datetime.fromisoformat(q['started_at'])).total_seconds()
mean_val=statistics.mean(v['seconds'] for v in val);saves=a['checkpoint_events'];mean_save=statistics.mean(v['seconds'] for v in saves)
remaining_updates=8240-s['step'];remaining_val=4-len(val)
remaining_proxy=remaining_updates*seconds_per_update+remaining_val*(mean_val+mean_save)
summary={
 'A1_STATUS':'RUNNING_HEALTHY','query_started_utc':obs['utc'],'crosscheck_started_utc':x['started_utc'],'query_finished_utc':x['finished_utc'],'non_atomic_active_job_snapshot':True,
 'job_id':x['job_id'],'RUN_COMMIT':RUN,'evaluation_RUN_COMMIT':'445f1475cd1dd70d19881dbc1639bb311f7cccf7',
 'runner':{'status':q['status'],'exit_code':q['exit_code'],'supervisor_alive':q['supervisor_alive'],'worker_alive':q['worker_alive'],'workgroup_live':q['workgroup_live'],'process_identity_matches':matches,'supervisor_pid':q['supervisor']['pid'],'worker_pid':q['worker']['pid'],'supervisor_parent_pid':x['processes'][str(q['supervisor']['pid'])]['ppid']},
 'phase':'TRAINING_EPOCH_2',
 'training':{'steps':s['step'],'total_steps':8240,'percentage':100*s['step']/8240,'epoch':s['epoch'],'total_epochs':4,'ranks_consistent':True,'step_growth_this_check':[obs['rank']['0']['events'][-1]['step'],s['step']],'latest_event_utc':last['utc'],'latest_train_loss':last['values']['loss'],'current_optimizer_lr':s['lr'],'trainer_logged_previous_lr':last['values']['learning_rate'],'gradient_norm':s['grad_norm'],'recent_speed':{'from_step':steps[0]['step'],'to_step':steps[-1]['step'],'global_updates':updates,'wall_seconds':seconds,'seconds_per_global_update':seconds_per_update,'global_updates_per_second':updates/seconds,'global_presentations_per_second':64*updates/seconds,'selection':'Latest20 optimizer intervals wholly within current epoch, excluding epoch1 Validation/checkpoint pause'}},
 'validation':val,
 'best':{'checkpoint':state['best_model_checkpoint'],'eval_loss':state['best_metric'],'source':'checkpoint2060 trainer_state plus both-rank validation events','final_best_recovery_performed':False},
 'checkpoints':[{'path':c['path'],'step':c['seal']['global_step'],'status':c['status'],'members':c['member_count'],'logical_bytes':c['logical_bytes'],'small_member_hashes_checked':sum(v['sha256_checked'] for v in c['files']),'large_payloads_size_only':True,'full_tensor_load_or_recovery_test_performed':False} for c in x['checkpoint_checks']],
 'attempts':x['formal_ledger'],
 'recovery':{'formal_attempts':len(x['formal_ledger']['attempts']),'resume_values':[r.get('resume') for r in x['formal_ledger']['attempts']],'initial_global_steps':{r:v.get('initial_global_step') for r,v in x['rank_reports'].items()},'rotation_anomaly_observed':False,'note':'Single original process/attempt; no recovery entry. Only CP2060 exists; no later rotation yet.'},
 'test':{'state':'NOT_STARTED','users_completed':0,'users_planned':24772,'rank_reports_present':{r:v is not None for r,v in x['test_rank_reports'].items()},'failure_skip_counts':'NOT_APPLICABLE_TEST_NOT_STARTED','four_epochs_complete':False,'formal_best_recovered':False,'original_metrics':x['original_metrics'],'extended_pipeline':x['offline_pipeline'],'extended_CCE_bootstrap':'PENDING_SERVER_PIPELINE','new_evaluation_started_by_check':False},
 'health':{'anomaly_scan_matches':x['anomaly_lines_in_scanned_worklog'],'scanned_worklog_source':next(v for p,v in x['artifacts'].items() if p.endswith('/work.log')),'recent_progress_age_seconds':(finished-dt.datetime.fromisoformat(s['utc'])).total_seconds(),'resource_protection_trigger_observed':False,'historical_max_log_silence_seconds':x['resource_window_summary']['max_seconds_without_worklog_progress'],'historical_log_silence_explanation':'Epoch1 Validation took1836.9s, consistent with observed silence and below accepted7200s protection'},
 'resources':{'gpu_csv':x['gpu'],'compute_apps_csv':x['compute_apps'],'ram':x['ram'],'disk':x['disk'],'observed_window':x['resource_window_summary'],'latest_pipeline_sample':x['resource_samples_tail'][-1]},
 'timing':{'runner_started_utc':q['started_at'],'elapsed_wall_seconds':elapsed,'remaining_train_plus_validation_checkpoint_proxy_seconds':remaining_proxy,'remaining_training_updates':remaining_updates,'remaining_validation_passes':remaining_val,'validation_seconds_per_pass_proxy':mean_val,'checkpoint_seconds_per_save_proxy':mean_save,'complete_pipeline_remaining_planning_hours':[11,13],'test_time_assumption':'About1.6h based only on historical16-user preacceptance, not formal fulltest; loading/restore/CPU and throughput margin remain conditional','assumptions':['Latest20-interval training speed persists','Remaining3 validations comparable to real epoch1','Same hardware/resources, no recovery or long contention','Planning estimate, not promised completion time'],'formal_budget_wall_hours':96},
 'previous_network_failure':{'commit':'e394357e9166311f2e9e6471f934d17e7e8f3924','check_utc':'2026-11-13T15:37:10Z','interpretation':'Previous connection timeout receipt retained; this request reached original live job. No resubmit or recovery performed.'},
 'scope':['Existing job read-only state/metadata/log/process/resource check','No training, generation, evaluation, independent bootstrap or repair started','No experiment source/config/checkpoint/status modification','Commit only this new evidence directory'],
 'limitations':['Final scientific audit and metrics pending','Checkpoint seal check: required roles,all sizes,small-file SHA; large tensor contents not rehashed/loaded','Live job snapshot spans two bounded reads; latest steps legitimately differ','Single-controller status check, no independent second SSH/GPU reproduction'],
 'conflicts':[]}
(P/'status.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
for r,v in x['events'].items():(P/f'rank{r}_events.excerpt.jsonl').write_text(v['raw_jsonl_excerpt'])
for name,v in [('pipeline.excerpt.json',x['pipeline']),('checkpoint_integrity.json',x['checkpoint_checks']),('resources.excerpt.json',{'gpu':x['gpu'],'compute_apps':x['compute_apps'],'ram':x['ram'],'disk':x['disk'],'samples':x['resource_samples_tail'],'window_summary':x['resource_window_summary']}),('process_identity.excerpt.json',x['processes'])]:
    (P/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
(P/'worklog.tail.txt').write_text(x['worklog_tail'])
print(json.dumps({'status':summary['A1_STATUS'],'step':s['step'],'percentage':summary['training']['percentage'],'epoch':s['epoch'],'latest_log':last,'seconds_per_update':seconds_per_update,'elapsed_hours':elapsed/3600,'remaining_train_validation_hours':remaining_proxy/3600,'best':summary['best'],'checkpoints':summary['checkpoints'],'checkpoint_save_events':saves},indent=2))
