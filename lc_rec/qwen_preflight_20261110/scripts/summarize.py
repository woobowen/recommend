"""Derive conditional seed costs directly from retained two-rank measurements.

Never labels a complete seed CONFIRMED. Formal DS optimizer/scheduler/RNG saves
are forbidden in this preflight; their time is an explicit I/O proxy, not zero.
"""
import argparse, datetime, json, math, statistics, sys
from pathlib import Path
from common import MODELS,write,sha

DTYPE_BYTES={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8,'torch.int8':1,'torch.uint8':1,'torch.int64':8,'torch.int32':4}

def estimate(ranks,cpu,guard):
 r=ranks[0];updates=r['updates'];training=r['training'];steps=training['formal_updates']
 assert steps==4*math.ceil(r['formal_loader']['length_per_rank']/training['accumulation'])
 assert all(len(x['updates'])==len(updates) for x in ranks)
 mixed=[x for x in updates if x['kind']=='mixed' and x['index']>0]
 chosen=mixed or [x for x in updates if x['nonzero_lr']]
 assert chosen
 measured_mean_tokens=sum(x['global_effective_tokens'] for x in chosen)/(64*len(chosen))
 full_mean=cpu['lengths']['train']['mean_tokens'];token_scale=max(1.,full_mean/measured_mean_tokens)
 center_step=statistics.mean(x['seconds'] for x in chosen)*token_scale
 conservative_step=max(x['seconds'] for x in updates if x['index']>0)*token_scale
 cold_start_excess=max(0.,updates[0]['seconds']-statistics.mean(x['seconds'] for x in chosen))
 validation=r['validation'];center_v=next(x for x in validation if x['label']=='mixed')['seconds_including_collator']/16
 conservative_v=max(x['seconds_including_collator'] for x in validation)/16
 all_beams=[row for rank in ranks for row in rank['beam20']];assert len({x['user_id'] for x in all_beams})>=8
 groups=[]
 for label in ['short','middle','long','boundary']:
  rows=[x for x in all_beams if x['label']==label];assert len(rows)==2
  assert rows[0]['seconds_two_rank_max']==rows[1]['seconds_two_rank_max']
  groups.append({'label':label,'input_tokens':statistics.mean(x['input_tokens'] for x in rows),'seconds_per_user_dual_gpu':rows[0]['seconds_two_rank_max']/2})
 hist={int(k):v for k,v in cpu['lengths']['test_prompt0']['histogram'].items()};allocation={x['label']:0 for x in groups};test_center=0.
 for length,count in hist.items():
  group=min(groups,key=lambda x:(abs(x['input_tokens']-length),x['input_tokens']));allocation[group['label']]+=count;test_center+=count*group['seconds_per_user_dual_gpu']
 test_conservative=cpu['data']['test_users']*max(x['seconds_per_user_dual_gpu'] for x in groups)
 trainable=r['trainable']['parameters'];current_bytes=sum(x['numel']*DTYPE_BYTES[x['dtype_after_ds']] for x in r['parameter_inventory'].values())
 if r['route']=='L':
  saved=r['adapter_save'];saved_bytes=saved['file_bytes'];saved_seconds=saved['seconds'];best_restore=max(x['adapter_reload_seconds'] for x in ranks)
  extra_inference_init=max(x['inference_base_reload_seconds'] for x in ranks)
 else:
  saved=r['full_model_save'];saved_bytes=sum(x['bytes'] for x in saved['files']);saved_seconds=saved['seconds'];best_restore=max(x['full_model_reload_seconds'] for x in ranks);extra_inference_init=0.
 assert saved_seconds>0
 bandwidth=saved_bytes/saved_seconds
 # Exact serialized DS checkpoint size/format is NOT TESTED. Cover replicated
 # module-state copies, global FP32 master + 2 moments, plus an explicit 1GiB
 # bookkeeping/buffer allowance. This is an assumption for conditional planning.
 checkpoint_proxy_bytes=2*current_bytes+12*trainable+1024**3
 checkpoint_center=4*checkpoint_proxy_bytes/bandwidth+saved_seconds
 checkpoint_conservative=4*(checkpoint_proxy_bytes/(bandwidth/4)+60)+2*saved_seconds
 bounded_stage=next(x['elapsed_seconds'] for x in r['stages'] if x['stage']=='bounded_training')
 measured_launch_teardown=max(0.,guard['elapsed_seconds']-max(x['elapsed_seconds'] for x in ranks))
 center={'train':steps*center_step,'validation':4*cpu['data']['validation_rows']*center_v,'checkpoint':checkpoint_center,'best_restore':best_restore,'beam20_test':test_center,'init':bounded_stage+extra_inference_init+cold_start_excess}
 conservative={'train':steps*conservative_step,'validation':4*cpu['data']['validation_rows']*conservative_v,'checkpoint':checkpoint_conservative,'best_restore':best_restore*2,'beam20_test':test_conservative,'init':2*(bounded_stage+extra_inference_init+cold_start_excess)}
 center['overhead']=.10*sum(center.values())+measured_launch_teardown;conservative['overhead']=.25*sum(conservative.values())+measured_launch_teardown
 center_total=sum(center.values());conservative_total=sum(conservative.values())
 if conservative_total>96*3600:classification='UNLIKELY_WITHIN_96H'
 elif r['route']=='L' and len(mixed)>=2 and conservative_total<.8*96*3600:classification='LIKELY_WITHIN_96H'
 else:classification='INCONCLUSIVE'
 result={'status':classification,'formal_seed_status':'NOT CONFIRMED; no formal seed/epoch executed','measured':{'global_update_seconds':[{'index':x['index'],'kind':x['kind'],'seconds':x['seconds'],'samples':64,'effective_tokens':x['global_effective_tokens'],'padded_tokens':x['global_padded_tokens'],'supervised_tokens':x['global_supervised_tokens'],'lr_before':x['lr_before']} for x in updates],'post_first_mixed_updates':len(mixed),'chosen_update_indices':[x['index'] for x in chosen],'full_training_distribution_mean_tokens':full_mean,'measured_mean_tokens':measured_mean_tokens,'upward_only_linear_token_scale':token_scale,'validation':validation,'test_strata':groups,'test_histogram_assigned_users':allocation,'model_or_adapter_save_bytes':saved_bytes,'model_or_adapter_save_seconds':saved_seconds,'measured_write_bytes_per_second':bandwidth,'actual_weight_or_adapter_reload_seconds':best_restore,'actual_launcher_teardown_proxy_seconds':measured_launch_teardown},'workload':{'original_rows_per_epoch':cpu['data']['train_rows'],'epochs_budget_only':4,'original_rows_four_epochs':4*cpu['data']['train_rows'],'actual_loader':r['formal_loader'],'effective_formal_optimizer_steps':steps,'global64_equivalent_presentations':64*steps,'validation_presentations':4*cpu['data']['validation_rows'],'beam20_test_users':cpu['data']['test_users'],'complete_epoch_traversed':False,'tail_identity':'Actual loader and batch-sampler inspected; no epoch tail actually traversed'},'checkpoint_proxy':{'current_parameter_bytes_unique_parameter_inventory':current_bytes,'trainable_elements':trainable,'formula':'2*current_parameter_bytes + 12*trainable_elements + 1GiB metadata/buffer assumption per checkpoint, four checkpoints','bytes_per_checkpoint_assumption':checkpoint_proxy_bytes,'center_io_bandwidth_source':'this route native adapter/model-only measured save; page-cache/durability not separately measured','conservative_bandwidth':'one quarter observed bandwidth, plus60seconds/checkpoint','actual_formal_optimizer_checkpoint':'NOT TESTED / NOT WRITTEN','actual_optimizer_scheduler_rng_restore':'NOT TESTED'},'center':{'seconds_by_component':center,'total_seconds':center_total,'hours':center_total/3600,'days':center_total/86400},'conservative':{'seconds_by_component':conservative,'total_seconds':conservative_total,'hours':conservative_total/3600,'days':conservative_total/86400},'unknown_costs_and_conditions':['No formal Trainer/DS optimizer/scheduler/RNG checkpoint serialization or best-checkpoint selection/restore was executed; checkpoint terms are conditional proxies, never measured full-checkpoint seconds.','Four-epoch thermal/allocator/network-filesystem contention and long-run stability untested.','Complete validation loader aggregation, all-user CPU decoding/filtering/Recall-NDCG are untested;10%/25% margins are assumptions, not confidence intervals.','Only2 full-finetune updates allowed: one nonzero-LR length-envelope timing cannot establish a stable mixed-update rate.' if r['route']=='F' else 'Only2 post-first mixed global updates plus a length-envelope update; fixed finite real-row pool, not a full epoch.','Training timings include CPU collator, synchronization, gradient guards, JSON logging and communication; not a kernel-only benchmark.','Model-only/adapter strict restore is distinct from a complete training-state restore.']}

 all_entries=[entry for rank in ranks for update in rank['updates'] for micro in update['microsteps'] for entry in micro['rows']]
 chosen_indices={x['index'] for x in chosen}
 timed_entries=[entry for rank in ranks for update in rank['updates'] if update['index'] in chosen_indices for micro in update['microsteps'] for entry in micro['rows']]
 result['measured']['sampling']={'declared_selection_pool_rows':256,'actual_presentations':len(all_entries),'actual_unique_train_rows':len({x['row_idx'] for x in all_entries}),'actual_unique_row_prompt_pairs':len({(x['row_idx'],x['prompt_id']) for x in all_entries}),'center_timing_presentations':len(timed_entries),'center_timing_unique_rows':len({x['row_idx'] for x in timed_entries}),'boundary_note':'last update repeats one real longest-history row across17 original author templates; pool size is not unique rows actually trained'}
 result['measured']['cold_start_excess_counted_once_in_init_seconds']=cold_start_excess
 result['checkpoint_proxy']['extra_final_model_or_adapter_save_seconds_center']=saved_seconds
 result['checkpoint_proxy']['extra_final_model_or_adapter_save_seconds_conservative']=2*saved_seconds
 result['unknown_costs_and_conditions'].append('Final native model/adapter save adds one measured save proxy; Trainer.save_state JSON and complete final Trainer/DS state behavior remain untested.')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',type=Path,default=Path(__file__).parents[1]);p.add_argument('--run-commit',default='UNSPECIFIED');p.add_argument('--outcomes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();jobs=a.task/'evidence/jobs';outcomes=json.loads(a.outcomes.read_text());result={'analysis':{'run_commit':a.run_commit,'script_sha256':sha(Path(__file__)),'common_sha256':sha(Path(__file__).with_name('common.py')),'argv':sys.argv,'interpreter':sys.executable,'cwd':str(Path.cwd()),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},'source_outcomes_sha256':sha(a.outcomes),'models':{},'recommendation_metrics':'NOT TESTED; technical preflight cannot rank formal Recall/NDCG'}
 for key,identity in MODELS.items():
  model={};cpu=json.loads((jobs/('qwen-20261110-cpu-'+key+'-r0')/'output/audit/audit.json').read_text());model['cpu_status']=cpu['status'];model['data']=cpu['data'];model['tokenizer']=cpu['tokenizer'];model['model']=cpu['model'];model['static_full_parameter_resources']=cpu['full_parameter_static'];model['routes']={}
  for route in ['L','F']:
   item=outcomes[key][route];record={'status':item['status'],'reason':item.get('reason'),'jobs':item.get('jobs',[])}
   if item.get('training_job'):
    directory=jobs/item['training_job'];ranks=[json.loads(path.read_text()) if path.exists() else {'status':'MISSING_RANK_RECORD','updates':[],'missing_path':str(path)} for path in [directory/'output/probe'/f'rank{i}.json' for i in [0,1]]]
    record['run_commit']=ranks[0].get('run_commit',json.loads((directory/'request.json').read_text())['capture_version']);record['core_training_status']=[r.get('core_training_status','NOT PASSED') for r in ranks];record['updates_per_rank']=[len(r.get('updates',[])) for r in ranks];record['nonzero_lr_updates_per_rank']=[sum(x['nonzero_lr'] for x in r.get('updates',[])) for r in ranks];record['training_peak_per_rank']=[r.get('training_peak',r.get('failure_peak')) for r in ranks];record['recorded_job_phase_peak_per_rank']=[r.get('job_peak_from_recorded_phases') for r in ranks];record['complete_allocator_phase_coverage_per_rank']=['validation_and_post_train_check_peak' in r for r in ranks];record['trainable']=ranks[0].get('trainable');record['precision']=ranks[0].get('precision','NOT OBSERVED');record['training']=ranks[0].get('training')
    if all(r.get('status')=='PASS_TECHNICAL' for r in ranks):
     guard=json.loads((directory/'output/resources/resource-summary.json').read_text());record['seed_estimate']=estimate(ranks,cpu,guard)
    else:record['seed_estimate']={'status':'BLOCKED' if record['core_training_status']!=['PASS','PASS'] else 'INCONCLUSIVE','center':None,'conservative':None,'reason':'Missing composite training/restore/generation measurements; never substitute zero for unknown costs'}
   else:record['seed_estimate']={'status':'BLOCKED','center':None,'conservative':None,'reason':item.get('reason','No safe authorized training run')}
   model['routes'][route]=record
  result['models'][key]=model
 sample_sets={key:json.loads((jobs/('qwen-20261110-cpu-'+key+'-r0')/'output/audit/samples.json').read_text()) for key in MODELS}
 reference=sample_sets['q06'];fields=['label','row_idx','prompt_id','input_text','full_text','input_text_sha256','full_text_sha256']
 for key,samples in sample_sets.items():
  assert [{field:x[field] for field in fields} for x in samples]==[{field:x[field] for field in fields} for x in reference]
  assert result['models'][key]['data']==result['models']['q06']['data']
  assert result['models'][key]['tokenizer']['sid_tokens']==result['models']['q06']['tokenizer']['sid_tokens']
 selections={key:sha(jobs/('qwen-20261110-cpu-'+key+'-r0')/'output/audit/selection.json') for key in MODELS};assert len(set(selections.values()))==1
 result['cross_model_inputs']={'actual_raw_train_sample_texts_identical':True,'samples_per_model':len(reference),'sample_identity_sha256':{key:sha(jobs/('qwen-20261110-cpu-'+key+'-r0')/'output/audit/samples.json') for key in MODELS},'full_selection_files_sha256':selections,'fixed_selection_rows_prompts_users_and_validation_indices_identical':True,'original_data_hashes_and_counts_identical':True,'all_918_sid_token_strings_order_identical':True,'different_tokenizer_ids_lengths_and_padding_retained':True}
 result['analysis']['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();result['analysis']['status']='PASS_ANALYSIS_FROM_RAW_RECORDS';write(a.output,result);print(json.dumps({key:{route:data['seed_estimate']['status'] for route,data in value['routes'].items()} for key,value in result['models'].items()},indent=2))
if __name__=='__main__':main()
