#!/usr/bin/env python3
"""Conditional96h estimate from this run's real six-task load; no LETTER rate."""
import argparse,hashlib,json,math,statistics,time,itertools
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scenario(components):
 known=sum(v for v in components.values() if v is not None)
 return {'component_hours':{k:None if v is None else v/3600 for k,v in components.items()},'known_subtotal_hours':known/3600,'total_hours':None if any(v is None for v in components.values()) else known/3600,'missing_components':[k for k,v in components.items() if v is None]}
def nonnegative_fb_fit(ranks,micro,full):
 import numpy as np
 X=[];y=[];widths=[];whole_rank_fb_total=0.
 for left,right in zip(ranks[0].get('updates',[]),ranks[1].get('updates',[])):
  if not(left.get('status')=='PASS' and right.get('status')=='PASS' and left.get('nonzero_learning_rate') and right.get('nonzero_learning_rate')):continue
  chosen=max([left,right],key=lambda z:z['rank_seconds']);whole_rank_fb_total+=chosen['phase_seconds']['forward_seconds']+chosen['phase_seconds']['backward_seconds']
  for a,b in zip(left['microsteps'],right['microsteps']):
   width=max(a['counts']['shape'][1],b['counts']['shape'][1]);slots=micro*width;quad=micro*width*width
   X.append([1.,slots/1000,quad/1000000]);y.append(max(a['forward_seconds']+a['backward_seconds'],b['forward_seconds']+b['backward_seconds']));widths.append(width)
 if len(X)<3:return None
 X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float);best=None
 for k in range(1,4):
  for active in itertools.combinations(range(3),k):
   c=np.linalg.lstsq(X[:,active],y,rcond=None)[0]
   if np.any(c<0):continue
   coef=np.zeros(3);coef[list(active)]=c;residual=X@coef-y;err=float(residual@residual)
   if best is None or err<best[0]:best=(err,coef,residual)
 assert best is not None
 err,coef,res=best;target=np.asarray([full['rank_micro_steps'],full['synchronized_wall_token_slots_max_rank_sum']/1000,full['synchronized_wall_attention_quadratic_max_rank_sum']/1000000],dtype=float)
 raw_seconds=float(target@coef)*4;rmse=float(np.sqrt(err/len(y)));normalizer=whole_rank_fb_total/float(y.sum());seconds=raw_seconds*normalizer
 return {'method':'Nonnegative3-term least-squares of paired-rank MAX(forward+backward) per actual nonzero-LR microstep; exhaustive7 active feature subsets, no test metrics','features':['constant_per_microstep','max_rank_padded_slots/1000','max_rank_micro_times_max_length_squared/1e6'],'coefficients_seconds':coef.tolist(),'observations':len(y),'observed_width_min':min(widths),'observed_width_max':max(widths),'rmse_seconds':rmse,'observed_y_min_seconds':float(y.min()),'observed_y_max_seconds':float(y.max()),'raw_design_X':X.tolist(),'raw_timing_y_seconds':y.tolist(),'maximum_absolute_residual_seconds':float(np.max(np.abs(res))),'active_feature_indices':[int(i) for i in np.flatnonzero(coef)],'design_matrix_condition_number':float(np.linalg.cond(X)),'paired_max_sum_to_whole_update_rank_normalizer':normalizer,'normalizer_meaning':'Sum of paired microstep MAX(F+B) can exceed whole-update slow-rank F+B. Scale fitted compute by measured whole-rank/paired-MAX ratio before adding the separately measured whole-rank non-F+B overhead. No required unmeasured cost is subtracted.','four_epoch_raw_paired_max_compute_seconds':raw_seconds,'four_epoch_compute_seconds':seconds,'four_epoch_compute_plus_2rmse_each_micro_seconds':seconds+8*rmse*normalizer*full['rank_micro_steps'],'boundary':'Engineering elapsed-time predictor, not a confidence interval. Full data may exceed fitted length range; actual extreme F/B capacity probes and separate static sensitivity estimates are retained.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workload',required=True);ap.add_argument('--grouping',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--policy',required=True);ap.add_argument('--gpu',required=True);ap.add_argument('--output',required=True);ap.add_argument('--run-commit',required=True);a=ap.parse_args()
 w,g,p,policy=map(read,[a.workload,a.grouping,a.protocol,a.policy]);root=Path(a.gpu)
 paths=[root/('probe_rank%d.json'%i) for i in [0,1]];ranks=[read(q) for q in paths if q.exists()]
 r={'status':'INCONCLUSIVE','calculation_run_commit':a.run_commit,'scope':'One Instruments seed, official six tasks/full originalSID,4epochs,global128,original sampled SeqRecvalidation and17prompt beam20; no formal training launched','inputs':[{ 'path':str(q),'sha256':sha(q)} for q in [a.workload,a.grouping,a.protocol,a.policy,*[q for q in paths if q.exists()]]],'counts':{'train_per_epoch':w['concat_rows'],'nominal_full4epoch_presentations':4*w['concat_rows'],'validation_records_per_call':49544,'test_users':24772,'test_prompts':17,'test_user_prompt_pairs':24772*17},'unquantified_overheads':['Actual Trainer/DeepSpeed complete training-state serialization, disk flush, and restore (adapter alone is not a resume checkpoint)','Long-run GPU/host/I/O variation and allocator fragmentation','Full random-sampler epoch-tail gradient behavior under real Trainer+DeepSpeed','All-prompt stochastic long-tail layouts beyond this one realized workload; capacity diagnostics are bounded','Validation throughput at formal per-device microbatch; measured validation is bounded batch1','Full-test CPU dataloader/decode/evaluator aggregation across all users and17 prompts'],'limitations':['One seed42 stochastic Dataset realization; prompt/query/preference answers vary across epochs','Only two global128 updates targeted; at most one nonzero-LR steady timing if both pass','Technical update auxiliary samples have verified training pairing; their within-task lengths can differ from all raw auxiliary records','Original data mismatches,505EOS-only records and51token-boundary cases retained; no clean formal corpus claim','No optimizer checkpoint/selection/best restore/full epoch/full test was run','Calibration uses local bounded six-task runs only; LETTER14.66samples/s is not used']}
 components_center={k:None for k in ['train','validation','adapter_save','full_training_state_save','best_restore','test','test_cpu_decode_evaluator_aggregation','init','data_preparation','measured_python_transfer_optimizer_overhead']};components_slow=dict(components_center)
 if len(ranks)==2:
  r['gpu_statuses']=[x['status'] for x in ranks]
  defs=[x.get('training_definition',{}) for x in ranks];micro=defs[0].get('micro_batch')
  if micro:
   assert all(x.get('micro_batch')==micro and x.get('global_batch')==128 for x in defs)
   proto=next(x for x in p['layouts'] if x['micro']==micro);pol=next(x for x in policy['layouts'] if x['micro']==micro)
   steps=proto['trainer_max_steps_4epoch'];r['counts'].update(formal_scheduler_steps=steps,warmup=proto['warmup_steps'],actual_selected_micro=micro,actual_accum=defs[0]['accumulation'],effective_validation_calls_from_callback=pol['effective_validation_calls_under_control_simulation'])
   paired=[]
   for x,y in zip(ranks[0].get('updates',[]),ranks[1].get('updates',[])):
    if x.get('status')!='PASS' or y.get('status')!='PASS':continue
    assert x['update']==y['update'] and x['global_counts']==y['global_counts'] and x['global_counts']['samples']==128
    chosen=max([x,y],key=lambda z:z['rank_seconds']);compute=chosen['phase_seconds']['forward_seconds']+chosen['phase_seconds']['backward_seconds'];wall=max(x['max_rank_seconds'],y['max_rank_seconds']);raw_gap=wall-compute
    assert raw_gap>=0,'F+B phase times exceed complete slow-rank wall; do not conceal negative overhead'
    paired.append({'update':x['update'],'nonzero_lr':x['nonzero_learning_rate'] and y['nonzero_learning_rate'],'seconds':wall,'samples_per_second':128/wall,'effective_tokens_per_second':x['global_counts']['tokens']/wall,'compute_seconds':compute,'other_timed_seconds':raw_gap,'unclipped_wall_minus_chosen_rank_fb_seconds':raw_gap,'cpu_collator_seconds':max(x['cpu_collator_seconds'],y['cpu_collator_seconds']),'plan':'mixed_global128_'+str(x['update']%2),'global_counts':x['global_counts'],'padding_ratio':x['padding_ratio']})
   r['measured_joint_updates']=paired
   steady=[x for x in paired if x['nonzero_lr']]
   if steady:
    full=g['full_epoch_fixed_realization'][str(micro)]
    # Relative transformer matmul work: per layer4H²L+3HIL+2HL².
    # This is a cost covariate calibrated to measured wall seconds, not measuredFLOPs.
    H,I=4096,11008;eta=2*H/(4*H*H+3*H*I)
    fL=full['synchronized_wall_token_slots_max_rank_sum'];fQ=full['synchronized_wall_attention_quadratic_max_rank_sum'];central=[];linear=[];quad=[]
    for x in steady:
     m=g['mixed_global128'][x['plan']][str(micro)]['workload'];mL=m['synchronized_wall_token_slots_max_rank_sum'];mQ=m['synchronized_wall_attention_quadratic_max_rank_sum']
     central.append(4*x['compute_seconds']*(fL+eta*fQ)/(mL+eta*mQ));linear.append(4*x['compute_seconds']*fL/mL);quad.append(4*x['compute_seconds']*fQ/mQ)
    fit=nonnegative_fb_fit(ranks,micro,full)
    r['actual_microstep_fb_fit']=fit
    components_center['train']=fit['four_epoch_compute_seconds'] if fit else statistics.mean(central)
    # Conservative scenario is explicit, not a confidence interval or guarantee.
    candidates=central+linear+quad+([fit['four_epoch_compute_plus_2rmse_each_micro_seconds']] if fit else [])
    components_slow['train']=max(candidates)*1.25
    components_center['measured_python_transfer_optimizer_overhead']=steps*statistics.mean(x['other_timed_seconds'] for x in steady)
    components_slow['measured_python_transfer_optimizer_overhead']=steps*max(x['other_timed_seconds'] for x in paired)*1.25
    components_center['data_preparation']=steps*statistics.mean(x['cpu_collator_seconds'] for x in steady)
    components_slow['data_preparation']=steps*max(x['cpu_collator_seconds'] for x in paired)*1.25
    r['train_extrapolation']={'method':'Static sensitivity comparator: calibrate actual slow-rank F+B seconds to full221225-row mixed-padding workload (original data, including mismatches) using4H²L+3HIL+2HL²;4epochs as conservative work accounting,6912optimizer overheads','eta_quadratic_per_linear':eta,'central_compute_seconds':central,'linear_only_compute_seconds':linear,'quadratic_only_compute_seconds':quad,'conservative_multiplier_on_max_cost_covariate':1.25,'multiplier_meaning':'Explicit resource/runtime scenario; unknown checkpoint or evaluation overhead is never folded into it','counts_boundary':'Full4epoch workload slightly exceeds Trainer floor-step truncation; actual full Trainer+DS epoch tail not executed','full_epoch_workload':full}
   validation=[z for x in ranks for z in x.get('validation',{}).get('timings',[]) if z.get('finite')]
   if validation:
    mean=statistics.mean(z['seconds_max_rank'] for z in validation);mx=max(z['seconds_max_rank'] for z in validation)
    components_center['validation']=pol['total_validation_records_for_cost']*mean/2
    components_slow['validation']=4*49544*mx/2*1.25
    r['validation_extrapolation']={'batch1_forward_seconds_mean':mean,'batch1_forward_seconds_max':mx,'center_calls':pol['effective_validation_calls_under_control_simulation'],'conservative_calls':4,'dual_rank_denominator':2,'formal_microbatch_throughput_unmeasured':True}
  init=max(x.get('model_initialization_seconds',0) for x in ranks)
  if init:components_center['init']=init+max(x.get('zero2_initialization_seconds',0) for x in ranks)+max(x.get('data',{}).get('dataset_tokenizer_seconds',0) for x in ranks);components_slow['init']=components_center['init']*1.25
  saves=[x.get('save',{}) for x in ranks]
  if all(x.get('status')=='PASS' for x in saves):
   sec=max(x['seconds_max_rank'] for x in saves);components_center['adapter_save']=4*sec;components_slow['adapter_save']=4*sec*1.25
   r['adapter_save_extrapolation']={'seconds_each_measured':sec,'calls_assumed':4,'training_state_included':False,'serialized_bytes_rank0':sum(x['bytes'] for x in saves[0].get('files',[]))}
  reload=[x.get('reload',{}) for x in ranks]
  if all(x.get('status')=='PASS' for x in reload):
   sec=max(x['seconds_max_rank'] for x in reload);components_center['best_restore']=sec;components_slow['best_restore']=sec*1.25
   r['restore_extrapolation']={'measured_fresh_adapter_reload_seconds':sec,'scope':'Proxy for best-adapter inference restoration; actual Trainer/DS best-checkpoint path is not verified'}
  generation=ranks[0].get('generation',{})
  if generation.get('status')=='PASS' and generation.get('templates_measured')==17:
   components_center['test']=generation['central_full_test_two_rank_seconds'];components_slow['test']=generation['conservative_full_test_two_rank_seconds']*1.25
   r['test_extrapolation']={'per_prompt':generation['prompt_timings'],'batch_size':1,'beam':20,'all17_templates_measured':True,'users_per_template':3,'denominator':'Per-user GPU generation time,24,772users per prompt divided by2 simultaneous ranks; uneven3-user diagnostic wall not misused as saturated throughput','conservative_multiplier':1.25}
 r['center']=scenario(components_center);r['conservative']=scenario(components_slow)
 for key in ['center','conservative']:
  h=r[key]['component_hours'];r[key]['training_plus_measured_overhead_hours']=sum(v for k,v in h.items() if v is not None and k in ['train','data_preparation','measured_python_transfer_optimizer_overhead']);r[key]['evaluation_hours']=sum(v for k,v in h.items() if v is not None and k in ['validation','test'])
 # A known subtotal beyond96h suffices to reject the budget even with missing
 # checkpoints. Otherwise unmeasured required components keep the claim open.
 if r['conservative']['known_subtotal_hours']>96 and components_slow['train'] is not None and components_slow['test'] is not None:
  r['status']='UNLIKELY_WITHIN_96H';r['decision_reason']='Conservative measured-component scenario already exceeds96h; required unmeasured checkpoint/aggregate overhead can only add cost.'
 else:r['decision_reason']='Bounded timing does not sufficiently constrain all required full-seed components; explicit unmeasured checkpoint/long-tail/aggregation terms prevent a96h guarantee.'
 r['formal_data_status']=w['auxiliary_data_status']['formal_training_status'];r['created_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 q=Path(a.output);q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();q.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':r['status'],'center_known_hours':r['center']['known_subtotal_hours'],'conservative_known_hours':r['conservative']['known_subtotal_hours'],'unmeasured':r['center']['missing_components']}))
if __name__=='__main__':main()
