"""Evidence-bound one-seed time budget; missing measurements remain UNKNOWN."""
import argparse,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--train-dir');p.add_argument('--inference-dir');p.add_argument('--output',required=True);a=p.parse_args()
j=json.loads(Path(a.audit).read_text());n=131837;v=49544;t=24772
# Recover from the checked source audit when available; fail if expected current domain changes.
assert j.get('status')=='PASS','Only completed CPU audit may support a budget'
r={'status':'MISSING_GPU_MEASUREMENTS','observed_counts':{'train_per_epoch':n,'epochs':4,'train_presentations':n*4,'valid_texts_per_epoch':v,'total_validation_texts':v*4,'test_users_per_prompt':t,'test_prompt0_presentations':t,'test_17_prompt_presentations':t*17,'effective_batch':64,'updates_per_epoch':math.ceil(n/64),'optimizer_updates_four_epochs':4*math.ceil(n/64),'scheduler_warmup_steps':math.ceil(4*math.ceil(n/64)*.01)},'publication_prompt_protocol':'unknown; public run_test_ddp.sh specifies prompt0; 17 prompts is a separate cost scenario','identity_limit':'fixed author supplement is not bound to historical checkpoint/argv; time feasibility is not metric reproduction','measurement_notes':[]}
train=[];infer=[]
if a.train_dir:
 train=[json.loads(q.read_text()) for q in Path(a.train_dir).glob('rank[01].json')]
if a.inference_dir:
 infer=[json.loads(q.read_text()) for q in Path(a.inference_dir).glob('inference_rank[01].json')]
if len(train)==2 and all(x['status']=='PASS' for x in train) and len(infer)==2 and all(x['status']=='RELOAD_BEAM20_TECHNICAL_PASS' for x in infer):
 updates=train[0]['updates'];steady=[x for x in updates if x['phase']=='mixed_steady'];samples=[x['samples_per_second'] for x in steady]
 preparation=max(x['cpu_batch_preparation_and_mask_verification_seconds'] for z in train for x in z['updates'] if x['phase']=='mixed_steady')
 lower_rate=64/(max(x['max_rank_seconds'] for x in steady)+preparation)
 upper_rate=64/min(x['max_rank_seconds'] for x in steady)
 train_low=n*4/upper_rate/3600;train_high=n*4/lower_rate/3600
 valid_rows=[x for z in train for x in z['validation_timings'] if x['phase']!='warmup']
 vtime=max(x['max_rank_seconds']/x['global_samples'] for x in valid_rows)*v*4/3600
 itimes=[x['seconds'] for z in infer for x in z['timings'] if not x['warmup']]
 test_low=min(itimes)*t/2/3600;test_high=max(itimes)*t/2/3600
 save=max(x.get('adapter_save',{}).get('seconds_rank0',0) for x in train)/3600
 reload=max(x['base_load_seconds']+x['adapter_reload_seconds'] for x in infer)/3600
 # Four epoch adapter writes, final write, and four optimizer-state writes are
 # extrapolated rather than measured full checkpoint I/O. Reserve >=0.5h.
 checkpoint_reserve=max(.5,save*10)
 raw=train_high+vtime+checkpoint_reserve+reload+test_high
 high=raw*1.5
 r.update(status='MEASURED_BOUNDED_PROJECTION',training={'mixed_samples_s':samples,'runtime_warmup_updates':3,'scheduler_warmup_completed':False,'input_tokens_s':[x['effective_tokens_per_second'] for x in steady],'measured_length_updates':[x for x in updates if x['phase'] in ['short','middle','long']],'four_epochs_hours':[train_low,train_high]},validation={'four_epochs_conservative_hours':vtime,'method':'max of measured mixed/long no-grad BF16 forward seconds/sample; no selection or complete epoch run'},checkpoint={'adapter_save_observed_seconds':save*3600,'save_and_selection_reserve_hours':checkpoint_reserve,'full_deepspeed_optimizer_checkpoint_io_measured':False},reload={'observed_seconds':reload*3600},test={'single_prompt_hours_range':[test_low,test_high],'17_prompts_hours_upper':test_high*17,'method':'2 independent GPU replicas; max measured representative prompt0 beam20 user latency used as per-user upper proxy; all17 not run'},single_prompt_end_to_end={'raw_conservative_hours':raw,'with_50_percent_uncertainty_margin_hours':high,'within96_with_margin':high<=96},all17_prompt_end_to_end={'with_50_percent_uncertainty_margin_hours':(raw-test_high+17*test_high)*1.5,'within96_with_margin':(raw-test_high+17*test_high)*1.5<=96})
 r['measurement_notes']=['Runtime kernel warmup completed; 83-step scheduler warmup not completed in nine technical updates.','Repeated short/middle/long real samples measure length envelope, not population throughput.','Representative timing and 50% reserve do not guarantee a future full run.','Full optimizer checkpoint/selection, prompt17 cost and full-dataset evaluation remain projections.']
else:
 r['reason']='No approved successful real forward/backward/update and beam20 timing record set.'
 r['time_limit_assessment']='UNKNOWN; cannot certify <=96 hours or hardware execution'
 r['required_train_samples_per_second_if_other_hours']={str(h):n*4/((96-h)*3600) for h in [0,6,12,24]}
Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'output':a.output}))
