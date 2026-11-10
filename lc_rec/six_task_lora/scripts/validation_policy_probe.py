#!/usr/bin/env python3
"""Actual pinned Trainer callback on CPU loop-control states; no model/train."""
import argparse,hashlib,inspect,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from transformers import TrainingArguments,TrainerState,TrainerControl,DefaultFlowCallback
import torch
prior=json.loads(Path(a.protocol).read_text());result={'scope':'CPU actual callback plus loop-control simulation; no optimizer/model/validation scan','layouts':[]}
for layout in prior['layouts']:
 length=layout['ranks'][0]['loader_len'];accum=layout['accum'];maxsteps=layout['trainer_max_steps_4epoch'];state=TrainerState();state.max_steps=maxsteps;callback=DefaultFlowCallback();control=TrainerControl()
 args=TrainingArguments(output_dir=str(Path(a.output).parent/'unused_output'),no_cuda=True,evaluation_strategy='epoch',save_strategy='epoch',eval_delay=1,report_to=[],load_best_model_at_end=True)
 totalmicro=0;events=[]
 for epoch in range(4):
  before=state.global_step
  for step in range(length):
   totalmicro+=1
   if totalmicro%accum==0 or (length<=accum and step==length-1):
    state.global_step+=1;state.epoch=epoch+(step+1)/length
   if state.global_step>=maxsteps:break
  control=TrainerControl();control=callback.on_epoch_end(args,state,control)
  events.append({'epoch_loop_index':epoch,'state_epoch':state.epoch,'global_step':state.global_step,'global_updates_this_epoch':state.global_step-before,'should_evaluate':control.should_evaluate,'should_save':control.should_save})
  if state.global_step>=maxsteps:break
 count=sum(e['should_evaluate'] for e in events)
 result['layouts'].append({'micro':layout['micro'],'accum':accum,'callback_events':events,'effective_validation_calls_under_control_simulation':count,'records_per_validation':49544,'total_validation_records_for_cost':count*49544,'four_validation_upper_bound_records':4*49544,'limitation':'Model/DS iteration and epoch-end losses/checkpoint-selection are not run. Test is actual callback on states derived from pinned loop condition.'})
result['callback_source_sha256']=hashlib.sha256(inspect.getsource(DefaultFlowCallback.on_epoch_end).encode()).hexdigest();result['callback_source']=inspect.getsource(DefaultFlowCallback.on_epoch_end);result['cuda_initialized']=torch.cuda.is_initialized();assert not result['cuda_initialized'];result['status']='PASS'
q=Path(a.output);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'validation_calls':[r['effective_validation_calls_under_control_simulation'] for r in result['layouts']]}))
