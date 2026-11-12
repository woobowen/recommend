#!/usr/bin/env python3
"""Build a CPU-only navigation summary from original evidence, without rerunning training."""
import collections
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'evidence/resume_20261112'
PREFIX='letter-resume-20261112-'

def read(p):return json.loads(p.read_text())
def training(name):return ROOT/'jobs'/(PREFIX+name)/'run/training'
def events(root,rank):return [json.loads(s) for s in (root/f'events.rank{rank}.jsonl').read_text().splitlines()]
def number(a,b):
 delta=abs(b-a);limit=1e-6+1e-3*abs(a)
 return {'reference':a,'candidate':b,'abs_diff':delta,'fixed_limit':limit,'pass':delta<=limit,'relative_percent':delta/abs(a)*100 if a else None}
def main():
 out={'A0_RESUME_DIAGNOSIS':'ROOT_CAUSE_NARROWED','A0_RESUME_GATE':'RESUME_GATE_FAIL','A0_FORMAL_NORMAL':'NOT_STARTED',
      'scope':'Bounded diagnosis complete; resume blocker remains. Not a formal training release.',
      'threshold':{'atol':1e-6,'rtol':1e-3,'discrete_scheduler_rng':'EXACT','changed':False},'pairs':[]}
 old=BASE/'evidence/formal_20261111/jobs'
 pairs=[('historical',old/'letter-a0-20261111-acceptance-cont-r3/run/training',old/'letter-a0-20261111-acceptance-resume-r3/run/training'),
        ('default',training('default-cont-r2'),training('default-resume-r2')),
        ('repeat',training('load-r1'),training('repeat-r2')),
        ('trace',training('trace1-r3'),training('trace2-r3')),
        ('torchdet',training('torchdet-cont-r3'),training('torchdet-resume-r3'))]
 for name,left,right in pairs:
  report=ROOT/'jobs'/(PREFIX+'cpu-compare-'+name)/'run/numeric_comparison.json';numeric=read(report)
  row={'label':name,'diagnostic_only':name=='torchdet','numeric_report':str(report.relative_to(BASE)),'numeric_status':numeric['status'],'ranks':[]}
  for rank in [0,1]:
   a,b=events(left,rank),events(right,rank)
   def pick(es,kind,condition):return [e for e in es if e['event']==kind and condition(e)]
   aa,bb=[pick(es,'batch',lambda e:e['step']>=2) for es in [a,b]];fields=['step','loader_yielded','input_sha256','label_sha256','shape']
   la,lb=[pick(es,'micro_loss',lambda e:e['step']>=2) for es in [a,b]]
   ua,ub=[pick(es,'optimizer_step',lambda e:e['step']>2) for es in [a,b]]
   r={'rank':rank,'batch_input_label_position_shape_exact':[{k:e[k] for k in fields} for e in aa]==[{k:e[k] for k in fields} for e in bb],
      'interval_microbatches':[len(aa),len(bb)],'loss':[number(x['loss'],y['loss']) for x,y in zip(la,lb)],
      'grad_norm':{str(x['step']):number(x['grad_norm'],y['grad_norm']) for x,y in zip(ua,ub)},
      'post_update_scheduler_lr_exact':[(x['step'],x['lr']) for x in ua]==[(x['step'],x['lr']) for x in ub]}
   n=numeric['ranks'][rank];fails=n['state_failures'];r['state_failure_reasons']=dict(collections.Counter(x['reason'] for x in fails))
   r['state_failure_categories']={'trainable':sum("['trainable']" in x['path'] for x in fails),'master':sum('single_partition_of_fp32_groups' in x['path'] for x in fails),
                                'exp_avg':sum("['exp_avg']" in x['path'] for x in fails),'exp_avg_sq':sum("['exp_avg_sq']" in x['path'] for x in fails)}
   r['all_tensors_finite']=n['all_model_and_optimizer_tensors_finite'];r['tensor_comparisons']=len(n['tensor_comparisons'])
   r['exact_required_tensor_failures']=[x['path'] for x in n['tensor_comparisons'] if x['requires_exact'] and not x['exact_equal']]
   r['exact_required_numpy_failures']=[x['path'] for x in n['numpy_array_comparisons'] if not x['exact_equal']]
   r['state_files']=n['files'];r['loss_exact_first_four']=all(x['abs_diff']==0 for x in r['loss'][:4]);r['loss_violations']=sum(not x['pass'] for x in r['loss'])
   if name!='historical':
    r['effective_lr_before_adam']={str(s+1):{'reference':[g['lr'] for g in read(left/f'S4-step{s}.rank{rank}.json')['optimizer']['param_groups']],
                                         'candidate':[g['lr'] for g in read(right/f'S4-step{s}.rank{rank}.json')['optimizer']['param_groups']]} for s in [2,3]}
   row['ranks'].append(r)
  out['pairs'].append(row)
 resource=read(ROOT/'jobs'/(PREFIX+'cpu-final-receipt')/'run/final_resources.json')
 out['gpu_device_minutes']=resource['charged_gpu_device_minutes'];out['resource_receipt_bytes_at_observation']=resource['total_new_persistent_bytes'];out['original_checkpoint_rehashed_unchanged_files']=len(resource['historical_checkpoint_rehashed_unchanged'])
 out['earliest_observed_difference']={'pair':'trace1-r3 vs trace2-r3','rank':0,'pre_update_global_step':2,'loader_yielded':9,'raw_event_line_each':351,
   'module':'base_model.model.model.layers.28.self_attn.q_proj','point':'gradient entering local projection backward; before reduction of that parameter gradient and Adam',
   'measurement':'full tensor byte hash differs; norm equal0.009680621325969696; no raw tensor saved at this point, so no elementwise magnitude claim',
   'not_claimed':'No unique kernel cause; ZeRO may already be reducing other parameters concurrently.'}
 out['unverified_state']=['Nonpersistent rotary cos_cached/sin_cached and max_seq_len_cached were not directly fingerprinted.',
   'Attention internal q/kv/dout/dq at the observed layer28 boundary were not captured; isolated case has no explicit layer binding.',
   'Device workspace, allocator/stream execution context and custom kernel internal state are not serialized-state equivalence claims.']
 out['training_code_correction_groups_used']=3;out['further_training_or_probe_changes']=False;out['new_installations']=[]
 (ROOT/'final_summary.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'status':out['A0_RESUME_DIAGNOSIS'],'gate':out['A0_RESUME_GATE'],'gpu_minutes':out['gpu_device_minutes'],
                   'pairs':[{k:r[k] for k in ['label','numeric_status']} for r in out['pairs']]}))
if __name__=='__main__':main()
