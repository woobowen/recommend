#!/usr/bin/env python3
"""Check nonauthor SSH-captured effective A0 fields without model imports."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).parent
    source=root/'terminal_effective_raw.json'
    raw=json.loads(source.read_text())
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    check('host',raw['host']=='zwlab-01' and raw['uid']==1005)
    check('run',raw['RUN_COMMIT']=='4c6a83d9ba42dc2fe737792759e410ef1874da76')
    expected={'num_train_epochs':4,'max_steps':-1,'per_device_train_batch_size':8,
        'per_device_eval_batch_size':8,'gradient_accumulation_steps':4,
        'optim':'adamw_torch','learning_rate':.0002,'weight_decay':.01,'warmup_ratio':.01,
        'lr_scheduler_type':'cosine','bf16':True,'fp16':False,'gradient_checkpointing':True,
        'evaluation_strategy':'epoch','save_strategy':'epoch','eval_delay':1,
        'load_best_model_at_end':True,'save_total_limit':2,'metric_for_best_model':'loss',
        'greater_is_better':False,'seed':42}
    rank_fields=[]
    for rank,r in enumerate(raw['ranks']):
        p='rank'+str(rank)+'_'
        ta=r['training_arguments'];ds=r['effective_ds']
        check(p+'identity',r['rank']==rank and r['run_commit']==raw['RUN_COMMIT'])
        check(p+'complete',r['status']=='TRAINING_COMPLETE' and r['global_step']==8240 and r['epoch']==4)
        check(p+'training_arguments',all(ta[k]==v for k,v in expected.items()))
        check(p+'zero2',ds['zero_optimization']['stage']==2)
        check(p+'batch',ds['train_batch_size']==64 and ds['train_micro_batch_size_per_gpu']==8
              and ds['gradient_accumulation_steps']==4)
        check(p+'precision',ds['bf16']['enabled'] is True and ds['fp16']['enabled'] is False)
        check(p+'trainable_scope',r['trainable_tensors']==450 and r['trainable_elements']==289652736
              and r['optimizer_membership_exact'] is True and r['trainable_policy_valid'] is True
              and r['original_embedding_head_frozen'] is True)
        check(p+'a_only',len(r['a_only_bf16']['targets'])==224
              and r['a_only_bf16']['parent_forward_identity']['target_count']==224)
        check(p+'residual',r['out_of_place']['target_count']==224)
        check(p+'data',r['data']=={'train_rows':131837,'valid_rows':49544,'tokenizer_vocab':32918})
        check(p+'model_revision',r['base_model_identity']['revision']=='4782ad278652c7c71b72204d462d6d01eaaf7549')
        check(p+'fa',r['formal_attention']['version']=='2.7.4.post1'
              and r['formal_attention']['enabled'] is True
              and r['formal_attention']['tensor_observers'] is False)
        rank_fields.append({'rank':rank,'training_arguments':{k:ta[k] for k in expected},
            'effective_batch':ds['train_batch_size'],'zero_stage':ds['zero_optimization']['stage'],
            'source_file':r['source_file']})
    result={'status':'PASS','observed_utc':raw['observed_utc'],'RUN_COMMIT':raw['RUN_COMMIT'],
        'checks':checks,'effective_fields':rank_fields,
        'input':{'path':str(source),'bytes':source.stat().st_size,
                 'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
        'metadata_notes':['DeepSpeed steps_per_print=Infinity is a logging-disable sentinel, '
            'not a nonfinite loss or gradient observation.',
            'a_only_bf16.adaptation inherits the historical string "probe-only" from compat; '
            'formal source/approval and actual formal argv establish approved use in this run. '
            'The original record is retained without mutation.'],
        'scope':'Nonauthor direct SSH read of final rank JSON and four small source files; '
                'no model/GPU/checkpoint payload work or remote mutation'}
    out=root/'terminal_effective_check.json'
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','checks':len(checks),'output':str(out)}))


if __name__=='__main__':main()
