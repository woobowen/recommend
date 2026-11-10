#!/usr/bin/env python3
"""CPU-only identity/tail/protocol check; no model, optimizer, training or CUDA."""
import argparse,hashlib,inspect,json,math,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cfg=json.loads(Path(a.config).read_text())
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from accelerate.data_loader import prepare_data_loader
from transformers import Trainer
r={'kind':'CPU_PROTOCOL_AND_SAMPLER_PROBE_NO_MODEL','observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cuda_initialized_before':torch.cuda.is_initialized(),'candidate_config':cfg,'layouts':[]}
# Actual Accelerate0.21 sharding iterator, not merely ceil arithmetic.
for micro,accum in [(8,8),(4,16),(2,32),(1,64)]:
    ranks=[]
    for rank in [0,1]:
        torch.manual_seed(42)
        loader=DataLoader(range(221225),batch_size=micro,sampler=RandomSampler(range(221225)),drop_last=False)
        prepared=prepare_data_loader(loader,num_processes=2,process_index=rank,split_batches=False,put_on_device=False,rng_types=[],even_batches=True)
        rows=[];tail=[];batches=0
        for b in prepared:
            rows.extend(b.tolist());tail.append(len(b));tail=tail[-accum:];batches+=1
        ranks.append({'rank':rank,'loader_len':len(prepared),'iterated_batches':batches,'presentations':len(rows),'unique_rows':len(set(rows)),'last_micro_sizes':tail,'first8':rows[:8],'last8':rows[-8:],'sorted_rows_sha256':hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()})
    length=ranks[0]['loader_len'];updates=max(length//accum,1)
    # Trainer4.31's update condition uses total_batched_samples across epochs;
    # this CPU simulation reproduces loop control only, never model computation.
    global_steps=0;total_micro=0;epoch_trace=[];max_steps=math.ceil(4*updates)
    for epoch in range(4):
        before=global_steps;start_micro=total_micro
        for step in range(length):
            total_micro+=1
            last_short=length<=accum and step==length-1
            if total_micro%accum==0 or last_short:global_steps+=1
            if global_steps>=max_steps:break
        epoch_trace.append({'epoch_index':epoch,'batches_executed':total_micro-start_micro,'updates':global_steps-before,'global_steps':global_steps,'unstepped_accumulation_microbatches':total_micro%accum})
        if global_steps>=max_steps:break
    r['layouts'].append({'micro':micro,'accum':accum,'world':2,'global':micro*accum*2,'ranks':ranks,'trainer_num_update_steps_per_epoch':updates,'trainer_max_steps_4epoch':max_steps,'warmup_steps':math.ceil(max_steps*.01),'loop_control_simulation':epoch_trace,'tail_limitation':'Dummy indices measure sampler/shard only; formal stochastic Dataset answers differ each access. Actual DS+Trainer epoch-tail gradient handling not run.'})
source=inspect.getsource(Trainer._inner_training_loop)
(out/'trainer_inner_loop.py.txt').write_text(source)
r['trainer_loop_source_sha256']=hashlib.sha256(source.encode()).hexdigest()
r['accelerate_prepare_data_loader_source_sha256']=hashlib.sha256(inspect.getsource(prepare_data_loader).encode()).hexdigest()
model=Path(cfg['base_model']);expected=[('model-00001-of-00002.safetensors',9976579217,1794238862607969478,78251776),('model-00002-of-00002.safetensors',3500297444,1794243142341035385,78251779)]
r['weight_reuse_identity']={'prior_full_hash_receipt':'6b9f336d0252b44a31a48fbcc59817c9af1197e2/lc_rec/offload/evidence/preparation/model_identity_readback_20261109.json','scope':'Reuse prior full SHA verification, check same file size/inode/mtime now; no redundant 13.5GB hash scan.','files':[]}
for name,size,mtime,inode in expected:
    q=model/name;s=q.stat();assert s.st_size==size and s.st_mtime_ns==mtime and s.st_ino==inode
    r['weight_reuse_identity']['files'].append({'name':name,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'unchanged':True})
r['small_model_files']=[]
for name in ['config.json','model.safetensors.index.json','tokenizer.model','tokenizer_config.json','generation_config.json']:
    q=model/name;r['small_model_files'].append({'name':name,'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
r['test_protocol']={'users':24772,'prompts':17,'num_beams':20,'max_new_tokens':10,'filter_items':False,'callback':'Original position token sets, not strict trie','aggregation':'per prompt mean across users, then arithmetic mean across17; also min/max','two_rank_sampler_drop_last':True,'real_full_count_divisible_by2':24772%2==0,'rank_test_users':[len(DistributedSampler(range(24772),num_replicas=2,rank=i,drop_last=True)) for i in [0,1]],'odd_diagnostic_subset_warning':'drop_last=True would omit1 user for odd count; use explicit even fixed subset','oom_fallback':'Original silently decrements beam; technical harness must log OOM and keep beam20 requirement'}
r['validation_protocol']={'seqrec_prompts_per_user':2,'records_per_epoch':49544,'sample_valid':True,'selection':'lowest eval_loss; Trainer default greater_is_better=False for loss','eval_delay_epochs':1,'strategy':'epoch','best_checkpoint_restore':'not proved by adapter reload; requires actual Trainer/DS state restore'}
r['cuda_initialized_after']=torch.cuda.is_initialized();assert not r['cuda_initialized_after']
r['status']='PASS'
(out/'protocol.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'layouts':[{k:x[k] for k in ['micro','accum','trainer_max_steps_4epoch']} for x in r['layouts']]}))
