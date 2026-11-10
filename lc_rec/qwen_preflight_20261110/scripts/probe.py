"""Finite dual-4090 LETTER SeqRec probe, never an epoch or formal seed."""
import argparse, datetime as dt, gc, hashlib, importlib.metadata as md, inspect, json, math, os, random, resource, sys, time, traceback
from pathlib import Path
from common import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--key',choices=MODELS,required=True);p.add_argument('--route',choices=['L','F'],required=True);p.add_argument('--precision',choices=['int8','bf16'],required=True);p.add_argument('--model',required=True);p.add_argument('--letter',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);p.add_argument('--run-commit',required=True);p.add_argument('--micro',type=int,default=8);a=p.parse_args()
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2 and a.micro in [8,4,2,1]
 assert not (a.route=='F' and a.precision!='bf16');assert not (a.key=='q35' and a.route=='L'),'Qwen3.5 LoRA mapping not approved'
 accumulation=32//a.micro;out=Path(a.output);out.mkdir(parents=True,exist_ok=True);path=out/f'rank{rank}.json';start=time.monotonic()
 r={'status':'RUNNING','key':a.key,'route':a.route,'precision':a.precision,'run_commit':a.run_commit,'rank':rank,'local_rank':local,'world_size':world,'argv':sys.argv,'cwd':str(Path.cwd()),'interpreter':sys.executable,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'updates':[],'validation':[],'stages':[],'technical_only':True}
 torch=None;dist=None
 def stage(name):
  r['stages'].append({'stage':name,'elapsed_seconds':time.monotonic()-start});write(path,r);print(json.dumps({'rank':rank,'stage':name}),flush=True)
 try:
  import numpy as np
  import torch, torch.distributed as dist, torch.nn.functional as F
  import deepspeed
  from transformers import Trainer,TrainingArguments
  from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training,get_peft_model_state_dict,PeftModel
  torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=dt.timedelta(minutes=5));device=torch.device('cuda',local)
  def sync():torch.cuda.synchronize(device)
  def max_time(value):
   t=torch.tensor(value,device=device);dist.all_reduce(t,op=dist.ReduceOp.MAX);return t.item()
  def total(value):
   t=torch.tensor(value,device=device,dtype=torch.float64);dist.all_reduce(t);return t.item()
  def all_equal(value):
   gathered=[None]*2;dist.all_gather_object(gathered,value);assert gathered[0]==gathered[1];return gathered
  def digest_tensor(tensor):
   x=tensor.detach().contiguous().cpu();return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
  def peak():return {'allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3,'reserved_gib':torch.cuda.max_memory_reserved(device)/1024**3,'ru_maxrss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
  random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  r['versions']={name:md.version(name) for name in ['torch','transformers','peft','accelerate','bitsandbytes','deepspeed']}
  r['gpu']={'name':torch.cuda.get_device_name(local),'total_bytes':torch.cuda.get_device_properties(local).total_memory,'visible_devices':os.environ['CUDA_VISIBLE_DEVICES']}
  train,valid,test=datasets(a.letter);tok,tokenmap=tokenizer_for(a.model,train.get_new_tokens());collator=Collator(tok)
  audit=json.loads((Path(a.audit)/'audit.json').read_text());selection=json.loads((Path(a.audit)/'selection.json').read_text())
  assert tokenmap=={k:v for k,v in audit['tokenizer'].items() if k not in ['save_reload_vocab_exact','all_9922_item_sids_encode_decode_before_and_after_reload_exact']}
  r['data']={k:v for k,v in audit['data'].items()};r['tokenizer']=tokenmap;r['audit_sha256']=sha(Path(a.audit)/'audit.json');r['selection_sha256']=sha(Path(a.audit)/'selection.json')
  # Static analysis already precedes the first model GPU allocation.
  r['static_resources']=audit['full_parameter_static'];r['checkpoint_static']=audit['checkpoint_static']
  official=json.loads((Path(__file__).parents[1]/'metadata'/MODELS[a.key][0]/'manifest.json').read_text());formal=json.loads((Path(a.model)/'ASSET_MANIFEST.json').read_text());assert official==formal
  assert Path(a.model).name==MODELS[a.key][1]
  for entry in official['files']:assert (Path(a.model)/entry['path']).stat().st_size==entry['size']
  r['formal_asset']={'model_id':official['model_id'],'revision':official['revision'],'manifest_path':str(Path(a.model)/'ASSET_MANIFEST.json'),'manifest_sha256':sha(Path(a.model)/'ASSET_MANIFEST.json'),'all_file_sizes_match_pinned_manifest':True,'weight_hash_verification':'reuse prior formal publish job complete LFS SHA verification; no repeated payload hashing each GPU job'}
  stage('load_model');t=time.monotonic();model,loading=load_model(a.model,MODELS[a.key][2],a.precision=='int8',device if a.precision=='int8' else None)
  r['load_info']=loading;r['original_parameter_count']=sum(x.numel() for x in model.parameters());r['sharing_before_resize']=sharing(model)
  r['model_identity']={'class':type(model).__module__+'.'+type(model).__name__,'source_file':inspect.getfile(type(model)),'source_sha256':sha(inspect.getfile(type(model))),'config_model_type':model.config.model_type}
  assert r['original_parameter_count']==audit['model']['original_unique_parameters']
  model.resize_token_embeddings(len(tok),mean_resizing=False);r['sharing_after_resize']=sharing(model);assert r['sharing_after_resize']['same_parameter_object']
  r['expanded_base_parameter_count']=sum(x.numel() for x in model.parameters());assert r['expanded_base_parameter_count']==audit['model']['expanded_unique_parameters']
  if a.route=='L':
   if a.precision=='int8':model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
   actual_targets=[name for name,module in model.named_modules() if name.rsplit('.',1)[-1] in TARGETS]
   assert len(actual_targets)==28*7
   config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=TARGETS,modules_to_save=['embed_tokens','lm_head'],ensure_weight_tying=True,bias='none',task_type=TaskType.CAUSAL_LM)
   model=get_peft_model(model,config)
   for name,param in model.named_parameters():
    if '.original_module.' in name:param.requires_grad=False
   r['lora']={'config':model.peft_config['default'].to_dict(),'actual_target_names':actual_targets,'actual_lora_parameter_count':sum(x.numel() for n,x in model.named_parameters() if 'lora_' in n),'full_active_embedding_and_head':True,'ensure_weight_tying':True}
   # Config's set fields need stable JSON serialization.
   r['lora']['config']=json.loads(json.dumps(r['lora']['config'],default=lambda x:sorted(x) if isinstance(x,set) else str(x)))
  else:
   for param in model.parameters():param.requires_grad=True
   r['full_parameter_scope']={'official_text_class_only':a.key=='q35','excluded_vision_tensors':153 if a.key=='q35' else 0,'excluded_vision_parameters':100592896 if a.key=='q35' else 0,'excluded_mtp_tensors':15 if a.key=='q35' else 0,'excluded_mtp_parameters':20452864 if a.key=='q35' else 0,'all_instantiated_parameters_trainable':all(x.requires_grad for x in model.parameters())}
  model.config.use_cache=False;model.config.pad_token_id=tok.pad_token_id
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  if a.precision=='bf16':model.to(device)
  r['sharing_after_peft']=sharing(model);assert r['sharing_after_peft']['same_parameter_object']
  named=dict(model.named_parameters());trainable={n:x for n,x in named.items() if x.requires_grad};train_ids={id(x) for x in trainable.values()}
  r['parameter_inventory']={n:{'shape':list(x.shape),'numel':x.numel(),'dtype_before_ds':str(x.dtype),'requires_grad':x.requires_grad} for n,x in named.items()}
  r['trainable']={'tensors':len(trainable),'parameters':sum(x.numel() for x in trainable.values()),'current_unique_allocated_parameters_including_frozen_original_clones':sum(x.numel() for x in named.values()),'effective_expanded_base_parameters':audit['model']['expanded_unique_parameters']}
  r['trainable']['fraction_of_allocated_with_clones']=r['trainable']['parameters']/r['trainable']['current_unique_allocated_parameters_including_frozen_original_clones']
  r['trainable']['fraction_of_effective_base']=r['trainable']['parameters']/r['trainable']['effective_expanded_base_parameters']
  training=TrainingArguments(output_dir=str(out/'unused_trainer'),seed=42,per_device_train_batch_size=a.micro,gradient_accumulation_steps=accumulation,num_train_epochs=4,learning_rate=2e-4,weight_decay=.01,warmup_ratio=.01,lr_scheduler_type='cosine',bf16=True,optim='adamw_torch',max_grad_norm=1.,gradient_checkpointing=True,report_to=[],local_rank=local,ddp_find_unused_parameters=False)
  trainer=Trainer(model=model,args=training,train_dataset=train,data_collator=collator)
  def ident(x):
   r={'class':type(x).__module__+'.'+type(x).__name__}
   for key in ['batch_size','drop_last','even_batches','split_batches','num_processes','process_index']:
    val=getattr(x,key,None)
    if isinstance(val,(str,int,float,bool,type(None))):r[key]=val
   return r
  loader=trainer.get_train_dataloader();loader_info={'length_per_rank':len(loader),'loader':ident(loader),'batch_sampler':ident(loader.batch_sampler),'sampler':ident(loader.sampler),'iteration_executed':False}
  inner=getattr(loader.batch_sampler,'batch_sampler',None)
  if inner is not None:loader_info['inner_batch_sampler']=ident(inner);loader_info['inner_sampler']=ident(inner.sampler)
  r['formal_loader']=loader_info;formal_updates=math.ceil(len(loader)/accumulation)*4;del loader
  trainer.create_optimizer();optimizer=trainer.optimizer;trainer.create_scheduler(num_training_steps=formal_updates,optimizer=optimizer);scheduler=trainer.lr_scheduler
  selected=[x for g in optimizer.param_groups for x in g['params']];assert len(selected)==len({id(x) for x in selected}) and {id(x) for x in selected}==train_ids
  name_by_id={id(x):n for n,x in named.items()};r['optimizer']={'class':type(optimizer).__module__+'.'+type(optimizer).__name__,'groups':[{'weight_decay':g['weight_decay'],'names':[name_by_id[id(x)] for x in g['params']]} for g in optimizer.param_groups],'membership_exact_unique':True,'cpu_offload':False}
  r['training']={'micro':a.micro,'accumulation':accumulation,'global_batch':64,'seed':42,'learning_rate':2e-4,'weight_decay':.01,'temperature':.8,'scheduler':'cosine','warmup_ratio':.01,'formal_updates':formal_updates,'warmup_steps':training.get_warmup_steps(formal_updates),'gradient_checkpointing':True,'checkpointing_reentrant':False,'attention':'official sdpa, hybrid DeltaNet official torch fallback when q35','global_autocast':'DeepSpeed BF16 computation; FP32 author CE'}
  ds={'bf16':{'enabled':True},'zero_optimization':{'stage':2,'allgather_partitions':True,'allgather_bucket_size':50000000,'overlap_comm':True,'reduce_scatter':True,'reduce_bucket_size':50000000,'contiguous_gradients':True},'gradient_accumulation_steps':accumulation,'gradient_clipping':1.,'train_batch_size':64,'train_micro_batch_size_per_gpu':a.micro,'steps_per_print':100000,'wall_clock_breakdown':False,'zero_allow_untested_optimizer':True}
  r['deepspeed_config']=ds;stage('deepspeed_initialize')
  engine,_,_,scheduler=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,model_parameters=list(trainable.values()),config=ds,dist_init_required=False)
  groups=engine.optimizer.bit16_groups;flatten=[x for group in groups for x in group];assert len(flatten)==len(train_ids) and {id(x) for x in flatten}==train_ids
  r['optimizer']['zero_original_groups_exact']=True
  masters=engine.optimizer.single_partition_of_fp32_groups
  assert all(x.device.type=='cuda' and x.dtype==torch.float32 for x in masters)
  r['optimizer']['fp32_master_partitions']=[{'numel':x.numel(),'device':str(x.device),'dtype':str(x.dtype)} for x in masters]
  r['optimizer']['actual_gradient_accumulation_dtype']=str(getattr(engine.optimizer,'gradient_accumulation_dtype',None))
  sync();r['init_seconds']=time.monotonic()-t;r['sharing_after_deepspeed']=sharing(model);assert r['sharing_after_deepspeed']['same_parameter_object']
  for n,x in named.items():r['parameter_inventory'][n].update(dtype_after_ds=str(x.dtype),device_after_ds=str(x.device))
  if a.precision=='int8':
   import bitsandbytes as bnb
   quantized=[(n,m) for n,m in model.named_modules() if isinstance(m,bnb.nn.Linear8bitLt)];assert quantized;r['int8_actual_modules']=[{'name':n,'weight_dtype':str(m.weight.dtype)} for n,m in quantized];assert all(m.weight.dtype==torch.int8 for _,m in quantized)
  sid_ids=tokenmap['sid_ids'];emb=model.get_input_embeddings().weight
  input_used=sorted({x for entry in selection['train_mixed'] for x in tok(pair(train,entry)['input_ids'])['input_ids'] if x in set(sid_ids)})
  monitors={}
  for n,x in trainable.items():
   if 'lora_' in n or x is emb or n.endswith(('q_proj.weight','up_proj.weight','in_proj_qkv.weight','A_log','norm.weight')):monitors[n]=x
  def sampled(x):
   if x is emb:return x.detach()[sid_ids].float().cpu().clone()
   flat=x.detach().reshape(-1);indices=torch.linspace(0,flat.numel()-1,min(8192,flat.numel()),device=flat.device).long();return flat[indices].float().cpu().clone()
  before={n:sampled(x) for n,x in monitors.items()};grad_observation={n:{'calls':0,'nonzero_calls':0,'finite':True,'max_l2_norm':0.} for n in trainable};handles=[]
  for n,x in trainable.items():
   def hook(g,name=n,param=x):
    obs=grad_observation[name];obs['calls']+=1;finite=bool(torch.isfinite(g).all());norm=float(g.norm(dtype=torch.float32));obs['finite'] &= finite;obs['max_l2_norm']=max(obs['max_l2_norm'],norm);obs['nonzero_calls']+=int(norm>0)
    assert finite,'Nonfinite gradient: '+name
    if param is emb:
     obs['sid_gradient_max_abs']=max(obs.get('sid_gradient_max_abs',0.),float(g[sid_ids].abs().max()));obs['input_sid_gradient_max_abs']=max(obs.get('input_sid_gradient_max_abs',0.),float(g[input_used].abs().max()))
    return g
   handles.append(x.register_hook(hook))
  activations={'input_embedding_backward_calls':0,'head_logits_backward_calls':0};sid_id_tensor=torch.tensor(sid_ids,device=device)
  def embed_forward(mod,inp,output):
   if output.requires_grad:
    actual_sid_positions=torch.isin(inp[0],sid_id_tensor)
    def backward(g):
     activations['input_embedding_backward_calls']+=1;activations['input_embedding_gradient_abs_max']=max(activations.get('input_embedding_gradient_abs_max',0.),float(g.abs().max()))
     if bool(actual_sid_positions.any()):activations['actual_input_sid_position_gradient_abs_max']=max(activations.get('actual_input_sid_position_gradient_abs_max',0.),float(g[actual_sid_positions].abs().max()))
     return g
    output.register_hook(backward)
  handles.append(model.get_input_embeddings().register_forward_hook(embed_forward))
  budget=Path('/home/bwwu/lc-rec/qwen-preflight-20261110/update-budget.json')
  counts=json.loads(budget.read_text()) if budget.exists() else {'attempted_updates':[]}
  consumed=sum(x['key']==a.key and x['route']==a.route for x in counts['attempted_updates']);maximum=(4 if a.route=='L' else 2)-consumed;assert maximum>0
  r['remaining_update_allowance_at_start']=maximum
  def charge_update(index):
   if rank==0:
    state=json.loads(budget.read_text()) if budget.exists() else {'attempted_updates':[]}
    entries=state['attempted_updates'];assert sum(x['route']==a.route for x in entries)<(12 if a.route=='L' else 6)
    assert sum(x['key']==a.key and x['route']==a.route for x in entries)<(4 if a.route=='L' else 2)
    entries.append({'job_id':os.environ['GRID_JOB_ID'],'key':a.key,'route':a.route,'index_in_job':index,'run_commit':a.run_commit,'charged_before_boundary_step':True,'utc':dt.datetime.now(dt.timezone.utc).isoformat()});write(budget,state)
   dist.barrier()
  stage('bounded_training');engine.train()
  for update in range(maximum):
   # Mixed throughput uses the same 256 distinct real rows across models.
   # Last update covers the real longest-history envelope, without shortening.
   kind='boundary' if update==maximum-1 else 'mixed'
   rowinfo={'index':update,'kind':kind,'microsteps':[],'lr_before':[float(g['lr']) for g in optimizer.param_groups],'engine_step_before':engine.global_steps};r['current_update']=rowinfo;write(path,r)
   sync();global_start=time.monotonic();effective=0;padded=0;supervised=0;losses=[]
   for micro in range(accumulation):
    cpu_start=time.monotonic()
    if kind=='mixed':entries=selection['train_mixed'][(update*64+micro*2*a.micro+rank*a.micro):(update*64+micro*2*a.micro+rank*a.micro+a.micro)]
    else:entries=[{'row_idx':selection['train_representatives']['boundary']['row_idx'],'prompt_id':(micro*2*a.micro+rank*a.micro+j)%17} for j in range(a.micro)]
    assert len(entries)==a.micro
    rows=[pair(train,e) for e in entries];batch=collator(rows);effective+=int(batch['attention_mask'].sum());padded+=int(batch['input_ids'].numel());supervised+=int((batch['labels']!=-100).sum());cpu_seconds=time.monotonic()-cpu_start
    sync();phase=time.monotonic();batch={k:v.to(device) for k,v in batch.items()};sync();transfer=time.monotonic()-phase
    phase=time.monotonic();output=engine(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False);logits=output.logits
    def head_grad(g):
     activations['head_logits_backward_calls']+=1;activations['output_sid_gradient_abs_max']=max(activations.get('output_sid_gradient_abs_max',0.),float(g[...,sid_ids].abs().max()))
     targets=batch['labels'][:,1:];actual_sid_targets=torch.isin(targets,sid_id_tensor);values=g[:,:-1,:].gather(-1,targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)[actual_sid_targets]
     if values.numel():activations['actual_supervised_sid_logit_gradient_abs_max']=max(activations.get('actual_supervised_sid_logit_gradient_abs_max',0.),float(values.abs().max()))
     return g
    logits.register_hook(head_grad);value=loss(logits,batch['labels']);assert bool(torch.isfinite(value));losses.append(float(value.detach()))
    if update==0 and micro==0:
     with torch.no_grad():
      shifted=batch['labels'][:,1:];mask=shifted!=-100;selected_logits=logits.detach().float()[:,:-1,:][mask]/.8;independent=(-F.log_softmax(selected_logits,dim=-1).gather(1,shifted[mask].unsqueeze(1)).mean());delta=float((independent-value.detach()).abs());assert delta<2e-5;r['temperature_ce_independent_abs_diff']=delta
      del selected_logits,independent,shifted,mask
    sync();forward=time.monotonic()-phase;phase=time.monotonic();engine.backward(value);sync();backward=time.monotonic()-phase
    boundary=engine.is_gradient_accumulation_boundary()
    if boundary:charge_update(update)
    phase=time.monotonic();engine.step();sync();step=time.monotonic()-phase
    rowinfo['microsteps'].append({'micro':micro,'rows':entries,'loss':float(value.detach()),'input_shape':list(batch['input_ids'].shape),'cpu_collator_seconds':cpu_seconds,'transfer_seconds':transfer,'forward_seconds':forward,'backward_seconds':backward,'step_seconds':step,'boundary':boundary,'global_steps_after':engine.global_steps});write(path,r)
    del output,logits,value,batch
   sync();seconds=max_time(time.monotonic()-global_start);assert engine.global_steps==rowinfo['engine_step_before']+1
   rowinfo.update(seconds=seconds,global_effective_tokens=int(total(effective)),global_padded_tokens=int(total(padded)),global_supervised_tokens=int(total(supervised)),global_samples=64,mean_loss_rank=sum(losses)/len(losses),nonzero_lr=any(x>0 for x in rowinfo['lr_before']),lr_after=scheduler.get_last_lr(),peak=peak())
   rowinfo['samples_per_second']=64/seconds;rowinfo['effective_tokens_per_second']=rowinfo['global_effective_tokens']/seconds
   inner=engine.optimizer.optimizer;r['optimizer']['state_layout']=[{'param_dtype':str(param.dtype),'param_device':str(param.device),'numel':param.numel(),'state':{k:{'dtype':str(v.dtype),'device':str(v.device),'shape':list(v.shape)} if isinstance(v,torch.Tensor) else v for k,v in state.items()}} for param,state in inner.state.items()]
   for param,state in inner.state.items():
    assert param.device.type=='cuda' and param.dtype==torch.float32
    for key in ['exp_avg','exp_avg_sq']:assert state[key].device.type=='cuda' and state[key].dtype==torch.float32
   steps=sorted({int(state['step']) for state in inner.state.values()});assert steps==[update+1];rowinfo['adam_step_values']=steps
   r['updates'].append(rowinfo);r.pop('current_update',None);write(path,r);print(json.dumps({'rank':rank,'update':update,'seconds':seconds,'loss':rowinfo['mean_loss_rank'],'lr':rowinfo['lr_before']}),flush=True)
  r['training_peak']=peak();r['gradient_observations']=grad_observation;r['activation_gradients']=activations
  assert all(x['calls']>0 and x['finite'] for x in grad_observation.values())
  assert any(x['nonzero_lr'] for x in r['updates'])
  changed={n:{'sampled_elements':before[n].numel(),'changed_elements':int((sampled(x)!=before[n]).sum()),'max_abs_change':float((sampled(x)-before[n]).abs().max())} for n,x in monitors.items()}
  r['parameter_changes']=changed;assert changed[next(n for n,x in monitors.items() if x is emb)]['changed_elements']>0
  if a.route=='L':
   assert all(x['nonzero_calls']>0 for n,x in grad_observation.items() if 'lora_' in n)
   assert all(x['changed_elements']>0 for n,x in changed.items() if 'lora_' in n)
  else:assert any(x['changed_elements']>0 for n,x in changed.items() if monitors[n] is not emb)
  assert activations['input_embedding_backward_calls']>0 and activations['head_logits_backward_calls']>0
  assert activations['input_embedding_gradient_abs_max']>0 and activations['output_sid_gradient_abs_max']>0
  assert activations['actual_input_sid_position_gradient_abs_max']>0 and activations['actual_supervised_sid_logit_gradient_abs_max']>0
  for x in trainable.values():assert bool(torch.isfinite(x).all())
  # Full LoRA tensors, every SID row, representative full-finetune layer slices.
  hashes={n:digest_tensor(x if 'lora_' in n else x[sid_ids] if x is emb else sampled(x)) for n,x in monitors.items()};r['all_rank_parameter_hashes']=all_equal(hashes)
  r['all_rank_optimizer_steps']=all_equal([int(x) for x in steps]);r['all_rank_update_count']=all_equal(len(r['updates']))
  r['core_training_status']='PASS'
  for h in handles:h.remove()
  stage('validation');engine.eval()
  with torch.no_grad():
   for label,indices in selection['valid'].items():
    startval=time.monotonic();rows=[valid.valid_text_data[i] for i in indices[rank*8:(rank+1)*8]];batch=collator(rows);tokens=int(batch['attention_mask'].sum());supervised=int((batch['labels']!=-100).sum());padding=int(batch['input_ids'].numel())-tokens;batch={k:v.to(device) for k,v in batch.items()};sync();fstart=time.monotonic();output=engine(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False);value=loss(output.logits,batch['labels']);sync();forward=max_time(time.monotonic()-fstart);seconds=max_time(time.monotonic()-startval);assert bool(torch.isfinite(value));r['validation'].append({'label':label,'global_rows':16,'seconds_including_collator':seconds,'forward_seconds':forward,'global_tokens':int(total(tokens)),'global_supervised_tokens':int(total(supervised)),'global_padding':int(total(padding)),'loss_rank':float(value),'indices':indices});del output,value,batch
  stage('save_and_reload')
  torch.cuda.reset_peak_memory_stats(device)
  r['full_training_state_restore']='NOT TESTED';r['full_optimizer_checkpoint_written']=False
  if a.route=='L':
   adapter=out/'adapter';expected=get_peft_model_state_dict(model,save_embedding_layers=False)
   expected_cpu={k:v.detach().cpu().clone() for k,v in expected.items()}
   if rank==0:
    t=time.monotonic();model.save_pretrained(adapter,safe_serialization=False,save_embedding_layers=False);tok.save_pretrained(adapter);save_seconds=time.monotonic()-t
    disk=torch.load(adapter/'adapter_model.bin',map_location='cpu',weights_only=True);assert set(disk)==set(expected_cpu)
    for k,v in disk.items():assert torch.equal(v,expected_cpu[k]),k
    r['adapter_save']={'path':str(adapter),'seconds':save_seconds,'file_bytes':(adapter/'adapter_model.bin').stat().st_size,'sha256':sha(adapter/'adapter_model.bin'),'keys':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':digest_tensor(v)} for k,v in disk.items()},'strict_all_saved_values_match':True,'format':'native PEFT bin with full shared active embedding/head; no optimizer'}
    del disk
   dist.barrier()
   # Release the engine/optimizer before loading a complete BF16 inference base.
   del expected,trainer,scheduler,optimizer,engine,model,named,trainable,monitors,groups,flatten,selected,emb,inner,masters,param,state
   if a.precision=='int8':del quantized
   gc.collect();torch.cuda.empty_cache();dist.barrier()
   stage('reload_bf16_base');t=time.monotonic();model,info=load_model(a.model,MODELS[a.key][2],False,None);model.resize_token_embeddings(len(tok),mean_resizing=False);model.to(device);sync();r['inference_base_reload_seconds']=time.monotonic()-t
   t=time.monotonic();model=PeftModel.from_pretrained(model,adapter,is_trainable=False,autocast_adapter_dtype=False);sync();r['adapter_reload_seconds']=time.monotonic()-t
   loaded=get_peft_model_state_dict(model,save_embedding_layers=False)
   assert set(loaded)==set(expected_cpu),(set(loaded)-set(expected_cpu),set(expected_cpu)-set(loaded))
   for k,v in loaded.items():assert torch.equal(v.detach().cpu(),expected_cpu[k].to(v.dtype)),k
   del expected_cpu
   r['adapter_reload']={'all_keys_and_values_exact':True,'sharing':sharing(model),'load_info':info};assert sharing(model)['same_parameter_object']
   reloaded_tok=__import__('transformers').AutoTokenizer.from_pretrained(adapter,local_files_only=True);assert reloaded_tok.get_vocab()==tok.get_vocab();r['adapter_reload']['tokenizer_vocab_and_sid_order_exact']=True;del loaded
  else:
   # Model-only save is within the explicitly allowed optional interface test:
   # <3.5 GB/model, <6.5 GB all three. No optimizer/scheduler/RNG is written.
   folder=out/'full_model';expected_full={n:{'shape':list(x.shape),'dtype':str(x.dtype),'sha256':digest_tensor(x)} for n,x in model.state_dict().items()}
   if rank==0:
    t=time.monotonic();model.save_pretrained(folder,safe_serialization=True,max_shard_size='4GB');tok.save_pretrained(folder);seconds=time.monotonic()-t
    files=[{'path':str(x),'bytes':x.stat().st_size,'sha256':sha(x)} for x in sorted(folder.glob('*.safetensors'))]
    assert sum(x['bytes'] for x in files)<4*1024**3
    r['full_model_save']={'path':str(folder),'seconds':seconds,'files':files,'parameter_state':expected_full,'model_only':True,'optimizer_scheduler_rng_saved':False}
   dist.barrier()
   del trainer,scheduler,optimizer,engine,model,named,trainable,monitors,groups,flatten,selected,emb,inner,masters,param,state;gc.collect();torch.cuda.empty_cache();dist.barrier()
   t=time.monotonic();model,info=load_model(folder,MODELS[a.key][2],False,device);sync();r['full_model_reload_seconds']=time.monotonic()-t
   actual=model.state_dict();assert set(actual)==set(expected_full)
   for name,x in actual.items():
    assert list(x.shape)==expected_full[name]['shape'] and str(x.dtype)==expected_full[name]['dtype'] and digest_tensor(x)==expected_full[name]['sha256'],name
   r['full_model_reload']={'all_keys_shapes_dtypes_values_exact':True,'state_tensors':len(actual),'sharing':sharing(model),'loading_info':info};assert sharing(model)['same_parameter_object'];del actual,expected_full
  r['save_reload_phase_peak']=peak()
  model.eval();model.config.use_cache=True;model.generation_config.eos_token_id=tok.eos_token_id;model.generation_config.pad_token_id=tok.pad_token_id
  model.gradient_checkpointing_disable();r['beam20']=[];canonical=train.get_all_items()
  stage('beam20')
  with torch.no_grad():
   for index in range(4):
    torch.cuda.reset_peak_memory_stats(device)
    entry=selection['test_users'][index*2+rank];row=test[entry['row_idx']];inputs=tok(row['input_ids'],return_tensors='pt',padding=False,return_attention_mask=True);callback=Allowed(tok,train.indices);callback.bind(inputs['input_ids'][0].tolist(),row['input_ids']);inputs={k:v.to(device) for k,v in inputs.items()};sync();gstart=time.monotonic()
    generated=model.generate(**inputs,num_beams=20,num_return_sequences=20,max_new_tokens=10,prefix_allowed_tokens_fn=callback,output_scores=True,return_dict_in_generate=True,early_stopping=True,do_sample=False,use_cache=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
    sync();seconds=max_time(time.monotonic()-gstart);outputs=generated.sequences[:,inputs['input_ids'].shape[1]:].tolist();sequences=[]
    for ids in outputs:
     assert len(ids)>=5 and ids[4]==tok.eos_token_id
     for pos in range(4):assert ids[pos] in callback.allowed[pos]
     assert all(x==tok.pad_token_id for x in ids[5:]);sid=tok.decode(ids[:4],skip_special_tokens=False);sequences.append({'token_ids':ids,'sid':sid,'canonical':sid in canonical})
    assert set(callback.calls)==set(range(5));scores=generated.sequences_scores.tolist();assert all(math.isfinite(x) for x in scores)
    r['beam20'].append({**entry,'input_text_sha256':hashlib.sha256(row['input_ids'].encode()).hexdigest(),'input_tokens':inputs['input_ids'].shape[1],'seconds_two_rank_max':seconds,'users_per_second_dual_gpu':2/seconds,'sequences':sequences,'sequence_scores':scores,'callback_calls':callback.calls,'callback_events':callback.events,'context_response_separator_ids':callback.sep,'canonical_fraction':sum(s['canonical'] for s in sequences)/20,'peak':peak(),'beam':20,'num_return_sequences':20,'max_new_tokens':10,'temperature_training_only':True});write(path,r);del generated,inputs
  r['final_phase_peak']=peak();r['job_peak_from_recorded_phases']={key:max(part[key] for part in [r['training_peak'],r['save_reload_phase_peak']]+[x['peak'] for x in r['beam20']]) for key in ['allocated_gib','reserved_gib','ru_maxrss_gib']};r['status']='PASS_TECHNICAL';r['completed_global_updates']=len(r['updates']);r['nonzero_lr_updates']=sum(x['nonzero_lr'] for x in r['updates']);r['elapsed_seconds']=time.monotonic()-start;r['completed_utc']=dt.datetime.now(dt.timezone.utc).isoformat();write(path,r);stage('complete')
 except Exception:
  r['status']='BLOCKED';r['traceback']=traceback.format_exc()
  if torch is not None and torch.cuda.is_initialized():r['failure_peak']={'allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
  write(path,r);print(r['traceback'],flush=True);raise
 finally:
  if dist is not None and dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
