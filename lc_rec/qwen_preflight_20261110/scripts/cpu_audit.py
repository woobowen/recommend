"""CPU-only real-tokenizer/data checks and exact meta-model resource analysis."""
import argparse, collections, hashlib, json, math, os, random, time, traceback
from pathlib import Path
from common import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--key',choices=MODELS,required=True);p.add_argument('--letter',required=True);p.add_argument('--output',required=True);p.add_argument('--run-commit',required=True);a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);r={'status':'RUNNING','argv':__import__('sys').argv,'run_commit':a.run_commit,'key':a.key,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write(out/'audit.json',r)
 try:
  import torch
  from accelerate import init_empty_weights
  train,valid,test=datasets(a.letter);tokens=train.get_new_tokens();tok,identity=tokenizer_for(a.model,tokens)
  r['data']={'sha256':ASSET_SHA,'train_rows':len(train),'validation_rows':len(valid),'test_users':len(test),'templates':len(train.prompts),'item_count':len(train.indices),'sid_tokens':len(tokens)}
  r['tokenizer']=identity;selection=make_selection(train,valid,test);write(out/'selection.json',selection)
  collator=Collator(tok);samples=[]
  entries=list(selection['train_representatives'].items())+[('mixed'+str(i),x) for i,x in enumerate(selection['train_mixed'][:16])]
  for label,entry in entries:
   row=pair(train,entry);batch=collator([row]);samples.append({'label':label,**entry,'input_text':row['input_ids'],'full_text':row['labels'],'input_text_sha256':hashlib.sha256(row['input_ids'].encode()).hexdigest(),'full_text_sha256':hashlib.sha256(row['labels'].encode()).hexdigest(),'input_ids':batch['input_ids'][0].tolist(),'labels':batch['labels'][0].tolist(),'valid_tokens':int(batch['attention_mask'].sum()),'supervised_tokens':int((batch['labels']!=-100).sum())})
  mixed=collator([pair(train,x) for x in selection['train_mixed'][:8]])
  r['mask']={'checked_samples':len(samples),'batch_padding_positions':int((mixed['attention_mask']==0).sum()),'padding_labels_all_ignored':bool((mixed['labels'][mixed['attention_mask']==0]==-100).all()),'supervision':'exactly4 SID+realEOS, answer-only','automatic_chat_template':False}
  tok.save_pretrained(out/'tokenizer')
  from transformers import AutoTokenizer
  restored=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True);assert restored.get_vocab()==tok.get_vocab()
  r['tokenizer']['save_reload_vocab_exact']=True;write(out/'samples.json',samples)
  for parts in train.indices.values():
   expected=''.join(parts)
   for tokenizer in [tok,restored]:
    ids=tokenizer.encode(expected,add_special_tokens=False)
    assert ids==[tokenizer.convert_tokens_to_ids(token) for token in parts] and len(ids)==4
    assert tokenizer.decode(ids,skip_special_tokens=False)==expected
  r['tokenizer']['all_9922_item_sids_encode_decode_before_and_after_reload_exact']=True
  # Full CPU length denominators: one fixed seed42 author-template draw per
  # original training row, all already-fixed validation texts, all test prompt0.
  histograms={};rng=random.Random(42);train_prompt_ids=[rng.randrange(17) for _ in range(len(train))]
  for name,dataset in [('train',train),('validation',valid),('test_prompt0',test)]:
   hist=collections.Counter();valid_total=0;maximum=0;max_row=None
   for begin in range(0,len(dataset),512):
    texts=[]
    for i in range(begin,min(begin+512,len(dataset))):
     if name=='train':x=pair(train,{'row_idx':i,'prompt_id':train_prompt_ids[i]});text=x['labels']+tok.eos_token
     elif name=='validation':text=dataset.valid_text_data[i]['labels']+tok.eos_token
     else:text=dataset[i]['input_ids']
     texts.append(text)
    encoded=tok(texts,add_special_tokens=True,truncation=False)['input_ids']
    for offset,ids in enumerate(encoded):
     size=len(ids);hist[size]+=1;valid_total+=size
     if size>maximum:maximum=size;max_row=begin+offset
   assert maximum<=2048,'Original fixed-history data exceeds authorized max_length; stop dependent training rather than silently dropping history'
   histograms[name]={'rows':len(dataset),'histogram':dict(sorted(hist.items())),'total_tokens':valid_total,'mean_tokens':valid_total/len(dataset),'max_tokens':maximum,'max_row_idx':max_row,'template_policy':'fixed seed42 random author-template per row' if name=='train' else 'author Dataset text','truncation_used':False}
   r['lengths']=histograms;write(out/'audit.json',r)
  # No weight payload allocation. Count actual architecture with tied weights.
  cls,config=model_class_and_config(a.model,MODELS[a.key][2]);base_vocab=config.vocab_size
  with init_empty_weights():model=cls(config)
  model.tie_weights();base_numel=sum(x.numel() for x in model.parameters())
  model.resize_token_embeddings(len(tok),mean_resizing=False);model.tie_weights()
  expanded_numel=sum(x.numel() for x in model.parameters());assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
  inventory={n:{'shape':list(x.shape),'numel':x.numel()} for n,x in model.named_parameters()}
  r['model']={'class':cls.__name__,'base_vocab_rows':base_vocab,'expanded_vocab_rows':len(tok),'physical_rows_added':len(tok)-base_vocab,'original_unique_parameters':base_numel,'expanded_unique_parameters':expanded_numel,'tied_embedding_head':True,'extension_initialization':'legacy LETTER-compatible normal initializer, mean_resizing=False; all918IDs are new tokenizer identities, some IDs reuse padded original weight rows','excluded_official_source_components':['visual:100592896','MTP:20452864'] if a.key=='q35' else [],'parameter_inventory':inventory}
  max_len=histograms['train']['max_tokens'];n=expanded_numel
  # FP32 master + two FP32 moments, BF16 weights, partitioned BF16 gradients.
  memory={'bf16_weight_bytes_per_gpu':2*n,'zero2_gradient_bytes_per_gpu':n,'zero2_fp32_master_bytes_per_gpu':2*n,'zero2_adam_moments_bytes_per_gpu':4*n,'persistent_zero2_bytes_per_gpu':9*n,'ddp_persistent_bytes_per_gpu':16*n,'max_real_train_length':max_len,'max_allowed_length':2048,'micro8_fp32_logits_bytes':8*max_len*len(tok)*4,'communication_bucket_elements':50000000,'activation_and_temporary_reserve_bytes':5*1024**3,'estimate_not_measurement':True}
  memory['zero2_static_with_logits_and_reserve_gib']=(9*n+memory['micro8_fp32_logits_bytes']*3+memory['activation_and_temporary_reserve_bytes'])/1024**3
  r['full_parameter_static']=memory
  r['checkpoint_static']={'model_weights_bytes':2*n,'fp32_master_plus_adam_bytes_global':12*n,'gradient_bytes_not_normally_saved':2*n,'scheduler_rng_cost':'nonzero unknown, NOT TESTED','full_optimizer_checkpoint_created':False}
  if a.key=='q35':r['lora_status']='BLOCKED_MAPPING_REQUIRES_APPROVAL';r['lora_reason']='18/24 mixers are DeltaNet with fused QKV/gates/step projections; author seven-target list omits them. No new mapping authorized.'
  else:
   from peft import LoraConfig,TaskType,get_peft_model
   config_l=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=TARGETS,modules_to_save=['embed_tokens','lm_head'],ensure_weight_tying=True,bias='none',task_type=TaskType.CAUSAL_LM)
   model=get_peft_model(model,config_l)
   assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
   r['lora_static']={'lora_parameters':sum(x.numel() for name,x in model.named_parameters() if 'lora_' in name),'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),'targets':TARGETS,'ensure_weight_tying':True,'modules_to_save':['embed_tokens','lm_head'],'meta_only':True}
  callback=Allowed(tok,train.indices);r['allowed_position_sizes']=[len(callback.allowed[i]) for i in range(5)];delimiter_checks=[]
  for user in selection['test_users']:
   text=test[user['row_idx']]['input_ids'];ids=tok(text)['input_ids'];callback.bind(ids,text);assert set(callback(0,torch.tensor(ids)))==callback.allowed[0]
   current=ids[:]
   for pos in range(4):current.append(min(callback.allowed[pos]));assert set(callback(0,torch.tensor(current)))==callback.allowed[pos+1]
   delimiter_checks.append({'row_idx':user['row_idx'],'actual_context_separator_ids':callback.sep,'standalone_response_ids':tok.encode('Response:',add_special_tokens=False),'input_tokens':len(ids)})
  r['delimiter_checks']=delimiter_checks
  assert not torch.cuda.is_initialized();r['cuda_initialized']=False;r['status']='PASS_CPU_TOKENIZER_AND_STATIC';write(out/'audit.json',r)
 except Exception:
  r['status']='BLOCKED';r['traceback']=traceback.format_exc();write(out/'audit.json',r);raise
if __name__=='__main__':main()
