"""LETTER text/data semantics plus explicit Qwen tokenizer/architecture adapters."""
import hashlib, json, random, sys
from pathlib import Path
from types import SimpleNamespace

MODELS={
 'q06':('Qwen3-0.6B-Base','da87bfb608c14b7cf20ba1ce41287e8de496c0cd','qwen3'),
 'q17':('Qwen3-1.7B-Base','ea980cb0a6c2ae4b936e82123acc929f1cec04c1','qwen3'),
 'q35':('Qwen3.5-0.8B-Base','dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68','qwen35')}
ASSET_SHA={'index':'33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd','inter':'0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743','item':'1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2'}
TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def write(path,value):
 def container_default(obj):
  if isinstance(obj,(set,frozenset)):return sorted(obj)
  raise TypeError('Unsupported evidence JSON type: '+type(obj).__name__)
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=container_default)+'\n');tmp.replace(path)

def datasets(letter):
 import numpy as np
 random.seed(42);np.random.seed(42)
 letter=Path(letter);sys.path.insert(0,str(letter/'LETTER-LC-Rec'))
 from data import SeqRecDataset
 args=SimpleNamespace(dataset='Instruments',data_path=str(letter/'data'),index_file='.index.json',max_his_len=20,his_sep=', ',add_prefix=False,only_train_response=True,sample_valid=True,valid_prompt_id=0)
 for name,digest in ASSET_SHA.items():assert sha(letter/'data/Instruments'/('Instruments.'+name+'.json'))==digest
 train=SeqRecDataset(args,mode='train',prompt_sample_num=1,sample_num=0)
 valid=SeqRecDataset(args,mode='valid',prompt_sample_num=2)
 test=SeqRecDataset(args,mode='test',prompt_id=0,sample_num=-1)
 assert (len(train),len(valid),len(test))==(131837,49544,24772)
 assert len(train.get_new_tokens())==918
 return train,valid,test

def tokenizer_for(path,tokens):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False,padding_side='right',model_max_length=2048)
 original=len(tok);added=tok.add_tokens(tokens);assert added==918
 ids=[tok.encode(t,add_special_tokens=False) for t in tokens]
 assert all(len(i)==1 for i in ids);ids=[i[0] for i in ids]
 assert len(set(ids))==918 and all(i not in tok.all_special_ids for i in ids)
 # Preserve real Base EOS; PAD may equal EOS. Masks use attention_mask.
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 assert tok.eos_token_id is not None and tok.pad_token_id is not None
 return tok,{'original_tokenizer_size':original,'expanded_tokenizer_size':len(tok),'added_sid_tokens':added,'sid_ids':ids,'sid_tokens':tokens,'special_tokens_map':tok.special_tokens_map,'bos_token_id':tok.bos_token_id,'eos_token_id':tok.eos_token_id,'pad_token_id':tok.pad_token_id,'encodes_automatic_bos':tok.encode('Response:')!=tok.encode('Response:',add_special_tokens=False),'response_separator_ids':tok.encode('Response:',add_special_tokens=False)}

def pair(train,entry):
 inp,full=train._get_text_data(train.inter_data[entry['row_idx']],train.prompts[entry['prompt_id']])
 return {'input_ids':inp,'labels':full}

class Collator:
 def __init__(self,tokenizer):self.tokenizer=tokenizer
 def __call__(self,rows):
  import torch
  t=self.tokenizer
  inputs=t([r['labels']+t.eos_token for r in rows],padding='longest',truncation=True,max_length=2048,return_tensors='pt',return_attention_mask=True)
  labels=inputs['input_ids'].clone();labels[inputs['attention_mask']==0]=-100
  for i,r in enumerate(rows):
   prefix=t(r['input_ids'],truncation=True,max_length=2048)['input_ids']
   valid=int(inputs['attention_mask'][i].sum());full=inputs['input_ids'][i,:valid].tolist()
   assert full[:len(prefix)]==prefix,'Tokenizer changed original response boundary'
   labels[i,:len(prefix)]=-100
   expected=t(r['labels'][len(r['input_ids']):],add_special_tokens=False)['input_ids']+[t.eos_token_id]
   assert len(expected)==5 and labels[i][labels[i]!=-100].tolist()==expected,'Response mask must supervise four SID plus EOS'
   assert max(expected)<len(t)
  inputs['labels']=labels
  return inputs

def loss(logits,labels):
 import torch.nn.functional as F
 # This is the author's float32, full-vocabulary, causal CE at temperature .8.
 return F.cross_entropy((logits.float()[:,:-1,:]/.8).reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='mean')

def make_selection(train,valid,test):
 # Same rows/prompts and original natural-language bytes for every tokenizer.
 rng=random.Random(42);pool=[{'row_idx':i,'prompt_id':rng.randrange(17)} for i in rng.sample(range(len(train)),256)]
 ordered=sorted(range(len(train)),key=lambda i:(len(train.inter_data[i]['inters']),i))
 representatives={k:{'row_idx':ordered[min(len(ordered)-1,int(q*(len(ordered)-1)))],'prompt_id':0} for k,q in [('short',.1),('middle',.5),('long',.9),('boundary',1.)]}
 test_order=sorted(range(len(test)),key=lambda i:(len(test[i]['input_ids']),i))
 users=[]
 for label,q in [('short',.1),('middle',.5),('long',.9),('boundary',1.)]:
  at=min(len(test_order)-2,int(q*(len(test_order)-1)))
  users.extend({'label':label,'row_idx':test_order[i],'user_id':list(test.inters)[test_order[i]]} for i in [at,at+1])
 val_order=sorted(range(len(valid)),key=lambda i:(len(valid.valid_text_data[i]['input_ids']),i))
 vals={k:[val_order[min(len(val_order)-1,int(q*(len(val_order)-1)))]]*16 for k,q in [('short',.1),('middle',.5),('long',.9),('boundary',1.)]}
 vals['mixed']=[rng.randrange(len(valid)) for _ in range(16)]
 return {'rule':'seed42 unique random256 train rows with fixed random author template; representatives by raw history character length then row index; test prompt0 by raw prompt character length then row index; never tokenizer-dependent selection','seed':42,'train_mixed':pool,'train_representatives':representatives,'valid':vals,'test_users':users}

def model_class_and_config(path,family):
 from transformers import AutoConfig
 config=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False)
 if family=='qwen35':
  from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
  assert config.model_type in ['qwen3_5','qwen3_5_text']
  # Official text class and checkpoint conversion; no hand-edited weights.
  if config.model_type=='qwen3_5':config=Qwen3_5TextConfig(**config.text_config.to_dict())
  cls=Qwen3_5ForCausalLM
 else:
  from transformers import Qwen3ForCausalLM
  assert config.model_type=='qwen3';cls=Qwen3ForCausalLM
 return cls,config

def load_model(path,family,quantized=False,device=None):
 import torch
 cls,config=model_class_and_config(path,family)
 kwargs={'config':config,'local_files_only':True,'trust_remote_code':False,'torch_dtype':torch.bfloat16,'attn_implementation':'sdpa','output_loading_info':True}
 if quantized:
  from transformers import BitsAndBytesConfig
  kwargs.update(quantization_config=BitsAndBytesConfig(load_in_8bit=True),device_map={'':device})
 model,info=cls.from_pretrained(path,**kwargs)
 assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),info
 assert not info.get('unexpected_keys'),info
 if device is not None and not quantized:model.to(device)
 return model,info

def sharing(model):
 inp=model.get_input_embeddings().weight;out=model.get_output_embeddings().weight
 return {'same_parameter_object':inp is out,'same_storage':inp.data_ptr()==out.data_ptr(),'input_shape':list(inp.shape),'output_shape':list(out.shape),'input_dtype':str(inp.dtype),'output_dtype':str(out.dtype),'input_requires_grad':inp.requires_grad,'output_requires_grad':out.requires_grad}

class Allowed:
 def __init__(self,tok,indices):
  self.tok=tok;self.sep=None;self.prefix=None;self.allowed={i:set() for i in range(4)};self.eos=tok.eos_token_id;self.calls={};self.events=[]
  for parts in indices.values():
   assert len(parts)==4
   for i,t in enumerate(parts):
    ids=tok.encode(t,add_special_tokens=False);assert len(ids)==1;self.allowed[i].add(ids[0])
  self.allowed[4]={self.eos};assert [len(self.allowed[i]) for i in range(5)]==[150,256,256,256,1]
 def bind(self,prompt_ids,text):
  # Byte BPE encodes the space before Response in its token. Derive the
  # delimiter from its real context, then bind the known original prompt end.
  # This preserves position-only constraints without inventing a trie.
  assert text.endswith('### Response:')
  actual=self.tok(text,add_special_tokens=True,return_offsets_mapping=True)
  assert actual['input_ids']==list(prompt_ids)
  begin=len(text)-len('Response:')
  self.sep=[token for token,(lo,hi) in zip(actual['input_ids'],actual['offset_mapping']) if hi>begin]
  assert self.sep and list(prompt_ids)[-len(self.sep):]==self.sep
  self.prefix=list(prompt_ids)
 def __call__(self,batch_id,sentence):
  assert self.prefix is not None,'Bind the actual original prompt first'
  sent=sentence.tolist();assert sent[:len(self.prefix)]==self.prefix
  start=len(self.prefix);assert sent[start-len(self.sep):start]==self.sep
  pos=len(sent)-start;assert pos in self.allowed,('Unexpected generation position',pos)
  self.calls[pos]=self.calls.get(pos,0)+1
  if len(self.events)<16:self.events.append({'position':pos,'batch_id':batch_id,'allowed_count':len(self.allowed[pos])})
  return sorted(self.allowed[pos])
