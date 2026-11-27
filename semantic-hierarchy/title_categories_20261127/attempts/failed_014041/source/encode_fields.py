"""Text-only LLM2Vec worker: SID files are never opened by this process."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',USE_TF='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,fcntl,json,time
from pathlib import Path
import numpy as np,torch
from original_encoder import load_encoder
from runtime_common import ROOT,read_json,read_jsonl,write_json,sha,status

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',choices=['beauty','instruments'],required=True);a=p.parse_args();run=a.run
 cfg=read_json(run/'encoding_config.json');domain=a.dataset;out=run/domain;out.mkdir(exist_ok=True)
 locks=[]
 for gpu in sorted(os.environ['CUDA_VISIBLE_DEVICES'].split(','),key=int):
  for path in [ROOT/'stage_gpu_locks'/f'gpu_{gpu}.lock',ROOT/'codex/letter_lc_rec_smoke_20261121'/f'gpu{gpu}.lock']:
   f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 snapshot=__import__('subprocess').check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
 procs=__import__('subprocess').check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 for row in snapshot.splitlines():
  idx,uuid,mem=map(str.strip,row.split(','))
  if idx in os.environ['CUDA_VISIBLE_DEVICES'].split(','):assert int(mem)<512 and uuid not in procs,'GPU became occupied'
 write_json(out/'gpu_start.json',dict(devices=os.environ['CUDA_VISIBLE_DEVICES'],snapshot=snapshot,processes=procs))
 torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
 rows=read_jsonl(run/'inputs'/domain/'unique_texts.jsonl');maps=read_jsonl(run/'inputs'/domain/'item_text_map.jsonl')
 audit=read_json(run/'inputs'/domain/'text_manifest.json');assert sha(run/'inputs'/domain/'unique_texts.jsonl')==audit['text_sha256']
 assert sha(run/'inputs'/domain/'item_text_map.jsonl')==audit['map_sha256']
 status(out/'encoding_status.json','loading_model',dataset=domain)
 encoder=load_encoder(cfg);assert not encoder.training and encoder.skip_instruction and encoder.pooling_mode=='mean'
 def batch(ids):
  prepared=[encoder.prepare_for_tokenization(encoder._convert_to_str(cfg['instruction'],rows[i]['text'])) for i in ids]
  feat=encoder.tokenize(prepared);feat={k:v.to('cuda:0') for k,v in feat.items()}
  with torch.inference_mode():raw=encoder(feat).float().cpu()
  assert raw.shape==(len(ids),4096) and torch.isfinite(raw).all() and (raw.norm(dim=1)>1e-12).all()
  return raw.numpy()
 # Test longest documents in both fields and repeat one text, before any full encoding.
 selected=[]
 for label in ['Title:','Categories:']:
  ids=[i for i,r in enumerate(rows) if r['text'].startswith(label)]
  selected.extend(sorted(ids,key=lambda i:(-len(rows[i]['text']),i))[:2])
 for start in range(0,len(selected),2):batch(selected[start:start+2])
 x=batch([selected[0]])[0];y=batch([selected[0]])[0];assert np.max(np.abs(x-y))<1e-4
 paired=batch(selected[:2]);single=batch([selected[0]])
 padding_cos=float(torch.nn.functional.cosine_similarity(torch.tensor(paired[:1]),torch.tensor(single)).item());assert padding_cos>.999
 write_json(out/'encoding_preflight.json',dict(passed=True,longest_text_ids=selected,repeated_max_abs=float(np.max(np.abs(x-y))),batch_padding_cosine=padding_cos,output_dimension=4096,no_vector_quantization=True,read_inputs=['encoding_config.json',f'inputs/{domain}/text_manifest.json',f'inputs/{domain}/unique_texts.jsonl',f'inputs/{domain}/item_text_map.jsonl']))
 vectors=np.lib.format.open_memmap(out/'unique_raw.npy',mode='w+',dtype=np.float32,shape=(len(rows),4096))
 order=sorted(range(len(rows)),key=lambda i:(-len(rows[i]['text']),i));start=time.monotonic()
 for at in range(0,len(order),cfg['batch_size']):
  ids=order[at:at+cfg['batch_size']];vectors[ids]=batch(ids)
  if at%128==0 or at+2>=len(order):
   vectors.flush();done=min(at+2,len(order));rate=done/(time.monotonic()-start)
   status(out/'encoding_status.json','encoding',completed=done,total=len(rows),texts_per_second=rate,remaining_seconds=(len(rows)-done)/rate,devices=os.environ['CUDA_VISIBLE_DEVICES'])
 vectors.flush();n=len(maps);shape=(n,4096)
 title=np.lib.format.open_memmap(out/'title_raw.npy',mode='w+',dtype=np.float32,shape=shape)
 category=np.lib.format.open_memmap(out/'categories_raw.npy',mode='w+',dtype=np.float32,shape=shape)
 fused=np.lib.format.open_memmap(out/'fused_normalized.npy',mode='w+',dtype=np.float32,shape=shape)
 titlez=np.lib.format.open_memmap(out/'title_normalized.npy',mode='w+',dtype=np.float32,shape=shape)
 masks=[];norms=[]
 for i,r in enumerate(maps):
  ti,ci=r['title_text_id'],r['categories_text_id'];t=vectors[ti] if ti>=0 else np.zeros(4096,dtype=np.float32);c=vectors[ci] if ci>=0 else np.zeros(4096,dtype=np.float32)
  title[i]=t;category[i]=c;m=(t+c)/(int(ti>=0)+int(ci>=0)) if ti>=0 or ci>=0 else t
  norm=float(np.linalg.norm(m.astype(np.float64)));tn=float(np.linalg.norm(t.astype(np.float64)));cn=float(np.linalg.norm(c.astype(np.float64)))
  assert norm>1e-12 or (ti<0 and ci<0)
  fused[i]=m/norm if norm else m;titlez[i]=t/tn if tn else t
  masks.append([ti>=0,ci>=0]);norms.append([tn,cn,norm])
 for v in [title,category,fused,titlez]:v.flush()
 np.save(out/'field_valid.npy',np.array(masks));np.save(out/'field_norms.npy',np.array(norms,dtype=np.float64))
 valid=np.array(masks).any(axis=1);assert np.isfinite(fused).all() and np.max(abs(np.linalg.norm(fused[valid].astype(np.float64),axis=1)-1))<1e-5
 lengths=[len(encoder.tokenizer(r['text'],add_special_tokens=False)['input_ids']) for r in rows]
 write_json(out/'encoding_complete.json',dict(complete=True,items=n,unique_texts=len(rows),seconds=time.monotonic()-start,input_text_sha256=audit['text_sha256'],map_sha256=audit['map_sha256'],
  outputs={p.name:sha(p) for p in out.glob('*.npy')},tokens_over_doc_limit=sum(v>224 for v in lengths),tokens_p50_p95_max=np.quantile(lengths,[.5,.95,1]).tolist(),physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'],max_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in [0,1]],field_average_before_l2=True))
 status(out/'encoding_status.json','complete',items=n)
if __name__=='__main__':main()
