"""Freeze text inputs before reading SID; verify common axes without changing originals."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
from pathlib import Path
import argparse,ast,csv,html,json,re
from runtime_common import ROOT,read_json,read_jsonl,write_json,write_jsonl,sha

def clean(x):
 return re.sub(r'\s+',' ',html.unescape(str(x or ''))).strip()
def title(x):
 s=clean(x)
 return '' if s.lower() in ['','nan','none','unknown'] or re.fullmatch(r'title_\d+',s) else s
def categories(x):
 if isinstance(x,list):
  if any(isinstance(v,list) for v in x):return '\n'.join(filter(None,(categories(v) for v in x)))
  return ' > '.join(filter(None,(clean(v) for v in x)))
 s=clean(x)
 return '' if s.lower() in ['','nan','none','unknown','[]'] else s
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
 import numpy as np,tensorflow as tf,torch
 tf.config.set_visible_devices([],'GPU');opts=tf.data.Options();opts.threading.private_threadpool_size=1
 source=run/'source';identity=list(csv.DictReader((source/'beauty_g3_item_mapping.csv').open()))
 assert len(identity)==12101
 assert [int(r['sid_column']) for r in identity]==[int(r['grid_item_id']) for r in identity]==list(range(12101))
 assert [int(r['p5_item_id']) for r in identity]==list(range(1,12102)) and len({r['original_item_id'] for r in identity})==12101
 raw={};paths=sorted((ROOT/'original_assets/data/amazon_data/beauty/items').glob('*.tfrecord.gz'))
 for data in tf.data.TFRecordDataset([str(p) for p in paths],compression_type='GZIP').with_options(opts):
  f=tf.train.Example.FromString(data.numpy()).features.feature;i=int(f['id'].int64_list.value[0]);assert i not in raw
  text=f['text'].bytes_list.value[0].decode();m=re.fullmatch(r'Title: (.*); Brand: (.*); Categories: (.*); Price: (.*);\s*',text,flags=re.DOTALL);assert m,(i,text)
  raw[i]=dict(title=title(m[1]),categories=categories(ast.literal_eval(m[3])),asin=identity[i]['original_item_id'])
 assert set(raw)==set(range(12101))
 author=ROOT/'codex/letter_lc_rec_smoke_20261121/vendor/LETTER/data'
 items=read_json(author/'Beauty/Beauty.item.json');prior=read_json(ROOT/'codex/letter_upstream_transformer_20261115/inputs/beauty/mapping_audit.json')
 assert prior['complete'] and prior['title_mismatch_count']==0 and all(v['mismatched_sequences']==0 for v in prior['splits'].values())
 for i,r in raw.items():
  if r['title']:assert re.sub(r'[^\w]','',r['title']).lower()==re.sub(r'[^\w]','',title(items[str(i)]['title'])).lower(),i
 alignment=ROOT/'codex/instruments_letter_alignment_20261127/aligned'
 check=read_json(alignment/'alignment_verification.json');assert check['complete'] and check['aligned_items']==9922 and check['mapping_one_to_one']
 for name,row in check['outputs'].items():assert sha(alignment/name)==row['sha256'],name
 common=read_jsonl(alignment/'items_title_categories.jsonl');mapping=read_jsonl(alignment/'item_mapping.jsonl')
 assert len(common)==len(mapping)==9922
 instrument=[]
 for i,r in enumerate(common):
  assert r['item_id']==mapping[i]['item_id']==i and r['asin']==mapping[i]['asin']
  instrument.append(dict(item_id=i,asin=r['asin'],title=title(r['title']),categories=categories(r['categories'])))
 domains={'beauty':[dict(item_id=i,**raw[i]) for i in range(12101)],'instruments':instrument}
 # Text construction/validity masks are independent of SID values.
 audits={}
 for domain,rows in domains.items():
  dest=run/'inputs'/domain;dest.mkdir(parents=True)
  write_jsonl(dest/'items.jsonl',rows)
  unique=[];lookup={}
  for r in rows:
   for field,label in [('title','Title'),('categories','Categories')]:
    text=f'{label}: '+r[field] if r[field] else None
    if text is not None and text not in lookup:lookup[text]=len(unique);unique.append(dict(item_id=len(unique),text=text))
    r[field+'_text_id']=lookup.get(text,-1)
  write_jsonl(dest/'item_text_map.jsonl',rows);write_jsonl(dest/'unique_texts.jsonl',unique)
  audits[domain]=dict(items=len(rows),unique_texts=len(unique),valid_title=sum(bool(r['title']) for r in rows),valid_categories=sum(bool(r['categories']) for r in rows),
   complete_fields=sum(bool(r['title']) and bool(r['categories']) for r in rows),fallback_only=sum(bool(r['title'])!=bool(r['categories']) for r in rows),
   neither=sum(not r['title'] and not r['categories'] for r in rows),text_sha256=sha(dest/'unique_texts.jsonl'),map_sha256=sha(dest/'item_text_map.jsonl'))
  write_json(dest/'text_manifest.json',audits[domain])
 # Only now associate frozen code tables with the independent text axis.
 bq=read_jsonl(ROOT/'codex/llm2vec_sid_hierarchy_20261113/inputs/beauty/item_sids.jsonl')
 assert [r['item_id'] for r in bq]==list(range(12101))
 for i,r in enumerate(bq):assert r['sid']==[int(identity[i]['sid_'+str(d)]) for d in range(1,5)],i
 def decode(path):
  rows=read_json(path)
  return [dict(item_id=i,sid=[int(re.fullmatch('<'+l+r'_(\d+)>',t)[1]) for l,t in zip('abcd',rows[str(i)])]) for i in range(len(rows))]
 codes={'beauty':{'qwen':bq,'letter':decode(author/'Beauty/Beauty.index.json')},'instruments':{'qwen':read_jsonl(alignment/'qwen_item_sids.jsonl'),'letter':decode(alignment/'Instruments.index.json')}}
 old_path=ROOT/'codex/instrument_semantic_hierarchy_20261115/runs/20261115_104723/upstream/semantic_ids.pt'
 assert sha(old_path)=='0f12825d1de428e8cba9d9827537be229446f0f076effbf6bd695efa42c83015'
 old=torch.load(old_path,map_location='cpu',weights_only=False)
 for i,r in enumerate(codes['instruments']['qwen']):assert r['sid']==old[:,mapping[i]['qwen_original_item_id']].tolist()
 for domain,methods in codes.items():
  for method,rows in methods.items():
   assert len(rows)==audits[domain]['items'] and [r['item_id'] for r in rows]==list(range(len(rows)))
   write_jsonl(run/'inputs'/domain/(method+'_sids.jsonl'),rows)
  audits[domain]['sid_sha256']={m:sha(run/'inputs'/domain/(m+'_sids.jsonl')) for m in methods}
 write_json(run/'input_audit.json',dict(passed=True,domains=audits,common_embedding_per_dataset=True,alignment_verification_sha256=sha(alignment/'alignment_verification.json'),beauty_identity_sha256=sha(source/'beauty_g3_item_mapping.csv'),beauty_mapping_audit_sha256=sha(ROOT/'codex/letter_upstream_transformer_20261115/inputs/beauty/mapping_audit.json')))
 cfg=dict(seed=20261113,max_length=256,doc_max_length=224,batch_size=2,output_dimension=4096,instruction='Represent this product for semantic similarity:',dtype='float16',pooling='mean',skip_instruction=True,field_l2_before_average=False,final_l2=True)
 write_json(run/'encoding_config.json',cfg)
 print(json.dumps(audits),flush=True)
if __name__=='__main__':main()
