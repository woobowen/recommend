"""Verify uploaded derived inputs and write CPU SID tensors; protect all original runs."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,re

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
ORIGINAL=ROOT/'codex/instrument_semantic_hierarchy_20261115'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jsonl(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 import torch,numpy as np
 aligned=HERE/'aligned';audit=json.loads((aligned/'alignment_verification.json').read_text())
 for name,row in audit['outputs'].items():assert sha(aligned/name)==row['sha256'],name
 mapping=jsonl(aligned/'item_mapping.jsonl');ids=[r['qwen_original_item_id'] for r in mapping]
 oldpath=ORIGINAL/'runs/20261115_104723/upstream/semantic_ids.pt'
 assert sha(oldpath)=='0f12825d1de428e8cba9d9827537be229446f0f076effbf6bd695efa42c83015'
 old=torch.load(oldpath,map_location='cpu',weights_only=False).long();assert list(old.shape)==[4,10611]
 original_mapping=jsonl(ORIGINAL/'inputs/instrument/item_mapping.jsonl')
 assert sha(ORIGINAL/'inputs/instrument/item_mapping.jsonl')=='262c3bc796ece5826f2e171f585c1664750e9d5b3ef336b5d13899f9715246e8'
 assert all(original_mapping[r['qwen_original_item_id']]['asin']==r['asin'] for r in mapping)
 selected=old[:,ids].contiguous()
 expected=torch.tensor([r['sid'] for r in jsonl(aligned/'qwen_item_sids.jsonl')],dtype=torch.long).T.contiguous()
 assert torch.equal(selected,expected) and len(torch.unique(selected.T,dim=0))==9922
 official=ROOT/'third_party/LETTER-8d0154e/data/Instruments'
 for name in ['Instruments.index.json','Instruments.inter.json','Instruments.item.json']:assert (official/name).read_bytes()==(aligned/name).read_bytes(),name
 codes=json.loads((aligned/'Instruments.index.json').read_text())
 letter=[]
 for i in range(9922):
  row=[]
  for level,token in zip('abcd',codes[str(i)]):
   match=re.fullmatch('<'+level+r'_(\d+)>',token);assert match;row.append(int(match.group(1)))
  assert len(codes[str(i)])==4;letter.append(row)
 letter=torch.tensor(letter,dtype=torch.long).T.contiguous()
 for name,tensor in [('qwen_semantic_ids.pt',selected),('letter_semantic_ids.pt',letter)]:
  path=aligned/name;assert not path.exists(),'Derived tensor already exists; do not overwrite'
  torch.save(tensor,path);assert torch.equal(torch.load(path,map_location='cpu',weights_only=False),tensor)
 np.save(aligned/'qwen_original_item_ids.npy',np.array(ids,dtype=np.int64))
 # Preserve reindexed original Qwen fields separately from shared LETTER evaluation text.
 original_meta=jsonl(ORIGINAL/'inputs/instrument/items_metadata.jsonl')
 with (aligned/'qwen_original_metadata_aligned.jsonl').open('w') as f:
  for r in mapping:
   record=original_meta[r['qwen_original_item_id']];assert record['asin']==r['asin']
   f.write(json.dumps({**record,'item_id':r['item_id'],'qwen_original_item_id':r['qwen_original_item_id']},ensure_ascii=False)+'\n')
 result=dict(passed=True,server=str(HERE),items=9922,removed=689,common_axis='LETTER original IDs 0..9921',
  qwen_shape=list(selected.shape),letter_shape=list(letter.shape),qwen_sid_value_differences=0,
  mapping_all_9922_asins_checked=True,official_three_files_byte_identical=True,
  qwen_unique_full_sids=len(torch.unique(selected.T,dim=0)),letter_unique_full_sids=len(torch.unique(letter.T,dim=0)),
  gpu_used=False,original_assets_modified=False,
  original_qwen_categories_nonempty=sum(bool(original_meta[i].get('category')) for i in ids),
  source_qwen_sha256=sha(oldpath),
  outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in aligned.iterdir() if p.is_file()})
 write(HERE/'server_verification.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
