#!/usr/bin/env python3
"""Bind a saved preflight adapter and frozen 32-user axis to the audited Shuffle core."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'shuffle'))
from core import transform
from formal_evaluate import catalog_and_users
from preflight_users import validate
from uneven_sampler import assignment, validate_assignments

p=argparse.ArgumentParser()
for x in ('adapter','model','data-root','source-root','manifest','normal-test-dir','output'):p.add_argument('--'+x,required=True)
a=p.parse_args()
sys.path.insert(0,a.source_root)
from data import SeqRecDataset
from transformers import LlamaTokenizer
adapter=Path(a.adapter); out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
args=SimpleNamespace(dataset='Beauty',data_path=a.data_root,max_his_len=20,his_sep=', ',index_file='.index.json',add_prefix=False)
data=SeqRecDataset(args,mode='test',sample_num=-1);data.set_prompt(0)
_,_,users,identities=catalog_and_users(a.data_root)
shards=[assignment(len(data),rank,2) for rank in (0,1)];validate_assignments(shards,len(data))
manifest_path=Path(a.manifest);manifest=json.loads(manifest_path.read_text())
assert manifest['asset_sha256']=={k:v['sha256'] for k,v in identities.items()}
selected=validate(manifest,users,shards)
tok=LlamaTokenizer.from_pretrained(adapter,local_files_only=True);tok.pad_token_id=0;tok.model_max_length=2048;tok.padding_side='left'
ref=LlamaTokenizer.from_pretrained(a.model,local_files_only=True);ref.pad_token_id=0;ref.add_tokens(data.get_new_tokens())
assert len(tok)==len(ref)==32965 and tok.get_vocab()==ref.get_vocab()
tokens=sorted({t for parts in data.indices.values() for t in parts})
lookup={t:tok.encode(t,add_special_tokens=False)[0] for t in tokens}
assert len(tokens)==len(set(lookup.values()))==965 and all(tok.encode(t,add_special_tokens=False)==[v] for t,v in lookup.items())
checks=[]
for rows in selected:
    for i in rows:
        uid=users[i]['user_id'];seq=data.inters[uid]
        groups=[[lookup[t] for t in data.indices[str(item)]] for item in seq[:-1][-20:]]
        row=data[i];assert row['labels']==users[i]['target']
        encoded=tok(row['input_ids'],return_attention_mask=True)
        ids,mask=encoded['input_ids'],encoded['attention_mask']
        for n in (0,3,4):
            changed,evidence=transform(ids,mask,groups,set(lookup.values()),n,i)
            assert changed==ids if n==0 else changed!=ids
        checks.append(i)
assert len(checks)==len(set(checks))==32
result={'status':'PASS_CPU_PREFLIGHT_BOUND','model_key':'a0','preflight_assignments':{str(i):selected[i] for i in (0,1)},
        'manifest':{'path':str(manifest_path.resolve()),'sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
        'token_ids':lookup,'normal_test_dir':a.normal_test_dir,
        'saved_tokenizer_files':{f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
          for f in adapter.iterdir() if f.is_file() and (f.name.startswith('tokenizer') or f.name in ('added_tokens.json','special_tokens_map.json','vocab.json','merges.txt'))},
        'scope':'32 fixed users; relies on earlier full-user Beauty CPU preflight for full-axis checks'}
with (out/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'users':len(checks),'manifest_sha256':result['manifest']['sha256']}))
