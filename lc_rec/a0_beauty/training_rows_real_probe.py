"""Compare tracked/untracked real Beauty train collation on a small CPU sample."""
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader, RandomSampler

source, data_root, model, tracker = sys.argv[1:]
sys.path[:0]=[source,tracker]
from data import SeqRecDataset
from collator import Collator
from transformers import LlamaTokenizer
from training_rows import IndexedDataset, IndexedCollator, take_indices

args=SimpleNamespace(dataset='Beauty',data_path=data_root,max_his_len=20,his_sep=', ',
    index_file='.index.json',add_prefix=False,only_train_response=True)
data=SeqRecDataset(args,mode='train',prompt_sample_num=1,sample_num=0)
assert len(data)==131413
tok=LlamaTokenizer.from_pretrained(model,local_files_only=True,model_max_length=2048,padding_side='right')
tok.pad_token_id=0
assert tok.add_tokens(data.get_new_tokens())==965
base=Collator(args,tok)


def sample(tracked):
    random.seed(42);torch.manual_seed(42)
    gen=torch.Generator().manual_seed(42)
    dataset=IndexedDataset(data) if tracked else data
    loader=DataLoader(dataset,batch_size=8,num_workers=0,sampler=RandomSampler(dataset,generator=gen),
        collate_fn=IndexedCollator(base) if tracked else base)
    batches=[];rows=[]
    for i,batch in enumerate(loader):
        if i==3:break
        if tracked:rows.extend(take_indices(batch))
        batches.append({k:v.cpu().clone() for k,v in batch.items()})
    return batches,rows


reference,_=sample(False)
tracked,rows=sample(True)
assert len(rows)==24 and len(set(rows))==24
assert len(reference)==len(tracked)==3
for left,right in zip(reference,tracked):
    assert set(left)==set(right) and all(torch.equal(left[k],right[k]) for k in left)
print(json.dumps({'status':'PASS','batches':3,'samples':24,'row_indices':rows,
                  'scope':'single-process CPU sample of actual Beauty data, not distributed training'},indent=2))
