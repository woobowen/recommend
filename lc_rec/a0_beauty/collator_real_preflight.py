#!/usr/bin/env python3
"""CPU-only regression of real Beauty rows and the actual AuditTrainer eval method."""
import ast
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

source, root, model = sys.argv[1:]
sys.path.insert(0, source)
import torch
from transformers import LlamaTokenizer, Trainer, TrainingArguments
from utils import load_datasets, set_seed
from collator import Collator
from training_rows import IndexedDataset, IndexedCollator, take_indices, KEY

args=SimpleNamespace(dataset='Beauty',data_path=root,max_his_len=20,his_sep=', ',
    index_file='.index.json',add_prefix=False,tasks='seqrec',train_prompt_sample_num='1',
    train_data_sample_num='0',sample_valid=True,valid_prompt_id=0,valid_prompt_sample_num=2,
    only_train_response=True)
set_seed(42)
train, valid=load_datasets(args)
assert (len(train),len(valid))==(131413,44726)
tok=LlamaTokenizer.from_pretrained(model,local_files_only=True,model_max_length=2048,padding_side='right')
tok.pad_token_id=0
assert tok.add_tokens(train.datasets[0].get_new_tokens())==965
base=Collator(args,tok)
indexed=IndexedDataset(train)
train_data=train.datasets[0].inter_data
valid_data=valid.inter_data
length=lambda row: len(row['inters'].split(', '))
train_ids=list(dict.fromkeys([0,len(train)-1,min(range(len(train_data)),key=lambda i:length(train_data[i])),max(range(len(train_data)),key=lambda i:length(train_data[i])),17,200,2000,20000]))
valid_ids=list(dict.fromkeys([0,len(valid)-1,min(range(len(valid_data)),key=lambda i:length(valid_data[i]))*2,max(range(len(valid_data)),key=lambda i:length(valid_data[i]))*2,17,200,2000,20000]))
random.seed(42)
train_pairs=[indexed[i] for i in train_ids]
assert all(isinstance(x,tuple) and len(x)==2 and isinstance(x[1],dict) for x in train_pairs)
train_samples=[x[1] for x in train_pairs]
tracked=IndexedCollator(base)(train_pairs)
assert take_indices(tracked)==train_ids and KEY not in tracked
reference=base(train_samples)
assert set(tracked)==set(reference)=={'input_ids','attention_mask','labels'}
assert all(torch.equal(tracked[k],reference[k]) for k in reference)
valid_samples=[valid[i] for i in valid_ids]
assert all(isinstance(row,dict) for row in valid_samples)
for i,row in zip(valid_ids,valid_samples):
    assert valid_data[i//2]['item'] in row['labels']

# Extract the exact production override, retaining its super() semantics.
tree=ast.parse((Path(__file__).parent/'formal_train.py').read_text())
classes=[n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='AuditTrainer']
assert len(classes)==1
method=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='get_eval_dataloader']
assert len(method)==1
module=ast.Module(body=[ast.ClassDef(name='AuditTrainer',bases=[ast.Name(id='Trainer',ctx=ast.Load())],keywords=[],body=method,decorator_list=[])],type_ignores=[])
namespace={'Trainer':Trainer,'IndexedCollator':IndexedCollator}
exec(compile(ast.fix_missing_locations(module),'formal_train.py','exec'),namespace)
class Tiny(torch.nn.Module):
    def forward(self,input_ids=None,attention_mask=None,labels=None):
        return {'loss':torch.tensor(0.)}
from torch.utils.data import Subset
with tempfile.TemporaryDirectory() as tmp:
    ta=TrainingArguments(output_dir=tmp,no_cuda=True,per_device_eval_batch_size=len(valid_ids),remove_unused_columns=False,report_to=[])
    wrapper=IndexedCollator(base)
    trainer=namespace['AuditTrainer'](model=Tiny(),args=ta,train_dataset=indexed,eval_dataset=Subset(valid,valid_ids),data_collator=wrapper)
    loader=trainer.get_eval_dataloader()
    assert trainer.data_collator is wrapper
    actual=next(iter(loader))
    expected=base(valid_samples)
    assert set(actual)==set(expected)=={'input_ids','attention_mask','labels'}
    assert all(torch.equal(actual[k],expected[k]) for k in expected)
    assert all(v.dtype==torch.int64 for v in actual.values())
    assert actual['input_ids'].shape==actual['attention_mask'].shape==actual['labels'].shape
    assert torch.equal(actual['attention_mask'],(actual['input_ids']!=tok.pad_token_id).long())
    for j,row in enumerate(valid_samples):
        prompt=tok(row['input_ids'],add_special_tokens=True)['input_ids']
        full=tok(row['labels']+tok.eos_token,add_special_tokens=True)['input_ids']
        labels=actual['labels'][j].tolist()
        assert all(x==-100 for x in labels[:len(prompt)])
        assert labels[len(prompt):len(full)]==full[len(prompt):]
        assert all(x==-100 for x in labels[len(full):])
print(json.dumps({'status':'PASS','scope':'CPU real Beauty samples; actual eval override in installed Trainer, no model execution','train_indices':train_ids,'validation_indices':valid_ids,'train_history_lengths':[length(train_data[i]) for i in train_ids],'validation_history_lengths':[length(valid_data[i//2]) for i in valid_ids],'train_shape':list(tracked['input_ids'].shape),'validation_shape':list(actual['input_ids'].shape),'exact_native_collator_tensors':True,'audit_indices_removed':True,'validation_targets_match':True},indent=2))
