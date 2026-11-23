"""Check author assets, dataset tasks, tokenization and response-only batches."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
vendor = HERE / 'vendor/LETTER'
sys.path.insert(0, str(vendor/'LETTER-LC-Rec'))
import torch
from transformers import LlamaTokenizer
from data import SeqRecDataset, ItemFeatDataset
from collator import Collator
from modeling_letter import LETTER


def main():
    audit = json.loads((HERE/'expected_source_assets.json').read_text())
    report = {'time_unix':time.time(),'source_commit':audit['source_commit'],
              'model_id':audit['model_id'],'model_revision':audit['model_revision'],
              'packages':{},'datasets':{},'model_files':{},'code_files':{}}
    for p,h in audit['official_files'].items():
        report['code_files'][p] = hashlib.sha256((vendor/p).read_bytes()).hexdigest()==h
    assert all(report['code_files'].values())
    for p in audit['model_files']:
        f=ROOT/'models/huggyllama-llama-7b'/p['name']
        report['model_files'][p['name']]={'expected_bytes':p['bytes'],
            'exists':f.exists(),'actual_bytes':f.stat().st_size if f.exists() else 0}
    for k in ['torch','transformers','peft','bitsandbytes','accelerate','sentencepiece','safetensors']:
        report['packages'][k]=importlib.metadata.version(k)
    report['gpus']=[{'name':torch.cuda.get_device_name(i),
                    'capability':torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]
    for ds in ['Beauty','Instruments']:
        cfg=argparse.Namespace(dataset=ds,data_path=str(vendor/'data'), max_his_len=20,
            his_sep=', ',index_file='.index.json',add_prefix=False,only_train_response=True,
            sample_valid=False,valid_prompt_id=0,valid_prompt_sample_num=2)
        seq=SeqRecDataset(cfg,mode='train',prompt_sample_num=1,sample_num=-1)
        token=LlamaTokenizer.from_pretrained(HERE/'model_metadata',local_files_only=True,
                                            model_max_length=512,padding_side='right')
        token.pad_token_id=0
        added=token.add_tokens(seq.get_new_tokens())
        assert added==audit['datasets'][ds]['sid_tokens']
        assert all(len(token.encode(t,add_special_tokens=False))==1 for t in seq.get_new_tokens())
        test=SeqRecDataset(cfg,mode='test',prompt_sample_num=1,sample_num=-1)
        test.prompt_id=0
        first=test[0]
        prefix=token.encode(first['input_ids'])
        answer=token.encode(first['labels'],add_special_tokens=False)
        assert len(answer)==4
        constraint=test.get_prefix_allowed_tokens_fn(token)
        for i,next_token in enumerate(answer+[token.eos_token_id]):
            permitted=constraint(0,torch.tensor(prefix+answer[:i]))
            assert permitted and next_token in permitted,(ds,i,permitted)
        tasks={'seqrec':seq}
        if ds=='Beauty':
            tasks.update({t:ItemFeatDataset(cfg,task=t,prompt_sample_num=1) for t in ['item2index','index2item']})
            # All author feature prompts must be formattable for every item.
            for task in ['item2index','index2item']:
                ds_object=tasks[task]
                for record in ds_object.feat_data:
                    for prompt in ds_object.prompts:
                        prompt['instruction'].format(**record)
                        prompt['response'].format(**record)
        details={}
        for name,dataset in tasks.items():
            sample=dataset[0]
            batch=Collator(cfg,token)([sample])
            labels=int((batch['labels']!=-100).sum())
            assert labels>0, (ds,name)
            details[name]={'samples':len(dataset),'first_batch_tokens':int(batch['input_ids'].shape[-1]),
                           'first_batch_supervised_tokens':labels}
        report['datasets'][ds]={'tasks':details,'added_sid_tokens':added,
                               'official_generation_constraint_checked':True,**audit['datasets'][ds]}
    report['data_code_tokenization_ready']=True
    report['model_weights_complete']=all(r['exists'] and r['actual_bytes']==r['expected_bytes']
                                         for r in report['model_files'].values())
    (HERE/'readiness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['code_files','model_files']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
