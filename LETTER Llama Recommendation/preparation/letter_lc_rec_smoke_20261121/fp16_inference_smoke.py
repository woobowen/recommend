"""Check author-style unquantized inference using FP16 on 2080 Ti."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import sys
import time
from shuffle_grid_protocol import PROTOCOL_VERSION, transform

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
from hardware_compat import disable_optional_bnb_triton
disable_optional_bnb_triton()
import torch
from transformers import LlamaForCausalLM,LlamaTokenizer
from peft import PeftModel
sys.path.insert(0,str(HERE/'vendor/LETTER/LETTER-LC-Rec'))
from data import SeqRecDataset
from evaluate import get_topk_results,get_metrics_results

torch.set_num_threads(4)
queue=json.loads((HERE/'latest_queue.json').read_text());run=Path(queue['launch'])
assert json.loads((run/'status.json').read_text())['complete']
records={}
for ds in ['Instruments','Beauty']:
    adapter=run/ds/'adapter'
    tokenizer=LlamaTokenizer.from_pretrained(adapter,local_files_only=True,padding_side='left')
    tokenizer.pad_token_id=0
    base=LlamaForCausalLM.from_pretrained(ROOT/'models/huggyllama-llama-7b',
        torch_dtype=torch.float16,device_map='balanced',max_memory={0:'8GiB',1:'8GiB'},
        attn_implementation='sdpa',local_files_only=True)
    assert all(str(v) not in ('cpu','disk') for v in base.hf_device_map.values())
    base.resize_token_embeddings(len(tokenizer))
    model=PeftModel.from_pretrained(base,adapter,is_trainable=False)
    assert not getattr(model,'is_loaded_in_8bit',False) and not getattr(model,'is_loaded_in_4bit',False)
    model.eval();device=model.get_input_embeddings().weight.device
    cfg=argparse.Namespace(dataset=ds,data_path=str(HERE/'vendor/LETTER/data'),max_his_len=20,
        his_sep=', ',index_file='.index.json',add_prefix=False,only_train_response=True,
        sample_valid=False,valid_prompt_id=0,valid_prompt_sample_num=2)
    test=SeqRecDataset(cfg,mode='test',prompt_sample_num=1,sample_num=-1);test.prompt_id=0
    allowed=test.get_prefix_allowed_tokens_fn(tokenizer)
    sid_ids={tokenizer.convert_tokens_to_ids(t) for t in test.get_new_tokens()}
    all_items=test.get_all_items();metrics_by_mode={}
    histories=list(test.inters.values())
    user_ids=list(test.inters.keys())
    for mode,n in [('unshuffle',0),('shuffle_3',3),('shuffle_4',4)]:
        rows=[]
        for row_idx in range(4):
            sample=test[row_idx];original=tokenizer.encode(sample['input_ids']);ids=list(original)
            expected_groups=[[tokenizer.convert_tokens_to_ids(t) for t in test.indices[str(i)]]
                             for i in histories[row_idx][:-1][-20:]]
            ids,audit=transform(original,[1]*len(original),expected_groups,sid_ids,n,
                                int(user_ids[row_idx]),ds.lower(),seed=2026)
            changed=sum(g['nonidentity'] for g in audit['groups'])
            assert n or changed==0
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                result=model.generate(input_ids=torch.tensor([ids],device=device),
                    attention_mask=torch.ones((1,len(ids)),dtype=torch.long,device=device),
                    num_beams=20,num_return_sequences=20,max_new_tokens=5,do_sample=False,
                    prefix_allowed_tokens_fn=allowed,output_scores=True,return_dict_in_generate=True,
                    pad_token_id=0,eos_token_id=tokenizer.eos_token_id,early_stopping=True,use_cache=True)
            predictions=tokenizer.batch_decode(result.sequences,skip_special_tokens=True)
            metrics=get_metrics_results(get_topk_results(predictions,result.sequences_scores.cpu().tolist(),
                [sample['labels']],20,all_items=None),['hit@5','hit@10','ndcg@5','ndcg@10'])
            candidates=[tokenizer.decode(s[len(ids):],skip_special_tokens=True).strip().replace(' ','') for s in result.sequences]
            assert len(candidates)==20
            rows.append({'user_row':row_idx,'target':sample['labels'],'metrics':metrics,'candidates':candidates,
                'history_items':len(expected_groups),'changed_history_items':changed,'shuffle_audit':audit,
                'valid_catalog_candidates':sum(s in all_items for s in candidates)})
        aggregate={k:sum(r['metrics'][k] for r in rows)/4 for k in rows[0]['metrics']}
        data={'passed':True,'dataset':ds,'mode':mode,'precision':'unquantized_fp16','beam':20,
            'users':4,'metrics':aggregate,'rows':rows,'readiness_only':True,
            'shuffle_protocol':PROTOCOL_VERSION,'shuffle_seed':2026,'adapter':str(adapter)}
        (run/ds/f'test_fp16_grid_{mode}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        metrics_by_mode[mode]=aggregate
        print(json.dumps({'stage':'fp16_mode_complete','dataset':ds,'mode':mode,'metrics':aggregate}),flush=True)
    records[ds]={'passed':True,'precision':'unquantized_fp16','tests':metrics_by_mode,
        'max_allocated_mib':[torch.cuda.max_memory_allocated(i)/2**20 for i in [0,1]]}
    del result,model,base
    import gc
    gc.collect();torch.cuda.empty_cache()
(HERE/'fp16_grid_verified.json').write_text(json.dumps({'complete':True,'datasets':records,
    'shuffle_protocol':PROTOCOL_VERSION,'shuffle_seed':2026},indent=2))
print(json.dumps({'complete':True,'datasets':records}),flush=True)
