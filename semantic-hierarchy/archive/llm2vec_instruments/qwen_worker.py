"""Original Qwen2.5/WMH pooling protocol on a new text-only item table."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  USE_TF='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import argparse
import fcntl
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from runtime_common import ROOT, read_json, read_jsonl, sha, status, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    args=parser.parse_args();run=Path(args.run);out=run/'upstream';out.mkdir(exist_ok=True)
    cfg=read_json(run/'qwen_config.json');source=Path(cfg['input_path'])
    assert sha(source)==cfg['input_sha256']
    rows=read_jsonl(source);assert len(rows)==cfg['items']
    assert [row['item_id'] for row in rows]==list(range(len(rows)))
    locks=[];devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(devices)==2
    for gpu in sorted(devices,key=int):
        stream=(ROOT/'stage_gpu_locks'/f'gpu_{gpu}.lock').open('a')
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(stream)
    assert torch.cuda.device_count()==2
    for index in range(2):assert torch.cuda.mem_get_info(index)[0]>9*1024**3
    state=out/'qwen_status.json';status(state,'loading_model',physical_gpus=devices)
    torch.set_num_threads(2);torch.manual_seed(42)
    model_path=ROOT/'models/Qwen2.5-7B-Instruct'
    model_cfg=AutoConfig.from_pretrained(model_path,local_files_only=True)
    assert model_cfg.model_type=='qwen2' and model_cfg.hidden_size==3584
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token;tokenizer.padding_side='right'
    model=AutoModel.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float16,
                                   device_map='auto',max_memory={0:'9GiB',1:'9GiB','cpu':'32GiB'},
                                   low_cpu_mem_usage=True).eval()
    assert all(str(device) not in ['cpu','disk'] for device in model.hf_device_map.values())
    model.config.use_cache=False
    sys.path.insert(0,str(ROOT))
    from src.models.components.network_blocks.embedding_aggregator import EmbeddingAggregator
    from src.models.components.network_blocks.aggregation_strategy import MeanAggregation
    aggregator=EmbeddingAggregator(MeanAggregation())
    device=model.get_input_embeddings().weight.device
    path=out/'item_embeddings_raw.npy';checkpoint=out/'encoding_checkpoint.json'
    signature={'config':cfg,'source_sha256':sha(Path(__file__))}
    shape=(len(rows),3584)
    if checkpoint.exists():
        old=read_json(checkpoint);assert old['signature']==signature
        cursor=old['completed'];vectors=np.lib.format.open_memmap(path,mode='r+')
        assert vectors.shape==shape
    else:
        cursor=0;vectors=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape)
    start=time.monotonic();initial=cursor;oom_splits=0
    def batch(ids):
        nonlocal oom_splits
        features=tokenizer([rows[i]['text'] for i in ids],max_length=128,padding='max_length',
                           truncation=True,add_special_tokens=True,return_tensors='pt').to(device)
        try:
            with torch.inference_mode():
                hidden=model(**features,use_cache=False).last_hidden_state
                raw=aggregator(hidden,features.attention_mask.to(hidden.device)).float().cpu()
            assert raw.shape==(len(ids),3584) and torch.isfinite(raw).all()
            return raw.numpy()
        except torch.cuda.OutOfMemoryError:
            if len(ids)==1:raise
            del features;torch.cuda.empty_cache();oom_splits+=1
            middle=len(ids)//2
            return np.concatenate([batch(ids[:middle]),batch(ids[middle:])])
    while cursor<len(rows):
        stop=min(cursor+128,len(rows))
        for first in range(cursor,stop,cfg['batch_size']):
            ids=list(range(first,min(first+cfg['batch_size'],stop)))
            vectors[ids]=batch(ids)
        cursor=stop;vectors.flush()
        write_json(checkpoint,{'signature':signature,'completed':cursor})
        speed=(cursor-initial)/(time.monotonic()-start)
        status(state,'encoding',completed=cursor,total=len(rows),items_per_second=speed,
               estimated_remaining_seconds=(len(rows)-cursor)/speed,physical_gpus=devices)
    assert np.isfinite(vectors).all()
    output=out/'item_embeddings.pt';temporary=output.with_suffix('.partial')
    torch.save(torch.from_numpy(np.array(vectors)),temporary);temporary.replace(output)
    info={'complete':True,'dataset':'instrument','architecture':'qwen2','shape':list(shape),
          'model':str(model_path),'pooling':'WMH MeanAggregation, right padding','max_length':128,
          'weight_dtype':'float16','stored_dtype':'float32','output_unit_normalized':False,
          'device_map':model.hf_device_map,'source':str(source),'source_sha256':sha(source),
          'sha256':sha(output),'oom_batch_splits':oom_splits,
          'model_config_sha256':sha(model_path/'config.json'),
          'gpu_peak_allocated_bytes':[torch.cuda.max_memory_allocated(i) for i in range(2)]}
    write_json(output.with_suffix('.json'),info);status(state,'complete',items=len(rows))


if __name__=='__main__':main()
