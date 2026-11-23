"""Frozen assets and hardware setup shared by formal LETTER workers."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time

os.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled')
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
PREP=ROOT/'codex/letter_lc_rec_smoke_20261121'
AUTHOR=PREP/'vendor/LETTER'
MODEL=ROOT/'models/huggyllama-llama-7b'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def event(out,stage,**kwargs):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    item={'time_unix':time.time(),'stage':stage,**kwargs}
    with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    write(out/'progress.json',item)
    print(json.dumps(item,ensure_ascii=False),flush=True)


def runtime():
    from hardware_compat import disable_optional_bnb_triton
    disable_optional_bnb_triton()
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(AUTHOR/'LETTER-LC-Rec'))
    return torch


def verify_assets(out):
    audit=json.loads((PREP/'assets_audit.json').read_text())
    assert audit['model_id']=='huggyllama/llama-7b'
    assert audit['source_commit']=='8d0154e28de37dbb6e24871c508ad8ddb1921cda'
    for name,h in audit['official_files'].items():assert sha(AUTHOR/name)==h,name
    for f in audit['model_files']:
        path=MODEL/f['name']
        assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256'],str(path)
    write(Path(out)/'assets_provenance.json',audit)
    return audit


def config(dataset,micro):
    return {'dataset':dataset,'tasks':['seqrec'],
        'train_seed':42,'shuffle_seed':2026,'shuffle_protocol':'GRID-PLAN-V1_1','identity_allowed':True,
        'temperature':1.0 if dataset=='Beauty' else .8,'epochs':4,'learning_rate':.0002,
        'weight_decay':.01,'warmup_ratio':.01,'lr_scheduler_type':'cosine',
        'lora_r':8,'lora_alpha':16,'lora_dropout':.05,
        'lora_targets':['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj'],
        'modules_to_save':['embed_tokens','lm_head'],'only_train_response':True,
        'model_max_length':2048,'max_history':20,'micro_batch':micro,
        'effective_batch':64,
        'gradient_accumulation_steps':64//micro,
        'valid_prompt_samples_per_user':2,'select_best':'lowest eval_loss after epoch',
        'train_backbone':'int8','compute_dtype':'float16','test_backbone':'unquantized float16',
        'num_beams':20,'test_batch':1,'test_prompts':[0],'global_gpu_ids':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'author_reference':'User 2026-11-22: Beauty seqrec only, effective batch64; other settings inherited, including temperature1.0',
        'hardware_adaptations':['single-process two-GPU model parallelism',
            'FP16 in place of unsupported BF16','native SDPA; no FastChat FlashAttention or DeepSpeed',
            'per-device physical batch adapted; gradient accumulation preserves effective batch',
            'compilerless optional-package import guards','Accelerate GradScaler initial scale128'],
        'checkpoint_policy':'each epoch plus recovery checkpoints; retain best and latest, choose by epoch validation',
        'long_prompt_policy':'Only zero-target rows: trim prompt head/tail to keep response supervision within2048; other author batches unchanged',
        'full_test':True,'train_from_base':True}


def datasets(dataset,tokenizer):
    from data import SeqRecDataset,ItemFeatDataset
    from torch.utils.data import ConcatDataset
    args=argparse.Namespace(dataset=dataset,data_path=str(AUTHOR/'data'),max_his_len=20,
        his_sep=', ',index_file='.index.json',add_prefix=False,only_train_response=True,
        sample_valid=True,valid_prompt_id=0,valid_prompt_sample_num=2)
    seq=SeqRecDataset(args,mode='train',prompt_sample_num=1,sample_num=-1)
    parts=[seq]
    # User-authorized seqrec-only run; no auxiliary product-text training.
    valid=SeqRecDataset(args,mode='valid',prompt_sample_num=2)
    added=tokenizer.add_tokens(seq.get_new_tokens())
    assert added==(965 if dataset=='Beauty' else 918)
    assert all(len(tokenizer.encode(t,add_special_tokens=False))==1 for t in seq.get_new_tokens())
    return args,seq,parts,ConcatDataset(parts),valid,added


def build_model(cfg,tokenizer):
    import torch
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from modeling_letter import LETTER
    from hardware_compat import restore_legacy_bnb_state
    model=LETTER.from_pretrained(MODEL,quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,device_map='balanced',max_memory={0:'8GiB',1:'8GiB'},
        attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True)
    assert all(str(v) not in ['cpu','disk'] for v in model.hf_device_map.values())
    model.resize_token_embeddings(len(tokenizer));model.set_hyper(cfg['temperature'])
    restore_legacy_bnb_state(model)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
        target_modules=cfg['lora_targets'],modules_to_save=cfg['modules_to_save'],
        task_type='CAUSAL_LM',bias='none'))
    model.is_parallelizable=True;model.model_parallel=True
    return model


def seed_all(seed=42):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
