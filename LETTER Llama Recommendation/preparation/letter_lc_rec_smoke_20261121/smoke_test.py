"""Short LETTER-LC-Rec preflight using untouched author model/data/collator.

Hardware adaptations: FP16 int8 LoRA and two-GPU model parallelism, no
FastChat/FlashAttention/DeepSpeed. This is a readiness test, not paper metrics.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from shuffle_grid_protocol import PROTOCOL_VERSION, transform

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor/LETTER'


def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['Beauty', 'Instruments'], required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu-count', type=int, default=2)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    events = args.out / 'events.jsonl'

    def event(stage, **kw):
        record = {'time_unix': time.time(), 'stage': stage, **kw}
        with events.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        write(args.out / 'status.json', record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    try:
        import numpy as np
        import torch
        from hardware_compat import disable_optional_bnb_triton, restore_legacy_bnb_state
        disable_optional_bnb_triton()
        import bitsandbytes as bnb
        from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, TaskType, PeftModel, get_peft_model, prepare_model_for_kbit_training
        sys.path.insert(0, str(VENDOR / 'LETTER-LC-Rec'))
        from data import SeqRecDataset, ItemFeatDataset
        from collator import Collator
        from modeling_letter import LETTER
        from evaluate import get_topk_results, get_metrics_results

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.set_num_threads(4)
        expected = json.loads((HERE / 'assets_audit.json').read_text())
        for name, digest in expected['official_files'].items():
            assert sha(VENDOR / name) == digest, name
        # Independently validate the complete model snapshot, not just filenames.
        for f in expected['model_files']:
            path = args.model / f['name']
            assert path.exists() and path.stat().st_size == f['bytes'], str(path)
            if f['sha256']:
                assert sha(path) == f['sha256'], str(path)
        event('assets_verified', dataset=args.dataset,
              packages={k: importlib.metadata.version(k) for k in
                        ['torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'sentencepiece']})

        cfg = argparse.Namespace(dataset=args.dataset, data_path=str(VENDOR / 'data'),
            max_his_len=20, his_sep=', ', index_file='.index.json', add_prefix=False,
            only_train_response=True, sample_valid=False, valid_prompt_id=0,
            valid_prompt_sample_num=2, max_his_len_unused=20)
        train = SeqRecDataset(cfg, mode='train', prompt_sample_num=1, sample_num=-1)
        valid = SeqRecDataset(cfg, mode='valid', prompt_sample_num=1)
        test = SeqRecDataset(cfg, mode='test', prompt_sample_num=1, sample_num=-1)
        tasks = {'seqrec': train}
        if args.dataset == 'Beauty':
            tasks['item2index'] = ItemFeatDataset(cfg, task='item2index', prompt_sample_num=1)
            tasks['index2item'] = ItemFeatDataset(cfg, task='index2item', prompt_sample_num=1)
        tokenizer = LlamaTokenizer.from_pretrained(args.model, local_files_only=True,
                                                   model_max_length=512, padding_side='right')
        tokenizer.pad_token_id = 0
        tokens = train.get_new_tokens()
        added = tokenizer.add_tokens(tokens)
        assert added == expected['datasets'][args.dataset]['sid_tokens']
        for token in tokens:
            assert len(tokenizer.encode(token, add_special_tokens=False)) == 1, token
        temperature = 1.0 if args.dataset == 'Beauty' else 0.8
        setup = {'dataset': args.dataset, 'tasks': list(tasks), 'added_tokens': added,
            'temperature': temperature, 'train_seed': 42, 'shuffle_seed': 2026,
            'max_history': 20, 'smoke_max_length': 512, 'formal_author_max_length': 2048,
            'train_samples': {name: len(ds) for name, ds in tasks.items()},
            'validation_users': len(valid), 'test_users': len(test),
            'hardware_adaptations': ['FP16 compute instead of BF16', 'int8 frozen backbone as in author training',
                                    'single process across GPU0/1 instead of DDP/DeepSpeed',
                                    'native SDPA instead of FastChat FlashAttention patch',
                                    'optional bnb SwitchBack Triton imports and unused DeepSpeed unwrap disabled; native CUDA int8 retained'],
            'lora': {'r': 8, 'alpha': 16, 'dropout': .05,
                     'targets': ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                     'modules_to_save': ['embed_tokens','lm_head']},
            'train_loss_scope': 'response_only', 'learning_rate': .0002,
            'beam': 20, 'generation_constraint': 'official per-position token sets, not full catalog Trie',
            'shuffle': 'Exact GRID plan_v1_1: all permutations including identity, seed2026, lowercase dataset and actual uid; only inputs changed',
            'shuffle_protocol': PROTOCOL_VERSION,
            'test_metrics_are_not_formal_results': True}
        write(args.out / 'config.json', setup)
        event('datasets_ready', **setup)
        quant = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs = dict(quantization_config=quant, torch_dtype=torch.float16,
            device_map='balanced', max_memory={i: '8GiB' for i in range(args.gpu_count)},
            attn_implementation='sdpa', local_files_only=True, low_cpu_mem_usage=True)

        def load_base(native_inference=False):
            cls = LlamaForCausalLM if native_inference else LETTER
            base = cls.from_pretrained(str(args.model), **load_kwargs)
            assert all(str(v) not in ('cpu', 'disk') for v in base.hf_device_map.values()), base.hf_device_map
            base.resize_token_embeddings(len(tokenizer))
            if not native_inference:
                base.set_hyper(temperature)
            restore_legacy_bnb_state(base)
            return base

        base = prepare_model_for_kbit_training(load_base(), use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={'use_reentrant': False})
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
            target_modules=setup['lora']['targets'], modules_to_save=['embed_tokens','lm_head'],
            task_type=TaskType.CAUSAL_LM, bias='none'))
        active = {name: p for name,p in model.named_parameters() if p.requires_grad}
        assert any('embed_tokens.modules_to_save' in n for n in active)
        assert any('lm_head.modules_to_save' in n for n in active)
        assert any('lora_' in n for n in active)
        write(args.out / 'trainable_parameters.json',
              {'total': sum(p.numel() for p in active.values()),
               'parameters': {n: {'shape': list(p.shape), 'dtype': str(p.dtype)} for n,p in active.items()},
               'device_map': base.hf_device_map})
        event('model_loaded', trainable=sum(p.numel() for p in active.values()), device_map=base.hf_device_map)
        device = model.get_input_embeddings().weight.device
        collator = Collator(cfg, tokenizer)

        def batch_of(rows):
            batch = collator(rows)
            assert (batch['labels'] != -100).sum() > 0
            return {k: v.to(device) for k,v in batch.items()}

        optimizer = torch.optim.AdamW(list(active.values()), lr=.0002, weight_decay=.01, foreach=False)
        scaler = torch.amp.GradScaler('cuda', init_scale=128.0)
        # Two updates per task, with author dataset prompt sampling retained.
        schedule = list(tasks) * 2 if len(tasks) > 1 else ['seqrec'] * 6
        initial_embedding = model.get_input_embeddings().weight[-added:].detach().float().cpu().clone()
        losses = []
        model.train()
        for step, name in enumerate(schedule):
            ds = tasks[name]
            batch = None
            for offset in range(32):
                example = ds[(step + offset) % len(ds)]
                candidate = collator([example])
                if (candidate['labels'] != -100).sum().item() > 0:
                    batch = {k:v.to(device) for k,v in candidate.items()}
                    break
            assert batch is not None, name
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(**batch).loss
            assert torch.isfinite(loss), loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norms = {}
            for label, needle in [('lora', 'lora_'), ('embedding','embed_tokens.modules_to_save'),
                                  ('head','lm_head.modules_to_save')]:
                grads = [p.grad for n,p in active.items() if needle in n and p.grad is not None]
                assert grads and all(torch.isfinite(g).all().item() for g in grads), label
                norms[label] = sum(g.float().square().sum().item() for g in grads) ** .5
                assert norms[label] > 0, (step,label)
            torch.nn.utils.clip_grad_norm_(list(active.values()), 1.0, foreach=False)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            event('optimizer_step', step=step+1, task=name, loss=losses[-1], gradient_norms=norms,
                  max_allocated_mib=[torch.cuda.max_memory_allocated(i)/2**20 for i in range(args.gpu_count)])
        changed = (model.get_input_embeddings().weight[-added:].detach().float().cpu()-initial_embedding).abs().max().item()
        assert changed > 0, 'New SID embeddings did not update'
        optimizer.zero_grad(set_to_none=True)
        del optimizer, active, initial_embedding, loss, batch, grads
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            vbatch = batch_of([valid[0], valid[1]])
            validation_loss = float(model(**vbatch).loss)
            assert math.isfinite(validation_loss)
            reference = model(**batch_of([valid[0]])).logits[0,-1].float().cpu()
        adapter = args.out / 'adapter'
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        event('adapter_saved', validation_loss=validation_loss, sid_embedding_max_change=changed)
        del model, base, vbatch
        gc.collect()
        torch.cuda.empty_cache()
        # Re-load saved adapter into a fresh quantized backbone and compare logits.
        # Author test_ddp.py also loads native LlamaForCausalLM at inference.
        reload_base = prepare_model_for_kbit_training(load_base(native_inference=True), use_gradient_checkpointing=False)
        reload_base.config.use_cache = True
        model = PeftModel.from_pretrained(reload_base, adapter, is_trainable=False)
        model.eval()
        device = model.get_input_embeddings().weight.device
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            reloaded = model(**batch_of([valid[0]])).logits[0,-1].float().cpu()
        diff = (reference-reloaded).abs().max().item()
        assert torch.allclose(reference, reloaded, atol=.03, rtol=.005), diff
        event('adapter_reload_verified', max_logit_error=diff)
        tokenizer.padding_side = 'left'
        # Official closure depends on response marker and generated position.
        allowed = test.get_prefix_allowed_tokens_fn(tokenizer)
        all_items = test.get_all_items()
        test.prompt_id = 0
        indices = [0, 1, 2, 3]
        original_histories = [[train.indices[str(i)] for i in
                              list(test.inters.values())[idx][:-1][-20:]] for idx in indices]
        sid_ids = {tokenizer.convert_tokens_to_ids(t) for t in tokens}
        eval_results = {}
        for mode,n in [('unshuffle',0),('shuffle_3',3),('shuffle_4',4)]:
            rows = []
            for row_idx,groups in zip(indices,original_histories):
                sample = test[row_idx]
                encoded = tokenizer(sample['input_ids'], return_tensors='pt', add_special_tokens=True)
                ids = encoded['input_ids'][0].tolist()
                expected_groups = [[tokenizer.convert_tokens_to_ids(t) for t in group] for group in groups]
                output, shuffle_audit = transform(ids, encoded['attention_mask'][0].tolist(),
                    expected_groups, sid_ids, n, int(list(test.inters.keys())[row_idx]),
                    args.dataset.lower(), seed=2026)
                changed_groups = sum(g['nonidentity'] for g in shuffle_audit['groups'])
                assert n or changed_groups == 0
                model_inputs = {'input_ids': torch.tensor([output],device=device),
                                'attention_mask': torch.ones((1,len(output)),device=device,dtype=torch.long)}
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    generated = model.generate(**model_inputs, num_beams=20,num_return_sequences=20,
                        max_new_tokens=5,do_sample=False, prefix_allowed_tokens_fn=allowed,
                        output_scores=True,return_dict_in_generate=True,early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
                candidates = tokenizer.batch_decode(generated.sequences,skip_special_tokens=True)
                scores = generated.sequences_scores.cpu().tolist()
                hit_rows = get_topk_results(candidates,scores,[sample['labels']],20,all_items=None)
                metrics = get_metrics_results(hit_rows,['hit@5','hit@10','ndcg@5','ndcg@10'])
                outputs = [tokenizer.decode(seq[len(output):], skip_special_tokens=True).strip().replace(' ','')
                           for seq in generated.sequences]
                rows.append({'user_row':row_idx,'target':sample['labels'], 'history_items':len(groups),
                    'changed_history_items':changed_groups,'shuffle_audit':shuffle_audit,
                    'metrics':metrics, 'candidates':outputs,
                    'valid_catalog_candidates':sum(s in all_items for s in outputs)})
            result = {'dataset':args.dataset,'mode':mode,'users':len(rows),'beam':20,
                      'metrics':{key:sum(row['metrics'][key] for row in rows)/len(rows) for key in rows[0]['metrics']},
                      'rows':rows,'readiness_only':True,'adapter':str(adapter),
                      'shuffle_protocol':PROTOCOL_VERSION,'shuffle_seed':2026}
            write(args.out/f'test_{mode}.json',result)
            eval_results[mode] = result['metrics']
            event('test_mode_complete', mode=mode, users=len(rows), metrics=result['metrics'])
        result = {'complete':True,'dataset':args.dataset,'optimizer_steps':len(schedule),'losses':losses,
                  'validation_loss':validation_loss,'reload_max_logit_error':diff,
                  'sid_embedding_max_change':changed,'tests':eval_results,
                  'smoke_users_per_mode':4,'formal_experiment_started':False}
        write(args.out/'result.json',result)
        event('complete', **result)
    except Exception as exc:
        import traceback
        event('failed',error=repr(exc),traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
