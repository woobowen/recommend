"""Official LETTER data/model/collator with a model-parallel HF Trainer."""
from pathlib import Path
import argparse
import json
import math
import os
import random
import sys
import time
from common import MODEL,AUTHOR,config,datasets,build_model,event,write,sha,seed_all,verify_assets,runtime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['Instruments','Beauty'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--micro',type=int,required=True)
    parser.add_argument('--capacity',action='store_true')
    parser.add_argument('--memory-only',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    try:
        torch=runtime()
        from transformers import LlamaTokenizer,Trainer,TrainingArguments,TrainerCallback
        import numpy as np
        from collator import Collator
        from response_collator import ResponseSafeCollator
        seed_all()
        cfg=config(args.dataset,args.micro)
        assert cfg['effective_batch']%args.micro==0
        write(args.out/'config.json',cfg)
        audit=verify_assets(args.out)
        tokenizer=LlamaTokenizer.from_pretrained(MODEL,model_max_length=2048,padding_side='right',local_files_only=True)
        tokenizer.pad_token_id=0
        ds_args,seq,parts,train,valid,added=datasets(args.dataset,tokenizer)
        collator=ResponseSafeCollator(Collator(ds_args,tokenizer),tokenizer)
        cfg.update(train_samples=len(train),train_task_samples={name:len(ds) for name,ds in zip(cfg['tasks'],parts)},
            validation_examples=len(valid),test_users=audit['datasets'][args.dataset]['users'],added_tokens=added)
        write(args.out/'config.json',cfg)
        event(args.out,'data_ready',**cfg)
        model=build_model(cfg,tokenizer)
        params=[p for p in model.parameters() if p.requires_grad]
        names={n:p for n,p in model.named_parameters() if p.requires_grad}
        device=model.get_input_embeddings().weight.device
        write(args.out/'trainable.json',{'count':sum(p.numel() for p in params),
            'device_map':model.hf_device_map,'parameter_names':list(names)})
        optimizer=torch.optim.AdamW(params,lr=.0002,weight_decay=.01,foreach=False)
        if args.capacity:
            model.train();scaler=torch.amp.GradScaler('cuda',init_scale=128.)
            cases=[]
            record=max(seq.inter_data,key=lambda r:len(r['inters']))
            rows=[seq._get_text_data(record,prompt) for prompt in seq.prompts]
            x,y=max(rows,key=lambda pair:len(tokenizer.encode(pair[1])))
            cases.append(('seqrec',{'input_ids':x,'labels':y}))
            for task,ds in zip(cfg['tasks'][1:],parts[1:]):
                longest=sorted(ds.feat_data,key=lambda r:len(r['title'])+len(str(r['description'])),reverse=True)[:8]
                options=[ds._get_text_data(record,prompt) for record in longest for prompt in ds.prompts]
                x,y=max(options,key=lambda pair:len(tokenizer.encode(pair[1])))
                cases.append((task,{'input_ids':x,'labels':y}))
            stats=[]
            for task,row in cases:
                batch={k:v.to(device) for k,v in collator([row]*args.micro).items()}
                assert (batch['labels']!=-100).any()
                optimizer.zero_grad(set_to_none=True)
                for accumulation in range(2):
                    with torch.autocast('cuda',dtype=torch.float16):loss=model(**batch).loss/2
                    assert torch.isfinite(loss)
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params)
                torch.nn.utils.clip_grad_norm_(params,1.,foreach=False)
                scaler.step(optimizer);scaler.update()
                item={'task':task,'tokens':list(batch['input_ids'].shape),'loss':loss.item()*2,'accumulations_checked':2,
                    'max_allocated_mib':[torch.cuda.max_memory_allocated(i)/2**20 for i in [0,1]]}
                event(args.out,'capacity_step',**item);stats.append(item)
                del loss,batch
            write(args.out/'capacity_memory_result.json',{'passed':True,'micro':args.micro,'cases':stats})
            event(args.out,'capacity_memory_passed',micro=args.micro)
            optimizer.zero_grad(set_to_none=True)
            if args.memory_only:
                write(args.out/'capacity_result.json',{'passed':True,'micro':args.micro,
                    'memory_accumulation_checked':True,'memory_cases':stats})
                return

        class HealthTrainer(Trainer):
            def compute_loss(self,model,inputs,return_outputs=False):
                if not (inputs['labels']!=-100).any():raise RuntimeError('Batch has no supervised target tokens')
                result=super().compute_loss(model,inputs,return_outputs=return_outputs)
                loss=result[0] if return_outputs else result
                if not torch.isfinite(loss).item():raise FloatingPointError('Nonfinite train/evaluation loss')
                return result

            def _load_rng_state(self,checkpoint):
                if not checkpoint:return
                rng_path=Path(checkpoint)/'rng_state.pth'
                if not rng_path.exists():raise RuntimeError('Missing recovery RNG state')
                rng=torch.load(rng_path,map_location='cpu',weights_only=False)
                random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['cpu'])
                all_cuda=Path(checkpoint)/'cuda_all_rng.pt'
                if all_cuda.exists():torch.cuda.set_rng_state_all(torch.load(all_cuda,weights_only=False))
                else:torch.cuda.set_rng_state(rng['cuda'])

        start=time.monotonic()

        class Progress(TrainerCallback):
            def on_train_begin(self,training_args,state,control,**kwargs):
                event(args.out,'training_started',global_step=state.global_step,max_steps=state.max_steps,
                    effective_batch=cfg['effective_batch'],micro=args.micro,
                    gradient_accumulation_steps=cfg['gradient_accumulation_steps'],from_base=args.resume is None)

            def on_log(self,training_args,state,control,logs=None,**kwargs):
                logs=dict(logs or {})
                for key in ['loss','eval_loss','grad_norm']:
                    if key in logs and not math.isfinite(float(logs[key])):
                        raise FloatingPointError(f'Nonfinite {key}: {logs[key]}')
                event(args.out,'train_log',global_step=state.global_step,max_steps=state.max_steps,
                    epoch=state.epoch,logs=logs,elapsed_seconds=time.monotonic()-start,
                    repaired_long_prompt_rows=collator.repaired_rows,
                    max_allocated_mib=[torch.cuda.max_memory_allocated(i)/2**20 for i in [0,1]],
                    scaler_scale=float(trainer.accelerator.scaler.get_scale()))

            def on_step_end(self,training_args,state,control,**kwargs):
                # Recovery saves do not trigger validation or change best-model selection.
                if state.global_step>0 and state.global_step%100==0:control.should_save=True

            def on_save(self,training_args,state,control,**kwargs):
                path=Path(training_args.output_dir)/f'checkpoint-{state.global_step}'
                torch.save(torch.cuda.get_rng_state_all(),path/'cuda_all_rng.pt')
                adapter=path/'adapter_model.safetensors'
                assert adapter.exists() and (path/'optimizer.pt').exists() and (path/'trainer_state.json').exists()
                write(path/'COMPLETE.json',{'step':state.global_step,'adapter_sha256':sha(adapter),
                    'trainer_state_sha256':sha(path/'trainer_state.json'),'best_model_checkpoint':state.best_model_checkpoint})
                event(args.out,'checkpoint_saved',global_step=state.global_step,checkpoint=str(path),
                    best_checkpoint=state.best_model_checkpoint,best_metric=state.best_metric)

        if args.capacity:
            from torch.utils.data import Subset
            train=Subset(train,range(cfg['effective_batch']*2))
            valid=Subset(valid,range(4))
        training_args=TrainingArguments(output_dir=str(args.out/'checkpoints'),seed=42,data_seed=42,
            per_device_train_batch_size=args.micro,per_device_eval_batch_size=args.micro,
            gradient_accumulation_steps=cfg['gradient_accumulation_steps'],num_train_epochs=4,
            learning_rate=.0002,weight_decay=.01,warmup_ratio=.01,lr_scheduler_type='cosine',
            fp16=True,bf16=False,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},
            evaluation_strategy='steps' if args.capacity else 'epoch',
            save_strategy='steps' if args.capacity else 'epoch',eval_delay=0 if args.capacity else 1,
            eval_steps=2 if args.capacity else None,save_steps=2 if args.capacity else 500,
            max_steps=2 if args.capacity else -1,
            save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
            logging_steps=1,logging_nan_inf_filter=False,report_to=[],disable_tqdm=True,
            dataloader_num_workers=0,save_safetensors=True,optim='adamw_torch',max_grad_norm=1.0)
        trainer=HealthTrainer(model=model,args=training_args,train_dataset=train,eval_dataset=valid,
            tokenizer=tokenizer,data_collator=collator,optimizers=(optimizer,None),callbacks=[Progress()])
        assert trainer.is_model_parallel and trainer.args._n_gpu==1
        trainer.accelerator.scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        event(args.out,'trainer_ready',is_model_parallel=trainer.is_model_parallel,logical_n_gpu=trainer.args._n_gpu)
        if args.resume:
            assert (args.resume/'COMPLETE.json').exists()
        result=trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        trainer.save_state()
        best=Path(trainer.state.best_model_checkpoint)
        assert best.exists() and (best/'adapter_model.safetensors').exists()
        outcome={'complete':True,'dataset':args.dataset,'best_checkpoint':str(best),
            'best_adapter_sha256':sha(best/'adapter_model.safetensors'),
            'best_eval_loss':trainer.state.best_metric,'global_step':trainer.state.global_step,
            'epoch':trainer.state.epoch,'training_metrics':result.metrics,'test_started':False,
            'train_seed':42,'shuffle_seed':2026,'full_training':not args.capacity}
        if args.capacity:
            write(args.out/'capacity_result.json',{'passed':True,'micro':args.micro,
                'memory_cases':stats,'trainer_integration':outcome})
            event(args.out,'capacity_and_trainer_passed',micro=args.micro)
        else:
            write(args.out/'training_result.json',outcome);event(args.out,'training_complete',**outcome)
    except Exception as exc:
        import traceback
        event(args.out,'failed',error=repr(exc),traceback=traceback.format_exc())
        if 'torch' in locals() and isinstance(exc,torch.cuda.OutOfMemoryError) and args.capacity:sys.exit(42)
        raise


if __name__=='__main__':main()
