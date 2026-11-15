"""Explicit CPU test boundary for production_pipeline, never selectable by formal CLI."""
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from formal_control import durable_write
from formal_train import checkpoint_required_files, write
from checkpoint_receipt import write_receipt
from production_identity import ASSETS, SYNTHETIC, STEPS, need, sha, identity, load
from production_pipeline import execute_stage


def compute(plan,stage,output,resume=None):
    need(plan['evidence_scope']==SYNTHETIC and SYNTHETIC in plan['root'],'test isolation')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    data={k:{'sha256':v} for k,v in ASSETS.items()}
    training=Path(plan['root'])/'training'
    if stage=='train':
        import torch
        import ast, importlib.util, logging, re
        from typing import List
        source=Path(importlib.util.find_spec('transformers').origin).parent/'trainer.py'
        raw=source.read_text();tree=ast.parse(raw)
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Trainer')
        methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('_rotate_checkpoints','_sorted_checkpoints')]
        need(len(methods)==2,'installed Trainer rotation methods')
        namespace={'Path':Path,'os':os,'re':re,'shutil':shutil,'logger':logging.getLogger('rotation'), 'List':List,'PREFIX_CHECKPOINT_DIR':'checkpoint'}
        for method in methods:exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        Trainer=type('InstalledTrainerRotation',(),{x.name:namespace[x.name] for x in methods})
        trainer=Trainer()

        trainer.args=SimpleNamespace(save_total_limit=2)
        trainer.state=SimpleNamespace(best_model_checkpoint=None)
        training.mkdir(exist_ok=False)
        receipts=[];losses=[4.,2.,2.,3.];best=None
        for epoch,(step,loss) in enumerate(zip(STEPS,losses),1):
            cp=training/f'checkpoint-{step}';cp.mkdir()
            for name in checkpoint_required_files(step):
                p=cp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'SYNTHETIC_CHECKPOINT_NOT_A_MODEL\n')
            torch.save({'SYNTHETIC_TEST_TENSOR':torch.tensor([float(step)])},cp/'adapter_model.bin')
            write(cp/'trainer_state.json',{'global_step':step});(cp/'latest').write_text(f'global_step{step}\n')
            if best is None or loss<best[1]:best=(step,loss);trainer.state.best_model_checkpoint=str(cp)
            rows=[{k:v for k,v in identity(p).items() if k!='path'}|{'path':str(p.relative_to(cp))} for p in sorted(cp.rglob('*')) if p.is_file()]
            manifest={'run_commit':plan['run_commit'],'config_sha256':plan['config_sha256'],'global_step':step,'files':rows,'collective_ranks':2}
            write(cp/'complete.json',manifest)
            r=write_receipt(cp,training,run_commit=plan['run_commit'],config_sha256=plan['config_sha256'],data_identity=data,
                epoch=float(epoch),step=step,eval_loss=loss,best_checkpoint=trainer.state.best_model_checkpoint,
                best_metric=best[1],attempt=plan['task_id'],required_files=checkpoint_required_files(step),expected_manifest=manifest,scope=SYNTHETIC)
            receipts.append(identity(r))
            # Actual installed Transformers 4.31 rotation, including best protection.
            Trainer._rotate_checkpoints(trainer,use_mtime=False,output_dir=str(training))
        adapter=training/'best_adapter';adapter.mkdir()
        selected=training/f'checkpoint-{best[0]}'
        for name in ('adapter_model.bin','adapter_config.json','tokenizer.model','tokenizer_config.json','added_tokens.json','special_tokens_map.json'):
            shutil.copyfile(selected/name,adapter/name)
        write(training/'best_selection.json',{'checkpoint':str(selected),'metric':'eval_loss','greater_is_better':False,'value':best[1],
            'adapter_sha256':sha(adapter/'adapter_model.bin'),'run_commit':plan['run_commit'],'completed_training_step':8214,'loaded_best_checkpoint_step':best[0]})
        for rank in (0,1):
            write(training/f'rank{rank}.json',{'scope':SYNTHETIC,'status':'TRAINING_COMPLETE','rank':rank,'probe':False,
                'run_commit':plan['run_commit'],'config_sha256':plan['config_sha256'],'global_step':8214,'epoch':4.,'data_identity':data,
                'data':{'train_rows':131413,'valid_rows':44726,'validation_subset_preflight_only':False,'tokenizer_vocab':32965}})
            ev=[{'event':'log','attempt_job_id':plan['task_id'],'step':s,'epoch':float(i+1),'values':{'eval_loss':loss}} for i,(s,loss) in enumerate(zip(STEPS,losses))]
            ev.extend({'event':'checkpoint_complete','step':step,'attempt_job_id':plan['task_id'],'path':str(training/f'checkpoint-{step}')} for step in STEPS)
            ev.append({'event':'strict_ds_restored','path':str(selected),'global_step':best[0]})
            (training/f'events.rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ev))
        durable_write(output/'fixture_boundary.json',{'scope':SYNTHETIC,'external_model_compute_replaced':True,
            'receipts':receipts,'retained_checkpoints':[p.name for p in training.glob('checkpoint-*')],
            'rotation_source':identity(source)})
    else:
        condition={'normal':'Normal','s3':'S3','s4':'S4'}[stage]
        origin=Path(plan['fixture_root']);summary=load(origin/'summary.json')
        need(load(origin/'SYNTHETIC_NOTICE.json')['designation']==SYNTHETIC,'fixture designation')
        adapter=training/'best_adapter'
        files={p.name:{k:v for k,v in identity(p).items() if k!='path'} for p in adapter.iterdir() if p.is_file()}
        test=output/'test';test.mkdir()
        for rank in (0,1):
            source=origin/condition/'synthetic_predictions'/f'predictions.rank{rank}.jsonl'
            expected=summary['conditions'][condition]['prediction_files'][rank]
            need(sha(source)==expected['sha256'] and source.stat().st_size==expected['bytes'],'frozen synthetic prediction identity')
            shutil.copyfile(source,test/source.name) # byte reuse; no candidate generation
            old=load(source.parent/f'rank{rank}.json')
            report={'scope':SYNTHETIC,'status':'PASS','rank':rank,'preflight_only':False,'RUN_COMMIT':plan['prediction_run_commit'],
                 'run_commit':plan['prediction_run_commit'],'completed_users':11182 if rank==0 else 11181,
                 'adapter_files':files,'data_identity':data,'author_sources':{'scope':SYNTHETIC},
                 'generation':{'scope':SYNTHETIC},'allowed_callback_sha256':'SYNTHETIC_NOT_GPU',
                 'elapsed_generation_seconds':0.,'assignment_sha256':old['assignment_sha256']}
            if stage!='normal':report['shuffle']={'n':int(stage[-1]),'protocol':'GRID-LETTER-SHUFFLE-V1','seed':42}
            durable_write(test/f'rank{rank}.json',report)
        durable_write(output/'fixture_boundary.json',{'scope':SYNTHETIC,'external_gpu_generation_replaced':True,
             'original_predictions':[identity(origin/condition/'synthetic_predictions'/f'predictions.rank{r}.jsonl') for r in (0,1)]})


def main():
    import argparse
    p=argparse.ArgumentParser()
    for n in ('plan','stage','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();plan=load(a.plan)
    execute_stage(plan,a.stage,a.output,compute=compute)

if __name__=='__main__':main()
