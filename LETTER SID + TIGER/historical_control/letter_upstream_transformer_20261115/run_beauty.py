"""One fresh own-TIGER fit on official LETTER SIDs, then three inference modes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE=ROOT/'codex/letter_upstream_transformer_20261115'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--preflight',action='store_true')
    args=ap.parse_args()
    out=args.output.resolve()
    assert not out.exists(),'Fresh output required'
    out.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='2',
        TF_NUM_INTEROP_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',HYDRA_FULL_ERROR='1')
    for k in ['PROJECT_ROOT','METADATA_DIR','LOCAL_RANK','RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:
        os.environ.pop(k,None)
    sys.path.insert(0,str(ROOT))
    import rootutils
    rootutils.setup_root(__file__,indicator='.project-root',pythonpath=True)
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import Callback
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf,open_dict
    import src.utils.custom_hydra_resolvers
    from src.utils.launcher_utils import pipeline_launcher

    torch.use_deterministic_algorithms(True,warn_only=False)
    audit_path=HERE/'inputs/beauty/mapping_audit.json'
    audit=json.loads(audit_path.read_text())
    assert audit['complete'] and audit['item_mapping_verified']
    assert audit['all_split_sequences_exact_with_existing_history_policy']
    sid=Path(audit['sid_path'])
    assert digest(sid)==audit['sid_sha256']
    ids=torch.load(sid,map_location='cpu',weights_only=False)
    assert list(ids.shape)==[4,12101]
    assert ids.min()>=0 and ids.max()<256
    assert len(torch.unique(ids.T,dim=0))==12088
    for path,checksum in audit['official_sources'].items():assert digest(path)==checksum
    for path,checksum in audit['data_files'].items():assert digest(ROOT/path)==checksum
    paths=sorted(set(list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').rglob('*.yaml'))+
                     [ROOT/'codex/plan_v1_1/protocol.py',ROOT/'codex/sid_order_launch/sid_order_shuffle_model.py',
                      ROOT/'codex/sid_order_launch/sid_order_mlp_model.py',Path(__file__),HERE/'prepare_beauty.py']))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for p in paths:
        target=out/'source_snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(out/'source_manifest.json',{'files':hashes})
    shutil.copy2(audit_path,out/'mapping_audit.json')
    write(out/'input_provenance.json',dict(dataset='beauty',upstream='official_LETTER_released_SID',
        official_commit='8d0154e28de37dbb6e24871c508ad8ddb1921cda',
        train_seed=42,eval_seed=2026,fit_checkpoint=None,preflight_only=args.preflight,
        sid_path=str(sid),sid_sha256=digest(sid),mapping_audit_sha256=digest(audit_path),
        official_sid_values_unchanged=True,original_data_unchanged=True,
        collision_policy='Preserve official table; existing evaluator matches complete SID, not unique product identity.',
        variants=['unshuffle','shuffle_3','shuffle_4'],training_shuffle=False,
        shuffle_scope='Encoder historical item embeddings only; labels and decoder SID order stay unchanged.'))
    overrides=['experiment=tiger_train_flat',f'data_dir={audit["original_data_path"]}',
        f'semantic_id_path={sid}',f'paths.root_dir={ROOT}',f'paths.output_dir={out}',
        f'paths.work_dir={ROOT}',f'paths.profile_dir={out}/profile_output',f'paths.metadata_dir={out}/metadata',
        'num_hierarchies=4','trainer.devices=1','++trainer.strategy=auto','trainer.precision=32-true',
        'trainer.val_check_interval=1600','trainer.log_every_n_steps=10','trainer.deterministic=true',
        '++trainer.enable_progress_bar=false','train=true','test=true','ckpt_path=null',
        '++callbacks.restart_job=null','++callbacks.model_summary=null','extras.print_config=false','seed=42',
        '++model.top_k_for_generation=10','model._target_=codex.plan_v1_1.protocol.ShuffleModel',
        '++model.dataset_name=beauty','++model.eval_seed=2026','++model.shuffle_embedding_on_test=false',
        '++model.hierarchies_to_shuffle=3','++model.save_attentions_on_test=false',
        '++model.save_attentions_on_predict=false']
    for split in ['train','val','test']:
        p=f'data_loading.{split}_dataloader_config.dataloader.'
        overrides += [f'++{p}num_workers=0',f'++{p}timeout=0',f'++{p}persistent_workers=false']
    if args.preflight:
        overrides+=['trainer.max_steps=2','trainer.max_epochs=1','trainer.val_check_interval=32',
            '++trainer.limit_train_batches=32','++trainer.limit_val_batches=2','++trainer.limit_test_batches=2']
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        cfg=compose(config_name='train',overrides=overrides)
    with open_dict(cfg):
        cfg.paths.output_dir=str(out);cfg.paths.profile_dir=str(out/'profile_output')
        cfg.paths.metadata_dir=str(out/'metadata');cfg.trainer.default_root_dir=str(out)
        cfg.callbacks.model_checkpoint.dirpath=str(out/'checkpoints');cfg.logger.csv.save_dir=str(out)
    (out/'.hydra').mkdir()
    OmegaConf.save(cfg,out/'.hydra/config.yaml',resolve=True)
    write(out/'.hydra/overrides.json',overrides)
    torch.set_float32_matmul_precision('medium')

    class Recorder(Callback):
        def __init__(self):
            self.last=0;self.mode=None;self.test_users=set();self.test_totals={}
        def record(self,t,state,**extra):
            row=dict(state=state,time=time.time(),pid=os.getpid(),physical_gpu=args.gpu,
                     global_step=t.global_step,epoch=t.current_epoch,mode=self.mode,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated(),**extra)
            write(out/'progress.json',row)
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        def on_train_start(self,t,m):
            assert t.global_step==0
            self.record(t,'training_started')
        def on_train_batch_end(self,t,m,outputs,batch,batch_idx):
            value=float((outputs['loss'] if isinstance(outputs,dict) else outputs).detach().float().cpu())
            assert math.isfinite(value),'Non-finite loss'
            if time.monotonic()-self.last>=10:
                self.record(t,'training',loss=value,batch_idx=batch_idx);self.last=time.monotonic()
        def on_validation_start(self,t,m):
            assert not m.shuffle_embedding_on_test and m.shuffle_calls==0
            self.record(t,'validating')
        def on_validation_end(self,t,m):
            self.record(t,'validation_completed',metrics={k:float(v) for k,v in t.callback_metrics.items()})
        def on_train_end(self,t,m):self.record(t,'training_completed')
        def on_test_start(self,t,m):
            self.test_users=set();self.test_totals={}
            self.record(t,'test_started')
        def on_test_batch_end(self,t,m,outputs,batch,batch_idx,dataloader_idx=0):
            users=[int(v) for v in batch[0].transformed_sequences['user_id'][:,0].detach().cpu().tolist()]
            assert len(set(users))==len(users) and not self.test_users.intersection(users)
            self.test_users.update(users)
            # log_metrics resets custom metric counters at epoch end, so capture them here.
            self.test_totals={k:v.total_values for k,v in m.evaluator.metrics.items()}
        def on_test_end(self,t,m):self.record(t,'test_completed')
        def on_exception(self,t,m,exception):self.record(t,'failed',error=str(exception))

    recorder=Recorder()
    with pipeline_launcher(cfg) as p:
        p.trainer.callbacks.append(recorder)
        assert p.model.get_embedding_table('encoder') is p.model.get_embedding_table('decoder')
        assert torch.equal(p.model.codebooks.cpu(),ids.T)
        # All architecture, optimizer and data settings must equal the user's latest Qwen run.
        old=ROOT/'experiments/sid_order_5seed_v1/beauty/shuffle/seed_42/.hydra/config.yaml'
        reference=OmegaConf.load(old)
        keys=['model.huggingface_model.config','model.decoder.config','model.mlp_layers',
              'model.weight_tying','optim','sequence_length','trainer.accumulate_grad_batches',
              'trainer.val_check_interval','callbacks.early_stopping','eval',
              'data_loading.train_dataloader_config.dataloader.batch_size_per_device',
              'data_loading.train_dataloader_config.dataloader.collate_fn.max_batch_size']
        for key in keys:
            a=OmegaConf.select(cfg,key);b=OmegaConf.select(reference,key)
            if OmegaConf.is_config(a):a=OmegaConf.to_container(a,resolve=True)
            if OmegaConf.is_config(b):b=OmegaConf.to_container(b,resolve=True)
            if args.preflight and key=='trainer.val_check_interval':continue
            assert a==b,(key,a,b)
        write(out/'model_verified.json',dict(model=type(p.model).__module__+'.'+type(p.model).__name__,
            parameters=sum(t.numel() for t in p.model.parameters()),shared_embedding=True,
            official_sid_tensor_exact=True,baseline_configuration_matches=True,
            compared_configuration_keys=keys,reference_config=str(old),preflight_only=args.preflight,
            attention_export=False,max_steps=p.trainer.max_steps,max_epochs=p.trainer.max_epochs))
        L.seed_everything(42,workers=True)
        p.trainer.fit(p.model,datamodule=p.datamodule,ckpt_path=None)
        assert p.model.shuffle_calls==0
        best=Path(p.trainer.checkpoint_callback.best_model_path).resolve()
        assert best.is_file() and best.is_relative_to(out)
        best_hash=digest(best)
        result=dict(dataset='beauty',upstream='official_LETTER_SID',train_seed=42,eval_seed=2026,
                    checkpoint=str(best),checkpoint_sha256=best_hash,trained_from_scratch=True,
                    global_step=p.trainer.global_step,preflight_only=args.preflight,
                    best_validation_recall5=float(p.trainer.checkpoint_callback.best_model_score),
                    test_metrics={},completed=False)
        write(out/'stage_result.json',result)
        for mode,enabled,count in [('unshuffle',False,3),('shuffle_3',True,3),('shuffle_4',True,4)]:
            recorder.mode=mode
            L.seed_everything(2026,workers=True)
            p.model.shuffle_embedding_on_test=enabled;p.model.hierarchies_to_shuffle=count
            before=p.model.shuffle_calls
            metrics=p.trainer.test(p.model,datamodule=p.datamodule,ckpt_path=str(best))
            calls=p.model.shuffle_calls-before
            assert bool(calls)==enabled
            assert metrics and all(math.isfinite(float(v)) for v in metrics[0].values())
            totals=recorder.test_totals
            expected=16 if args.preflight else audit['users']
            assert all(n==expected for n in totals.values()),totals
            assert len(recorder.test_users)==expected
            info=dict(checkpoint=str(best),checkpoint_sha256=best_hash,metrics=metrics,
                      full_test_set=not args.preflight,evaluated_users=expected,
                      shuffle_calls=calls,shuffle_enabled=enabled,hierarchies_to_shuffle=count,
                      eval_seed=2026,ambiguous_official_sid_test_targets=audit['splits']['testing']['ambiguous_sid_targets'])
            write(out/f'test_{mode}.json',info)
            result['test_metrics'][mode]=info;write(out/'stage_result.json',result)
        assert digest(sid)==audit['sid_sha256']
        for path,checksum in audit['official_sources'].items():assert digest(path)==checksum
        result['completed']=True;write(out/'stage_result.json',result)
        rows=['| 模式 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |','|---|---:|---:|---:|---:|']
        for mode,info in result['test_metrics'].items():
            values=info['metrics'][0]
            row=[mode]+[f'{values["test/"+k]:.6f}' for k in ['recall@5','recall@10','ndcg@5','ndcg@10']]
            rows.append('| '+' | '.join(row)+' |')
        (out/'metrics.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
        recorder.record(p.trainer,'completed')
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
