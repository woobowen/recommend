"""One v1.1 task: fresh fit, best-val full tests, frozen-user attention."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASELINES = {'yelp_aligned_v1': 'yelp_rkmeans_qwen_aligned_v1'}
DATA_DIR = Path('/home/jcy/data/yelp_aligned_v1')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=BASELINES, required=True)
    ap.add_argument('--variant', choices=['shuffle', 'mlp'], required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--seed', type=int, choices=range(42, 47), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--sample-dir', type=Path, required=True)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--prepare-only', action='store_true', help='Verify loaders/model/samples without fit or test')
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        ap.error('Fresh output required; no implicit checkpoint reuse')
    out.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='3', TF_NUM_INTRAOP_THREADS='2',
        TF_NUM_INTEROP_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ('PROJECT_ROOT', 'METADATA_DIR', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(key, None)
    sys.path.insert(0, str(ROOT))
    import rootutils
    rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import Callback
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    import src.utils.custom_hydra_resolvers
    from src.utils.launcher_utils import pipeline_launcher
    from codex.plan_v1_1.attention import prepare_samples, export_attention
    from codex.plan_v1_1.protocol import user_ids

    torch.use_deterministic_algorithms(True, warn_only=False)
    baseline = ROOT / 'experiments' / BASELINES[args.dataset]
    sid = baseline / 'semantic_ids.pt'
    vectors = baseline / 'item_embeddings.pt'
    vm = json.loads(vectors.with_suffix('.json').read_text())
    qm = json.loads((baseline / 'quantizer_train/stage_result.json').read_text())
    quantizer = Path(qm['checkpoint'].replace('/home/jcyin/', str(Path.home())+'/')).resolve()
    assert quantizer.is_relative_to(baseline)
    assert vm['dataset'] == args.dataset and vm['architecture'] == 'qwen2'
    assert digest(vectors) == vm['sha256']
    ids = torch.load(sid, map_location='cpu', weights_only=False)
    assert list(ids.shape) == [4, vm['shape'][0]]
    assert ids.min() >= 0 and ids.max() < 256
    assert torch.unique(ids.T, dim=0).size(0) == ids.size(1)
    inputs = {name: {'path': str(path), 'sha256': digest(path)} for name, path in
              [('text_vectors',vectors),('quantizer',quantizer),('semantic_ids',sid)]}
    raw_files = sorted((DATA_DIR).rglob('*.tfrecord*'))
    assert raw_files
    raw_hashes = {str(path): digest(path) for path in raw_files if path.is_file()}
    input_hashes = dict(sid=inputs['semantic_ids']['sha256'], data=raw_hashes)
    source_paths = sorted(set(list((ROOT/'src').rglob('*.py')) + list((ROOT/'configs').rglob('*.yaml')) +
        list((ROOT/'codex/plan_v1_1').glob('*.py')) + list((ROOT/'codex/yelp_aligned_v1').glob('*.py')) + list((ROOT/'codex/sid_order_launch').glob('*_model.py'))))
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    source_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    for source in source_paths:
        target = out/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_json(out/'source_manifest.json', {'sha256':source_digest,'files':source_hashes})
    write_json(out/'input_provenance.json', dict(plan_version='1.1', dataset=args.dataset,
        variant=args.variant, train_seed=args.seed, eval_seed=2026, gpu=args.gpu,
        fit_ckpt_path=None, inputs=inputs, data_files=raw_hashes, preflight_only=args.preflight,
        embedding_tables='shared_encoder_decoder', train=True, test=True,
        full_test_attention_export=False, analysis_attention_export=True, source_sha256=source_digest))
    del ids
    overrides = ['experiment=tiger_train_flat',
        f'data_dir={DATA_DIR}', f'semantic_id_path={sid}',
        f'paths.root_dir={ROOT}', f'paths.output_dir={out}', f'paths.work_dir={ROOT}',
        f'paths.profile_dir={out}/profile_output', f'paths.metadata_dir={out}/metadata',
        'num_hierarchies=4','trainer.devices=1','++trainer.strategy=auto','trainer.precision=32-true',
        'trainer.val_check_interval=1600','trainer.log_every_n_steps=10','trainer.deterministic=true',
        '++trainer.enable_progress_bar=false','train=true','test=true','ckpt_path=null',
        '++callbacks.restart_job=null','++callbacks.model_summary=null','extras.print_config=false',
        f'seed={args.seed}','++model.top_k_for_generation=10',
        f'model._target_=codex.plan_v1_1.protocol.{"ShuffleModel" if args.variant=="shuffle" else "MLPModel"}',
        f'++model.dataset_name={args.dataset}','++model.eval_seed=2026']
    for split in ('train','val','test'):
        prefix = f'data_loading.{split}_dataloader_config.dataloader.'
        overrides += [f'++{prefix}num_workers=0', f'++{prefix}timeout=0', f'++{prefix}persistent_workers=false']
    if args.variant == 'shuffle':
        overrides += ['++model.shuffle_embedding_on_test=false','++model.hierarchies_to_shuffle=3',
            '++model.save_attentions_on_test=false','++model.save_attentions_on_predict=false']
    if args.preflight:
        # Retain the formal batch sizes and accumulation; cap only duration/data coverage.
        overrides += ['trainer.max_steps=2','trainer.max_epochs=1','trainer.val_check_interval=32',
            '++trainer.limit_train_batches=32','++trainer.limit_val_batches=2','++trainer.limit_test_batches=2']
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=overrides)
    with open_dict(cfg):
        cfg.paths.output_dir = str(out)
        cfg.paths.profile_dir = str(out/'profile_output')
        cfg.paths.metadata_dir = str(out/'metadata')
        cfg.trainer.default_root_dir = str(out)
        cfg.callbacks.model_checkpoint.dirpath = str(out/'checkpoints')
        cfg.logger.csv.save_dir = str(out)
    (out/'.hydra').mkdir()
    OmegaConf.save(cfg, out/'.hydra/config.yaml', resolve=True)
    write_json(out/'.hydra/overrides.json', overrides)
    torch.set_float32_matmul_precision('medium')

    class Recorder(Callback):
        def __init__(self):
            self.last = 0
            self.mode = None
        def record(self, trainer, state, **extra):
            record = dict(state=state, time=time.time(), pid=os.getpid(), physical_gpu=args.gpu,
                global_step=trainer.global_step, epoch=trainer.current_epoch, mode=self.mode,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(), **extra)
            write_json(out/'progress.json',record)
            with (out/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == 0
            self.record(trainer,'training_started')
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs['loss'] if isinstance(outputs,dict) else outputs
            value = float(loss.detach().float().cpu())
            assert math.isfinite(value), 'Non-finite loss'
            if time.monotonic()-self.last >= 10:
                self.record(trainer,'training',loss=value,batch_idx=batch_idx,
                    actual_batch_size=int(batch[0][0].mask.size(0)))
                self.last=time.monotonic()
        def on_validation_start(self, trainer, pl_module):
            self.record(trainer,'validating')
        def on_validation_end(self, trainer, pl_module):
            self.record(trainer,'validation_completed', metrics={k:float(v) for k,v in trainer.callback_metrics.items()})
        def on_train_end(self, trainer, pl_module):
            self.record(trainer,'training_completed')
        def on_test_start(self, trainer, pl_module):
            self.record(trainer,'test_started')
        def on_test_end(self, trainer, pl_module):
            self.record(trainer,'test_completed')
        def on_exception(self, trainer, pl_module, exception):
            self.record(trainer,'failed',error=str(exception))

    recorder=Recorder()
    with pipeline_launcher(cfg) as p:
        p.trainer.callbacks.append(recorder)
        assert p.model.get_embedding_table('encoder') is p.model.get_embedding_table('decoder')
        if args.variant=='mlp':
            assert p.model.sep_token is None and hasattr(p.model,'item_aggregation_mlp')
        samples, sample_manifest = prepare_samples(p,args.sample_dir,args.dataset,input_hashes)
        write_json(out/'sample_manifest.json',sample_manifest)
        p.model.protocol_sample_ids = {s['user_id'] for s in samples}
        # Sampling and dataloader setup may use RNG; reset before fit, consistently across tasks.
        L.seed_everything(args.seed,workers=True)
        write_json(out/'model_verified.json',dict(model=type(p.model).__name__,
            parameters=sum(t.numel() for t in p.model.parameters()),shared_embedding=True,
            deterministic=p.trainer.deterministic if hasattr(p.trainer,'deterministic') else cfg.trainer.deterministic,
            train_source_batch=cfg.data_loading.train_dataloader_config.dataloader.batch_size_per_device,
            train_collate_cap=cfg.data_loading.train_dataloader_config.dataloader.collate_fn.max_batch_size,
            full_test_attention_export=False,preflight_only=args.preflight))
        assert cfg.train and cfg.test
        if args.prepare_only:
            from codex.yelp_aligned_v1.preflight import inspect_without_training
            report = inspect_without_training(p, cfg, args.variant)
            report.update(fit_called=False, test_called=False, formal_training_started=False,
                          frozen_users=len(samples), sample_manifest_sha256=digest(out/'sample_manifest.json'))
            write_json(out/'preparation_verified.json', report)
            print(json.dumps(report, indent=2), flush=True)
            return
        p.trainer.fit(p.model,datamodule=p.datamodule,ckpt_path=None)
        assert p.model.shuffle_calls==0,'Training or validation unexpectedly shuffled'
        best=Path(p.trainer.checkpoint_callback.best_model_path).resolve()
        assert best.is_relative_to(out) and best.is_file()
        best_hash=digest(best)
        result=dict(plan_version='1.1', dataset=args.dataset,variant=args.variant,
            train_seed=args.seed,eval_seed=2026,checkpoint=str(best),checkpoint_sha256=best_hash,
            best_validation_recall5=float(p.trainer.checkpoint_callback.best_model_score),
            global_step=p.trainer.global_step,trained_from_scratch=True,
            preflight_only=args.preflight,train_completed=True,test_metrics={},attention_groups={})
        write_json(out/'stage_result.json',result)
        modes=[('mlp',False,0)] if args.variant=='mlp' else [
            ('unshuffled',False,3),('shuffle_3',True,3),('shuffle_4',True,4)]
        references={}
        for mode,enabled,count in modes:
            recorder.mode=mode
            L.seed_everything(2026,workers=True)
            p.model.protocol_references={}
            if args.variant=='shuffle':
                p.model.shuffle_embedding_on_test=enabled
                p.model.hierarchies_to_shuffle=count
            before=p.model.shuffle_calls
            metrics=p.trainer.test(p.model,datamodule=p.datamodule,ckpt_path=str(best))
            calls=p.model.shuffle_calls-before
            assert bool(calls)==enabled
            assert metrics and all(math.isfinite(float(v)) for v in metrics[0].values())
            references[mode]=p.model.protocol_references
            torch.save(references[mode],out/f'full_test_reference_{mode}.pt')
            info=dict(checkpoint=str(best),checkpoint_sha256=best_hash,metrics=metrics,
                train_seed=args.seed,eval_seed=2026,full_test_set=not args.preflight,
                attention_export=False,shuffle_calls=calls)
            write_json(out/f'test_{mode}.json',info)
            result['test_metrics'][mode]=info
            write_json(out/'stage_result.json',result)
        for mode,enabled,count in modes:
            recorder.mode=mode
            recorder.record(p.trainer,'attention_analysis_started')
            if args.variant=='shuffle':
                p.model.shuffle_embedding_on_test=enabled
                p.model.hierarchies_to_shuffle=count
            metadata=dict(plan_version='1.1',dataset=args.dataset,variant=args.variant,mode=mode,
                train_seed=args.seed,eval_seed=2026,checkpoint_sha256=best_hash,
                source_sha256=source_digest,sample_manifest_sha256=digest(out/'sample_manifest.json'))
            analysis=export_attention(p.model,samples[:3] if args.preflight else samples,
                out/'attention_analysis'/mode,metadata,references[mode],require_references=not args.preflight)
            result['attention_groups'][mode]={'users':analysis['users'],'complete':True}
            write_json(out/'stage_result.json',result)
        result['completed']=True
        write_json(out/'stage_result.json',result)
        recorder.record(p.trainer,'completed')
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        traceback.print_exc()
        # Launcher also records failures before Trainer is constructed.
        raise
