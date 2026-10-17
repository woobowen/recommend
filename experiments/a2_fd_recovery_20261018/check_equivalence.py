"""Bounded CPU checks on the preserved real Sports item/embedding pipeline.

No GPU training, new SID, or recommendation results are produced here.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import resource
from types import SimpleNamespace

import numpy as np
import torch


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def storages(value):
    if isinstance(value, torch.Tensor):
        return {(value.untyped_storage().data_ptr(), value.untyped_storage().nbytes())}
    if isinstance(value, dict):
        return set().union(*(storages(x) for x in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(storages(x) for x in value))
    return set()


def compare(a, b, reference):
    ids_a = torch.stack(list(a.item_ids), dim=0)
    ids_b = torch.stack(list(b.item_ids), dim=0)
    assert ids_a.dtype == ids_b.dtype and torch.equal(ids_a, ids_b)
    assert a.transformed_features.keys() == b.transformed_features.keys()
    for k, v in a.transformed_features.items():
        w = b.transformed_features[k]
        assert v.shape == w.shape and v.dtype == w.dtype and torch.equal(v, w)
    ids = ids_b.reshape(-1)
    assert torch.equal(b.transformed_features['input_embedding'], reference[ids])
    return dict(items=len(ids), unique_ids=len(torch.unique(ids)), id_shape=list(ids_b.shape),
                id_dtype=str(ids_b.dtype), ids_sha256=digest(ids_b),
                embedding_shape=list(b.transformed_features['input_embedding'].shape),
                embedding_dtype=str(b.transformed_features['input_embedding'].dtype),
                embedding_sha256=digest(b.transformed_features['input_embedding']),
                old_id_storages=len(storages(a.item_ids)), new_id_storages=len(storages(b.item_ids)),
                old_total_storages=len(storages([a.item_ids,a.transformed_features])),
                new_total_storages=len(storages([b.item_ids,b.transformed_features])),
                exact_values_dtype_order_all_features_pairing=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from experiments.beauty_extensions.author.bootstrap import bind_author_source
    bind_author_source()
    import hydra
    from lightning import seed_everything
    from lightning.fabric.utilities.seed import pl_worker_init_function
    from lightning.pytorch.trainer.states import TrainerFn
    from omegaconf import OmegaConf
    from src.data.loading.components.collate_functions import collate_fn_items
    from src.data.loading.components.interfaces import ItemData
    from experiments.launch_20261017.a2_upstream import components
    from experiments.a2_fd_recovery_20261018.adapter import (collate_items_contiguous,
        configure_process, adapt_config, process_record)

    torch.set_num_threads(4)
    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    limits = configure_process(args.output)
    old_job = Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261017-sportsa2-quantize-v1/output')
    used = json.loads((old_job/'used_config.json').read_text())
    cfg = OmegaConf.create(used)
    embedding_path = Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs/ga-20261017-sportsa2-encode-v1/output/item_embeddings.pt')
    raw = embedding_path.read_bytes()
    reference = torch.load(embedding_path, map_location='cpu', weights_only=True)
    assert reference.shape == (18357,512) and reference.dtype == torch.float32
    config_diff = adapt_config(cfg)
    (args.output/'config-equivalence.json').write_text(json.dumps(config_diff,indent=2)+'\n')
    seed_everything(42, workers=True)
    dm = hydra.utils.instantiate(OmegaConf.create(used['data_loading']['datamodule']))
    dm.trainer = SimpleNamespace(world_size=1, global_rank=0)
    dm.setup()
    dataset = dm.get_dataloader(TrainerFn.PREDICTING)[0].dataset
    samples = list(dataset)
    ids = [int(x['id'].item()) for x in samples]
    assert len(ids)==18357 and len(set(ids))==18357 and set(ids)==set(range(18357))
    assert all(torch.equal(x['embedding'],reference[i]) for x,i in zip(samples,ids))
    sample = samples[:2048]
    kwargs = dict(item_id_field='id', feature_to_input_name={'embedding':'input_embedding'})
    random_before=random.getstate();np_before=np.random.get_state();torch_before=torch.get_rng_state().clone()
    a = collate_fn_items(sample, **kwargs)
    b = collate_items_contiguous(sample, **kwargs)
    assert isinstance(a,ItemData) and isinstance(b,ItemData)
    assert random.getstate()==random_before and torch.equal(torch.get_rng_state(),torch_before)
    np_after=np.random.get_state();assert np_before[0]==np_after[0] and np.array_equal(np_before[1],np_after[1]) and np_before[2:]==np_after[2:]
    direct=compare(a,b,reference)
    assert direct['old_id_storages']==2048 and direct['new_id_storages']==1
    # Exercise the original dict input branch, including an additional feature.
    grouped={'id':[x['id'] for x in sample], 'embedding':[x['embedding'] for x in sample],
             'probe':[torch.tensor([i],dtype=torch.int16) for i in range(len(sample))]}
    extra=dict(item_id_field='id',feature_to_input_name={'embedding':'input_embedding','probe':'other'})
    other=compare(collate_fn_items(grouped,**extra),collate_items_contiguous(grouped,**extra),reference)

    # A deterministic initialized CPU fixture for actual preserved model_step and predict_step.
    # These centroids are test-only; no training or historical quantizer is claimed.
    model=hydra.utils.instantiate(cfg.model)
    model.trainer=SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.PREDICTING))
    for j,layer in enumerate(model.quantization_layer_list):
        layer.centroids.data.copy_(torch.nn.functional.normalize(reference[j*256:(j+1)*256],dim=-1))
        layer.is_initialized=True
    model.eval()
    with torch.inference_mode():
        outputs_a=model.model_step(a);outputs_b=model.model_step(b)
        for x,y in zip(outputs_a,outputs_b):
            assert x is None and y is None or isinstance(x,torch.Tensor) and torch.equal(x,y)
        pred_a=model.predict_step(a);pred_b=model.predict_step(b)
    assert pred_a.keys==pred_b.keys==[int(x) for x in b.item_ids.reshape(-1)]
    assert torch.equal(pred_a.predictions,pred_b.predictions)
    _,Writer=components();wa=Writer(args.output,'cluster_ids');wb=Writer(args.output,'cluster_ids')
    wa.on_predict_batch_end(None,None,pred_a,a,0);wb.on_predict_batch_end(None,None,pred_b,b,0)
    assert all(x['item_id']==y['item_id'] and torch.equal(x['cluster_ids'],y['cluster_ids']) for x,y in zip(wa.rows,wb.rows))

    ipc={}
    # Small bounded before/after IPC fixture; never recreate 50K-FD failure.
    for mode,fn in [('original',collate_fn_items),('contiguous',collate_items_contiguous)]:
        seed_everything(42,workers=True)
        loader=torch.utils.data.DataLoader(samples[:2048],batch_size=512,num_workers=2,
            prefetch_factor=2,persistent_workers=True,collate_fn=functools.partial(fn,**kwargs),
            worker_init_fn=functools.partial(pl_worker_init_function,rank=0))
        it=iter(loader);rows=[]
        for index in range(4):
            value=next(it)
            rows.append(dict(batch=index,process=process_record(),
                item_ids=torch.stack(list(value.item_ids)).reshape(-1).tolist(),
                item_storages=len(storages(value.item_ids)),embedding_sha256=digest(value.transformed_features['input_embedding'])))
        it._shutdown_workers()
        ipc[mode]=rows
    assert [r['item_ids'] for r in ipc['original']]==[r['item_ids'] for r in ipc['contiguous']]
    assert [r['embedding_sha256'] for r in ipc['original']]==[r['embedding_sha256'] for r in ipc['contiguous']]
    native=[]
    # Original native 12-worker/2048 loader, adapted collate only. Rank is CPU
    # simulation here; actual Lightning 2-GPU/DDP is checked in the formal job.
    for rank in (0,1):
        os.environ['LOCAL_RANK']=str(rank)
        seed_everything(42,workers=True)
        dm=hydra.utils.instantiate(cfg.data_loading.datamodule)
        dm.trainer=SimpleNamespace(world_size=2,global_rank=rank);dm.setup()
        loader=dm.get_dataloader(TrainerFn.FITTING)[0]
        assert loader.num_workers==12 and loader.batch_size==2048 and loader.prefetch_factor==2
        assert loader.worker_init_fn is None
        loader.worker_init_fn=functools.partial(pl_worker_init_function,rank=rank)
        it=iter(loader);rows=[]
        for i in range(3):
            value=next(it);data=value.transformed_features['input_embedding']
            itemids=value.item_ids.reshape(-1)
            assert torch.equal(data,reference[itemids])
            rows.append(dict(batch=i,items=len(itemids),id_dtype=str(itemids.dtype),
                ids_sha256=digest(itemids),embedding_sha256=digest(data),
                process=process_record(),item_storages=len(storages(value.item_ids)),exact_pairing=True))
        children=[process_record(x.pid) for x in it._workers]
        assert len(children)==12 and all(x['soft']==limits['after'][0] and x['fd']<8192 for x in children)
        native.append(dict(simulated_rank=rank,num_workers=12,prefetch_factor=loader.prefetch_factor,
            raw_batch=2048,persistent_workers=loader.persistent_workers,pin_memory=loader.pin_memory,
            workers=children,batches=rows,worker_seeding='same Lightning function explicitly called in CPU simulation'))
        it._shutdown_workers()
    report=dict(status='CPU_EQUIVALENCE_AND_BOUNDED_IPC_PASS',scientific_result='NOT_YET_AVAILABLE',
        input=dict(path=str(embedding_path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
        real_item_axis=dict(items=len(ids),unique=len(set(ids)),all_dense_ids_exact=True,all_embedding_pairs_exact=True),
        direct=direct,dict_and_extra_features=other,rng_states_unchanged=True,
        actual_original_model_step_predict_step_and_writer_equal=True,
        model_fixture='deterministically filled initialized CPU centroids; not a trained quantizer',
        ipc=ipc,native_loader_cpu_simulation=native,process_limits=limits,
        original_process_limits=list(before),gpu_requested=0,
        adapted_source_sha256=hashlib.sha256(Path(__file__).with_name('adapter.py').read_bytes()).hexdigest())
    (args.output/'equivalence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
