"""Read batches and compute a loss without optimization or metric testing."""
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from src.data.loading.components.interfaces import SequentialModelInputData, SequentialModuleLabelData


def inspect_without_training(p, cfg, variant):
    root=Path(__file__).resolve().parents[2]
    reference=root/f'experiments/sid_order_5seed_v1/beauty/{variant}/seed_42/.hydra/config.yaml'
    old=OmegaConf.to_container(OmegaConf.load(reference),resolve=True)
    new=OmegaConf.to_container(cfg,resolve=True)
    diffs=[]
    def compare(a,b,key):
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(set(a)|set(b)):
                compare(a.get(k),b.get(k),key+'.'+k)
        elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
            for i,(x,y) in enumerate(zip(a,b)):
                compare(x,y,f'{key}.{i}')
        elif a!=b:
            diffs.append(dict(key=key,amazon=a,yelp=b))
    for key in ('model','trainer','optim','loss','eval','data_loading','sequence_length','num_hierarchies','seed',
                'callbacks','logger','train','test','ckpt_path'):
        compare(old[key],new[key],key)
    # Only dataset identity, dataset input paths and output paths can differ.
    allowed=('model.dataset_name','trainer.default_root_dir','callbacks.model_checkpoint.dirpath','logger.csv.save_dir')
    def accepted(d):
        key=d['key']
        if key in allowed:
            return True
        if key.startswith('data_loading.') or key.startswith('model.codebooks.'):
            return (key.endswith('.data_folder') or key.endswith('.file_path'))
        return False
    unexpected=[d for d in diffs if not accepted(d)]
    assert not unexpected, json.dumps(unexpected,indent=2)
    dm=p.datamodule
    dm.trainer=p.trainer
    dm.setup('fit')
    p.model.eval()
    # CPU forward is sufficient to verify the interface; no optimizer is constructed.
    batch_reports=[]
    for name,loader in [('training',dm.train_dataloader()),('validation',dm.val_dataloader()),('testing',dm.test_dataloader())]:
        if isinstance(loader,(tuple,list)):
            loader=loader[0]
        batch=next(iter(loader))
        # The training loader can wrap a batch as a singleton list.
        while isinstance(batch,(list,tuple)) and len(batch)==1:
            batch=batch[0]
        inputs,labels=batch
        size=inputs.mask.shape[0]
        assert 0<size<= (128 if name=='training' else 8)
        assert inputs.mask.shape[1]==120
        assert set(inputs.transformed_sequences)=={'user_id','sequence_data'}
        assert all(t.reshape(size,-1).shape[1]==4 for t in labels.labels.values())
        n=min(size,2)
        x=SequentialModelInputData(user_id_list=inputs.transformed_sequences['user_id'][:n,0].tolist(),mask=inputs.mask[:n],
            transformed_sequences={k:v[:n] for k,v in inputs.transformed_sequences.items()})
        y=SequentialModuleLabelData(labels={k:v.reshape(size,-1)[:n] for k,v in labels.labels.items()})
        with torch.inference_mode():
            _,loss=p.model.model_step(model_input=x,label_data=y)
        assert torch.isfinite(loss)
        batch_reports.append(dict(split=name,batch_size=size,mask_shape=list(inputs.mask.shape),
                                  label_sid_slots=4,forward_loss=float(loss)))
    assert p.model.shuffle_calls==0
    return dict(passed=True,reference_config=str(reference),allowed_config_differences=diffs,
                unexpected_config_differences=[],batch_checks=batch_reports,shared_encoder_decoder_embedding=True,
                optimizer_steps=0,attention_samples_only=True)
