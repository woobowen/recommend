"""Validation-only convergence diagnostics; preserve training RNG and parameters."""
from collections import Counter
import json
import random
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
from codex.plan_v1_1.protocol import user_ids


class ExtendedMonitor(Callback):
    def __init__(self, pipeline, samples, output):
        self.output=Path(output)
        self.best=-float('inf')
        self.wait=0
        self.shadow_stop=None
        chosen={s['user_id'] for s in samples}
        dm=pipeline.datamodule
        dm.trainer=pipeline.trainer
        dm.setup('validate')
        loader=dm.val_dataloader()
        if isinstance(loader,(tuple,list)):
            loader=loader[0]
        selected=[]
        for inputs,labels in loader:
            size=inputs.mask.size(0)
            for i,uid in enumerate(user_ids(inputs)):
                if uid in chosen:
                    selected.append(dict(user_id=uid,mask=inputs.mask[i:i+1].clone(),
                        inputs={k:v[i:i+1].clone() for k,v in inputs.transformed_sequences.items()},
                        label=next(iter(labels.labels.values())).reshape(size,-1)[i:i+1].clone()))
        assert len(selected)==len(chosen)==100
        self.samples=sorted(selected,key=lambda x:x['user_id'])
        torch.save(self.samples,self.output/'fixed_validation_users.pt')
        (self.output/'diagnostic_protocol.json').write_text(json.dumps(dict(
            variant='shuffle',seed=43,changed_training_parameter={'early_stopping_patience':[10,40]},
            diagnostic_users=100,diagnostic_split='validation',
            selection='Same previously frozen user IDs; validation histories and validation labels only',
            test_used_for_model_selection=False,shadow_patience=10,
            original_seed43_test_checkpoint_step=200,original_seed43_stop_step=1200,
            note='Fresh run; original TFRecord shuffle has no explicit TF seed, so this is not promised to replay its exact trajectory.'),indent=2))

    def on_validation_end(self,trainer,model):
        if trainer.sanity_checking:
            return
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        training=model.training
        old_uids=getattr(model,'_protocol_user_ids',None)
        try:
            score=float(trainer.callback_metrics['val/recall@5'])
            if score>self.best:
                self.best=score
                self.wait=0
            else:
                self.wait+=1
            if self.shadow_stop is None and self.wait>=10:
                self.shadow_stop=int(trainer.global_step)
            model.eval()
            top1=[];top10=[];hits=[];scores=[]
            with torch.inference_mode():
                for start in range(0,len(self.samples),8):
                    group=self.samples[start:start+8]
                    mask=torch.cat([s['mask'] for s in group]).to(model.device)
                    fields={k:torch.cat([s['inputs'][k] for s in group]).to(model.device)
                            for k in group[0]['inputs']}
                    ids,probs=model.generate(attention_mask=mask,output_attentions=False,
                        **{model.feature_to_model_input_map.get(k,k):v for k,v in fields.items()})
                    labels=torch.cat([s['label'] for s in group]).to(ids.device)
                    hits.extend((ids[:,:5]==labels[:,None,:]).all(-1).any(-1).cpu().tolist())
                    for row,prob in zip(ids.cpu(),probs.cpu()):
                        top1.append(tuple(row[0].tolist()))
                        top10.append(tuple(row.reshape(-1).tolist()))
                        scores.append(float(prob[0]))
            record=dict(step=int(trainer.global_step),val_recall5=score,best_val_recall5=self.best,
                validations_without_improvement=self.wait,original_patience_would_first_stop_at=self.shadow_stop,
                validation_users=100,unique_top1=len(set(top1)),unique_top10_lists=len(set(top10)),
                largest_top1_share=Counter(top1).most_common(1)[0][1]/100,
                fixed_validation_recall5=sum(hits)/100,mean_top1_score=float(np.mean(scores)))
            with (self.output/'convergence_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            (self.output/'latest_convergence.json').write_text(json.dumps(record,indent=2))
        finally:
            model.train(training)
            if old_uids is not None:
                model._protocol_user_ids=old_uids
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])


def verify_config(cfg,reference,output):
    a=OmegaConf.to_container(OmegaConf.load(reference),resolve=True)
    b=OmegaConf.to_container(cfg,resolve=True)
    differences=[]
    def compare(x,y,key):
        if isinstance(x,dict) and isinstance(y,dict):
            for k in sorted(set(x)|set(y)):
                compare(x.get(k),y.get(k),key+'.'+k)
        elif isinstance(x,list) and isinstance(y,list) and len(x)==len(y):
            for i,(u,v) in enumerate(zip(x,y)):
                compare(u,v,f'{key}.{i}')
        elif x!=y:
            differences.append(dict(key=key,reference=x,diagnostic=y))
    for k in ('model','trainer','callbacks','optim','loss','eval','data_loading','seed','sequence_length','num_hierarchies'):
        compare(a[k],b[k],k)
    allowed={'callbacks.early_stopping.patience','callbacks.model_checkpoint.dirpath','trainer.default_root_dir'}
    assert all(d['key'] in allowed for d in differences),differences
    assert cfg.callbacks.early_stopping.patience==40
    Path(output,'configuration_comparison.json').write_text(json.dumps(differences,indent=2))
