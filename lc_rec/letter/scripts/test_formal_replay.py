#!/usr/bin/env python3
"""CPU-only synthetic boundary tests of the actual nested ReplayLoader AST.

Uses the installed real DataLoaderShard/RandomSampler, with zero CUDA devices.
Does not load a model or claim a real training/epoch-boundary GPU test.
"""
import argparse
import ast
import hashlib
import json
import random
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Dataset, RandomSampler
    from accelerate.data_loader import prepare_data_loader
    assert not torch.cuda.is_initialized()
    source=Path(__file__).with_name('formal_train.py').read_text()
    tree=ast.parse(source)
    cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='ReplayLoader')
    class Data(Dataset):
        def __len__(self):return 131
        def __getitem__(self,i):return (i,random.randrange(17),int(np.random.randint(100)))
    def capture(loader=None):
        r={'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':[]}
        if loader is not None:r['loader_generator']=loader.synchronized_generator.get_state()
        return r
    def restore(r,loader=None):
        random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'])
        if loader is not None:loader.synchronized_generator.set_state(r['loader_generator'])
    events=[]
    ns={'rng_capture':capture,'rng_restore':restore,'event':lambda kind,**kw:events.append({'event':kind,**kw})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'actual_ReplayLoader','exec'),ns)
    Replay=ns['ReplayLoader']
    def loader(rank,saved=None):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        data=Data();original=DataLoader(data,batch_size=8,sampler=RandomSampler(data),num_workers=0)
        shard=prepare_data_loader(original,device=None,num_processes=2,process_index=rank,
                                  split_batches=False,put_on_device=False,rng_types=[],even_batches=True)
        return Replay(shard,saved)
    def digest(batch):return [x.tolist() for x in batch]
    results=[]
    for rank in [0,1]:
        full=loader(rank);seq=[];saved=None
        iterator=iter(full)
        for i,batch in enumerate(iterator):
            seq.append(digest(batch))
            if i==3:saved=full.checkpoint(1)
        boundary=full.checkpoint(3)
        epoch2=[digest(b) for b in full]
        resumed=loader(rank,saved)
        remaining=[digest(b) for b in resumed]
        assert remaining==seq[4:]
        boundary_resume=loader(rank,boundary)
        # Match native Trainer's first-batch-only replay of completed epochs.
        for _ in boundary_resume:break
        restored_epoch2=[digest(b) for b in boundary_resume]
        assert restored_epoch2==epoch2
        results.append({'rank':rank,'batches_per_epoch':len(seq),'mid_epoch_remaining':len(remaining),
                        'mid_epoch_all_rows_prompts_numpy_exact':True,'next_epoch_all_rows_prompts_numpy_exact':True})
    assert not torch.cuda.is_initialized()
    Path(a.output).write_text(json.dumps({'status':'PASS','synthetic_cpu_only':True,
        'actual_class_ast_sha256':hashlib.sha256(ast.dump(cls).encode()).hexdigest(),
        'torch_version':torch.__version__,'cases':results,'replay_events':events},indent=2)+'\n')
    print(json.dumps({'status':'PASS','cases':results}))


if __name__=='__main__':main()
