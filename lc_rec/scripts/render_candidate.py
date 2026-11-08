#!/usr/bin/env python3
"""Print candidate argv only. This script never launches training."""
import argparse,json,pathlib,shlex
p=argparse.ArgumentParser();p.add_argument('--config',default=str(pathlib.Path(__file__).resolve().parents[1]/'config/formal_five_seed_candidate.json'));a=p.parse_args()
c=json.loads(pathlib.Path(a.config).read_text());assert c['world_size']*c['args']['per_device_batch_size']*c['args']['gradient_accumulation_steps']==128
for seed in c['seeds']:
    argv=['torchrun','--nproc_per_node='+str(c['world_size']),'finetune.py','--seed',str(seed),'--output_dir',f'/data/bwwu/lc-rec/formal-candidate/seed{seed}']
    for k,v in c['args'].items():
        if isinstance(v,bool):
            if v: argv+=['--'+k]
        else: argv+=['--'+k,str(v)]
    print(shlex.join(argv))
