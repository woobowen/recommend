#!/usr/bin/env python3
"""CPU S0/S1/S2/after-replay joins, including small native and loader RNG files."""
import argparse
import json
import os
from pathlib import Path
from resume_analyze import differences,effective_bnb
from resume_probe import tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--continuous',required=True);p.add_argument('--resumed',required=True);p.add_argument('--inspection',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 import torch
 left,right,ins=map(Path,[a.continuous,a.resumed,a.inspection]);cp=left/'checkpoint-2'
 def read(p):return json.loads(p.read_text())
 def load(p):return torch.load(p,map_location='cpu',weights_only=False)
 model=read(ins/'mp_rank_00_model_states.pt.json')['module'];sched=tree(load(cp/'external_scheduler.pth'));report={'points':{'S0':'immediately before native checkpoint save, update2 complete','S1':'actual checkpoint files','S2':'on_train_begin after strict load and zero_grad, BEFORE loader replay','before_first_forward':'after replay, next batch yielded9, before actual forward'},'ranks':[]}
 for rank in [0,1]:
  s0=read(left/f'S0-before-save-step2.rank{rank}.json');s2=read(right/f'S2-train-begin.rank{rank}.json');pre0=read(left/f'before_first_forward.rank{rank}.json');pre2=read(right/f'before_first_forward.rank{rank}.json')
  opt=read(ins/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt.json')['optimizer_state_dict'];loader=load(cp/f'loader_state_{rank}.pth');native=load(cp/f'rng_state_{rank}.pth')
  row={'rank':rank,'S0_S1':{'model':differences(s0['model'],model),'optimizer':differences(s0['optimizer'],opt),'scheduler':differences(s0['scheduler'],sched),'checkpoint_rng':differences(s0['rng'],tree(loader['checkpoint_rng']))},
       'S1_S2':{'model':differences(model,s2['model']),'optimizer':differences(opt,s2['optimizer']),'scheduler':differences(sched,s2['scheduler'])},
       'paired_before_forward':{k:differences(pre0[k],pre2[k]) for k in ['model','optimizer','scheduler','rng','parameters','gradients']},'canonical_bnb':differences(effective_bnb(pre0),effective_bnb(pre2)),
       'loader':{k:loader[k] for k in ['schema','epoch_index','yielded','resume_epoch','global_step']},'native_rng_vs_loader_checkpoint':{k:differences(tree(v),tree(loader['checkpoint_rng'][k])) for k,v in native.items()},'cuda_initialized':torch.cuda.is_initialized()}
  report['ranks'].append(row)
 Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
