"""Bounded real CPU Gloo ranks for runner identity/exit/timeout acceptance."""
import argparse
import datetime
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from formal_control import durable_write

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['success','failure','timeout'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['TORCHELASTIC_MAX_RESTARTS']=='0'
rank=int(os.environ['RANK']);root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
stat=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()
durable_write(root/f'rank{rank}.identity.json',{'rank':rank,'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgid(0),'session':os.getsid(0),
 'start_ticks':stat[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'job_id':os.environ['GRID_JOB_ID'],'job_dir':os.environ['GRID_JOB_DIR'],'cuda_visible':os.environ['CUDA_VISIBLE_DEVICES']})
dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=20))
x=torch.tensor([rank+1]);dist.all_reduce(x);assert x.item()==3
if a.mode=='failure' and rank==1:raise RuntimeError('INTENTIONAL_CPU_RANK_FAILURE')
if a.mode in ('failure','timeout'):time.sleep(90)
dist.barrier();dist.destroy_process_group()
durable_write(root/f'rank{rank}.done.json',{'rank':rank,'sum':x.item(),'status':'CPU_COMPLETE'})
