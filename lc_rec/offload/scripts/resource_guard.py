#!/usr/bin/env python3
"""Task-local resource guard around one fixed torchrun command; not a scheduler."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(p,v):
    q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)


def meminfo():
    return {x.split(':',1)[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if len(x.split())>=2}


def swap_io():
    return {n:int(v) for n,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines()) if n in ('pswpin','pswpout')}


def allocated(roots):
    total=0
    for root in roots:
        if not root.exists():continue
        for directory,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[n for n in dirs if not (Path(directory)/n).is_symlink()]
            for name in files:
                p=Path(directory)/name
                try:
                    if not p.is_symlink():total+=p.stat().st_blocks*512
                except FileNotFoundError:pass
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--job-id',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    assert a.command and a.command[0]=='--';command=a.command[1:]
    cfg=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert cfg['single_gpu_seconds']<=5400 and cfg['all_gpu_seconds']<=7200 and cfg['cpu_reserve_bytes']>=40*1024**3
    assert os.environ.get('GRID_JOB_ID')==a.job_id and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    import psutil
    ledger_path=Path(cfg['budget_ledger']);ledger_path.parent.mkdir(parents=True,exist_ok=True)
    with ledger_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'runs':[],'prior_stage_a_seconds':cfg['prior_stage_a_seconds']}
        assert all(r['job_id']!=a.job_id for r in ledger['runs']),'query original job; do not relaunch'
        assert all(r.get('end_utc') for r in ledger['runs']),'previous GPU run unresolved'
        elapsed=sum(r['elapsed_seconds'] for r in ledger['runs'])+ledger['prior_stage_a_seconds']
        limit=min(cfg['single_gpu_seconds'],cfg['all_gpu_seconds']-elapsed)
        assert limit>0
        baseline=meminfo();baseline_swap_io=swap_io();page_size=os.sysconf('SC_PAGE_SIZE');assert baseline['MemAvailable']>=cfg['cpu_reserve_bytes']
        start=time.monotonic();entry={'job_id':a.job_id,'candidate':cfg['candidate'],'argv':command,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limit_seconds':limit}
        ledger['runs'].append(entry);atomic(ledger_path,ledger)
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL)
        owner=psutil.Process(child.pid);tracked={(owner.pid,owner.create_time()):owner};reason=None
        roots=[Path(r) for r in cfg['new_disk_roots']]
        peaks={'cpu_rss_sum_peak_bytes':0,'min_mem_available_bytes':baseline['MemAvailable'],'new_disk_allocated_peak_bytes':0,'max_swap_used_increase_bytes':0,'task_vm_swap_peak_bytes':0}
        samples=0
        try:
            with (out/'resource-samples.jsonl').open('a',buffering=1) as stream:
                while True:
                    stamp=time.monotonic()-start;mem=meminfo();now=[]
                    try:
                        for q in [owner]+owner.children(recursive=True):
                            if q.uids().real==os.getuid():tracked.setdefault((q.pid,q.create_time()),q)
                    except psutil.Error:pass
                    for ident,q in list(tracked.items()):
                        try:
                            if q.create_time()==ident[1] and q.is_running():now.append(q)
                        except (psutil.Error,OSError):pass
                    rss=0;task_swap=0
                    for q in now:
                        try:
                            rss+=q.memory_info().rss
                            for line in Path('/proc',str(q.pid),'status').read_text().splitlines():
                                if line.startswith('VmSwap:'):task_swap+=int(line.split()[1])*1024
                        except (psutil.Error,OSError):pass
                    swap_increase=max(0,(baseline['SwapFree']-mem['SwapFree']))
                    io=swap_io();io_delta={n:(io[n]-baseline_swap_io[n])*page_size for n in io}
                    row={'elapsed_seconds':stamp,'mem_available_bytes':mem['MemAvailable'],'task_tree_rss_sum_bytes':rss,'swap_used_increase_bytes':swap_increase,'task_vm_swap_bytes':task_swap,'host_swap_io_delta_bytes':io_delta,'tracked_pids':[q.pid for q in now]}
                    if samples%3==0:
                        row['new_disk_allocated_bytes']=allocated(roots)
                        v=os.statvfs('/data/bwwu');row['data_free_bytes']=v.f_bavail*v.f_frsize;row['data_total_bytes']=v.f_blocks*v.f_frsize
                        g=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
                        row['nvidia_smi']=g.stdout.strip();row['nvidia_smi_exit']=g.returncode
                        peaks['new_disk_allocated_peak_bytes']=max(peaks['new_disk_allocated_peak_bytes'],row['new_disk_allocated_bytes'])
                        if row['new_disk_allocated_bytes']>100*1024**3:reason='NEW_DISK_100GIB_BOUND'
                        if row['data_free_bytes']<max(30*1024**3,row['data_total_bytes']//10):reason='DISK_RESERVE_BOUND'
                    peaks['cpu_rss_sum_peak_bytes']=max(peaks['cpu_rss_sum_peak_bytes'],rss)
                    peaks['min_mem_available_bytes']=min(peaks['min_mem_available_bytes'],mem['MemAvailable'])
                    peaks['max_swap_used_increase_bytes']=max(peaks['max_swap_used_increase_bytes'],swap_increase)
                    peaks['task_vm_swap_peak_bytes']=max(peaks['task_vm_swap_peak_bytes'],task_swap)
                    stream.write(json.dumps(row)+'\n');samples+=1
                    if mem['MemAvailable']<cfg['cpu_reserve_bytes']:reason='CPU_40GIB_RESERVE_BOUND'
                    if swap_increase>256*1024**2:reason='HOST_SWAP_INCREASE_256MIB_BOUND'
                    if task_swap>cfg.get('task_vm_swap_limit_bytes',64*1024**2):reason='TASK_SWAP_BOUND'
                    if io_delta['pswpout']>cfg.get('host_swap_io_limit_bytes',256*1024**2):reason='HOST_SWAP_OUT_IO_BOUND'
                    if stamp>=limit:reason='GPU_WALL_CLOCK_BOUND'
                    if reason or child.poll() is not None:break
                    time.sleep(5)
        finally:
            if child.poll() is None:
                # Only descendants whose pid AND start-time were observed under our child.
                for ident,q in reversed(list(tracked.items())):
                    try:
                        if q.create_time()==ident[1] and q.uids().real==os.getuid():q.terminate()
                    except psutil.Error:pass
                psutil.wait_procs(list(tracked.values()),timeout=5)
                for ident,q in reversed(list(tracked.items())):
                    try:
                        if q.create_time()==ident[1] and q.is_running():q.kill()
                    except psutil.Error:pass
            exitcode=child.wait(timeout=15)
            entry.update(elapsed_seconds=time.monotonic()-start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),child_exit_code=exitcode,stop_reason=reason,peaks=peaks,samples=samples)
            atomic(ledger_path,ledger);atomic(out/'resource-summary.json',entry)
        sys.exit(124 if reason else (exitcode if exitcode>=0 else 128-exitcode))

if __name__=='__main__':main()
