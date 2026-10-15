"""User-authorized shared-GPU checks; stop only our own bounded computation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

GIB = 1024**3
CURRENT = None


def observe():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    devices=[]; tasks=[]
    for line in gpu.splitlines():
        idx,uid,total,used,util=[v.strip() for v in line.split(',')]
        devices.append({'index':int(idx),'uuid':uid,'total_bytes':int(total)*1024**2,
            'used_bytes':int(used)*1024**2,'free_bytes':(int(total)-int(used))*1024**2,'utilization_percent':int(util)})
    for line in processes.splitlines():
        uid,pid,used=[v.strip() for v in line.split(',')]
        tasks.append({'uuid':uid,'pid':int(pid),'used_bytes':int(used)*1024**2})
    return {'time_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'devices':devices,'processes':tasks}


def assess(observation, physical, shared=False, initial=True):
    device=next(v for v in observation['devices'] if v['uuid']==physical)
    foreign=[v for v in observation['processes'] if v['uuid']==physical and v['pid']!=os.getpid()]
    if foreign and not shared:return 'RESOURCE_BUSY: exclusive mode has existing processes'
    # Prior full-sized Beauty MLP: 1.95 GiB allocated / 3.12 GiB reserved.
    # Conservatively allow 6 GiB
    # for our unchanged models and preserve another 8 GiB (user asks about4).
    minimum=(14 if initial else 8)*GIB
    if device['free_bytes'] < minimum:return 'RESOURCE_MARGIN: insufficient physical free memory'
    return None


class ResourceMonitor:
    def __init__(self, output, physical, shared):
        self.output=Path(output);self.physical=physical;self.shared=shared
        self.stop=threading.Event();self.reason=None;self.observations=[]
    def sample(self, initial=False):
        try:
            value=observe();self.observations.append(value)
            reason=assess(value,self.physical,self.shared,initial)
            with (self.output/'resource-monitor.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
            if reason:self.reason=reason
        except Exception as exc:
            self.reason='RESOURCE_OBSERVATION_FAILED: '+repr(exc)
    def check(self):
        if self.reason:raise RuntimeError(self.reason+'; stopping only this preflight')
    def loop(self):
        while not self.stop.wait(10):self.sample()
    def __enter__(self):
        global CURRENT
        assert CURRENT is None;CURRENT=self
        self.sample(initial=True);self.check()
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self, *exception):
        global CURRENT
        self.stop.set();self.thread.join(timeout=15);self.sample()
        result={'shared_authorized':self.shared,'physical_uuid':self.physical,
            'expected_peak_gib':6,'initial_reserve_gib':8,'runtime_free_floor_gib':8,
            'samples':len(self.observations),'stop_reason':self.reason,
            'other_processes_signaled':False,'own_worker_pid':os.getpid()}
        (self.output/'resource-summary.json').write_text(json.dumps(result,indent=2)+'\n');CURRENT=None
        if not any(v is not None for v in exception):self.check()


def callback():
    from lightning import Callback
    class ResourceGuard(Callback):
        def on_train_batch_start(self,*args,**kwargs):
            if CURRENT:CURRENT.check()
        def on_validation_batch_start(self,*args,**kwargs):
            if CURRENT:CURRENT.check()
        def on_test_batch_start(self,*args,**kwargs):
            if CURRENT:CURRENT.check()
    return ResourceGuard()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=4);p.add_argument('--interval',type=float,default=5)
    a=p.parse_args();assert 2<=a.count<=6 and 1<=a.interval<=10
    observations=[]
    for i in range(a.count):
        observations.append(observe())
        if i+1<a.count:time.sleep(a.interval)
    a.output.write_text(json.dumps({'observations':observations,'models_loaded':False},indent=2)+'\n')
    print(json.dumps(observations[-1]))


if __name__=='__main__':main()
