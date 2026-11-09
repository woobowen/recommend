#!/usr/bin/env python3
"""Hash-checked public model chunk receiver. Does not overwrite existing assets."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();r=json.loads(a.request)
    base=Path('/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549');assert not base.is_symlink() and base.is_dir()
    target=base/r['name'];assert target.parent==base and r['name'] in ['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']
    journal=base/(r['name']+'.offload-20261108.receipt.json');partial=base/(r['name']+'.offload-20261108.partial')
    with (base/(r['name']+'.offload-20261108.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if target.exists():
            assert target.is_file() and not target.is_symlink() and target.stat().st_size==r['total_bytes']
            assert sha(target)==r['total_sha256']
            print(json.dumps({'complete':True,'bytes':target.stat().st_size,'sha256':r['total_sha256'],'reused':True}));return
        old=json.loads(journal.read_text()) if journal.exists() else {'offset':0,'chunks':[],'total_bytes':r['total_bytes'],'total_sha256':r['total_sha256']}
        assert old['total_bytes']==r['total_bytes'] and old['total_sha256']==r['total_sha256']
        assert (partial.stat().st_size if partial.exists() else 0)==old['offset']
        if r['mode']=='inspect':print(json.dumps(old));return
        assert r['offset']==old['offset']
        chunk=Path(r['chunk']);assert chunk.parent==Path('/home/bwwu/lc-rec/offload-20261108/chunks') and chunk.is_file() and not chunk.is_symlink()
        assert chunk.stat().st_size==r['bytes'] and sha(chunk)==r['sha256']
        assert old['offset']+r['bytes']<=old['total_bytes']
        disk=shutil.disk_usage(base);assert disk.free>max(30*1024**3,int(disk.total*.1))+r['bytes']
        mem=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024;assert mem>40*1024**3
        with partial.open('ab') as dst,chunk.open('rb') as src:
            shutil.copyfileobj(src,dst,4*1024**2);dst.flush();os.fsync(dst.fileno())
        old['offset']+=r['bytes'];old['chunks'].append({k:r[k] for k in ['offset','bytes','sha256']})
        old['complete']=old['offset']==old['total_bytes']
        if old['complete']:
            assert sha(partial)==old['total_sha256'],'whole shard LFS SHA mismatch; preserve partial'
            os.link(partial,target);partial.unlink()
            old['path']=str(target)
        tmp=journal.with_suffix('.json.new');tmp.write_text(json.dumps(old,indent=2)+'\n');tmp.replace(journal)
        # This exact chunk was consumed into a durable, hash-bound target; no other files removed.
        chunk.unlink();print(json.dumps(old))

if __name__=='__main__':main()
