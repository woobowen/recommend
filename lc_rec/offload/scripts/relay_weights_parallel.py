#!/usr/bin/env python3
"""Six bounded HTTP Range futures; serial SCP/receiver; same relay CLI.
Controller must stop the old serial writer first. Normal same-source transport
engineering only. No downloads run at import, no retries or full-file staging.
"""
import argparse, concurrent.futures as cf, fcntl, hashlib, json, os
from pathlib import Path
import re, shlex, sys, threading, time, urllib.request

CHUNK=256*1024**2; READ=4*1024**2; WORKERS=6; CAP=20*1024**3
REV='4782ad278652c7c71b72204d462d6d01eaaf7549'
NAMES=['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']
REMOTE='/home/bwwu/lc-rec/offload-20261108/chunks/'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(READ),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['transport-dir','receiver','summary','output','cpu-adam-evidence']: p.add_argument('--'+key,required=True)
    a=p.parse_args()
    assert json.loads(Path(a.cpu_adam_evidence).read_text())['status']=='PASS'
    sys.path.insert(0,a.transport_dir)
    from transport import Transport,atomic
    out=Path(a.output).absolute(); assert not out.is_symlink(); out.mkdir(parents=True,exist_ok=True); out=out.resolve()
    lockfile=(out/'parallel-relay.lock').open('a'); fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    t=Transport('new-server'); journal=out/'download.json'; mutex=threading.RLock(); stop=threading.Event()
    r=json.loads(journal.read_text()) if journal.exists() else {'started_monotonic':time.monotonic(),'downloaded_body_bytes':1048576,'initial_range_probe_bytes':1048576,'chunks':[],'formal_files':[]}
    assert type(r['downloaded_body_bytes']) is int and 1048576<=r['downloaded_body_bytes']<=CAP
    assert r['initial_range_probe_bytes']==1048576  # Preserve source probe accounting; never reset by disk lengths.
    summary=json.loads(Path(a.summary).read_text()); assert summary['revision']==REV and summary['model_id']=='huggyllama/llama-7b'
    rows={name:next(x for x in summary['siblings'] if x['rfilename']==name) for name in NAMES}
    assert sum(x['size'] for x in rows.values())<CAP
    assert all(type(x['size']) is int and re.fullmatch('[0-9a-f]{64}',x['lfs']['sha256']) for x in rows.values())
    identities={name:{'name':name,'total_bytes':x['size'],'total_sha256':x['lfs']['sha256']} for name,x in rows.items()}
    binding={'revision':REV,'model_id':summary['model_id'],'files':identities,'receiver':a.receiver}
    assert r.get('parallel_source_binding',binding)==binding
    r['parallel_source_binding']=binding; r.setdefault('prefetch',{}); reserved=0
    attempt={'script_sha256':sha(Path(__file__)),'argv':sys.argv,'started_monotonic':time.monotonic(),'workers':6,'max_futures':6,'chunk_bytes':CHUNK,'starting_downloaded_body_bytes':r['downloaded_body_bytes'],'http_retries':0,'engineering_only':True}
    r.setdefault('parallel_attempts',[]).append(attempt)
    held=set(out.glob('*.chunk'))
    def parse(path):
        for name in NAMES:
            m=re.fullmatch(re.escape(name)+r'\.(\d+)\.chunk',path.name)
            if m:
                offset=int(m.group(1)); assert offset%CHUNK==0 and 0<=offset<rows[name]['size']; return name,offset
        raise RuntimeError('Unknown chunk preserved: '+str(path))
    def check(path,length):
        assert path.parent==out and not path.is_symlink() and path.is_file()
        st=path.stat(); assert st.st_nlink==1 and 0<=st.st_size<=length; return st.st_size
    def save():
        with mutex:
            r['active_reserved_http_body_bytes']=reserved; r['staging_slots']=sorted(x.name for x in held); atomic(journal,r)
    def receive(request):
        return json.loads(t.ssh(shlex.join(['/usr/bin/python3','-I','-S','-B',a.receiver,'--request',json.dumps(request,separators=(',',':'))]),timeout=180))
    for path in held:
        name,offset=parse(path); check(path,min(CHUNK,rows[name]['size']-offset))
    assert len(held)<=6; r['status']='PARALLEL_RELAY_RUNNING'; save()

    def fetch(name,offset):
        nonlocal reserved
        path=out/(name+'.'+str(offset)+'.chunk'); length=min(CHUNK,rows[name]['size']-offset)
        began=time.monotonic(); remaining_reservation=0; key=name+':'+str(offset)
        try:
            if stop.is_set(): raise RuntimeError('Stopped after another transfer failure')
            present=check(path,length) if path.exists() else 0; h=hashlib.sha256()
            if present:
                with path.open('rb') as f:
                    for b in iter(lambda:f.read(READ),b''): h.update(b)
            previous=r['prefetch'].get(key,{})
            if present<length:
                start=offset+present; end=offset+length-1; left=length-present
                with mutex:
                    assert r['downloaded_body_bytes']+reserved+left<=CAP,'cumulative20GiB budget insufficient'
                    reserved+=left; remaining_reservation=left
                    r['prefetch'][key]={'name':name,'offset':offset,'present_bytes_at_resume':present,'http_range':[start,end],'complete':False}; save()
                url='https://huggingface.co/huggyllama/llama-7b/resolve/'+REV+'/'+name+'?download=true&range='+str(start)+'-'+str(end)
                req=urllib.request.Request(url,headers={'Range':'bytes='+str(start)+'-'+str(end),'Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=60) as response:
                    assert response.status==206 and response.headers.get('Content-Range')=='bytes '+str(start)+'-'+str(end)+'/'+str(rows[name]['size'])
                    assert response.headers.get('Content-Encoding') in (None,'identity')
                    if response.headers.get('Content-Length') is not None: assert int(response.headers['Content-Length'])==left
                    with path.open('ab' if present else 'xb') as dst:
                        while left:
                            if stop.is_set(): raise RuntimeError('Stopped after another transfer failure')
                            assert time.monotonic()-began<600,'bounded chunk request time exceeded'
                            b=response.read(min(READ,left)); assert b,'short body; keep partial, no retry'
                            with mutex:
                                r['downloaded_body_bytes']+=len(b); reserved-=len(b); remaining_reservation-=len(b)
                                assert r['downloaded_body_bytes']<=CAP; save()
                            dst.write(b); h.update(b); left-=len(b)
                            assert time.monotonic()-began<600,'bounded chunk request time exceeded'
                        dst.flush(); os.fsync(dst.fileno())
            digest=h.hexdigest(); assert check(path,length)==length
            if previous.get('complete') and previous.get('sha256'): assert previous['sha256']==digest
            item={'name':name,'offset':offset,'bytes':length,'sha256':digest,'elapsed_seconds':time.monotonic()-began}
            with mutex:
                r['prefetch'][key]=dict(item,complete=True,reused_local_bytes=present); save()
            return item
        except BaseException as exc:
            stop.set()
            with mutex:
                r.setdefault('prefetch_errors',[]).append({'name':name,'offset':offset,'error':repr(exc)}); save()
            raise
        finally:
            with mutex: reserved-=remaining_reservation; save()

    try:
        for name in NAMES:
            identity=identities[name]; current=receive(dict(identity,mode='inspect'))
            assert current.get('total_bytes',identity['total_bytes'])==identity['total_bytes'] and current.get('total_sha256',identity['total_sha256'])==identity['total_sha256']
            durable_end=identity['total_bytes'] if current.get('complete') else current['offset']
            # Lost reply recovery: only delete exact task buffers with durable hash-bound consumption evidence.
            known={x['offset']:x for x in r['chunks'] if x['name']==name}
            known.update({x['offset']:x for x in current.get('chunks',[])})
            for path in list(held):
                local_name,local_offset=parse(path)
                if local_name==name and local_offset<durable_end:
                    proof=known.get(local_offset); assert proof,'consumed buffer has no receipt; preserve and inspect'
                    assert check(path,proof['bytes'])==proof['bytes'] and sha(path)==proof['sha256']
                    path.unlink(); held.remove(path); save()
            if not current.get('complete'):
                offset=current['offset']; assert type(offset) is int and offset%CHUNK==0 and 0<=offset<rows[name]['size']
                upcoming=offset; futures={}; pool=cf.ThreadPoolExecutor(max_workers=6,thread_name_prefix='fixed-hf-range')
                try:
                    while offset<rows[name]['size']:
                        with mutex:
                            while upcoming<rows[name]['size'] and len(futures)<6 and not stop.is_set():
                                path=out/(name+'.'+str(upcoming)+'.chunk')
                                if path not in held and len(held)>=6: break
                                held.add(path); futures[upcoming]=pool.submit(fetch,name,upcoming)
                                upcoming+=min(CHUNK,rows[name]['size']-upcoming)
                            save()
                        assert offset in futures,'no safe slot for next chunk; preserve buffers'
                        item=futures[offset].result(); assert not stop.is_set(),'prefetch failed; no upload retry'
                        chunk=out/(name+'.'+str(offset)+'.chunk'); assert check(chunk,item['bytes'])==item['bytes'] and sha(chunk)==item['sha256']
                        remote=REMOTE+chunk.name; t.put(chunk,remote)  # sole Windows/SCP writer
                        current=receive(dict(identity,mode='append',offset=offset,chunk=remote,bytes=item['bytes'],sha256=item['sha256']))
                        end=offset+item['bytes']; assert current['offset']==end
                        assert any(x['offset']==offset and x['bytes']==item['bytes'] and x['sha256']==item['sha256'] for x in current['chunks'])
                        with mutex:
                            r['chunks'].append(dict(item,remote_offset=end,receiver_consumed_verified=True)); save()
                            chunk.unlink(); held.remove(chunk); del futures[offset]; offset=end; save()
                        print(json.dumps({'file':name,'offset':offset,'total':rows[name]['size'],'downloaded_bytes':r['downloaded_body_bytes'],'max_prefetch':6}),flush=True)
                finally:
                    if sys.exc_info()[0] is not None:
                        stop.set()
                        for future in futures.values(): future.cancel()
                    pool.shutdown(wait=True,cancel_futures=True)
            assert current['complete']  # receiver full-LFS check is required before this flag becomes true
            assert current.get('sha256',identity['total_sha256'])==identity['total_sha256']
            assert current.get('bytes',identity['total_bytes'])==identity['total_bytes']
            r['formal_files']=[x for x in r['formal_files'] if x['name']!=name]+[dict(identity,path=current.get('path'),verified=True,reused=current.get('reused',False))]; save()
        assert {x['name'] for x in r['formal_files'] if x.get('verified')}==set(NAMES)
        r['status']='ALL_REQUIRED_WEIGHT_SHARDS_VERIFIED'; attempt['status']='PASS'
    except BaseException as exc:
        stop.set(); r['status']='PARALLEL_RELAY_FAILED_PRESERVE_PARTIALS'; attempt.update(status='FAILED',error=repr(exc)); raise
    finally:
        attempt['elapsed_seconds']=time.monotonic()-attempt['started_monotonic']; r['elapsed_seconds']=time.monotonic()-r['started_monotonic']; save()
        lockfile.close()

if __name__=='__main__': main()
