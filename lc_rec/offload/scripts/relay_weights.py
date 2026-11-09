#!/usr/bin/env python3
"""Bounded WSL -> Windows SCP public HF relay; never stage a complete model locally."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
import urllib.request


def main():
    p=argparse.ArgumentParser();p.add_argument('--transport-dir',required=True);p.add_argument('--receiver',required=True);p.add_argument('--summary',required=True);p.add_argument('--output',required=True);p.add_argument('--cpu-adam-evidence',required=True);a=p.parse_args()
    assert json.loads(Path(a.cpu_adam_evidence).read_text())['status']=='PASS'
    sys.path.insert(0,a.transport_dir)
    from transport import Transport,atomic
    t=Transport('new-server');out=Path(a.output);out.mkdir(parents=True,exist_ok=True);journal=out/'download.json'
    r=json.loads(journal.read_text()) if journal.exists() else {'started_monotonic':time.monotonic(),'downloaded_body_bytes':1048576,'initial_range_probe_bytes':1048576,'chunks':[],'formal_files':[]}
    atomic(journal,r);summary=json.loads(Path(a.summary).read_text())
    revision='4782ad278652c7c71b72204d462d6d01eaaf7549'
    assert summary['revision']==revision and summary['model_id']=='huggyllama/llama-7b'
    required=[x for x in summary['siblings'] if x['rfilename'] in ['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']]
    assert len(required)==2 and sum(x['size'] for x in required)<20*1024**3
    for row in required:
        name=row['rfilename'];identity={'name':name,'total_bytes':row['size'],'total_sha256':row['lfs']['sha256']}
        def receive(request):
            return json.loads(t.ssh(shlex.join(['/usr/bin/python3','-I','-S','-B',a.receiver,'--request',json.dumps(request,separators=(',',':'))]),timeout=180))
        current=receive(dict(identity,mode='inspect'))
        if current.get('complete'):
            r['formal_files'].append(dict(identity,reused=True));atomic(journal,r);continue
        offset=current['offset']
        while offset<row['size']:
            end=min(offset+256*1024**2,row['size'])-1
            url='https://huggingface.co/huggyllama/llama-7b/resolve/'+revision+'/'+name+'?download=true&range='+str(offset)+'-'+str(end)
            chunk=out/(name+'.'+str(offset)+'.chunk');begin=time.monotonic()
            assert not chunk.exists(),'inspect partial transfer before restarting'
            request=urllib.request.Request(url,headers={'Range':'bytes='+str(offset)+'-'+str(end),'Accept-Encoding':'identity'})
            with urllib.request.urlopen(request,timeout=60) as response:
                assert response.status==206 and response.headers.get('Content-Range')=='bytes '+str(offset)+'-'+str(end)+'/'+str(row['size']), 'server did not honor exact byte range'
                total=0;digest=hashlib.sha256()
                with chunk.open('xb') as dst:
                    while b:=response.read(4*1024**2):
                        total+=len(b);r['downloaded_body_bytes']+=len(b)
                        atomic(journal,r)
                        assert total<=end-offset+1 and r['downloaded_body_bytes']<=20*1024**3
                        dst.write(b);digest.update(b)
                        assert time.monotonic()-begin<600,'bounded chunk request time exceeded'
                assert total==end-offset+1
            remote='/home/bwwu/lc-rec/offload-20261108/chunks/'+chunk.name
            t.put(chunk,remote)
            receipt=receive(dict(identity,mode='append',offset=offset,chunk=remote,bytes=total,sha256=digest.hexdigest()))
            assert receipt['offset']==end+1
            r['chunks'].append({'name':name,'offset':offset,'bytes':total,'sha256':digest.hexdigest(),'elapsed_seconds':time.monotonic()-begin,'remote_offset':receipt['offset']})
            offset=end+1;atomic(journal,r)
            chunk.unlink()  # exact self-created, consumed transfer buffer only
            print(json.dumps({'file':name,'offset':offset,'total':row['size'],'downloaded_bytes':r['downloaded_body_bytes']}),flush=True)
        assert receipt['complete'];r['formal_files'].append(dict(identity,path=receipt['path'],verified=True));atomic(journal,r)
    r['status']='ALL_REQUIRED_WEIGHT_SHARDS_VERIFIED';r['elapsed_seconds']=time.monotonic()-r['started_monotonic'];atomic(journal,r)

if __name__=='__main__':main()
