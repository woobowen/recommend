#!/usr/bin/env python3
"""CPU-only asset preparation. Never downloads model weights or uses a GPU."""
import argparse, datetime, hashlib, json, os, pathlib, shutil, subprocess, sys, time, urllib.request, zipfile

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();cfg=json.loads(pathlib.Path(a.config).read_text());out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();root=pathlib.Path(cfg['asset_root']);root.mkdir(parents=True,exist_ok=True)
    disk=shutil.disk_usage(root);reserve=max(cfg['limits']['disk_reserve_min_bytes'],int(disk.total*.1))
    if disk.free < reserve+1024**3: raise RuntimeError('disk reserve gate')
    report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':sys.argv,'cwd':os.getcwd(),'config_sha256':sha(a.config),'downloads':[],'weights_downloaded':False,'gpu_used':False,'initial_disk':disk._asdict()}
    def checkpoint(): (out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    def fetch(url,path,max_bytes,expected=None):
        path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if expected is not None and path.stat().st_size != expected: raise RuntimeError('existing asset size conflict: '+str(path))
            report['downloads'].append({'url':url,'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'reused':True});checkpoint();return
        partial=path.with_name(path.name+'.partial')
        # One request; transport failure is recorded, not a loop or access workaround.
        begin=time.monotonic()
        with urllib.request.urlopen(url,timeout=90) as response, partial.open('xb') as target:
            content_type=response.headers.get('Content-Type','');total=0
            if 'text/html' in content_type: raise RuntimeError('received HTML rather than public asset: '+url)
            while True:
                block=response.read(1024*1024)
                if not block: break
                total+=len(block)
                if total>max_bytes: raise RuntimeError('asset exceeds expected budget')
                target.write(block)
                if time.monotonic()-begin>300: raise RuntimeError('download wall time bound exceeded')
        if expected is not None and total != expected: raise RuntimeError('download size differs from provider metadata')
        partial.replace(path);report['downloads'].append({'url':url,'path':str(path),'bytes':total,'sha256':sha(path),'elapsed_seconds':time.monotonic()-begin,'reused':False});checkpoint()
    try:
        archive=root/'Instruments.zip'
        fetch('https://drive.google.com/uc?export=download&id='+cfg['data_drive_id'],archive,32*1024**2,cfg['data_archive_bytes'])
        data=root/'data';data.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            infos=z.infolist();names=[x.filename for x in infos]
            assert len(names)==len(set(names)), 'duplicate archive paths'
            assert sum(i.file_size for i in infos)<256*1024**2, 'unexpected expansion'
            for i in infos:
                p=pathlib.PurePosixPath(i.filename)
                assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
                assert not (i.external_attr>>16)&0o170000==0o120000, 'symlink member'
            assert z.testzip() is None
            report['archive_members']=[{'name':i.filename,'bytes':i.file_size,'crc32':i.CRC} for i in infos]
            for i in infos:
                if i.is_dir(): continue
                dest=data/i.filename;dest.parent.mkdir(parents=True,exist_ok=True)
                payload=z.read(i)
                if dest.exists(): assert sha(dest)==hashlib.sha256(payload).hexdigest(), 'existing extracted asset conflict'
                else: dest.write_bytes(payload)
        tok=root/'llama-7b'/cfg['model_revision']
        for name in ['config.json','generation_config.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','LICENSE','README.md']:
            fetch('https://huggingface.co/'+cfg['model_id']+'/resolve/'+cfg['model_revision']+'/'+name,tok/name,4*1024**2)
        env=pathlib.Path(cfg['environment']);env_python=env/'bin/python'
        if not env_python.exists():
            subprocess.run([cfg['reused_python'],'-m','venv','--system-site-packages',str(env)],check=True)
            report['created_environment']=str(env)
        install=[str(env_python),'-m','pip','install','--disable-pip-version-check','--no-cache-dir','--no-deps',*cfg['overlay_packages']]
        report['install_argv']=install;checkpoint()
        with (out/'pip-install.log').open('w') as log: subprocess.run(install,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        with (out/'pip-freeze.txt').open('w') as log: subprocess.run([str(env_python),'-m','pip','freeze','--all'],check=True,stdout=log,timeout=60)
        code='import sys,json,importlib,importlib.metadata as m; pkgs=["torch","transformers","tokenizers","huggingface_hub","sentencepiece","numpy","safetensors"]; print(json.dumps({"python":sys.version,"executable":sys.executable,"prefix":sys.prefix,"packages":{p:{"version":m.version(p),"import_source":importlib.import_module(p).__file__} for p in pkgs}},indent=2))'
        p=subprocess.run([str(env_python),'-c',code],check=True,text=True,capture_output=True,timeout=90)
        (out/'environment.json').write_text(p.stdout);report['environment']=json.loads(p.stdout)
        report['status']='PREPARED_CPU_AUDIT_ONLY'
    except BaseException as error:
        report['status']='FAILED';report['error']=repr(error);raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        report['final_disk']=shutil.disk_usage(root)._asdict();checkpoint()
    print(json.dumps({'status':report['status'],'elapsed_seconds':report['elapsed_seconds']}))
if __name__=='__main__':main()
