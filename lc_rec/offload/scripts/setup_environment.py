#!/usr/bin/env python3
"""Isolated offline install; CPUAdam math and original LC-Rec are unmodified."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();report={'argv':sys.argv,'cwd':os.getcwd(),'config_sha256':sha(a.config),'commands':[],'status':'STARTED'}
    def save(): (out/'setup.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(argv,name,env=None,timeout=900):
        report['commands'].append({'argv':argv,'log':name});save()
        with (out/name).open('w') as log:
            subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=timeout)
    try:
        assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024>80*1024**3
        disk=shutil.disk_usage('/data/bwwu');assert disk.free>max(30*1024**3,int(disk.total*.1))+100*1024**3
        pkg=Path(cfg['package_dir']);manifest=json.loads((pkg/'manifest.json').read_text())
        for row in manifest['files']:
            q=pkg/row['name'];assert q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
        env=Path(cfg['environment']);py=str(env/'bin/python')
        assert not env.exists(), 'fresh environment required; inspect original attempt before a repair'
        run([cfg['base_python'],'-m','venv','--system-site-packages',str(env)],'venv.log')
        install=[py,'-m','pip','install','--no-index','--no-deps','--disable-pip-version-check','--no-cache-dir','--find-links',cfg['prior_wheelhouse'],'--find-links',str(pkg)]
        run(install+cfg['packages'],'pip-install.log')
        source=out/'deepspeed-build';source.mkdir()
        with tarfile.open(pkg/'deepspeed-0.16.9.tar.gz') as tf:
            for m in tf.getmembers():
                q=Path(m.name);assert not q.is_absolute() and '..' not in q.parts and (m.isfile() or m.isdir())
            tf.extractall(source)
        ds=source/'deepspeed-0.16.9';setup=ds/'setup.py';old=setup.read_text()
        old_probe='''        cuda_major_ver, cuda_minor_ver = installed_cuda_version()
        if (cuda_major_ver < 11) or ((cuda_major_ver == 11) and (cuda_minor_ver < 3)):
            cupy = f"cupy-cuda{cuda_major_ver}{cuda_minor_ver}"
        else:
            cupy = f"cupy-cuda{cuda_major_ver}x"'''
        new_probe='''        from op_builder.builder import MissingCUDAException
        try:
            cuda_major_ver, cuda_minor_ver = installed_cuda_version()
        except MissingCUDAException:
            cuda_major_ver, cuda_minor_ver = None, None
        if cuda_major_ver is not None:
            if (cuda_major_ver < 11) or ((cuda_major_ver == 11) and (cuda_minor_ver < 3)):
                cupy = f"cupy-cuda{cuda_major_ver}{cuda_minor_ver}"
            else:
                cupy = f"cupy-cuda{cuda_major_ver}x"'''
        assert old.count(old_probe)==1,'upstream source differs; refuse guessed patch'
        setup.write_text(old.replace(old_probe,new_probe))
        (out/'deepspeed-packaging.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),setup.read_text().splitlines(True),fromfile='upstream/setup.py',tofile='derived/setup.py')))
        build_env=os.environ.copy();build_env.update(DS_ACCELERATOR='cuda',DS_BUILD_OPS='0',DS_BUILD_CPU_ADAM='1',MAX_JOBS='2',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        # CPUAdamBuilder.builder() catches MissingCUDAException and builds CppExtension.
        # Preinstalled accelerator metadata remains cuda; no runtime JIT/skip-check patch.
        run([py,'-m','pip','install','--no-index','--no-deps','--no-build-isolation','--no-cache-dir',str(ds)],'deepspeed-build.log',build_env,1200)
        run([py,'-m','pip','freeze','--all'],'pip-freeze.txt',timeout=60)
        smoke=Path(__file__).with_name('verify_cpu_adam.py')
        run([py,'-B',str(smoke),'--output',str(out/'cpu-adam.json')],'cpu-adam.log',build_env,180)
        report['status']='ENVIRONMENT_AND_CPU_ADAM_VERIFIED'
    except BaseException as e:
        report['status']='FAILED';report['error']=repr(e);raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start;save()

if __name__=='__main__':main()
