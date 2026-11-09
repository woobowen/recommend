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
    p.add_argument('--reuse-new-environment',action='store_true',help='bounded repair of this task environment only')
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
        pkg=Path(cfg['package_dir']);manifest=json.loads((pkg/cfg.get('package_manifest','manifest.json')).read_text())
        for row in manifest['files']:
            q=pkg/row['name'];assert q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
        if cfg.get('cuda_home'):
            cuda=Path(cfg['cuda_home']);assert str(cuda)=='/data/bwwu/lc-rec/offload-20261108/cuda-12.4' and not cuda.exists()
            cuda.mkdir(parents=True)
            for name in cfg['cuda_packages']:
                with tarfile.open(pkg/name) as tf:
                    for member in tf.getmembers():
                        q=Path(member.name)
                        assert not q.is_absolute() and '..' not in q.parts
                        assert member.isfile() or member.isdir() or member.issym()
                        if member.issym():assert not Path(member.linkname).is_absolute() and (cuda/q).parent.joinpath(member.linkname).resolve().is_relative_to(cuda.resolve())
                    # Keep each vendor info/ license independently; payloads share CUDA prefix.
                    info=out/('vendor-info-'+name);info.mkdir()
                    for member in tf.getmembers():
                        if member.name.startswith('info/'):
                            tf.extract(member,info)
                        else:tf.extract(member,cuda)
            lib=cuda/'lib64';lib.mkdir(exist_ok=True)
            base=Path('/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages/nvidia')
            for component,library in [('cuda_runtime','libcudart.so.12'),('cublas','libcublas.so.12'),('cublas','libcublasLt.so.12'),('curand','libcurand.so.10')]:
                original=base/component/'lib'/library;assert original.is_file()
                for soname in [library,library.split('.so.')[0]+'.so']:
                    dest=lib/soname
                    if dest.is_symlink():assert dest.resolve()==original.resolve()
                    elif not dest.exists():dest.symlink_to(original)
            report['cuda_home']=str(cuda)
        env=Path(cfg['environment']);py=str(env/'bin/python')
        if env.exists():
            assert a.reuse_new_environment and str(env)=='/data/bwwu/envs/lc-rec-offload-20261108'
            assert not env.is_symlink() and (env/'pyvenv.cfg').is_file()
            report['existing_task_environment_reused_after_recorded_round0_failure']=True
        else:
            run([cfg['base_python'],'-m','venv','--system-site-packages',str(env)],'venv.log')
        install=[py,'-m','pip','install','--no-index','--no-deps','--disable-pip-version-check','--no-cache-dir','--find-links',cfg['prior_wheelhouse'],'--find-links',str(pkg)]
        run(install+cfg['packages'],'pip-install.log')
        source=out/'deepspeed-build';source.mkdir()
        with tarfile.open(pkg/'deepspeed-0.16.9.tar.gz') as tf:
            for m in tf.getmembers():
                q=Path(m.name);assert not q.is_absolute() and '..' not in q.parts and (m.isfile() or m.isdir())
            tf.extractall(source)
        ds=source/'deepspeed-0.16.9';setup=ds/'setup.py';old=setup.read_text()
        protected=['op_builder/cpu_adam.py','op_builder/builder.py','csrc/adam/cpu_adam.cpp','csrc/adam/cpu_adam_impl.cpp','csrc/includes/cpu_adam.h','deepspeed/ops/adam/cpu_adam.py']
        before={n:sha(ds/n) for n in protected};report['upstream_compute_sources']=before
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
        if not cfg.get('cuda_home'):setup.write_text(old.replace(old_probe,new_probe))
        (out/'deepspeed-packaging.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),setup.read_text().splitlines(True),fromfile='upstream/setup.py',tofile='derived/setup.py')))
        build_env=os.environ.copy();build_env.update(DS_ACCELERATOR='cuda',DS_BUILD_OPS='0',DS_BUILD_CPU_ADAM='1',MAX_JOBS='2',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        if cfg.get('cuda_home'):
            build_env.update(CUDA_HOME=cfg['cuda_home'],PATH=cfg['cuda_home']+'/bin:'+str(env/'bin')+':'+build_env.get('PATH',''),LD_LIBRARY_PATH=cfg['cuda_home']+'/lib64:'+build_env.get('LD_LIBRARY_PATH',''))
            run([cfg['cuda_home']+'/bin/nvcc','-V'],'nvcc.log',build_env,30)
            report['deepspeed_setup_unmodified']=sha(setup)==hashlib.sha256(old.encode()).hexdigest()
        # CPUAdamBuilder.builder() catches MissingCUDAException and builds CppExtension.
        # Preinstalled accelerator metadata remains cuda; no runtime JIT/skip-check patch.
        run([py,'-m','pip','install','--no-index','--no-deps','--no-build-isolation','--no-cache-dir',str(ds)],'deepspeed-build.log',build_env,1200)
        assert before=={n:sha(ds/n) for n in protected},'compute source modified during build'
        report['compute_sources_unchanged']=True
        run([py,'-m','pip','freeze','--all'],'pip-freeze.txt',timeout=60)
        smoke=Path(__file__).with_name('verify_cpu_adam.py')
        run([py,'-B',str(smoke),'--output',str(out/'cpu-adam.json')],'cpu-adam.log',build_env,180)
        report['status']='ENVIRONMENT_AND_CPU_ADAM_VERIFIED'
    except BaseException as e:
        report['status']='FAILED';report['error']=repr(e);raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start;save()

if __name__=='__main__':main()
