#!/usr/bin/env python3
"""Task-local deployment/submission/light-evidence client using the accepted runner.

Windows SSH/SCP and the existing namespace remain unchanged. No package setup,
source authoring, cleanup, or GPU execution is performed by deploy/collect.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[3]
SUPPORT = Path('/home/addaswsw/recommend-support/lc-six-task-20261110')
REMOTE = '/home/bwwu/lc-rec/six-task-20261110'
PYTHON = '/data/bwwu/envs/lc-six-task-20261110/bin/python'
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import transport
import gridctl


def remote_python(self, code, timeout=45):
    # Preserve the prior task's proven -c transport, avoiding stdin/EOF stalls.
    return self.ssh(shlex.join(['python3', '-I', '-S', '-B', '-c', code]), timeout=timeout)


transport.Transport.python = remote_python


def write(path, value):
    transport.atomic(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'deploy', 'submit', 'query', 'logs', 'collect'])
    parser.add_argument('--commit')
    parser.add_argument('--job')
    parser.add_argument('--kind', choices=['setup', 'cpu', 'protocol', 'gpu'])
    parser.add_argument('--train-job')
    args = parser.parse_args()
    SUPPORT.mkdir(parents=True, exist_ok=True)
    t = transport.Transport('new-server')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    assert branch == 'wbw-lc-rec'
    if args.mode == 'preflight':
        code = """import hashlib,json,os,subprocess,time
from pathlib import Path
r={'observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'memory':Path('/proc/meminfo').read_text(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True),'disk':subprocess.check_output(['df','-B1','/home/bwwu','/data/bwwu'],text=True)}
r['runner_sha256']=hashlib.sha256(Path('/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-398768a732bbfebd.py').read_bytes()).hexdigest()
r['required_retained_paths']={p:Path(p).exists() for p in ['/data/bwwu/envs/lc-rec-offload-20261108/bin/python','/data/bwwu/lc-rec/assets/data/Instruments/Instruments.inter.json','/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549/config.json']}
r['ledger']=json.loads(Path('/home/bwwu/lc-rec/six-task-20261110/gpu-budget.json').read_text()) if Path('/home/bwwu/lc-rec/six-task-20261110/gpu-budget.json').exists() else {'runs':[],'prior_stage_a_seconds':0}
print(json.dumps(r))"""
        r = json.loads(t.python(code))
        assert r['runner_sha256'] == t.profile['runner_sha256']
        assert all(r['required_retained_paths'].values())
        stamp = r['observed_utc'].replace(':', '')
        write(SUPPORT / 'receipts' / ('preflight-' + stamp + '.json'), r)
        print(json.dumps(r, indent=2))
        return
    if args.mode == 'deploy':
        commit = subprocess.check_output(['git', 'rev-parse', args.commit or 'HEAD'], cwd=REPO, text=True).strip()
        archive = SUPPORT / (commit + '.tar.gz')
        subprocess.run(['git', 'archive', '--format=tar.gz', '--output=' + str(archive), commit, 'lc_rec/six_task_lora', 'lc_rec/vendor/original'], cwd=REPO, check=True)
        upload = t.profile['remote_support'] + '/transfers/lc-six-task-' + commit + '.tar.gz'
        receipt = t.put(archive, upload)
        release = REMOTE + '/releases/' + commit
        code = f"""from pathlib import Path,PurePosixPath\nimport hashlib,json,tarfile\na=Path({upload!r});root=Path({release!r});rows=[]\nassert hashlib.sha256(a.read_bytes()).hexdigest()=={receipt['sha256']!r}\nwith tarfile.open(a,'r:gz') as tf:\n members=tf.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))\n for m in members:\n  p=PurePosixPath(m.name);assert not p.is_absolute() and '..' not in p.parts and (m.isfile() or m.isdir())\n if not root.exists():\n  root.mkdir(parents=True);tf.extractall(root,filter='data')\n for m in members:\n  if m.isfile():\n   b=tf.extractfile(m).read();p=root/m.name;assert p.is_file() and not p.is_symlink() and p.read_bytes()==b\n   rows.append({{'path':m.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}})\nr={{'RUN_COMMIT':{commit!r},'release':str(root),'not_a_git_checkout':True,'archive_sha256':{receipt['sha256']!r},'files':rows}}\nmanifest=root/'RUN_VERSION.json'\nif manifest.exists():assert json.loads(manifest.read_text())==r\nelse:manifest.write_text(json.dumps(r,indent=2)+'\\n')\nprint(json.dumps(r))"""
        r = json.loads(t.python(code, timeout=60))
        write(SUPPORT / 'receipts' / ('deployment-' + commit + '.json'), r)
        print(json.dumps({'RUN_COMMIT': commit, 'release': release, 'files_verified': len(r['files'])}, indent=2))
        return
    assert args.job and args.job.startswith('lc-six-20261110-')
    gridctl.valid(args.job)
    if args.mode == 'submit':
        assert args.commit and args.kind
        commit = subprocess.check_output(['git', 'rev-parse', args.commit], cwd=REPO, text=True).strip()
        release = REMOTE + '/releases/' + commit
        scripts = release + '/lc_rec/six_task_lora/scripts/'
        source = release + '/lc_rec/vendor/original'
        jobdir = t.profile['job_root'] + '/' + args.job
        output = jobdir + '/output/'
        model = '/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549'
        old = json.loads((REPO / 'lc_rec/letter/evidence/jobs/letter-20261110-gpu-r1/request.json').read_text())
        req = {k: old[k] for k in ['schema', 'host', 'user', 'uid', 'namespace', 'deployment_sha256']}
        env = dict(old['env'])
        env['PYTHONUNBUFFERED']='1'
        env['TORCH_EXTENSIONS_DIR']=REMOTE+'/torch_extensions'
        env['NCCL_P2P_DISABLE']='1'
        env['OMP_NUM_THREADS']='4'
        if args.kind in ('cpu','setup','protocol'):
            env['CUDA_VISIBLE_DEVICES']=''
            env.pop('DS_ACCELERATOR',None)
            if args.kind=='setup':
                argv=['/data/bwwu/envs/grid-20261009-e61baf3/bin/python',scripts+'setup_env.py','--output',output+'environment']
            elif args.kind=='protocol':
                argv=[PYTHON,scripts+'protocol_probe.py','--config',release+'/lc_rec/six_task_lora/config/candidate.json','--output',output+'protocol']
            else:
                argv=[PYTHON,scripts+'analyze_workload.py','--original',source,'--data-root','/data/bwwu/lc-rec/assets/data','--model',model,'--output',output+'workload']
        else:
            assert args.train_job and args.train_job.startswith('lc-six-20261110-cpu-')
            workload=t.profile['job_root']+'/'+args.train_job+'/output/workload'
            previous=json.loads(t.python("from pathlib import Path;import json;p=Path('/home/bwwu/lc-rec/six-task-20261110/update-budget.json');print(p.read_text() if p.exists() else '{\"used\":0}')"))
            assert 0<=previous['used']<6
            inner=[scripts+'gpu_probe.py','--config',release+'/lc_rec/six_task_lora/config/candidate.json','--samples',workload+'/samples.json','--output-dir',output+'probe','--run-commit',commit,'--max-updates',str(min(2,6-previous['used'])),'--candidate','bf16','--micro-batch','8']
            argv=[PYTHON,scripts+'resource_guard.py','--config',release+'/lc_rec/six_task_lora/config/guard.json','--output',output+'resources','--job-id',args.job,'--',PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',*inner]
        deploy = json.loads((SUPPORT/'receipts'/('deployment-'+commit+'.json')).read_text())
        prefixes = ('lc_rec/six_task_lora/scripts/', 'lc_rec/six_task_lora/config/', 'lc_rec/vendor/original/')
        inputs = [{'path':release+'/'+row['path'], 'sha256':row['sha256']} for row in deploy['files'] if row['path'].startswith(prefixes)]
        req.update(job_id=args.job, capture_version=commit, cwd=release, argv=argv, env=env, inputs=inputs, timeout_seconds=2850 if args.kind=='gpu' else 3600)
        path = SUPPORT/'requests'/(args.job+'.json')
        if path.exists():
            assert json.loads(path.read_text()) == req
        else:
            write(path,req)
        r = gridctl.submit(t,path)
    elif args.mode in ('query','logs'):
        r = gridctl.query(t,args.job)
        if args.mode=='logs':
            print(t.ssh(shlex.join(['tail','-n','50',t.profile['job_root']+'/'+args.job+'/work.log'])))
    else:
        r = gridctl.query(t,args.job)
        assert r['cleanup_verified'] and not r['worker_alive'] and not r['workgroup_live'] and r['status'] not in ['RUNNING','ACCEPTED','CONTROL_LOST']
        jobpath=t.profile['job_root']+'/'+args.job
        destination=REMOTE+'/light_exports/'+args.job+'.tar.gz'
        code=f"""from pathlib import Path\nimport hashlib,json,tarfile\nroot=Path({jobpath!r});archive=Path({destination!r});files=[];excluded=[]\nfor p in sorted(root.rglob('*')):\n if not p.is_file() or p.is_symlink() or p.name.endswith(('.lock','.partial','.tmp')):continue\n rel=str(p.relative_to(root))\n if p.stat().st_size>8*1024**2 or p.suffix in ['.bin','.safetensors','.pt','.pth']:\n  excluded.append({{'path':rel,'bytes':p.stat().st_size,'reason':'large/payload remains on server'}});continue\n files.append({{'path':rel,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}})\nmanifest=root/'light-export-manifest.json'\nfiles=[x for x in files if x['path']!='light-export-manifest.json']\nmanifest.write_text(json.dumps({{'files':files,'excluded':excluded}},indent=2)+'\\n')\narchive.parent.mkdir(exist_ok=True)\nwith tarfile.open(archive,'w:gz') as tf:\n for row in files:tf.add(root/row['path'],arcname=row['path'],recursive=False)\n tf.add(manifest,arcname=manifest.name,recursive=False)\nprint(json.dumps({{'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}}))"""
        meta=json.loads(t.python(code,timeout=60))
        local=SUPPORT/'exports'/(args.job+'.tar.gz')
        t.get_verified(meta['path'],local,meta,timeout=60)
        dest=REPO/'lc_rec/six_task_lora/evidence/jobs'/args.job
        dest.mkdir(parents=True,exist_ok=True)
        with tarfile.open(local,'r:gz') as tf:
            names=[m.name for m in tf.getmembers()]
            assert len(names)==len(set(names))
            for m in tf.getmembers():
                assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            tf.extractall(dest,filter='data')
        manifest=json.loads((dest/'light-export-manifest.json').read_text())
        assert set(names)=={x['path'] for x in manifest['files']}|{'light-export-manifest.json'}
        for row in manifest['files']:
            p=dest/row['path'];assert p.stat().st_size==row['bytes'] and transport.sha(p)==row['sha256']
        write(dest/'controller-query.json',r)
        r={'job_id':args.job,'status':r['status'],'exit_code':r['exit_code'],'cleanup_verified':r['cleanup_verified'],'files_verified':len(manifest['files']),'excluded_payloads':manifest['excluded']}
    write(SUPPORT/'receipts'/(args.mode+'-'+args.job+'.json'),r)
    summary_keys=('job_id','status','exit_code','cleanup_verified','worker_alive','workgroup_live','files_verified','excluded_payloads','submission')
    print(json.dumps({k:r[k] for k in summary_keys if k in r},indent=2))


if __name__=='__main__':
    main()
