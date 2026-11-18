#!/usr/bin/env python3
"""Bounded G3 preparation only. Called by the accepted independent runner."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from g3_identity import EXPERIMENT_ID, verify_assets, verify_base_model, sha, source_identity, need, EXECUTION_TOPOLOGY, EXECUTION_PROFILE, verify_execution_profile
from preflight_subprocess import run


def write(path, value):
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--release',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--large-root',required=True,type=Path)
    p.add_argument('--run-commit',required=True)
    p.add_argument('--mode',required=True,choices=['train','generate'])
    p.add_argument('--wall-seconds',required=True,type=int)
    p.add_argument('--cpu-validation',required=True,type=Path)
    a=p.parse_args();need(0<a.wall_seconds<=3600,'bounded preparation wall time')
    job=Path(os.environ['GRID_JOB_DIR']);out=job/'output';out.mkdir(exist_ok=True)
    scripts=a.release/'lc_rec/a0_beauty';source=a.release/'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'
    version=json.loads((a.release/'RUN_VERSION.json').read_text())
    need(version['run_commit']==a.run_commit,'run snapshot commit')
    for row in version['files']:
        need(sha(row['path'])==row['sha256'],'changed run snapshot')
    cpu=json.loads(a.cpu_validation.read_text())
    need(cpu['status']=='PASS' and cpu['exit_code']==0 and cpu['gpu_used'] is False and
         cpu['experiment_id']==EXPERIMENT_ID,'necessary CPU preflight must pass first')
    need(cpu['scope']=='PREFLIGHT_CPU_VALIDATION' and cpu.get('execution_topology')==EXECUTION_TOPOLOGY,
         'CPU validation single GPU scope/topology')
    verify_execution_profile(cpu.get('execution_profile'))
    need(cpu['config']['sha256']==sha(scripts/'config/formal_g3_beauty_seed42.json'), 'CPU validated effective config')
    for row in cpu['actual_import_files']:
        imported=Path(row['path'])
        need(imported.is_relative_to(a.release) or str(imported).startswith('/data/bwwu/envs/'),
             'unexpected CPU import location')
        need(sha(imported)==row['sha256'],'CPU validation dependency changed: '+str(imported))
    inputs=verify_assets(a.data_root);base=verify_base_model(a.model)
    expected='GPU-59cac3bd-3645-668b-79c8-839e0d878893'
    gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    device=gpu.strip().split(',')
    need(len(device)==2 and device[0]==expected and int(device[1])>=20*1024,'authorized GPU1 identity/free memory')
    need(os.environ['CUDA_VISIBLE_DEVICES']=='1','single physical GPU1 to logical device0 mapping')
    need(not subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'authorized GPU1 occupied; never evict other tasks')
    a.large_root.mkdir(parents=True,exist_ok=True)
    # All attempts in this experiment share one ledger, regardless of job name.
    all_jobs=Path(os.environ['GRID_JOB_DIR']).parent
    charged=0.;prior=[]
    for request in all_jobs.glob('*/request.json'):
        req=json.loads(request.read_text())
        if req.get('g3_preparation_experiment')!=EXPERIMENT_ID or request.parent==job:continue
        if not req.get('g3_gpu_devices'):continue
        stat=request.parent/'status.json'
        if not stat.exists():raise RuntimeError('prior G3 GPU job unresolved: '+str(request.parent))
        s=json.loads(stat.read_text())
        need(s.get('ended_at'),'prior G3 GPU job not terminal')
        elapsed=(dt.datetime.fromisoformat(s['ended_at'])-dt.datetime.fromisoformat(s.get('started_at') or s['received_at'])).total_seconds()
        devices=req['g3_gpu_devices'];charged+=devices*elapsed;prior.append({'job_id':req['job_id'],'gpu_devices':devices,'device_seconds':devices*elapsed})
    accepted_request=json.loads((job/'request.json').read_text())
    reserve_seconds=accepted_request['timeout_seconds']
    need(accepted_request['g3_gpu_devices']==1 and accepted_request['capture_version']==a.run_commit and
         reserve_seconds>=a.wall_seconds+30,'single GPU accepted request and cleanup reserve')
    need(charged+reserve_seconds<=14400,'cumulative 240 GPU-device-minute preparation limit including cleanup')
    started=time.monotonic();record={'experiment_id':EXPERIMENT_ID,'scope':'PREFLIGHT','execution_topology':EXECUTION_TOPOLOGY,'execution_profile':EXECUTION_PROFILE,'physical_gpu_index':1,'logical_gpu_index':0,'dual_gpu_preflight_status':'NOT_APPLICABLE_THIS_RUN','mode':a.mode,
        'job_id':os.environ['GRID_JOB_ID'],'run_commit':a.run_commit,'argv':sys.argv,'cwd':os.getcwd(),
        'interpreter':sys.executable,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'inputs':inputs,'base_model_identity':base,'cpu_validation':source_identity([a.cpu_validation])[0],
        'gpu_start':gpu,'prior_attempts':prior,'reserved_current_device_seconds':reserve_seconds,
        'predeclared_resume_tolerance':{'atol':1e-6,'rtol':1e-3},'stages':[],'status':'RUNNING'}
    write(out/'g3_preflight.json',record)
    def observe():
        memory=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        free=shutil.disk_usage(a.large_root).free
        # Allocated bytes over this task's exact output subtree, not logical sparse sizes.
        task_roots=[a.large_root.parent,Path('/data/bwwu/lc-rec/g3-preparation-20261118'),
                    Path('/home/bwwu/grid-audits/sid-extensions-p0/g3-production-20261118')]
        resolved=set(root.resolve() for root in task_roots)
        task_roots=[root for root in resolved if not any(other!=root and root.is_relative_to(other) for other in resolved)]
        allocated=sum(x.stat().st_blocks*512 for root in task_roots for x in root.rglob('*')
                      if x.is_file() and not x.is_symlink())
        need(memory>=40*1024**3,'40GiB host reserve')
        need(free>=20*1024**3 and allocated<=80*1024**3,'G3 disk budget/reserve')
        observation={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'allocated_bytes':allocated,'free_bytes':free,
                     'memory_available':memory,'charged_device_seconds':charged+(time.monotonic()-started)}
        with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(observation)+'\n')
    launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=1']
    common=['--source-root',str(source),'--data-root',str(a.data_root),'--model',str(a.model),'--run-commit',a.run_commit]
    def refresh_devices():
        device=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        need(len(device)==2 and device[0]==expected and int(device[1])>=20*1024,'authorized GPU1 identity/free memory changed')
        need(not subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'authorized GPU1 occupied before new substage')
    try:
        if a.mode=='train':
            for mode in ('continuous','early','resumed'):
                root=a.large_root/('recovery' if mode!='continuous' else 'continuous')
                argv=launcher+[str(scripts/'formal_train.py'),*common,'--output',str(root),
                    '--config',str(scripts/'config/formal_g3_beauty_seed42.json'),
                    '--precision-approval',str(a.release/'lc_rec/letter/evidence/formal_20261111/precision_approval.json'),
                    '--boundary-preflight']
                if mode=='early':argv+=['--stop-after-epoch','1']
                if mode=='resumed':argv+=['--resume',str(root/'checkpoint-2')]
                stage={'mode':mode,'argv':argv,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat()};record['stages'].append(stage)
                refresh_devices()
                run(argv,started+a.wall_seconds,stage,observe)
                write(out/'g3_preflight.json',record)
                if mode=='early':
                    early=a.large_root/'early_evidence';early.mkdir()
                    for name in ('rank0.json','events.rank0.jsonl','epoch1_rows.json'):
                        shutil.copyfile(root/name,early/name)
                if mode=='continuous':
                    # These run-local checkpoints have finished all consumers. Preserve
                    # original manifests and adapter before freeing only their tensors.
                    from formal_train import verify_checkpoint
                    from g3_identity import verify_adapter_identity
                    verify_adapter_identity(root/'preflight_adapter',scope='PREFLIGHT',run_commit=a.run_commit)
                    archived=root/'verified_checkpoint_manifests';archived.mkdir()
                    removed=[]
                    for cp in sorted(root.glob('checkpoint-*')):
                        verify_checkpoint(cp,a.run_commit,sha(scripts/'config/formal_g3_beauty_seed42.json'))
                        shutil.copyfile(cp/'complete.json',archived/(cp.name+'.complete.json'))
                        removed.append({'path':str(cp),'complete_sha256':sha(cp/'complete.json'),
                                        'allocated_bytes':sum(x.stat().st_blocks*512 for x in cp.rglob('*') if x.is_file()),
                                        'reason':'verified preflight temporary checkpoint; adapter and raw comparison state retained'})
                        shutil.rmtree(cp)
                    write(out/'verified_temporary_cleanup.json',removed)
        else:
            argv=launcher+[str(scripts/'formal_evaluate.py'),*common,
                '--adapter',str(a.large_root/'continuous/preflight_adapter'),'--output',str(a.large_root/'generation'),
                '--preflight-users-per-rank','32','--preflight-manifest',str(scripts/'config/preflight_g3_users_seed42.json'),
                '--preflight-four-conditions','--max-seconds',str(a.wall_seconds)]
            stage={'mode':'four_conditions','argv':argv};record['stages'].append(stage)
            refresh_devices()
            run(argv,started+a.wall_seconds,stage,observe)
            write(out/'g3_preflight.json',record)
            reload_argv=launcher+[str(scripts/'formal_evaluate.py'),*common,
                '--adapter',str(a.large_root/'continuous/preflight_adapter'),
                '--output',str(a.large_root/'generation_reload_normal'),
                '--preflight-users-per-rank','32','--preflight-manifest',str(scripts/'config/preflight_g3_users_seed42.json'),
                '--condition','normal','--preflight-reference-normal',str(a.large_root/'generation/normal'),
                '--max-seconds',str(a.wall_seconds)]
            reloaded={'mode':'cross_process_normal_reload','argv':reload_argv};record['stages'].append(reloaded)
            refresh_devices()
            run(reload_argv,started+a.wall_seconds,reloaded,observe)
        record['status']='COMPUTE_EXITED_ZERO_PENDING_ARTIFACT_REVIEW'
    except BaseException as e:
        record['status']='FAILED';record['error']=repr(e);raise
    finally:
        record['wall_seconds']=time.monotonic()-started
        record['gpu_device_seconds_upper_bound']=record['wall_seconds']
        record['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        write(out/'g3_preflight.json',record)


if __name__=='__main__':main()
