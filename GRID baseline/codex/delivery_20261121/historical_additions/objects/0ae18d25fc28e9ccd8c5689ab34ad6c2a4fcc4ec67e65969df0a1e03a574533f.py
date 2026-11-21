from pathlib import Path
from datetime import datetime
import csv,hashlib,json,math,os,subprocess
from omegaconf import OmegaConf
root=Path('/home/jcyin/GRID_wmh_Qwen_TIGER_20261009')
plan=json.loads((root/'reconfigurations/latest_val1600_plan.json').read_text())
folder=Path(plan['plan_dir'])
report={'checked_at':datetime.now().astimezone().isoformat(),'datasets':{},'ready':True}
for dataset,job in plan['datasets'].items():
    launch=job['new_launch'];stage=Path(job['stage_dir'])
    row={'pid':launch['pid'],'alive':Path('/proc/'+str(launch['pid'])).exists(),'physical_gpu':job['physical_gpu'],'stage_dir':str(stage),'mode':job['mode']}
    status=json.loads((Path(job['run_dir'])/'pipeline_status.json').read_text())
    row['status']=status
    config_path=stage/'.hydra/config.yaml'
    if config_path.exists():
        cfg=OmegaConf.load(config_path)
        assert cfg.trainer.val_check_interval==1600 and cfg.trainer.accumulate_grad_batches==16
        assert cfg.train and cfg.test
        assert cfg.ckpt_path==(job['resume_checkpoint']['path'] if dataset=='beauty' else None)
        row['config_verified']=True
        row['val_check_interval']=cfg.trainer.val_check_interval
        row['test']=cfg.test
        row['ckpt_path']=cfg.ckpt_path
    else:
        row['config_verified']=False
    progress_path=stage/'progress.json'
    if progress_path.exists(): row['progress']=json.loads(progress_path.read_text())
    log=stage/job['console_filename']
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-10000))
            lines=f.read().decode('utf-8',errors='replace').splitlines()
        row['log_tail']=[line[:350] for line in lines[-6:]]
        row['restoration_messages']=[line[:500] for line in lines if 'Restor' in line or 'mid-epoch' in line or 'checkpoint' in line.lower()][-5:]
    training_pid=status.get('training_pid')
    if training_pid and Path('/proc/'+str(training_pid)).exists():
        env=dict(entry.split(b'=',1) for entry in Path('/proc/'+str(training_pid)+'/environ').read_bytes().split(b'\0') if b'=' in entry)
        assert env[b'CUDA_VISIBLE_DEVICES'].decode()==job['physical_gpu']
        row['training_pid']=training_pid
    progress=row.get('progress',{})
    starting_step=job['resume_checkpoint']['global_step'] if dataset=='beauty' else 0
    new_progress=progress.get('time','')>=launch['started_at'][:19].replace('T',' ')
    row['advancing']=new_progress and progress.get('global_step',0)>starting_step and progress.get('loss_finite',False)
    row['starting_step']=starting_step
    metrics=[]
    for path in sorted(stage.rglob('metrics.csv')):
        with path.open(newline='') as f:
            for r in csv.DictReader(f):
                if r.get('val/recall@5'):metrics.append({k:float(v) for k,v in r.items() if v and (k.startswith('val/') or k in ['epoch','step'])})
    row['latest_validation']=metrics[-1] if metrics else None
    row['ready']=row['alive'] and row['config_verified'] and row['advancing']
    report['ready']=report['ready'] and row['ready']
    report['datasets'][dataset]=row
backup=Path(plan['datasets']['beauty']['protected_best_backup'])
assert hashlib.sha256(backup.read_bytes()).hexdigest()==plan['datasets']['beauty']['resume_checkpoint']['sha256']
report['beauty_best_backup']={'path':str(backup),'sha256':plan['datasets']['beauty']['resume_checkpoint']['sha256'],'verified':True}
report['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
audit_path=folder/'runtime_audit.json'
report['two_observations_verified']=False
if audit_path.exists():
    previous=json.loads(audit_path.read_text())
    for dataset,row in report['datasets'].items():
        before=previous['datasets'][dataset].get('progress',{}).get('global_step',row['starting_step'])
        row['previous_global_step']=before
        row['step_increased']=row.get('progress',{}).get('global_step',0)>before
    report['two_observations_verified']=report['ready'] and all(row['step_increased'] for row in report['datasets'].values())
(folder/'runtime_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
