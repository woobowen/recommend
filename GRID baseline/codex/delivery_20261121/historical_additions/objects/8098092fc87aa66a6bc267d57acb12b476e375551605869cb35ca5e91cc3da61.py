from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,signal,subprocess,sys,time
root=Path('/home/jcyin/GRID_wmh_Qwen_TIGER_20261009')
plan=json.loads((root/'reconfigurations/latest_val1600_plan.json').read_text())
assert plan['state']=='prepared'
folder=Path(plan['plan_dir'])
plan_file=folder/'plan.json'
def save():
    plan_file.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (root/'reconfigurations/latest_val1600_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
def members(pgid):
    result=[]
    for line in subprocess.check_output(['ps','-eo','pid,pgid,stat,args'],text=True).splitlines()[1:]:
        fields=line.split(None,3)
        if len(fields)==4 and int(fields[1])==pgid and not fields[2].startswith('Z'):
            result.append({'pid':int(fields[0]),'pgid':int(fields[1]),'stat':fields[2],'args':fields[3]})
    return result
for dataset,job in plan['datasets'].items():
    pid=job['source_launch']['pid']
    assert os.getpgid(pid)==pid
    args=Path('/proc/'+str(pid)+'/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert str(root/'_运行/run_pipeline.py') in args and '--dataset '+dataset in args
    job['old_process_group']=members(pid)
    assert any('grid_stage.py' in row['args'] and 'experiment=tiger_train_flat' in row['args'] for row in job['old_process_group'])
# Freeze Beauty so the selected best file cannot change during backup.
beauty=plan['datasets']['beauty']
pid=beauty['source_launch']['pid']
os.killpg(pid,signal.SIGSTOP)
plan['state']='beauty_paused'
save()
try:
    source=Path(beauty['resume_checkpoint']['path'])
    expected=beauty['resume_checkpoint']['sha256']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,'Best checkpoint changed; reprepare while paused'
    backup_dir=folder/'beauty_best_backup'
    backup_dir.mkdir(exist_ok=False)
    backup=backup_dir/source.name
    shutil.copy2(source,backup)
    assert hashlib.sha256(backup.read_bytes()).hexdigest()==expected
    beauty['protected_best_backup']=str(backup)
    old_stage=Path(beauty['stage_dir'])
    if (old_stage/'.hydra').exists(): shutil.copytree(old_stage/'.hydra',folder/'beauty_previous_hydra')
    for name in ['progress.json','command.json']:
        if (old_stage/name).exists(): shutil.copy2(old_stage/name,folder/('beauty_previous_'+name))
    plan['state']='best_checkpoint_backed_up'
    save()
    print('BEAUTY_BACKUP_VERIFIED',str(backup),expected,flush=True)
except BaseException:
    os.killpg(pid,signal.SIGCONT)
    plan['state']='backup_failed_original_beauty_resumed'
    save()
    raise
for dataset,job in plan['datasets'].items():
    pgid=job['source_launch']['pid']
    os.killpg(pgid,signal.SIGTERM)
    if dataset=='beauty': os.killpg(pgid,signal.SIGCONT)
    print('STOP_SENT',dataset,pgid,flush=True)
deadline=time.monotonic()+25
while time.monotonic()<deadline:
    if not any(members(job['source_launch']['pid']) for job in plan['datasets'].values()): break
    time.sleep(.5)
for dataset,job in plan['datasets'].items():
    pgid=job['source_launch']['pid']
    remaining=members(pgid)
    if remaining:
        os.killpg(pgid,signal.SIGKILL)
        time.sleep(.5)
    assert not members(pgid),(dataset,members(pgid))
    job['old_processes_stopped_at']=datetime.now().astimezone().isoformat()
    status=Path(job['run_dir'])/'pipeline_status.json'
    status.write_text(json.dumps({'stage':Path(job['source_stage']).name,'state':'stopped_for_val1600_reconfiguration','previous_pid':pgid,'plan_file':str(plan_file)},indent=2),encoding='utf-8')
plan['state']='original_jobs_stopped'
save()
for dataset,job in plan['datasets'].items():
    name=dataset+'_rkmeans_qwen_'+folder.name.removeprefix('val1600_')+'_val1600'
    launch_log=root/'launches'/(name+'.log')
    command=[sys.executable,str(root/'_运行/run_tiger_job.py'),'--job',job['job_file']]
    with launch_log.open('w') as output:
        proc=subprocess.Popen(command,cwd=root,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    launch={'pid':proc.pid,'command':command,'run_dir':job['run_dir'],'stage_dir':job['stage_dir'],'launch_log':str(launch_log),'dataset':dataset,'method':'rkmeans','gpus':job['physical_gpu'],'train':True,'test':True,'started_at':datetime.now().astimezone().isoformat(),'validation_interval_batches':1600,'mode':job['mode'],'job_file':job['job_file']}
    (root/'launches'/(name+'.json')).write_text(json.dumps(launch,indent=2),encoding='utf-8')
    job['new_launch']=launch
    save()
    print('LAUNCHED',json.dumps(launch),flush=True)
plan['state']='new_jobs_launched'
save()
print('PLAN_FILE',str(plan_file),flush=True)
