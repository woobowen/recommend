from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv,json,subprocess
root=Path(__file__).resolve().parents[1]
report={'checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'datasets':{}}
for dataset in ['beauty','sports','toys']:
    launch=json.loads(sorted((root/'launches').glob(dataset+'_rkmeans_qwen_*.json'))[-1].read_text())
    run=Path(launch['run_dir']);stage=Path(launch.get('stage_dir',str(run/'tiger_train_test')))
    item={'run_dir':str(run),'pipeline_pid':launch['pid'],'pipeline_alive':Path('/proc/'+str(launch['pid'])).exists()}
    for filename,key in [('pipeline_status.json','pipeline_status')]:
        p=run/filename
        if p.exists(): item[key]=json.loads(p.read_text())
    for filename,key in [('progress.json','progress'),('stage_result.json','result')]:
        p=stage/filename
        if p.exists(): item[key]=json.loads(p.read_text())
    validation=[];testing=[];training=[]
    for f in sorted(stage.rglob('metrics.csv')):
        with f.open(newline='') as stream:
            for row in csv.DictReader(stream):
                base={k:float(v) for k,v in row.items() if v and k in ['step','epoch']}
                for prefix,target in [('val/',validation),('test/',testing),('train/',training)]:
                    values={k:float(v) for k,v in row.items() if v and k.startswith(prefix)}
                    if values: target.append(dict(base,**values))
        item['metrics_csv']=str(f)
    item['validation_count']=len([r for r in validation if 'val/recall@5' in r])
    item['latest_validation']=next((r for r in reversed(validation) if 'val/recall@5' in r),None)
    item['best_validation_recall5']=max((r for r in validation if 'val/recall@5' in r),key=lambda r:r['val/recall@5'],default=None)
    item['recent_validations']=[r for r in validation if 'val/recall@5' in r][-3:]
    item['latest_test']=testing[-1] if testing else None
    item['latest_train']=training[-1] if training else None
    item['checkpoints']=[p.name for p in (stage/'checkpoints').glob('*.ckpt')]
    log=stage/'console.log'
    if launch.get('job_file'):
        job=json.loads(Path(launch['job_file']).read_text())
        log=stage/job['console_filename']
        item['validation_interval_batches']=job['validation_interval_batches']
        item['mode']=job['mode']
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-6000))
            lines=f.read().decode('utf-8',errors='replace').splitlines()
        item['log_tail']=[line[:350] for line in lines[-3:]]
    report['datasets'][dataset]=item
report['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
(root/'codex/latest_three_dataset_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
