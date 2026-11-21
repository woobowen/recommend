from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv,json,re,os,subprocess,time
os.environ.update(CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='2')
root=Path('/home/jcyin/GRID_wmh_Qwen_TIGER_20261009')
report={'checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'datasets':{}}
for dataset in ['sports','toys']:
    launch=json.loads(sorted((root/'launches').glob(dataset+'_rkmeans_qwen_*.json'))[-1].read_text())
    job=json.loads(Path(launch['job_file']).read_text())
    stage=Path(launch['stage_dir'])
    rows=[]
    for path in sorted(stage.rglob('metrics.csv')):
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                if row.get('val/recall@5'):rows.append({k:float(v) for k,v in row.items() if v and (k.startswith('val/') or k in ('step','epoch'))})
    best=max(rows,key=lambda r:r['val/recall@5']) if rows else None
    best_index=next((i for i,r in enumerate(rows) if r==best),-1)
    log=stage/job['console_filename']
    events=[]
    for line in log.read_text(errors='replace').splitlines():
        clean=re.sub(r'\x1b\[[0-9;]*m','',line)
        m=re.search(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) Epoch 0, global step (\d+)',clean)
        if m: events.append({'time':m.group(1),'step':int(m.group(2))})
    item={'launch_time':launch['started_at'],'elapsed_hours':(datetime.now(ZoneInfo('Asia/Shanghai'))-datetime.fromisoformat(launch['started_at'])).total_seconds()/3600,'progress':json.loads((stage/'progress.json').read_text()),'status':json.loads((Path(launch['run_dir'])/'pipeline_status.json').read_text()),'validation_count':len(rows),'best_validation':best,'latest_validation':rows[-1] if rows else None,'checks_since_best':len(rows)-1-best_index,'recent_validation_events':events[-10:],'log_tail':[re.sub(r'\x1b\[[0-9;]*m','',line)[:400] for line in log.read_text(errors='replace').splitlines()[-8:]],'checkpoint_files':[{ 'name':p.name,'mtime':datetime.fromtimestamp(p.stat().st_mtime,ZoneInfo('Asia/Shanghai')).isoformat()} for p in (stage/'checkpoints').glob('*.ckpt')]}
    report['datasets'][dataset]=item
print(json.dumps(report,indent=2))
(root/'remaining_time_snapshot.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
