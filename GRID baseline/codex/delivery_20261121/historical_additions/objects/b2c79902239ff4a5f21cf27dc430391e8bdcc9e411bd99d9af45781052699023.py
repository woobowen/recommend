from pathlib import Path
import csv,json,datetime
root=Path('/home/jcyin/GRID_wmh_Qwen_TIGER_20261009')
stage=root/'experiments/beauty_rkmeans_qwen_20261010_012253/tiger_train_test'
rows=[]
for f in stage.rglob('metrics.csv'):
    with f.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('val/recall@5'):
                rows.append({k:float(v) for k,v in row.items() if v and (k.startswith('val/') or k in ('step','epoch'))})
report={'checked_at':datetime.datetime.now().astimezone().isoformat(),'csv':str(f),'validation_count':len(rows),'latest':rows[-1],'best_recall5':max(rows,key=lambda r:r['val/recall@5']),'training_progress':json.loads((stage/'progress.json').read_text()),'checkpoints':[str(p) for p in (stage/'checkpoints').glob('*.ckpt')]}
(root/'beauty_validation_snapshot.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
