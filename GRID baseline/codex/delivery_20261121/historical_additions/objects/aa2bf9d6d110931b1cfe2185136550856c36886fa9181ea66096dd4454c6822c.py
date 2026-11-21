from pathlib import Path
import json, os, subprocess, time
ROOT=Path('/home/jcyin/GRID_wmh_Qwen_TIGER_20261009')
print('SERVER_TIME',time.strftime('%Y-%m-%d %H:%M:%S %z'))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip())
print(subprocess.check_output(['df','-h',str(ROOT)],text=True).strip())
for dataset in ['beauty','sports','toys']:
    reports=sorted((ROOT/'launches').glob(dataset+'_rkmeans_qwen_*.json'))
    if not reports: continue
    launch=json.loads(reports[-1].read_text())
    run=Path(launch['run_dir'])
    print('RUN',dataset,'pid',launch['pid'],'alive',Path('/proc/'+str(launch['pid'])).exists(),run.name)
    for path in [run/'pipeline_status.json',run/'quantizer_train/progress.json',run/'tiger_train_test/progress.json']:
        if path.is_file():
            try: print(path.parent.name+'/'+path.name,json.loads(path.read_text()))
            except Exception as e: print('READ_ERROR',type(e).__name__)
    for path in ([run/'tiger_train_test/console.log'] if dataset=='beauty' else [run/'text_vectors/console.log',run/'tiger_train_test/console.log',Path(launch['launch_log'])]):
        if path.is_file():
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-1600))
                lines=f.read().decode('utf-8',errors='replace').splitlines()
            print(path.parent.name+'/'+path.name,'tail',[line[:220] for line in lines[-2:]])
print('RUNNER_SHA',subprocess.check_output(['sha256sum',str(ROOT/'_运行/run_pipeline.py')],text=True).strip())
