"""Compact progress snapshot for the download and finite smoke queue."""
from pathlib import Path
import datetime
import json
import subprocess
import time
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
manifest=json.loads((HERE/'expected_source_assets.json').read_text())
total=done=0
for f in manifest['model_files']:
    total+=f['bytes'];path=ROOT/'models/huggyllama-llama-7b'/f['name']
    state=path.with_name(path.name+'.chunks.json')
    if path.exists() and path.stat().st_size==f['bytes']:
        done+=f['bytes']
    elif state.exists():
        s=json.loads(state.read_text())
        done+=sum(min(s['block_size'],f['bytes']-i*s['block_size']) for i in s['done'])
    elif path.with_name(path.name+'.partial').exists():
        done+=min(f['bytes'],path.with_name(path.name+'.partial').stat().st_size)
out={'time_beijing':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
     'download_GB':round(done/1e9,3),'total_GB':round(total/1e9,3),
     'download_percent':round(done/total*100,2),'model_sha_verified':(HERE/'model_download_verified.json').exists()}
if (HERE/'latest_queue.json').exists():
    queue=json.loads((HERE/'latest_queue.json').read_text())
    launch=Path(queue['launch'])
    out['queue']=json.loads((launch/'status.json').read_text())
    out['workers']={}
    for ds in ['Instruments','Beauty']:
        p=launch/ds/'status.json'
        if p.exists():out['workers'][ds]=json.loads(p.read_text())
    out['log_tail']={ds:(launch/f'{ds}.log').read_text(errors='replace')[-1200:]
                     for ds in ['Instruments','Beauty'] if (launch/f'{ds}.log').exists()}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(out,ensure_ascii=False,indent=2))
(HERE/'last_snapshot.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
