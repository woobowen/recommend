"""Read-only, compact evidence for first-round training and detachment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ap=argparse.ArgumentParser()
ap.add_argument('group',type=Path)
args=ap.parse_args()
report={'time':time.time(),'jobs':[]}
for path in sorted(args.group.glob('T*_status.json')):
    item=json.loads(path.read_text())
    output=Path(item.get('output','/nonexistent'))
    for name in ('progress','model_verified','stage_result'):
        src=output/f'{name}.json'
        if src.exists():
            item[name]=json.loads(src.read_text())
    for name in ('pid','supervisor_pid'):
        pid=item.get(name)
        if pid:
            try:
                fields=Path(f'/proc/{pid}/stat').read_text().split()
                item[name+'_process']={'state':fields[2],'ppid':int(fields[3]),'pgrp':int(fields[4]),
                    'sid':int(fields[5]),'tty_nr':int(fields[6]),
                    'stdin':os.readlink(f'/proc/{pid}/fd/0'),'stdout':os.readlink(f'/proc/{pid}/fd/1')}
            except (FileNotFoundError,ProcessLookupError):
                item[name+'_process']=None
    log=args.group/f"{item['task_id']}.log"
    if log.exists():
        item['log_bytes']=log.stat().st_size
        item['log_tail']=log.read_text(errors='replace').splitlines()[-12:]
    report['jobs'].append(item)
report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu',
    '--format=csv,noheader'],text=True).strip()
print(json.dumps(report,ensure_ascii=False,indent=2))
