"""Read-only live status of formal training and queued full evaluation."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import subprocess

HERE=Path(__file__).resolve().parent
meta=json.loads((HERE/'latest_run.json').read_text());group=Path(meta['group'])
out={'time_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'group':str(group),
     'automatic_full_test':meta.get('automatic_full_test',False),'datasets':{}}
for ds in ['Instruments','Beauty']:
    path=group/ds;record={}
    for name,rel in [('pipeline','progress.json'),('training','training/progress.json'),('testing','testing/progress.json')]:
        p=path/rel
        if p.exists():record[name]=json.loads(p.read_text())
    if 'pipeline' in record and 'pid' in record['pipeline']:
        pid=record['pipeline']['pid'];proc=Path(f'/proc/{pid}')
        record['worker_alive']=proc.exists()
        if (proc/'cmdline').exists():record['worker_cmdline']=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    cfg=path/'training/config.json'
    if cfg.exists():
        c=json.loads(cfg.read_text());record['config']={k:c[k] for k in ['epochs','micro_batch','effective_batch',
            'gradient_accumulation_steps','train_samples','validation_examples','test_users','tasks','temperature']}
    out['datasets'][ds]=record
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(out,ensure_ascii=False,indent=2))
(HERE/'last_status.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
