"""Re-evaluate saved adapters with exact historical GRID permutations."""
from pathlib import Path
import json
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(HERE/'test_grid_alignment.py')],check=True)
csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={int(r.split(',')[0]):int(r.split(',')[1]) for r in csv.splitlines()}
assert all(used[i]<100 for i in [0,1]),used
with (HERE/'fp16_grid.log').open('wb',buffering=0) as f:
    worker=subprocess.Popen([sys.executable,str(HERE/'fp16_inference_smoke.py')],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),
        stdout=f,stderr=f,start_new_session=True)
record={'pid':worker.pid,'training_started':False,'shuffle_protocol':'GRID-PLAN-V1_1',
        'shuffle_seed':2026,'identity_allowed':True}
(HERE/'fp16_grid_launch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
