"""Freeze reviewed runner sources and launch independent GPU-pair pipelines."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
meta=json.loads((HERE/'latest_run.json').read_text());group=Path(meta['group'])
assert not meta.get('formal_started'), 'Already launched; inspect existing jobs'
runner=group/'runner';runner.mkdir(exist_ok=False)
for source in HERE.glob('*.py'):shutil.copy2(source,runner/source.name)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in runner.glob('*.py')}
(runner/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
jobs={}
for ds,cap in meta['capacity_jobs'].items():
    parent=group/ds
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=cap['global_gpus'],USE_TF='0',USE_FLAX='0',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (parent/'pipeline.log').open('ab',buffering=0) as log:
        worker=subprocess.Popen([sys.executable,str(runner/'pipeline.py'),'--dataset',ds,'--group',str(group),
            '--capacity',cap['capacity_out'],'--micro',str(cap['micro'])],env=env,
            stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
    jobs[ds]={'pipeline_pid':worker.pid,**cap}
meta.update(formal_started=True,jobs=jobs,runner=str(runner),source_manifest=manifest,
    automatic_full_test=True,test_modes=['unshuffle','shuffle_3','shuffle_4'])
(HERE/'latest_run.json').write_text(json.dumps(meta,indent=2));(group/'launch_manifest.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
