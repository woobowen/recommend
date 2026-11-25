
import datetime as dt,hashlib,json,os,pwd,shutil,socket,subprocess
from pathlib import Path
root=Path('/data/bwwu/grid-letter-rerun-20261125')
jobroot=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')
source=Path('/data/bwwu/grid-letter-rerun-20261125/source-b08c63940f5351dce63dc4cb8a0f9c6f1d8ac86e')
runner=Path('/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-398768a732bbfebd.py')
jobid='grid-letter-rerun-20261125-formal-r0'
interpreter=Path('/data/bwwu/envs/grid-20261009-e61baf3/bin/python')
inputs=['/data/bwwu/grid-letter-rerun-20261125/inputs/Instruments.index.json', '/data/bwwu/grid-letter-rerun-20261125/inputs/Instruments.inter.json', '/data/bwwu/grid-letter-rerun-20261125/inputs/Instruments.item.json', '/data/bwwu/grid-letter-rerun-20261125/inputs/Instruments.item2id', '/data/bwwu/grid-letter-rerun-20261125/inputs/assets_manifest.json', '/data/bwwu/grid-letter-rerun-20261125/inputs/item_embeddings.pt', '/data/bwwu/grid-letter-rerun-20261125/inputs/item_identity.jsonl', '/data/bwwu/grid-letter-rerun-20261125/inputs/items_qwen.jsonl', '/data/bwwu/grid-letter-rerun-20261125/inputs/prior_v2_replay_projection.jsonl', '/data/bwwu/grid-letter-rerun-20261125/inputs/user_axis.json']
out=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),hostname=socket.gethostname(),uid=os.getuid(),user=pwd.getpwuid(os.getuid()).pw_name,
 jobs=sorted(p.name for p in jobroot.glob('grid-letter-rerun-20261125-*')),
 saved_requests=sorted(p.name for p in (jobroot.parent/'requests').glob('grid-letter-rerun-20261125-*')),
 data_root=sorted(p.name for p in root.iterdir()),job_exists=os.path.lexists(jobroot/jobid),version=(source/'VERSION').read_text().strip(),
 interpreter=dict(path=str(interpreter),real_path=str(interpreter.resolve(strict=True)),version=subprocess.check_output([str(interpreter),'--version'],text=True).strip()),
 runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),deployment_sha256=hashlib.sha256(Path(str(runner)+'.deployment.json').read_bytes()).hexdigest(),
 assets=[dict(path=p,bytes=Path(p).stat().st_size,readable=os.access(p,os.R_OK)) for p in inputs],
 filesystems={str(p):dict(free=shutil.disk_usage(p).free,write=os.access(p,os.W_OK),inodes=os.statvfs(p).f_favail) for p in [jobroot,root]},
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),
 gpu_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True))
print(json.dumps(out))
