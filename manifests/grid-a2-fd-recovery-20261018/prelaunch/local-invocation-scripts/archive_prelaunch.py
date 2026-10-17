"""Copy fixed lightweight prelaunch evidence into wbw; no old evidence edits."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

B=Path(__file__).resolve().parent
R=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
T=R/'manifests/grid-a2-fd-recovery-20261018/prelaunch'
T.mkdir(parents=True,exist_ok=True)
review=json.loads((B/'reviews/fd/source-repair-admission.json').read_text())
assert review['fd_engineering_status']=='FD_ENGINEERING_EQUIVALENCE_PASS'
for rec in review['source_files']:
 assert hashlib.sha256(Path(rec['path']).read_bytes()).hexdigest()==rec['sha256'],rec['path']
sources={B/'initial-scene.json':T/'initial-scene.json',
 B/'fresh-ssh-limit-after-cpu.json':T/'fresh-ssh-limit-after-cpu.json',
 B/'cpu-candidate.json':T/'cpu-candidate.json'}
for p in (B/'reviews/fd').glob('*'):
 if p.is_file():sources[p]=T/'non-author-fd-review'/p.name
cpu=Path('/home/addaswsw/recommend-support/sid-extensions-p0/jobs/new-server/ga-20261018-a2-fd-cpu-equivalence-v1/retrieved')
for p in cpu.rglob('*'):
 if p.is_file():sources[p]=T/'cpu-equivalence-original'/p.relative_to(cpu)
for p in [B/'inspect_scene.py',B/'submit_cpu_check.py',B/'prepare_formal.py',Path(__file__)]:
 sources[p]=T/'local-invocation-scripts'/p.name
records=[]
for p,q in sources.items():
 q.parent.mkdir(parents=True,exist_ok=True)
 data=p.read_bytes()
 if q.exists():assert q.read_bytes()==data
 else:q.write_bytes(data)
 records.append(dict(path=q.relative_to(T).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),source=str(p)))
diff=subprocess.check_output(['git','diff','--','experiments/launch_20261017/a2_upstream.py','experiments/launch_20261017/run.py'],cwd=R)
(T/'legacy-engineering.diff').write_bytes(diff)
(T/'copy-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(members=len(records),bytes=sum(x['bytes'] for x in records),target=str(T))))
