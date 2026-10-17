"""Deploy hash-bound candidate bytes and submit one bounded CPU-only check."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport,atomic,sha
import gridctl

BASE=Path(__file__).resolve().parent
REPO=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
REMOTE='/home/bwwu/grid-audits/sid-extensions-p0/grid-a2-fd-recovery-20261018'
OLD='/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/snapshots/b6203414ee58a400'
files=sorted((REPO/'experiments/a2_fd_recovery_20261018').glob('*.py'))
records=[dict(path=p.relative_to(REPO).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
archive=BASE/('cpu-candidate-'+digest[:16]+'.tar.gz')
with tarfile.open(archive,'w:gz') as f:
 for p,r in zip(files,records):f.add(p,arcname=r['path'],recursive=False)
root=REMOTE+'/cpu-candidates/'+digest[:16]
t=Transport('new-server');t.identity()
t.put(archive,REMOTE+'/archives/'+archive.name)
code='''import pathlib,tarfile,hashlib,json,os
root=pathlib.Path(ROOT); records=RECORDS
if not root.exists():
 root.mkdir(parents=True)
 with tarfile.open(ARCHIVE) as tf:
  assert all(x.isfile() and not pathlib.Path(x.name).is_absolute() and '..' not in pathlib.Path(x.name).parts for x in tf.getmembers())
  tf.extractall(root,filter='data')
for r in records:
 p=root/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];os.chmod(p,0o444)
print(json.dumps({'root':str(root),'files':records}))
'''.replace('ROOT',repr(root)).replace('RECORDS',repr(records)).replace('ARCHIVE',repr(REMOTE+'/archives/'+archive.name))
deployment=json.loads(t.python(code));atomic(BASE/'cpu-candidate.json',deployment)
job='ga-20261018-a2-fd-cpu-equivalence-v1'
q=t.profile
request=dict(schema=1,job_id=job,host=q['hostname'],user=q['user'],uid=q['uid'],namespace=q['namespace'],
 deployment_sha256=q['deployment_sha256'],capture_version='ENGINEERING_CPU_CANDIDATE/'+digest,
 cwd=root,timeout_seconds=600,
 argv=['/data/bwwu/envs/grid-20261009-e61baf3/bin/python','-B',root+'/experiments/a2_fd_recovery_20261018/check_equivalence.py',
       '--output',q['job_root']+'/'+job+'/output'],
 env=dict(CUDA_VISIBLE_DEVICES='',PYTHONPATH=root+':'+OLD,OMP_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3',
          TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false'),
 inputs=[dict(path=root+'/'+r['path'],sha256=r['sha256']) for r in records])
path=BASE/'cpu-request.json';atomic(path,request)
# Existing ID check precedes the only submission. Never replace a received request.
exists=json.loads(t.python('import pathlib,json;print(json.dumps((pathlib.Path('+repr(q['job_root']+'/'+job)+')).exists()))'))
if exists:raise RuntimeError('CPU check ID already exists; query before any continuation')
receipt=gridctl.submit(t,path);atomic(BASE/'cpu-submission.json',receipt)
print(json.dumps(dict(deployment=deployment,submission=receipt),indent=2))
