"""Commit-bound immutable deployments through the already accepted runner."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from experiments.beauty_extensions.common import REPO, sha256, write_json

SUPPORT = Path('/home/addaswsw/recommend-support/grid-formal-20261015')
TRANSPORT = Path('/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
REMOTE = '/home/bwwu/grid-audits/sid-extensions-p0/grid-formal-20261015'
SOURCE_DIRS = ['GRID baseline/src', 'references/wmh-original/GRID/src',
    'experiments/beauty_extensions', 'experiments/preflight_20261015', 'experiments/formal_20261015']
GPUS = ['GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae', 'GPU-59cac3bd-3645-668b-79c8-839e0d878893']


def source_records():
    files = [p for d in SOURCE_DIRS for p in (REPO/d).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.yaml')]
    files += [REPO/p for p in ('GRID baseline/.project-root','references/wmh-original/GRID/.project-root') if (REPO/p).is_file()]
    return [{'path':str(p.relative_to(REPO)), 'bytes':p.stat().st_size, 'sha256':sha256(p)} for p in sorted(files)]


def snapshot(t):
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip() == 'wbw'
    records = source_records()
    for r in records:
        data = subprocess.check_output(['git','show',head+':'+r['path']],cwd=REPO)
        assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
    digest = hashlib.sha256(json.dumps({'source_commit':head,'files':records},sort_keys=True).encode()).hexdigest()
    manifest = {'source_commit':head,'snapshot_sha256':digest,'files':records,
        'identity_kind':'commit_plus_exact_manifest; immutable formal stage export'}
    data = (json.dumps(manifest,indent=2)+'\n').encode(); manifest_sha = hashlib.sha256(data).hexdigest()
    SUPPORT.mkdir(exist_ok=True)
    archive = SUPPORT/('snapshot-'+digest[:16]+'.tar.gz')
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tf:
            for r in records:tf.add(REPO/r['path'],arcname=r['path'],recursive=False)
            member=tarfile.TarInfo('DEPLOYMENT.json');member.size=len(data);tf.addfile(member,io.BytesIO(data))
    remote_archive=REMOTE+'/archives/'+archive.name;t.put(archive,remote_archive)
    root=REMOTE+'/snapshots/'+digest[:16]
    code='''import pathlib,tarfile,hashlib,json
p=pathlib.Path(ROOT);a=pathlib.Path(ARCHIVE)
assert hashlib.sha256(a.read_bytes()).hexdigest()==ARCHIVE_SHA
if not p.exists():
 p.mkdir(parents=True)
 with tarfile.open(a) as tf:
  m=tf.getmembers();assert len({v.name for v in m})==len(m)
  assert all(v.isfile() and not pathlib.Path(v.name).is_absolute() and '..' not in pathlib.Path(v.name).parts for v in m)
  tf.extractall(p,filter='data')
assert hashlib.sha256((p/'DEPLOYMENT.json').read_bytes()).hexdigest()==MANIFEST_SHA
d=json.loads((p/'DEPLOYMENT.json').read_text())
for r in d['files']:
 f=p/r['path'];assert not f.is_symlink() and f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
print(json.dumps({'root':str(p),'files_verified':len(d['files']),'manifest_sha256':MANIFEST_SHA}))
'''.replace('MANIFEST_SHA',repr(manifest_sha)).replace('ARCHIVE_SHA',repr(sha256(archive))).replace('ARCHIVE',repr(remote_archive)).replace('ROOT',repr(root))
    receipt=json.loads(t.python(code,timeout=60))
    write_json(SUPPORT/(digest[:16]+'-deployment.json'),{'manifest':manifest,'receipt':receipt})
    return root,manifest,manifest_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['identity-check','evaluate','metadata','train','attention'])
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--gpu',type=int,choices=[0,1]);p.add_argument('--timeout',type=int,required=True)
    a=p.parse_args();assert a.job.startswith('gf-20261015-') and 60<=a.timeout<=43200
    sys.path.insert(0,str(TRANSPORT));from transport import Transport
    import gridctl
    t=Transport('new-server');t.identity();gpu=''
    if a.mode not in ('identity-check','metadata'):
        assert a.gpu is not None;gpu=GPUS[a.gpu]
        observation=json.loads(t.python("import subprocess,json;print(json.dumps({'apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True),'devices':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)}))"))
        write_json(SUPPORT/(a.job+'-resource.json'),observation)
        assert not any(row.split(',')[0].strip()==gpu for row in observation['apps'].splitlines()),'GPU has another compute process'
        free=next(int(row.split(',')[1]) for row in observation['devices'].splitlines() if row.split(',')[0].strip()==gpu)
        assert free>=14336,'Need 14 GiB free before formal work'
    root,manifest,manifest_sha=snapshot(t)
    remote_spec=REMOTE+'/specs/'+sha256(a.spec)+'.json';t.put(a.spec,remote_spec)
    spec=json.loads(a.spec.read_text())
    inputs=[{'path':remote_spec,'sha256':sha256(a.spec)},{'path':root+'/DEPLOYMENT.json','sha256':manifest_sha}]
    for key in ('config','sid','checkpoint','test_index','evaluation_index','attention_cohort','nonauthor_gate','evaluation_result','evaluation_status'):
        if isinstance(spec.get(key),dict) and set(spec[key])>={'path','sha256'}:inputs.append({k:spec[key][k] for k in ('path','sha256')})
    request={'schema':1,'job_id':a.job,'host':t.profile['hostname'],'user':t.profile['user'],'uid':t.profile['uid'],
        'namespace':t.profile['namespace'],'deployment_sha256':t.profile['deployment_sha256'],
        'capture_version':manifest['source_commit']+'/'+manifest['snapshot_sha256'],
        'argv':['/data/bwwu/envs/grid-20261009-e61baf3/bin/python','-B','-m','experiments.formal_20261015.run',a.mode,'--spec',remote_spec],
        'cwd':root,'timeout_seconds':a.timeout,'env':{'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4','TF_CPP_MIN_LOG_LEVEL':'3',
        'TF_NUM_INTRAOP_THREADS':'2','TF_NUM_INTEROP_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':root,
        'HF_HUB_OFFLINE':'1','WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'},'inputs':inputs}
    path=SUPPORT/(a.job+'-request.json')
    if path.exists():assert json.loads(path.read_text())==request,'Existing request identity differs; query it'
    write_json(path,request);receipt=gridctl.submit(t,path);write_json(SUPPORT/(a.job+'-submission.json'),receipt)
    print(json.dumps({'deployment':root,'receipt':receipt},ensure_ascii=False))


if __name__=='__main__':main()
