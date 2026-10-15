"""Hash-bound deployment and accepted runner submission for preflight only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from experiments.beauty_extensions.common import REPO, sha256, write_json
from .run import require_preflight

SUPPORT = Path('/home/addaswsw/recommend-support/grid-shuffle-mlp-preflight-20261015')
TRANSPORT_SUPPORT = Path('/home/addaswsw/recommend-support/sid-extensions-p0')
REMOTE_STAGE = '/home/bwwu/grid-audits/sid-extensions-p0/grid-shuffle-mlp-preflight-20261015'
GPUS = ['GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae', 'GPU-59cac3bd-3645-668b-79c8-839e0d878893']


def snapshot(t):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert subprocess.check_output(['git','branch','--show-current'], cwd=REPO, text=True).strip() == 'wbw'
    files = []
    for directory in ['GRID baseline/src','references/wmh-original/GRID/src',
                      'experiments/beauty_extensions','experiments/preflight_20261015']:
        files.extend(f for f in (REPO/directory).rglob('*') if f.is_file() and
                     '__pycache__' not in f.parts and f.suffix in ('.py','.json','.yaml'))
    files.extend(REPO / x for x in ['GRID baseline/.project-root', 'references/wmh-original/GRID/.project-root'] if (REPO / x).is_file())
    files = sorted(files)
    records = [{'path':str(f.relative_to(REPO)), 'bytes':f.stat().st_size, 'sha256':sha256(f)} for f in files]
    # DEPLOYMENT.json contains source_commit. Bind that field into the directory
    # identity as well: identical files at a later commit must not reuse an old
    # manifest while submitting the hash of a freshly constructed manifest.
    digest = hashlib.sha256(json.dumps({'source_commit': head, 'files': records}, sort_keys=True).encode()).hexdigest()
    manifest = {'source_commit':head, 'snapshot_sha256':digest, 'files':records,
        'identity_kind':'commit_plus_file_manifest_v2; engineering patch may be uncommitted',
        'formal_approved':False}
    manifest_bytes = (json.dumps(manifest, indent=2)+'\n').encode()
    archive = SUPPORT / ('snapshot-'+digest[:16]+'.tar.gz')
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tf:
            for f in files: tf.add(f,arcname=str(f.relative_to(REPO)),recursive=False)
            info=tarfile.TarInfo('DEPLOYMENT.json'); info.size=len(manifest_bytes)
            tf.addfile(info,io.BytesIO(manifest_bytes))
    remote_archive=REMOTE_STAGE+'/archives/'+archive.name
    t.put(archive,remote_archive)
    remote_root=REMOTE_STAGE+'/snapshots/'+digest[:16]
    code='''import pathlib,tarfile,json,hashlib
p=pathlib.Path(TARGET); archive=pathlib.Path(ARCHIVE)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==ARCHIVE_SHA
if not p.exists():
 p.mkdir(parents=True)
 with tarfile.open(archive,'r:gz') as tf:
  members=tf.getmembers();assert len({v.name for v in members})==len(members)
  assert all(v.isfile() and not pathlib.Path(v.name).is_absolute() and '..' not in pathlib.Path(v.name).parts for v in members)
  tf.extractall(p,filter='data')
manifest=json.loads((p/'DEPLOYMENT.json').read_text());assert manifest['snapshot_sha256']==DIGEST
for r in manifest['files']:
 f=p/r['path'];assert not f.is_symlink() and f.is_file()
 assert f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
print(json.dumps({'root':str(p),'files_verified':len(manifest['files']),'snapshot_sha256':DIGEST}))
'''.replace('ARCHIVE_SHA',repr(sha256(archive))).replace('TARGET',repr(remote_root)).replace('ARCHIVE',repr(remote_archive)).replace('DIGEST',repr(digest))
    receipt=json.loads(t.python(code,timeout=60))
    write_json(SUPPORT/(digest[:16]+'-deployment.json'),{'manifest':manifest,'receipt':receipt})
    return remote_root,manifest,hashlib.sha256(manifest_bytes).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','shuffle','mlp-short','mlp-resume'])
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--gpu',choices=['0','1']);p.add_argument('--timeout',type=int,default=1200)
    p.add_argument('--share-gpu',action='store_true',help='Explicit user-authorized shared preflight; keep memory safety margins')
    args=p.parse_args();spec=json.loads(args.spec.read_text());require_preflight(spec)
    if not args.job.startswith('gsp-20261015-'):raise ValueError('New stage job namespace required')
    if not 60 <= args.timeout <= 1200:raise ValueError('Bounded timeout required')
    sys.path.insert(0,str(TRANSPORT_SUPPORT/'scripts'))
    from transport import Transport
    import gridctl
    t=Transport('new-server');t.identity()
    if args.phase != 'prepare':
        if args.gpu is None:raise ValueError('Explicit physical card required')
        gpu=GPUS[int(args.gpu)]
        code="import subprocess,json;print(json.dumps({'processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True),'devices':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used','--format=csv,noheader'],text=True)}))"
        observed=json.loads(t.python(code))
        write_json(SUPPORT/(args.job+'-resource-check.json'),observed)
        if args.share_gpu:
            from .resources import assess
            # Obtain the same standard-library observation from the saved
            # helper, before any model import or allocation.
            helper=Path(__file__).with_name('resources.py')
            remote_helper=REMOTE_STAGE+'/resource-tools/resources.py';t.put(helper,remote_helper)
            detailed=json.loads(t.python('import sys,json;sys.path.insert(0,'+repr(REMOTE_STAGE+'/resource-tools')+');import resources;print(json.dumps(resources.observe()))'))
            problem=assess(detailed,gpu,shared=True,initial=True)
            write_json(SUPPORT/(args.job+'-shared-resource-check.json'),detailed)
            if problem:raise RuntimeError(problem+'; no job submitted')
        elif any(line.split(',')[0].strip()==gpu for line in observed['processes'].splitlines()):
            raise RuntimeError('RESOURCE_BUSY: no job submitted; card has an existing compute process')
    else:gpu=''
    root,manifest,manifest_sha=snapshot(t)
    remote_spec=REMOTE_STAGE+'/specs/'+sha256(args.spec)+'.json';t.put(args.spec,remote_spec)
    request={'schema':1,'job_id':args.job,'host':t.profile['hostname'],'user':t.profile['user'],
        'uid':t.profile['uid'],'namespace':t.profile['namespace'],'deployment_sha256':t.profile['deployment_sha256'],
        'capture_version':manifest['source_commit']+'/'+manifest['snapshot_sha256'],
        'argv':['/data/bwwu/envs/grid-20261009-e61baf3/bin/python','-B','-m',
            'experiments.preflight_20261015.run',args.phase,'--spec',remote_spec],
        'cwd':root,'timeout_seconds':args.timeout,
        'env':{'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4','TF_CPP_MIN_LOG_LEVEL':'3',
            'TF_NUM_INTRAOP_THREADS':'2','TF_NUM_INTEROP_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
            'PYTHONPATH':root,'HF_HUB_OFFLINE':'1','WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'},
        'inputs':[{'path':remote_spec,'sha256':sha256(args.spec)},
                  {'path':root+'/DEPLOYMENT.json','sha256':manifest_sha}]}
    if args.share_gpu:request['env']['GRID_SHARED_PREFLIGHT']='1'
    request_path=SUPPORT/(args.job+'-request.json')
    if request_path.exists() and json.loads(request_path.read_text())!=request:raise RuntimeError('Existing job request differs')
    write_json(request_path,request);receipt=gridctl.submit(t,request_path)
    write_json(SUPPORT/(args.job+'-submission.json'),receipt)
    print(json.dumps({'deployment':root,'receipt':receipt},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
