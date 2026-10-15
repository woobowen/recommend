"""Collect terminal stage evidence, excluding every data/model payload."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile

from .deploy import REMOTE_STAGE, SUPPORT, TRANSPORT_SUPPORT
from experiments.beauty_extensions.common import sha256, write_json


def remote(plan):
    assert plan['job_root'] == '/home/bwwu/grid-audits/sid-extensions-p0/jobs'
    root = Path(plan['job_root']); target = Path(plan['target'])
    assert str(target).startswith(REMOTE_STAGE + '/evidence/') and not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    members = []; total = 0
    for job in plan['jobs']:
        assert job.startswith('gsp-20261015-') and '/' not in job and '..' not in job
        base = root / job
        status = json.loads((base/'status.json').read_text())
        assert status['status'] in ('SUCCEEDED','FAILED','TIMED_OUT')
        selected = [base/name for name in ('request.json','status.json','accepted.json','attempt.json',
            'cleanup.json','work.log','supervisor.log') if (base/name).is_file()]
        for path in (base/'output').rglob('*'):
            if path.is_file() and not path.is_symlink() and 'checkpoints' not in path.relative_to(base).parts and path.suffix in ('.json','.jsonl','.yaml','.csv','.txt'):
                selected.append(path)
        for path in sorted(set(selected)):
            assert not path.is_symlink() and path.stat().st_size <= 8*1024**2
            record = {'path':job+'/'+str(path.relative_to(base)), 'bytes':path.stat().st_size, 'sha256':sha256(path)}
            total += record['bytes']; assert total < 64*1024**2
            members.append((path, record))
    manifest = {'jobs':plan['jobs'], 'files':[r for _,r in members], 'bytes':total,
        'model_payloads_excluded':True, 'source_data_excluded':True}
    manifest_path = target.with_suffix('.manifest.json'); write_json(manifest_path, manifest)
    with tarfile.open(target, 'x:gz') as tf:
        for path, record in members: tf.add(path,arcname=record['path'],recursive=False)
        tf.add(manifest_path,arcname='COLLECTION_MANIFEST.json',recursive=False)
    return {'path':str(target), 'bytes':target.stat().st_size, 'sha256':sha256(target)}


def local(jobs):
    assert 1 <= len(jobs) <= 32 and len(jobs) == len(set(jobs))
    sys.path.insert(0,str(TRANSPORT_SUPPORT/'scripts'))
    from transport import Transport
    import gridctl
    t=Transport('new-server'); states={}
    for job in jobs:
        s=gridctl.query(t,job)
        assert s['status'] in ('SUCCEEDED','FAILED','TIMED_OUT') and s['cleanup_verified'] and not s['workgroup_live']
        states[job]=s
    tag=hashlib.sha256(json.dumps(jobs).encode()).hexdigest()[:16]
    plan={'jobs':jobs,'job_root':t.profile['job_root'],'target':REMOTE_STAGE+'/evidence/collection-'+tag+'.tar.gz'}
    local_plan=SUPPORT/('collect-'+tag+'.json');write_json(local_plan,plan)
    source=Path(__file__).resolve(); remote_script=REMOTE_STAGE+'/scripts/collect-'+sha256(source)[:16]+'.py'
    # Copy the self-contained collector invocation as a module inside the fixed
    # snapshot; source/helper identity is recorded in the deployment manifest.
    from .deploy import snapshot
    snapshot_root,manifest,_=snapshot(t)
    remote_plan=REMOTE_STAGE+'/specs/'+sha256(local_plan)+'.json';t.put(local_plan,remote_plan)
    command=shlex.join(['/usr/bin/python3','-B','-m','experiments.preflight_20261015.collect','remote',remote_plan])
    meta=json.loads(t.ssh('cd '+shlex.quote(snapshot_root)+' && '+command,timeout=60))
    archive=SUPPORT/('collection-'+tag+'.tar.gz');receipt=t.get_verified(meta['path'],archive,meta)
    dest=SUPPORT/('collected-'+tag);dest.mkdir(exist_ok=False)
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers();names=[v.name for v in members]
        assert len(names)==len(set(names)) and all(v.isfile() and not Path(v.name).is_absolute() and '..' not in Path(v.name).parts for v in members)
        tf.extractall(dest,filter='data')
    collected=json.loads((dest/'COLLECTION_MANIFEST.json').read_text())
    assert set(names)=={r['path'] for r in collected['files']}|{'COLLECTION_MANIFEST.json'}
    for r in collected['files']:
        path=dest/r['path'];assert path.stat().st_size==r['bytes'] and sha256(path)==r['sha256']
    write_json(SUPPORT/'terminal-states.json',states)
    write_json(SUPPORT/'collection-receipt.json',{**receipt,'directory':str(dest),'members_verified':True,
        'collector_snapshot_sha256':manifest['snapshot_sha256']})
    print(json.dumps({'directory':str(dest),'bytes':receipt['bytes'],'files':len(collected['files']),'verified':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('remote');q.add_argument('plan',type=Path)
    q=sub.add_parser('local');q.add_argument('jobs',nargs='+')
    a=p.parse_args()
    if a.mode=='remote':print(json.dumps(remote(json.loads(a.plan.read_text()))))
    else:local(a.jobs)


if __name__=='__main__':main()
