"""Collect only bounded audit artifacts from terminal successful formal jobs.

This delivery helper is separate from the frozen GPU/evaluation RUN source.
No model loading, prediction generation, or modification of server originals.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys
import zipfile
from control import connection,SUPPORT,DATA,REPO

RUN='f0242b9f90ae23a4979d688a287734ca9f88cea1'


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--job');choice.add_argument('--control',action='store_true')
    a=p.parse_args();t=connection()
    if a.control:
        destination=REPO/'lc_rec/shuffle/evidence/control';destination.mkdir(parents=True,exist_ok=True)
        paths=[SUPPORT/'receipts/github_prelaunch_readback.json',
               SUPPORT/'receipts'/('deployment-'+RUN+'.json'),
               SUPPORT/'receipts/submit-letter-shuffle-instruments-seed42-20261114-series-r0.json']
        paths+=list((SUPPORT/'requests').glob('shuffle-'+RUN+'.*.json'))
        plan=json.loads((SUPPORT/'requests'/('shuffle-'+RUN+'.series-plan.json')).read_text())
        paths += [SUPPORT/'requests'/(j['job_id']+'.json') for j in plan['jobs']]
        paths += [SUPPORT/'requests/letter-shuffle-instruments-seed42-20261114-series-r0.json']
        for source in paths:
            target=destination/source.parent.name/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            raw=source.read_bytes()
            if target.exists():assert target.read_bytes()==raw
            else:target.write_bytes(raw)
        print(json.dumps({'status':'PASS_CONTROL_CAPTURE','files':len(paths),'local':str(destination)}))
        return
    plan=json.loads((SUPPORT/'requests'/('shuffle-'+RUN+'.series-plan.json')).read_text())
    request=next(j for j in plan['jobs'] if j['job_id']==a.job)
    output=Path(request['argv'][request['argv'].index('--output')+1])
    code='''from pathlib import Path
import hashlib,json,subprocess,zipfile
job=Path(JOB_ROOT)/JOB;out=Path(OUTPUT)
state=json.loads(subprocess.check_output(['/usr/bin/python3','-I','-S','-B',RUNNER,'query',str(job)],text=True))
assert state['status']=='SUCCEEDED' and state['exit_code']==0 and state['cleanup_verified']
assert not state['worker_alive'] and not state['workgroup_live']
assert json.loads((out/'summary.json').read_text())['status']=='PASS'
assert json.loads((out/'evaluation/summary.json').read_text())['status']=='PASS'
members={}
for name in ['request.json','status.json','accepted.json','cleanup.json']:
 members['job/'+name]=job/name
for name in ['summary.json','execution_sources.json','resources.jsonl','generation.log','evaluation.log']:
 members[name]=out/name
for rank in [0,1]:
 for name in ['rank%d.json'%rank,'derived.rank%d.py'%rank,'derived.rank%d.diff'%rank]:
  members['test/'+name]=out/'test'/name
members['test/metrics.json']=out/'test/metrics.json'
for path in sorted((out/'evaluation').rglob('*')):
 if path.is_file() and path.suffix in ['.json','.log','.npz','.npy']:
  members[str(path.relative_to(out))]=path
assert all(p.is_file() and not p.is_symlink() and p.stat().st_size<10*1024**2 for p in members.values())
assert not any('predictions.rank' in k or 'adapter_model' in k for k in members)
manifest={'status':'PASS','job_id':JOB,'RUN_COMMIT':RUN,'query':state,'members':[]}
for name,path in sorted(members.items()):
 manifest['members'].append({'member':name,'source':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
dest=Path(DATA)/'delivery';dest.mkdir(exist_ok=True);archive=dest/(JOB+'.audit.zip')
if not archive.exists():
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for name,path in sorted(members.items()):z.write(path,name)
  z.writestr('COLLECTION_MANIFEST.json',json.dumps(manifest,indent=2)+'\\n')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 existing=json.loads(z.read('COLLECTION_MANIFEST.json'))
 assert existing['members']==manifest['members'] and existing['RUN_COMMIT']==RUN and existing['job_id']==JOB
 assert len(z.namelist())==len(set(z.namelist()))==len(members)+1
 for row in manifest['members']:assert hashlib.sha256(z.read(row['member'])).hexdigest()==row['sha256']
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'members':len(members)+1}))
'''
    prelude='JOB_ROOT='+repr(t.profile['job_root'])+'\nJOB='+repr(a.job)+'\nOUTPUT='+repr(str(output))+'\nRUNNER='+repr(t.profile['runner'])+'\nRUN='+repr(RUN)+'\nDATA='+repr(DATA)+'\n'
    meta=json.loads(t.python(prelude+code,timeout=90))
    archive=SUPPORT/'collections'/Path(meta['path']).name
    t.get_verified(meta['path'],str(archive),meta)
    destination=REPO/'lc_rec/shuffle/results'/a.job
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        for member in z.namelist():
            q=PurePosixPath(member)
            assert not q.is_absolute() and '..' not in q.parts
        manifest=json.loads(z.read('COLLECTION_MANIFEST.json'))
        for row in manifest['members']:
            raw=z.read(row['member']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        for member in z.namelist():
            raw=z.read(member);target=destination/member
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==raw,'Do not overwrite differing collected evidence'
            else:target.write_bytes(raw)
    print(json.dumps({'status':'PASS_COLLECTED','job_id':a.job,'source_archive':meta,'local':str(destination)}))


if __name__=='__main__':main()
