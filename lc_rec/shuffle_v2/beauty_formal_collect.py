"""Collect sealed and atomic sources for the Beauty launch evidence bundle."""
import argparse
from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path

from lc_rec.shuffle_v2 import beauty_control as bc, control
from lc_rec.shuffle_v2.beauty_formal_observe import PARENT, observe


REMOTE_RECEIPTS = '''
from datetime import datetime,timezone
from pathlib import Path
import base64,hashlib,json,os

parent=jobs/parent_id
request=json.loads((parent/'request.json').read_text())
plan_path=Path(request['argv'][request['argv'].index('--plan')+1])
plan_bytes=plan_path.read_bytes()
plan=json.loads(plan_bytes)
sources={'plan.json':plan_path,
         'prelaunch-gate.json':requests/(parent_id+'.review.json'),
         'runner.deployment.json':Path(plan['runner']+'.deployment.json'),
         'RUN_VERSION.json':Path(request['cwd'])/'RUN_VERSION.json'}
controller=Path(request['argv'][request['argv'].index('--output')+1])
if (controller/'summary.json').is_file():
    sources['controller-summary.json']=controller/'summary.json'
ids=[parent_id,*[r['job_id'] for conditions in plan['modes'].values() for r in conditions.values()],
     plan['paired']['job_id']]
for job in ids:
    if not (jobs/job/'accepted.json').is_file(): continue
    for file in ('request.json','accepted.json','attempt.json','status.json','cleanup.json'):
        path=jobs/job/file
        if path.is_file(): sources[f'jobs/{job}/{file}']=path
    actual=json.loads((jobs/job/'request.json').read_text())
    out=Path(actual['argv'][actual['argv'].index('--output')+1])
    phase=list(out.glob('*.request.json'))+list(out.glob('*.exit.json'))
    for member in phase:
        sources[f'worker/{job}/{member.name}']=member
    test=out/'test'
    for leaf in ('v2-binding.rank0.json','derived.rank0.diff','derived.rank0.py'):
        if (test/leaf).is_file(): sources[f'worker/{job}/{leaf}']=test/leaf
result={}
for name,path in sources.items():
    raw=path.read_bytes()
    assert len(raw)<2**20
    result[name]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
         'bytes':len(raw),'encoded':base64.b64encode(raw).decode('ascii'),
         'capture_utc':datetime.now(timezone.utc).isoformat()}
print(json.dumps(result))
'''


def collect(connection, output, parent, final_watermarks):
    roots = dict(jobs=Path(connection.profile['job_root']), requests=Path(control.BASE)/'requests', parent_id=parent)
    initialization = 'from pathlib import Path, PosixPath\n' + '\n'.join(
        f'{key}={value!r}' for key,value in roots.items())
    receipt = json.loads(control.remote_python(connection, initialization+'\n'+REMOTE_RECEIPTS))
    output.mkdir(parents=True,exist_ok=False)
    for name,item in receipt.items():
        destination = output/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        payload = b64decode(item.pop('encoded'),validate=True)
        assert (sha256(payload).hexdigest(),len(payload))==(item['sha256'],item['bytes'])
        with destination.open('xb') as stream:
            stream.write(payload)
    bc.immutable(output/'capture-manifest.json',receipt)
    if final_watermarks is not None:
        result = observe(connection,parent,final_watermarks)
        result['scope'] = 'single delivery check after the seven timed observations'
        bc.immutable(output/'delivery-observation.json',result)
    return {'captured_sources':len(receipt),'output':str(output)}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent',default=PARENT)
    p.add_argument('--previous',type=Path)
    args=p.parse_args()
    watermarks=json.loads(args.previous.read_text())['outputs'] if args.previous else None
    print(json.dumps(collect(control.connect(),args.output,args.parent,watermarks),indent=2))


if __name__=='__main__':
    main()
