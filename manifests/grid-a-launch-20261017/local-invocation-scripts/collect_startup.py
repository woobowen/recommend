"""Collect bounded original JSON/YAML/log evidence; exclude all model payloads."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport,atomic

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
jobs=['ga-20261017-assets-prepare-v1']
jobs += ['ga-20261017-'+n+'-metadata-v1' for n in ('toys5300','toys5000','sportsa1')]
jobs += ['ga-20261017-'+n+'-cpu-gate-v1' for n in ('toys5300','toys5000','sportsa1')]
jobs += ['ga-20261017-sportsa1-cpu-gate-v2']
jobs += ['ga-20261017-'+n+'-gpu-gate-v1' for n in ('toys5300','toys5000','sportsa1')]
code='''import pathlib,json,base64,hashlib
root=pathlib.Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs');out=[]
names=['request.json','accepted.json','status.json','work.log','supervisor.log','output/RESULT.json','output/invocation.json','output/loading.json','output/used_config.json','output/effective_config.yaml','output/structure.json','output/cpu_admission_batch_shapes.json','output/admission.json','output/progress.json','output/stage-entry.json','output/stage-exit.json','output/checkpoint_protocol.json','output/data_check.json','output/evaluation_coverage.json','output/test_coverage.json','output/fixed_attention_samples.json']
for job in JOBS:
 for name in names:
  p=root/job/name
  if not p.is_file():continue
  assert p.stat().st_size<2*1024**2
  data=p.read_bytes();out.append({'member':job+'/'+name,'source':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()})
print(json.dumps(out))
'''.replace('JOBS',repr(jobs))
records=json.loads(Transport('new-server').python(code,timeout=60))
manifest=[]
for r in records:
 member=Path(r['member']);assert not member.is_absolute() and '..' not in member.parts
 data=base64.b64decode(r.pop('base64'));assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
 p=args.output/member;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)
 manifest.append(r)
atomic(args.output/'COLLECTION_MANIFEST.json',{'files':manifest,'all_original_bytes_sha256_verified':True,'checkpoint_data_weights_included':False})
print(json.dumps({'files':len(manifest),'bytes':sum(r['bytes'] for r in manifest),'output':str(args.output)}))
