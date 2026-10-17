"""Read-only scoped hash check against the actual new fixed server export."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
P=Path(__file__).resolve().parent;S=P.parents[1]
d=json.loads((S/'source-deployment.json').read_text())
files=[x['path'] for x in json.loads((P/'scheduler_cpu_review.json').read_text())['source_files']]
files += ['experiments/launch_20261017/queue_runner.py','experiments/launch_20261017/queue.py','experiments/launch_20261018_shared_gpu/build.py','experiments/launch_20261018_shared_gpu/resume.py']
code='''from pathlib import Path
import json,hashlib
root=Path(ROOT)
manifest=json.loads((root/'DEPLOYMENT.json').read_text())
by={r['path']:r for r in manifest['files']}
out={'manifest':manifest,'manifest_sha256':hashlib.sha256((root/'DEPLOYMENT.json').read_bytes()).hexdigest(),'files':[]}
for rel in FILES:
 p=root/rel;digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==by[rel]['sha256']
 out['files'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest})
print(json.dumps(out))
'''.replace('ROOT',repr(d['root'])).replace('FILES',repr(files))
out=json.loads(Transport('new-server').python(code,timeout=45))
assert out['manifest']==d['manifest'] and out['manifest_sha256']==d['manifest_sha256']
out.pop('manifest');out['status']='ACTUAL_SERVER_FIXED_SOURCE_BINDING_PASS';out['source_commit']=d['manifest']['source_commit']
(P/'actual_server_source_binding-v2.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
