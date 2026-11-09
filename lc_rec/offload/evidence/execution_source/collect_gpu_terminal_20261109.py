import base64,datetime,hashlib,json,sys
from pathlib import Path
import transport
root=Path(__file__).resolve().parents[1];t=transport.Transport('new-server')
repo=Path('/home/addaswsw/recommend-worktrees/lc-rec');out=repo/'lc_rec/offload/evidence/gpu_20261109'
code='''from pathlib import Path
import base64,hashlib,json,datetime
root=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs/lc-rec-offload-20261109-bounded')
status=json.loads((root/'status.json').read_text());assert status['status'] not in ['RUNNING','ACCEPTED']
files=['request.json','accepted.json','status.json','attempt.json','cleanup.json','supervisor.log','work.log','worker-group.json','output/chain-status.json','output/model-validation-inherited.json']
for c in ['optimizer-only','optimizer-parameter']:
 for name in ['effective-guard.json','console.log','resource-summary.json','resource-samples.jsonl','probe/probe_rank0.json','probe/probe_rank1.json']:
  files.append('output/'+c+'/'+name)
rows=[]
for name in files:
 p=root/name
 if p.is_file():
  assert not p.is_symlink() and p.stat().st_size<25*1024**2
  b=p.read_bytes();rows.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
p=Path('/home/bwwu/lc-rec/offload-20261108/gpu-budget.json')
if p.is_file():
 b=p.read_bytes();rows.append({'path':'gpu-budget.json','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
print(json.dumps({'host':'zwlab-01','job_id':root.name,'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows}))'''
v=json.loads(t.python(code,timeout=45));out.mkdir(parents=True,exist_ok=True)
for r in v['rows']:
 b=base64.b64decode(r.pop('base64'));assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 p=out/r['path'];p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
v['transport']='existing Windows OpenSSH; remote bytes and SHA-256 matched local'
(out/'retrieval.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'files':len(v['rows']),'bytes':sum(r['bytes'] for r in v['rows']),'path':str(out)}))
