import datetime,json,sys
from pathlib import Path
import transport
root=Path(__file__).resolve().parents[1];t=transport.Transport('new-server')
code='''import json,datetime,subprocess
from pathlib import Path
root=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs/lc-rec-offload-20261109-bounded')
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for n in ['status.json','output/chain-status.json']:
 p=root/n
 if p.exists():out[n]=json.loads(p.read_text())
for candidate in ['optimizer-only','optimizer-parameter']:
 d=root/'output'/candidate
 p=d/'console.log'
 if p.exists():out[candidate+'_console_tail']=p.read_text(errors='replace')[-5000:]
 p=d/'resource-samples.jsonl'
 if p.exists():out[candidate+'_latest_resource']=json.loads(p.read_text().splitlines()[-1])
 for q in d.glob('probe/probe_rank*.json'):
  v=json.loads(q.read_text());out[str(q.relative_to(root))]={k:v.get(k) for k in ['rank','status','phase','elapsed_wall_seconds','model','gpu_allocator_peak','errors']}
  out[str(q.relative_to(root))]['forward_count']=len(v.get('losses',[]));out[str(q.relative_to(root))]['update_count']=len(v.get('optimizer_updates',[]))
p=Path('/home/bwwu/lc-rec/offload-20261108/gpu-budget.json')
if p.exists():out['budget']=json.loads(p.read_text())
print(json.dumps(out))'''
s=t.python(code,timeout=40)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
p=root/'monitor'/('gpu-'+stamp+'.json');p.parent.mkdir(exist_ok=True);p.write_text(s)
d=json.loads(s)
print(json.dumps(d,ensure_ascii=False))
