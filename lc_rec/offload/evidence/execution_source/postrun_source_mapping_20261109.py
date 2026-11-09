import hashlib,json,subprocess
from pathlib import Path
import transport
root=Path(__file__).resolve().parents[1];repo=Path('/home/addaswsw/recommend-worktrees/lc-rec');t=transport.Transport('new-server')
commit='c8fbe3fc8c627935f22483935ec59f0422b03602'
files=['lc_rec/offload/scripts/offload_probe.py','lc_rec/offload/scripts/resource_guard.py','lc_rec/offload/scripts/run_bounded_candidates.py','lc_rec/offload/config/ds_optimizer_only.json','lc_rec/offload/config/ds_optimizer_parameter.json','lc_rec/offload/config/probe_optimizer_only.json','lc_rec/offload/config/probe_optimizer_parameter.json']
code='''from pathlib import Path
import hashlib,json,datetime
root=Path(RELEASE)
files=FILES
model=Path('/data/bwwu/lc-rec/assets/llama-7b/4782ad278652c7c71b72204d462d6d01eaaf7549')
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':[{'path':p,'bytes':(root/p).stat().st_size,'sha256':hashlib.sha256((root/p).read_bytes()).hexdigest()} for p in files],'weights':{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'inode':p.stat().st_ino} for p in model.glob('*.safetensors')}}))'''.replace('RELEASE',repr('/home/bwwu/lc-rec/releases/'+commit)).replace('FILES',repr(files))
v=json.loads(t.python(code));pre=json.loads((repo/'lc_rec/offload/evidence/preparation/model_identity_readback_20261109.json').read_text())
for row in v['files']:
 fixed=subprocess.check_output(['git','show',commit+':'+row['path']],cwd=repo);local=(repo/row['path']).read_bytes()
 assert row['bytes']==len(fixed) and hashlib.sha256(fixed).hexdigest()==row['sha256'] and fixed==local
 row['server_equals_fixed_RUN_and_final_working_copy']=True
assert v['weights']==pre['stats']
v.update(RUN_COMMIT=commit,server_fixed_export='/home/bwwu/lc-rec/releases/'+commit,weight_stats_unchanged_since_preflight=True,weight_check_scope='size/mtime/inode only; inherit prior full-file hashes; no repeated13.48GB hash',status='PASS')
p=repo/'lc_rec/offload/evidence/preparation/postrun_server_runtime_mapping_20261109.json';p.write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'postrun_mapping':v['status'],'runtime_files':len(v['files']),'weight_stats_unchanged':True}))
