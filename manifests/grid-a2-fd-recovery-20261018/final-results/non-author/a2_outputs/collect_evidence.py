"""Bounded read-only A2 capture, split into preparation and terminal output sets."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,shutil,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parent
SUPPORT=ROOT.parents[1]
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
SOURCE_FILES=['experiments/launch_20261017/run.py','experiments/launch_20261017/statistics.py',
 'experiments/launch_20261017/train.py','experiments/launch_20261017/training_protocol.json',
 'experiments/a2_fd_recovery_20261018/adapter.py','experiments/a2_fd_recovery_20261018/controller.py',
 'experiments/formal_20261015/run.py','experiments/beauty_extensions/common.py','experiments/beauty_extensions/results.py',
 'experiments/beauty_extensions/cli.py','experiments/beauty_extensions/author/models.py',
 'experiments/beauty_extensions/attention/capture.py','experiments/beauty_extensions/attention/records.py',
 'experiments/beauty_extensions/attention/plot.py','experiments/beauty_extensions/observation.py',
 'experiments/beauty_extensions/data_checks.py','experiments/beauty_extensions/mlp_execution.py',
 'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py',
 'references/wmh-original/GRID/src/components/eval_metrics.py',
 'references/wmh-original/GRID/src/data/loading/components/label_function.py']

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','terminal']);a=p.parse_args()
 deployment=json.loads((SUPPORT/'source-deployment.json').read_text())
 destination=ROOT/('preparation-v2' if a.phase=='prepare' else 'terminal_outputs');destination.mkdir(exist_ok=False)
 assert shutil.disk_usage(ROOT).free>350_000_000
 remote_code=r'''
from pathlib import Path
import hashlib,io,json,sys,tarfile
jobs=Path('/home/bwwu/grid-audits/sid-extensions-p0/jobs')
snapshot=Path(SNAPSHOT)
items=[]
def add(p,rel):
 assert p.is_file() and not p.is_symlink(),str(p)
 items.append((p,rel))
if PHASE=='prepare':
 for role in ['metadata','gpu-gate']:
  job=jobs/f'ga-20261018-sportsa2-{role}-fd-recovery-v1'
  status=json.loads((job/'status.json').read_text());assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
  if role=='metadata':
   selected=sorted(job.rglob('*'))
  else:
   selected=[job/rel for rel in ['request.json','status.json','cleanup.json','output/RESULT.json','output/admission.json',
      'output/effective_config.yaml','output/used_config.json','output/invocation.json','output/training_protocol.json']]
  for f in selected:
   if f.is_file() and f.stat().st_size and not f.name.startswith('.'):
    assert f.suffix not in ['.ckpt','.pt'],str(f)
    add(f,'jobs/'+str(f.relative_to(jobs)))
 train=jobs/'ga-20261018-sportsa2-train-fd-recovery-v1'
 for rel in ['request.json','accepted.json','status.json','output/effective_config.yaml','output/used_config.json','output/invocation.json','output/training_protocol.json','output/structure.json','output/actual_train_loader.json','output/deadline.json']:
  f=train/rel
  if f.exists():add(f,'jobs/'+str(f.relative_to(jobs)))
 quant=jobs/'ga-20261018-sportsa2-quantize-fd-recovery-v1'
 for rel in ['request.json','status.json','cleanup.json','output/RESULT.json','output/stage_result.json','output/semantic_ids.pt']:
  f=quant/rel;add(f,'jobs/'+str(f.relative_to(jobs)))
 add(snapshot/'DEPLOYMENT.json','scientific_source/DEPLOYMENT.json')
 for rel in SOURCE_FILES:add(snapshot/rel,'scientific_source/'+rel)
else:
 for role in ['eval-0','eval-3','eval-4','attention-0','attention-3','attention-4','stats']:
  job=jobs/f'ga-20261018-sportsa2-{role}-fd-recovery-v1'
  status=json.loads((job/'status.json').read_text())
  assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified'],(job.name,status['status'])
  relatives=['request.json','accepted.json','status.json','cleanup.json','work.log','output/RESULT.json','output/invocation.json','output/stage-entry.json','output/stage-exit.json']
  if role!='stats':
   relatives+=['output/effective_config.yaml','output/used_config.json','output/loading.json','output/checkpoint_protocol.json']
  if role.startswith('eval-'):
   relatives+=['output/intervention_trace.json']+['output/test/'+name for name in ['predictions.jsonl','per_sample_metrics.jsonl','metrics.json','recomputed_metrics.json','lifecycle_cleanup.json']]
  if role.startswith('attention-'):
   for batch in range(4):
    prefix=f'output/batch-{batch:02d}/'
    relatives += [prefix+name for name in ['RESULT.json','off_on_checks.json','bundle/record.json','bundle/arrays.npz','figures/plot_receipt.json']]
    relatives += [prefix+f'figures/sample_{i:03d}_cross_attention.png' for i in range(8)]
  for rel in relatives:
   f=job/rel;assert f.suffix not in ['.ckpt','.pt'],str(f)
   add(f,'jobs/'+str(f.relative_to(jobs)))
 train=jobs/'ga-20261018-sportsa2-train-fd-recovery-v1'
 for rel in ['request.json','status.json','cleanup.json','output/RESULT.json','output/training_result.json','output/training_protocol.json','output/effective_config.yaml','output/used_config.json','output/invocation.json']:
  f=train/rel
  if f.exists():add(f,'jobs/'+str(f.relative_to(jobs)))
limit=18_000_000 if PHASE=='prepare' else 100_000_000
assert sum(p.stat().st_size for p,_ in items)<limit
assert len({r for _,r in items})==len(items)
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for path,relative in items:
  data=path.read_bytes();manifest.append({'relative':relative,'remote':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  ti=tarfile.TarInfo(relative);ti.size=len(data);ti.mode=0o444;tar.addfile(ti,io.BytesIO(data))
 data=json.dumps({'phase':PHASE,'non_atomic_read_of_explicit_sources':True,'server_mutations':0,'files':manifest},indent=2).encode()
 ti=tarfile.TarInfo('REMOTE_MANIFEST.json');ti.size=len(data);ti.mode=0o444;tar.addfile(ti,io.BytesIO(data))
'''
 code='SNAPSHOT='+repr(deployment['root'])+'\nPHASE='+repr(a.phase)+'\nSOURCE_FILES='+repr(SOURCE_FILES)+'\n'+remote_code
 transport=Transport('new-server');archive=destination/'evidence.tar.gz'
 with archive.open('xb') as out:
  run=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe',*transport.options,'-T',transport.host,'python3 -I -S -B -'],
    input=code.encode(),stdout=out,stderr=subprocess.PIPE,timeout=120)
 assert run.returncode==0,run.stderr.decode(errors='replace')
 with tarfile.open(archive,'r:gz') as tar:
  members=tar.getmembers();assert len({m.name for m in members})==len(members)
  manifest=json.load(tar.extractfile('REMOTE_MANIFEST.json'));expected={r['relative']:r for r in manifest['files']}
  for member in members:
   rel=PurePosixPath(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
   data=tar.extractfile(member).read()
   if member.name!='REMOTE_MANIFEST.json':
    e=expected[member.name];assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
   out=destination/'raw'/member.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
 if a.phase=='prepare':
  expected_sources={r['path']:r for r in deployment['manifest']['files']}
  for rel in SOURCE_FILES:
   actual=destination/'raw/scientific_source'/rel
   assert hashlib.sha256(actual.read_bytes()).hexdigest()==expected_sources[rel]['sha256'],rel
 receipt={'phase':a.phase,'returncode':0,'server_mutations':0,'archive':str(archive),'bytes':archive.stat().st_size,
          'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'verified_members':len(expected),
          'raw_bytes':sum(r['bytes'] for r in expected.values()),'remote_code_sha256':hashlib.sha256(code.encode()).hexdigest(),
          'scientific_commit':deployment['manifest']['source_commit'],'scientific_snapshot':deployment['manifest']['snapshot_sha256']}
 (destination/'collection-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
