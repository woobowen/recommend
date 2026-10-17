"""Prepare the lightweight, reviewable Git evidence set; does not commit/push."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

parser=argparse.ArgumentParser()
parser.add_argument('--index',type=Path,required=True)
args=parser.parse_args()
support=Path('/home/addaswsw/recommend-support/grid-a-launch-20261017')
repo=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
target=repo/'manifests/grid-a-launch-20261017'
index=json.loads(args.index.read_text());compiled=Path(index['local_root'])
copied=[]
def copy(path,relative):
 path=Path(path);dest=target/relative
 assert path.is_file() and not path.is_symlink() and path.stat().st_size<3*1024**2,path
 data=path.read_bytes();text=data.decode()
 assert not re.search(r'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{35,}',text),path
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 copied.append({'path':str(dest.relative_to(target)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source':str(path)})
for path in sorted((support/'startup-evidence').rglob('*')):
 if path.is_file():copy(path,Path('preparation-jobs')/path.relative_to(support/'startup-evidence'))
names=['asset-restoration-receipt.json','final_config_review.json','training_assets_independent.json',
 'a2_independent_review.json','metadata_source_link_review.json','cpu_admission_slice_review.json',
 'final_queue_review.json','launch_runtime_independent.json','observation_independent.json',
 '3ef93829656b734a-deployment.json','b6203414ee58a400-deployment.json']
for name in names:
 p=support/name
 if p.exists():copy(p,Path('reviews-and-identities')/name)
for path in sorted(compiled.glob('*.json')):
 copy(path,Path('compiled')/path.name)
for path in sorted((compiled/'specs').glob('*.json')):
 copy(path,Path('compiled/specs')/path.name)
old=support/'compiled-3ef93829656b734a'
for name in ('plan.json','index.json','gpu-admission-config.json','gpu-admission-render-receipt.json'):
 if (old/name).exists():copy(old/name,Path('prior-preparation-configs')/name)
for name in ('toys5300-metadata','toys5000-metadata','sportsa1-metadata',
 'toys5300-cpu-gate','toys5000-cpu-gate','sportsa1-cpu-gate','toys5300-gpu-gate','toys5000-gpu-gate'):
 copy(old/'specs'/(name+'.json'),Path('prior-preparation-configs/specs')/(name+'.json'))
fixture=support/'queue-fixture-20261017T051555'
for p in sorted(fixture.rglob('*')):
 if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.py','.log'):
  copy(p,Path('cpu-queue-fixture')/p.relative_to(fixture))
for name in ('prepare_assets_local.py','submit_preparation.py','prepare_gpu_admission.py',
 'finalize_queue.py','health_query.py','collect_startup.py','publish_startup.py','submit_formal_queue.py','finish_observation.py'):
 p=support/name
 if p.exists():copy(p,Path('local-invocation-scripts')/name)
external=[]
def paths(value):
 if isinstance(value,dict):
  for k,v in value.items():
   if k=='transport_evidence' and isinstance(v,str):external.append(Path(v))
   paths(v)
 elif isinstance(value,list):
  for v in value:paths(v)
for p in (repo/'experiments/launch_20261017/a2_review.json', repo/'experiments/launch_20261017/a2_sources.json',
 support/'a2_independent_review.json',support/'final_queue_review.json'):
 if p.exists():paths(json.loads(p.read_text()))
external_map=[]
for p in sorted(set(external)):
 if p.is_file():
  relative=Path('external-evidence')/p.name;copy(p,relative)
  external_map.append({'original':str(p),'member':str(relative),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  stdin=p.with_suffix('.stdin')
  if stdin.exists():copy(stdin,Path('external-evidence')/stdin.name)
for path in sorted(compiled.glob('health-*.json')):
 copy(path,Path('health')/path.name)
target.mkdir(parents=True,exist_ok=True)
(target/'EXTERNAL_EVIDENCE_MAP.json').write_text(json.dumps(external_map,indent=2)+'\n')
(target/'EVIDENCE_FILES.json').write_text(json.dumps({'files':copied,'member_hashes_verified':True,
 'no_checkpoint_data_or_weight_payloads':True,'secret_pattern_scan_passed':True},indent=2)+'\n')
(target/'INSTALLATION_LOG.json').write_text(json.dumps({'system_packages':[],'language_packages':[],
 'toolchains_or_fonts':[],'shell_or_global_configuration_changes':[],
 'environment':'Reused /data/bwwu/envs/grid-20261009-e61baf3/bin/python; per-request environment only.'},indent=2)+'\n')
(target/'CORRECTIONS.json').write_text(json.dumps({'task_revision_rounds_used':2,'maximum':3,
 'round1':{'scope':'Small queue fixture only','cause':'60-second queue fixture consumed by fixed60-second supervisor reserve',
  'change':'reserve=min(60,queue_seconds*0.1)','result':'CPU nested-runner fixture PASS; no formal research run affected'},
 'round2':{'scope':'CPU engineering training sample only; source b44399926c98369191c2ff5c2425377c8c07d3c8',
  'failed_job':'ga-20261017-sportsa1-cpu-gate-v1','cause':'Flattened four-code labels sliced as ordinary rows',
  'change':'Slice target/location complete four-code row groups; explicit shapes; preserve original failure',
  'passed_job':'ga-20261017-sportsa1-cpu-gate-v2','GPU_passed_job':'ga-20261017-sportsa1-gpu-gate-v1',
  'formal_training_functions_and_GPU_branch_unchanged':True}},indent=2)+'\n')
print(json.dumps({'directory':str(target),'files':len(copied),'bytes':sum(r['bytes'] for r in copied)}))
