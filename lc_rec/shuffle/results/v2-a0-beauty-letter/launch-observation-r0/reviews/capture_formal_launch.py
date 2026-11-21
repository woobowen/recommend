from pathlib import Path
import datetime, hashlib, json, os, subprocess
root=Path('/home/bwwu/grid-audits/sid-extensions-p0')
data=Path('/data/bwwu/lc-rec/letter-shuffle-v2-beauty')
ref='1182dfd914c20bd3dc11786587eae6879fba6b9d'
control_dir=root/'jobs/letter-beauty-v2-group-series-r0'
child_dir=root/'jobs/letter-beauty-v2-gpu-formal-shuffle2-r0'
read=lambda path:json.loads(path.read_bytes())
parent=read(control_dir/'request.json')
plan_file=Path(parent['argv'][parent['argv'].index('--plan')+1])
plan=read(plan_file)
parent_stage=read(data/'group-series-r0/summary.json')
child=read(child_dir/'request.json')
release=Path(child['cwd'])
run=read(release/'RUN_VERSION.json')
names=['generate.py','series.py','runtime.py','beauty.py','beauty_evaluate.py','statistics.py','a0_beauty_letter.json']
hashes={i:hashlib.sha256((release/'lc_rec/shuffle_v2'/i).read_bytes()).hexdigest() for i in names}
jobs=[p for p in (root/'jobs').glob('letter-beauty-v2-*') if 'gpu-formal' in p.name or 'group-series' in p.name]
controller=subprocess.check_output(['/usr/bin/python3','-I','-S','-B',plan['runner'],'query',str(control_dir)],text=True)
worker=subprocess.check_output(['/usr/bin/python3','-I','-S','-B',plan['runner'],'query',str(child_dir)],text=True)
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'plan':plan,'plan_sha256':hashlib.sha256(plan_file.read_bytes()).hexdigest(),'parent':parent,'child':child,
  'parent_stage':parent_stage,'parent_accepted':read(control_dir/'accepted.json'),
  'child_accepted':read(child_dir/'accepted.json'),'parent_query':json.loads(controller),
  'child_query':json.loads(worker),'manifest':run,'deployed_material':hashes,
  'existing_formal_jobs':sorted(p.name for p in jobs)},indent=2))
