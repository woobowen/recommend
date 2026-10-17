"""Prepare reviewed immutable source/config; submission is a separate action."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,'/home/addaswsw/recommend-worktrees/sid-extensions-p0')
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport,atomic,sha
from experiments.launch_20261017 import deploy
from experiments.a2_fd_recovery_20261018 import build

BASE=Path(__file__).resolve().parent
REMOTE='/home/bwwu/grid-audits/sid-extensions-p0/grid-a2-fd-recovery-20261018'
ROOT=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0')
p=argparse.ArgumentParser();p.add_argument('mode',choices=['deploy','specs','finalize'])
a=p.parse_args();t=Transport('new-server');t.identity()
scene=json.loads((BASE/'initial-scene.json').read_text())
scene['shared_config_path']='/home/bwwu/grid-audits/sid-extensions-p0/grid-a-launch-20261017/shared-20261018/compiled/queue-config-d5528b2c185395bd.json'
if a.mode=='deploy':
 deploy.SOURCE_DIRS=[*deploy.SOURCE_DIRS,'experiments/a2_fd_recovery_20261018']
 deployment=deploy.snapshot(t);atomic(BASE/'source-deployment.json',deployment)
 print(json.dumps(dict(root=deployment['root'],commit=deployment['manifest']['source_commit'],receipt=deployment['receipt']),indent=2))
 raise SystemExit(0)
deployment=json.loads((BASE/'source-deployment.json').read_text())
compiled=BASE/('compiled-'+deployment['manifest']['snapshot_sha256'][:16]);compiled.mkdir(exist_ok=True)
remote=REMOTE+'/compiled/'+deployment['manifest']['snapshot_sha256'][:16]
if a.mode=='specs':
 templates,plan=build.specs(deployment,scene,compiled,remote)
 atomic(BASE/'compiled-location.json',dict(local=str(compiled),remote=remote))
 t.put(compiled/'plan.json',remote+'/plan.json')
 print(json.dumps(dict(local=str(compiled),remote=remote,templates=len(templates),plan_sha256=sha(compiled/'plan.json')),indent=2))
 raise SystemExit(0)
review=BASE/'reviews/fd/launch-review-v2.json'
rv=json.loads(review.read_text());assert rv['fd_engineering_status']=='FD_ENGINEERING_EQUIVALENCE_PASS'
assert rv['plan']==dict(path=remote+'/plan.json',sha256=sha(compiled/'plan.json'))
t.put(review,remote+'/launch-review.json')
shared=scene['config']
paths=[scene['shared_config_path'],shared['state_dir']+'/queue-status.json',
       shared['continuation']['old_config_path'],shared['continuation']['old_queue_state_path']]
J=shared['runner']['job_root']
for job in ['ga-20261017-queue-v1','ga-20261018-shared-queue-v1','ga-20261017-sportsa2-encode-v1',
            'ga-20261017-sportsa2-quantize-v1','ga-20261018-a2-fd-cpu-equivalence-v1']:
 paths += [J+'/'+job+'/'+name for name in ['request.json','status.json']]
paths += [J+'/ga-20261018-a2-fd-cpu-equivalence-v1/output/equivalence.json',
          J+'/ga-20261018-a2-fd-cpu-equivalence-v1/output/config-equivalence.json',remote+'/launch-review.json']
frozen=json.loads(t.python('import pathlib,hashlib,json; print(json.dumps([dict(path=s,bytes=pathlib.Path(s).stat().st_size,sha256=hashlib.sha256(pathlib.Path(s).read_bytes()).hexdigest()) for s in '+repr(paths)+']))'))
atomic(compiled/'frozen-evidence.json',frozen)
config=build.compile_queue(deployment,scene,compiled,remote,review,frozen)
for path in sorted((compiled/'specs').glob('*.json')):t.put(path,remote+'/specs/'+path.name)
remote_config=remote+'/queue-config-'+sha(compiled/'queue-config.json')[:16]+'.json'
t.put(compiled/'queue-config.json',remote_config)
argv=['/data/bwwu/envs/grid-20261009-e61baf3/bin/python','-B','-m','experiments.a2_fd_recovery_20261018.controller']
prefix=shlex.join(['env','PYTHONPATH='+deployment['root'],*argv])
validation=json.loads(t.ssh(prefix+' validate --config '+shlex.quote(remote_config),timeout=60))
atomic(compiled/'queue-validation-server.json',validation)
wrapped=json.loads(t.ssh(prefix+' render-requests --config '+shlex.quote(remote_config),timeout=60))
atomic(compiled/'wrapped-requests.json',wrapped)
oldouter=scene['runner_statuses']['ga-20261018-shared-queue-v1']['supervisor']['argv'][4]
sidecar=compiled/'outer-runner.deployment.json'
sidecar.write_text(t.python('import pathlib;print(pathlib.Path('+repr(oldouter+'.deployment.json')+').read_text(),end="")'))
assert sha(sidecar)==t.profile['deployment_sha256']
runner_source=ROOT/'experiments/launch_20261017/queue_runner.py'
outer=REMOTE+'/runners/a2-queue-runner-'+sha(runner_source)[:16]+'.py'
t.put(sidecar,outer+'.deployment.json');t.put(runner_source,outer)
outer_meta=dict(script_sha256=sha(runner_source),deployment_sha256=sha(sidecar))
absolute=dt.datetime.fromisoformat(config['recovery']['original_absolute_queue_deadline'])
timeout=int((absolute-dt.datetime.now(dt.timezone.utc)).total_seconds())
request=deploy.request(t,deployment,config['queue_id'],[*argv,'supervise','--config',remote_config],timeout,
    inputs=[dict(path=remote_config,sha256=sha(compiled/'queue-config.json')),
            dict(path=outer,sha256=outer_meta['script_sha256']),
            dict(path=outer+'.deployment.json',sha256=outer_meta['deployment_sha256']),
            dict(path=remote+'/launch-review.json',sha256=sha(review))])
atomic(compiled/'supervisor-request.json',request)
receipt=dict(remote_config=remote_config,config_sha256=sha(compiled/'queue-config.json'),
             outer_runner=outer,outer_runner_sha256=outer_meta['script_sha256'],
             source_commit=deployment['manifest']['source_commit'],snapshot=deployment['root'],
             local_root=str(compiled),remote_root=remote,validation=validation,
             original_absolute_queue_deadline=absolute.isoformat(),submitted=False)
atomic(BASE/'queue-deployment.json',receipt)
print(json.dumps(receipt,indent=2))
