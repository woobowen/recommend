"""Compile the reviewed fixed plan into finite independent stage requests."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, '/home/addaswsw/recommend-worktrees/sid-extensions-p0')
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from experiments.launch_20261017.build import make_request, job, GPU0, GPU1, JOB_ROOT, PYTHON
from experiments.launch_20261017.deploy import REMOTE, request
from experiments.launch_20261017.queue import GROUP_CAPS, validate
from experiments.beauty_extensions.common import sha256, write_json
from transport import Transport, atomic

parser=argparse.ArgumentParser()
parser.add_argument('--index',type=Path,required=True)
parser.add_argument('--reviews',type=Path,nargs='+',required=True)
args=parser.parse_args()
index=json.loads(args.index.read_text())
local=Path(index['local_root']);remote=index['remote_root'];t=Transport('new-server');p=t.profile
pref_ids=['ga-20261017-toys5300-gpu-gate-v1','ga-20261017-toys5000-gpu-gate-v1','ga-20261017-sportsa1-gpu-gate-v1']
code='''import json,datetime,pathlib,hashlib
root=pathlib.Path(JOB_ROOT);out={}
for job in JOBS:
 p=root/job;s=json.loads((p/'status.json').read_text());r=json.loads((p/'output/RESULT.json').read_text())
 assert s['status']=='SUCCEEDED' and s['exit_code']==0 and s['cleanup']['verified']
 assert r['status'] in ('PASS','GPU_ADMISSION_PASS')
 seconds=(datetime.datetime.fromisoformat(s['ended_at'])-datetime.datetime.fromisoformat(s['started_at'])).total_seconds()
 out[job]={'status':s,'result':r,'gpu_request_seconds':seconds,
  'result_sha256':hashlib.sha256((p/'output/RESULT.json').read_bytes()).hexdigest()}
print(json.dumps(out))
'''.replace('JOB_ROOT',repr(JOB_ROOT)).replace('JOBS',repr(pref_ids))
pref=json.loads(t.python(code));write_json(local/'real-gpu-admission-receipts.json',pref)
prior_seconds=sum(v['gpu_request_seconds'] for v in pref.values())
assert prior_seconds<1800 and prior_seconds+900<=2700
review_records=[]
for path in args.reviews:
 v=json.loads(path.read_text())
 assert 'review' in json.dumps(v).lower() and path.stat().st_size>100
 target=remote+'/reviews/'+sha256(path)+'-'+path.name
 t.put(path,target);review_records.append({'path':target,'sha256':sha256(path)})
plan=local/'plan.json';remote_plan=remote+'/plan.json';t.put(plan,remote_plan)
approval={'status':'SOURCE_AND_LAUNCH_ADMITTED_CONDITIONALLY','future_results_approved':False,
 'kind':'USER_AUTHORIZED_20261017_LAUNCH_ONLY','explicit_user_approval_reference':'User full task and subsequent safe-sharing instruction, 2026-10-17',
 'plan':{'path':remote_plan,'sha256':sha256(plan)},
 'source_python_sha256':json.loads(plan.read_text())['source_python_sha256'],
 'independent_reviews':review_records,'real_prelaunch_gpu_jobs':pref_ids,
 'future_sid_best_and_results':'Only the fixed mechanical gates may advance dependencies; later human/nonauthor result review is still required.',
 'prior_gpu_request_seconds':prior_seconds}
approval_path=local/'launch-review.json';write_json(approval_path,approval)
remote_review=remote+'/reviews/'+sha256(approval_path)+'-launch-review.json';t.put(approval_path,remote_review)
approval_record={'path':remote_review,'sha256':sha256(approval_path)}
engineering={'metadata','admission-cpu','admission-gpu','train-admission'}
for name,path in index['templates'].items():
 spec=json.loads(Path(path).read_text())
 if index['modes'][name] not in engineering:
  spec['launch_review']=approval_record;write_json(path,spec)
 # C1 preparation IDs remain external evidence, never republished/restarted.
 if name.endswith(('-metadata','-cpu-gate','-gpu-gate')) and not name.startswith('sportsa2'):
  continue
 t.put(path,index['template_remote_paths'][name])

config={'schema':1,'ready':True,'queue_id':'ga-20261017-queue-v1',
 'runner':{'python':p['python'],'script':p['runner'],'script_sha256':p['runner_sha256'],
  'job_root':p['job_root'],'deployment_sha256':p['deployment_sha256']},
 'namespace_identity':{k:p['hostname'] if k=='host' else p[k] for k in ('host','user','uid','namespace')},
 'snapshot':{'root':index['source_deployment']['root'],'commit':index['source_deployment']['manifest']['source_commit']},
 'state_dir':remote+'/queue-state','gpu_locks_dir':REMOTE+'/gpu-locks','poll_seconds':30,'max_queue_seconds':172700,
 'free_space':[{'path':'/home/bwwu','min_free_bytes':20*1024**3},{'path':'/data/bwwu','min_free_bytes':20*1024**3}],
 'storage_roots':[REMOTE], 'max_storage_bytes':50*1024**3,
 'gpu_budget_seconds':int(math.floor(143100-prior_seconds)),
 'budget_groups':{**GROUP_CAPS,'engineering':int(math.floor(2700-prior_seconds))},
 'prior_gpu_request_usage':{'seconds':prior_seconds,'jobs':pref_ids,'counting':'actual original worker wall; all requests one GPU'},
 'blocked_workflows':{},'stages':[]}
prior_jobs=['ga-20261017-assets-prepare-v1','ga-20261017-sportsa1-cpu-gate-v2']
prior_jobs += ['ga-20261017-'+n+'-metadata-v1' for n in ('toys5300','toys5000','sportsa1')]
prior_jobs += ['ga-20261017-'+n+'-cpu-gate-v1' for n in ('toys5300','toys5000','sportsa1')]
prior_jobs += pref_ids
config['storage_roots'] += [JOB_ROOT+'/'+j for j in prior_jobs]

def external_gate(name,expected):
 j=job(name)
 result_gate={'path':JOB_ROOT+'/'+j+'/output/RESULT.json','json_equals':expected}
 if j in pref:result_gate['sha256']=pref[j]['result_sha256']
 return [{'path':JOB_ROOT+'/'+j+'/status.json','json_equals':{'status':'SUCCEEDED','exit_code':0,'cleanup.verified':True}},
  result_gate]

def add(name,kind,workflow,timeout,devices=(),deps=(),group=None,gates=(),share=False):
 stage={'id':name,'workflow':workflow,'kind':kind,'depends_on':list(deps),'budget_group':group,
  'request':make_request(t,index,name,timeout,','.join(devices)),'gates':list(gates)}
 if devices:stage['gpu_uuids']=list(devices)
 if share:stage['resource_policy']={'allow_shared':True,'min_free_mib':8192,'max_peer_memory_mib':2048,
  'max_prelaunch_utilization_percent':30,'sample_seconds':2,'required_samples':2}
 config['stages'].append(stage)
 config['storage_roots'].append(JOB_ROOT+'/'+job(name))

add('sportsa1-train','train','sports-a1',43200,[GPU0],group='sports-a1-train',
 gates=external_gate('sportsa1-gpu-gate',{'status':'GPU_ADMISSION_PASS','gpu_capacity_verified':True}))
add('sportsa2-encode','encode','sports-a2',10800,[GPU1],group='sports-a2-encode',share=True)
add('sportsa2-quantize','rkmeans','sports-a2',3600,[GPU0,GPU1],['sportsa2-encode'],'sports-a2-rkmeans',
 external_gate('sportsa2-encode',{'status':'SUCCEEDED','stage':'t5_encode','items':18357,'frozen_t5':True}))
add('sportsa2-metadata','cpu','sports-a2',1800,deps=['sportsa2-quantize'],
 gates=external_gate('sportsa2-quantize',{'status':'SUCCEEDED','shape':[4,18357],'quantization_layers':3,'world_size':2,'global_step':30}))
add('sportsa2-gpu-gate','admission','sports-a2',900,[GPU0],['sportsa2-metadata'],'engineering')
add('sportsa2-train','train','sports-a2',43200,[GPU0],['sportsa2-gpu-gate'],'sports-a2-train',
 external_gate('sportsa2-gpu-gate',{'status':'GPU_ADMISSION_PASS','gpu_capacity_verified':True}))
for name,w in (('toys5300','toys-a-5300'),('toys5000','toys-a-5000'),('sportsa1','sports-a1'),('sportsa2','sports-a2')):
 group={'toys5300':'toys-5300','toys5000':'toys-5000','sportsa1':'sports-a1-eval-attention','sportsa2':'sports-a2-eval-attention'}[name]
 deps=[name+'-train'] if name.startswith('sports') else []
 gates=external_gate(name+'-gpu-gate',{'status':'PASS','strict_checkpoint':True}) if name.startswith('toys') else []
 for n in (0,3,4):add(name+'-eval-'+str(n),'evaluate',w,1800,[GPU1],deps,group,gates)
 all_eval=[name+'-eval-'+str(n) for n in (0,3,4)]
 for n in (0,3,4):add(name+'-attention-'+str(n),'attention',w,1200,[GPU1],all_eval,group,
  external_gate(name+'-eval-'+str(n),{'status':'COMPLETE','formal':True,'test_complete':True,'shuffle_n':n}))
 add(name+'-stats','cpu',w,600,deps=[name+'-attention-'+str(n) for n in (0,3,4)])
assert len(config['stages'])==34
config_path=local/'queue-config.json';write_json(config_path,config)
summary=validate(config,config_path)
write_json(local/'queue-validation.json',summary)
remote_config=remote+'/queue-config-'+sha256(config_path)[:16]+'.json';t.put(config_path,remote_config)
render=[p['python'],'-I','-S','-B',index['source_deployment']['root']+'/experiments/launch_20261017/queue.py',
 'render-requests','--config',remote_config]
wrapped=json.loads(t.python('import subprocess;print(subprocess.check_output('+repr(render)+',text=True))'))
write_json(local/'queue-wrapped-requests.json',wrapped)
metadata=json.loads(t.python('from pathlib import Path;print(Path('+repr(p['runner']+'.deployment.json')+').read_text())'))
sidecar=local/'queue-runner.deployment.json';write_json(sidecar,metadata)
assert sha256(sidecar)==p['deployment_sha256']
runner_local=Path('/home/addaswsw/recommend-worktrees/sid-extensions-p0/experiments/launch_20261017/queue_runner.py')
outer=REMOTE+'/runners/queue-runner-'+sha256(runner_local)[:16]+'.py'
t.put(sidecar,outer+'.deployment.json');t.put(runner_local,outer)
outer_request=request(t,index['source_deployment'],'ga-20261017-queue-v1',
 [PYTHON,'-B','-m','experiments.launch_20261017.queue','supervise','--config',remote_config],172800,
 [{'path':remote_config,'sha256':sha256(config_path)}, {'path':outer,'sha256':sha256(runner_local)},
  {'path':outer+'.deployment.json','sha256':sha256(sidecar)},approval_record])
write_json(local/'queue-supervisor-request.json',outer_request)
write_json(local/'queue-deployment.json',{'outer_runner':outer,'outer_runner_sha256':sha256(runner_local),
 'remote_config':remote_config,'config_sha256':sha256(config_path),'summary':summary,
 'source_commit':config['snapshot']['commit'],'prior_gpu_seconds':prior_seconds})
print(json.dumps({'config':str(config_path),'remote':remote_config,'summary':summary,'prior_gpu_seconds':prior_seconds}))
