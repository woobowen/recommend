from pathlib import Path
import json,sys,shlex,os
sys.path.insert(0,'/home/addaswsw/recommend-worktrees/lc-rec/lc_rec/a0_beauty')
import closure_ops as op
import transport
kind=sys.argv[1];assert kind in ('followup','conformance','finalize','worker','dependencies','query','collect')
local=op.LOCAL;t=transport.Transport('new-server');t.profile=dict(t.profile,remote_support=op.BASE)
dep=json.loads((local/'deployment.json').read_text());commit=dep['run_commit'];release=dep['release']
old=os.environ.get('BEAUTY_PIPELINE_COMMIT','f6179273dd274f70151f003f469d321396d5c807');root=op.BASE+'/SYNTHETIC_CPU_INTEGRATION_ONLY_'+old[:12];planpath=root+'/plan.json'
if kind in ('followup','conformance','finalize','worker','dependencies'):
 meta=json.loads(t.python('import json,hashlib;from pathlib import Path;p=Path('+repr(planpath)+');print(json.dumps({"plan":json.loads(p.read_text()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}))'))
 p=meta['plan'];ns=p['namespace'];job='letter-a0-beauty-cpu-runner-check-'+kind+'-'+commit[:12]
 script={'followup':'closure_followup.py','conformance':'closure_conformance.py','finalize':'closure_finalize_check.py','worker':'closure_worker_guard.py','dependencies':'closure_dependencies.py'}[kind]
 req=dict(schema=1,**ns,capture_version=commit,cwd=release,argv=[p['python'],release+'/lc_rec/a0_beauty/'+script,planpath],env=p['env'],timeout_seconds=600,
  inputs=[{'path':x['path'],'sha256':x['sha256']} for x in dep['files']]+[{'path':planpath,'sha256':meta['sha256']}],job_id=job)
 path=local/(kind+'.request.json')
 if path.exists():
  previous=json.loads(path.read_text());transport.atomic(local/(kind+'-'+previous['capture_version'][:12]+'.request.json'),previous)
 transport.atomic(path,req);remote=op.BASE+'/'+job+'.request.json';t.put(path,remote)
 reply=json.loads(t.ssh(shlex.join(['/usr/bin/python3','-I','-S','-B',p['runner'],'submit',p['job_root'],remote]),timeout=60));transport.atomic(local/(kind+'.submit.json'),reply)
 print(json.dumps({k:reply.get(k) for k in ('job_id','status','submission')}))
else:
 request=json.loads((local/(sys.argv[2]+'.request.json')).read_text());job=request['job_id'];runner=op.BASE+'/runner-beauty-cc1a1e277faa.py';jobs=op.BASE+'/cpu-jobs'
 reply=json.loads(t.ssh(shlex.join(['/usr/bin/python3','-I','-S','-B',runner,'query',jobs+'/'+job])));transport.atomic(local/(sys.argv[2]+'.status.json'),reply)
 print(json.dumps({k:reply.get(k) for k in ('job_id','status','exit_code','cleanup_verified','worker_alive','workgroup_live')}))
 print(t.ssh(shlex.join(['tail','-n','18',jobs+'/'+job+'/work.log'])))
