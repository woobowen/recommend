"""One-shot task client. Reuses accepted Windows transport and remote runner."""
import argparse, hashlib, json, shlex, subprocess, sys, tarfile
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
TASK=REPO/'lc_rec/qwen_preflight_20261110'
SUPPORT=Path('/home/addaswsw/recommend-support/qwen-preflight-20261110')
REMOTE='/home/bwwu/lc-rec/qwen-preflight-20261110'
DATA='/data/bwwu/lc-rec/qwen-preflight-20261110'
LETTER='/home/bwwu/lc-rec/letter-20261110/releases/8e6399c2a1269dfde58d705eef25a499400dc59b/lc_rec/letter/vendor/LETTER'
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
import transport, gridctl
transport.Transport.python=lambda self,code,timeout=45:self.ssh(shlex.join(['python3','-I','-S','-B','-c',code]),timeout=timeout)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','deploy','submit','query','logs','collect']);p.add_argument('--commit');p.add_argument('--job');p.add_argument('--spec');a=p.parse_args()
 assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='wbw-lc-rec'
 t=transport.Transport('new-server');SUPPORT.mkdir(parents=True,exist_ok=True)
 if a.mode=='preflight':
  code="""import json,subprocess,time,hashlib;from pathlib import Path
r={'observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'memory':Path('/proc/meminfo').read_text(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True),'disk':subprocess.check_output(['df','-B1','/home/bwwu','/data/bwwu'],text=True)}
r['runner_sha256']=hashlib.sha256(Path('/home/bwwu/grid-audits/sid-extensions-p0/scripts/runner-398768a732bbfebd.py').read_bytes()).hexdigest()
print(json.dumps(r))"""
  r=json.loads(t.python(code));assert r['runner_sha256']==t.profile['runner_sha256'];transport.atomic(TASK/'evidence/preflight.json',r)
 elif a.mode=='deploy':
  commit=subprocess.check_output(['git','rev-parse',a.commit or 'HEAD'],cwd=REPO,text=True).strip()
  archive=SUPPORT/(commit+'.tar.gz');subprocess.run(['git','archive','--format=tar.gz','--output='+str(archive),commit,'lc_rec/qwen_preflight_20261110'],cwd=REPO,check=True)
  upload=t.profile['remote_support']+'/transfers/qwen-preflight-'+commit+'.tar.gz';sent=t.put(archive,upload);release=REMOTE+'/releases/'+commit
  code=f"""import json,hashlib,tarfile;from pathlib import Path,PurePosixPath
root=Path({release!r});archive=Path({upload!r});assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sent['sha256']!r}
rows=[]
with tarfile.open(archive) as tf:
 members=tf.getmembers();assert len(members)==len(set(m.name for m in members))
 for m in members:assert not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts and (m.isfile() or m.isdir())
 if not root.exists():root.mkdir(parents=True);tf.extractall(root,filter='data')
 for m in members:
  if m.isfile():
   data=tf.extractfile(m).read();p=root/m.name;assert not p.is_symlink() and p.read_bytes()==data
   rows.append({{'path':m.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}})
r={{'RUN_COMMIT':{commit!r},'release':str(root),'not_a_git_checkout':True,'archive_sha256':{sent['sha256']!r},'files':rows}}
p=root/'RUN_VERSION.json'
if p.exists():assert json.loads(p.read_text())==r
else:p.write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r))"""
  r=json.loads(t.python(code,timeout=90));transport.atomic(SUPPORT/('deployment-'+commit+'.json'),r)
 else:
  assert a.job and a.job.startswith('qwen-20261110-');gridctl.valid(a.job)
  if a.mode=='submit':
   assert a.spec and a.commit;commit=subprocess.check_output(['git','rev-parse',a.commit],cwd=REPO,text=True).strip()
   release=REMOTE+'/releases/'+commit;task=release+'/lc_rec/qwen_preflight_20261110';jobdir=t.profile['job_root']+'/'+a.job
   spec=json.loads(Path(a.spec).read_text());replace={'RELEASE':release,'TASK':task,'DATA':DATA,'REMOTE':REMOTE,'JOB':jobdir,'COMMIT':commit,'LETTER':LETTER}
   def expand(s):
    for k,v in replace.items():s=s.replace('{'+k+'}',v)
    return s
   old=json.loads((REPO/'lc_rec/letter/evidence/autograd_20261110/jobs/letter-20261110-autograd-gpu-r0/request.json').read_text())
   req={k:old[k] for k in ['schema','host','user','uid','namespace','deployment_sha256']}
   dep=json.loads((SUPPORT/('deployment-'+commit+'.json')).read_text())
   inputs=[{'path':release+'/'+x['path'],'sha256':x['sha256']} for x in dep['files'] if '/scripts/' in x['path'] or '/config/' in x['path'] or '/metadata/' in x['path']]
   req.update(job_id=a.job,capture_version=commit,cwd=release,argv=[expand(s) for s in spec['argv']],env={k:expand(v) for k,v in spec['env'].items()},inputs=inputs,timeout_seconds=spec['timeout_seconds'])
   path=SUPPORT/'requests'/(a.job+'.json')
   if path.exists():assert json.loads(path.read_text())==req
   else:transport.atomic(path,req)
   r=gridctl.submit(t,path)
  elif a.mode in ['query','logs']:
   r=gridctl.query(t,a.job)
   if a.mode=='logs':print(t.ssh(shlex.join(['tail','-n','45',t.profile['job_root']+'/'+a.job+'/work.log'])))
  else:
   r=gridctl.query(t,a.job);assert r['cleanup_verified'] and not r['worker_alive'] and not r['workgroup_live']
   job=t.profile['job_root']+'/'+a.job;dest=REMOTE+'/light_exports/'+a.job+'.tar.gz'
   code=f"""import json,hashlib,tarfile;from pathlib import Path
root=Path({job!r});archive=Path({dest!r});rows=[];excluded=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.is_symlink() or p.name.endswith(('.lock','.partial','.tmp')):continue
 rel=str(p.relative_to(root))
 if p.stat().st_size>8*1024**2 or p.suffix in ['.bin','.safetensors','.pt','.pth','.gz'] or p.name in ['tokenizer.json','vocab.json','merges.txt']:
  excluded.append({{'path':rel,'bytes':p.stat().st_size}});continue
 if rel!='light-export-manifest.json':rows.append({{'path':rel,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}})
manifest=root/'light-export-manifest.json';manifest.write_text(json.dumps({{'files':rows,'excluded':excluded}},indent=2)+'\\n');archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'w:gz') as tf:
 for row in rows:tf.add(root/row['path'],arcname=row['path'],recursive=False)
 tf.add(manifest,arcname=manifest.name,recursive=False)
print(json.dumps({{'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}}))"""
   meta=json.loads(t.python(code,timeout=90));local=SUPPORT/(a.job+'.tar.gz');t.get_verified(meta['path'],local,meta,timeout=90)
   dest=TASK/'evidence/jobs'/a.job;dest.mkdir(parents=True,exist_ok=True)
   with tarfile.open(local) as tf:
    names=[m.name for m in tf.getmembers()];assert len(names)==len(set(names))
    for m in tf.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
    tf.extractall(dest,filter='data')
   manifest=json.loads((dest/'light-export-manifest.json').read_text());assert set(names)=={x['path'] for x in manifest['files']}|{'light-export-manifest.json'}
   for x in manifest['files']:
    p=dest/x['path'];assert p.stat().st_size==x['bytes'] and transport.sha(p)==x['sha256']
   transport.atomic(dest/'controller-query.json',r)
  transport.atomic(SUPPORT/(a.mode+'-'+a.job+'.json'),r)
 keys=['RUN_COMMIT','release','not_a_git_checkout','archive_sha256','job_id','status','exit_code','cleanup_verified','worker_alive','supervisor_alive','workgroup_live','request_sha256','submission','observed_utc','runner_sha256']
 print(json.dumps({k:r[k] for k in keys if k in r},indent=2))
if __name__=='__main__':main()
