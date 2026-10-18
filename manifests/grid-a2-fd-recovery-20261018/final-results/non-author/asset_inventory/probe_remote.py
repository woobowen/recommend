#!/usr/bin/env python3
"""One exact-path stat and small metadata probe; no recursive walk/model/GPU calls."""
from pathlib import Path
import hashlib, json, sys

OUT=Path(__file__).resolve().parent
request=json.loads((OUT/'read-only-probe-request.json').read_text())
code='''from pathlib import Path
import datetime,hashlib,importlib.metadata,json,os,stat,sys
request=REQUEST
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'new-server','scope':'Exact saved paths only; no recursive traversal. Large assets stat only; only two explicitly bounded metadata/executable files may be hashed.','paths':[],'small_file_hashes':[],'current_versions':{},'remote_file_writes':0,'GPU_or_model_imports':0,'recursive_scans':0,'probe_python':sys.executable}
for name in request['paths']:
 p=Path(name)
 try:
  ls=p.lstat();s=p.stat()
  out['paths'].append({'path':name,'exists':True,'is_symlink':stat.S_ISLNK(ls.st_mode),'resolved':str(p.resolve()),'regular':stat.S_ISREG(s.st_mode),'directory':stat.S_ISDIR(s.st_mode),'bytes':s.st_size,'mode':oct(stat.S_IMODE(s.st_mode)),'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino})
 except OSError as e:out['paths'].append({'path':name,'exists':False,'error':str(e)})
for name in request['hash_small_files_only']:
 p=Path(name)
 if not p.is_file():
  out['small_file_hashes'].append({'path':name,'content_hashed':False,'reason':'absent'});continue
 size=p.stat().st_size
 if size>request['per_file_hash_upper_bytes']:
  out['small_file_hashes'].append({'path':name,'content_hashed':False,'bytes':size,'reason':'exceeds explicit bound'});continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 out['small_file_hashes'].append({'path':name,'content_hashed':True,'bytes':size,'sha256':h.hexdigest()})
def norm(s):return s.lower().replace('_','-')
target={norm(n):n for n in request['current_version_probe_packages']}
site='/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages'
out['version_metadata_site']=site
for d in importlib.metadata.distributions(path=[site]):
 n=d.metadata.get('Name','')
 if norm(n) in target:
  out['current_versions'][target[norm(n)]]={'version':d.version,'method':'Read distribution METADATA only; no corresponding package import.'}
out['missing_version_metadata']=sorted(set(request['current_version_probe_packages'])-set(out['current_versions']))
print(json.dumps(out))
'''.replace('REQUEST',repr(request),1)
sys.path.insert(0,'/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport
raw=json.loads(Transport('new-server').python(code,timeout=45))
raw['remote_script_sha256']=hashlib.sha256(code.encode()).hexdigest()
raw['request_file_sha256']=hashlib.sha256((OUT/'read-only-probe-request.json').read_bytes()).hexdigest()
(OUT/'targeted-remote-stat.json').write_text(json.dumps(raw,indent=2)+'\n')
print(json.dumps({'paths_statted':len(raw['paths']),'missing_paths':[a['path'] for a in raw['paths'] if not a['exists']],'small_file_hashes':raw['small_file_hashes'],'current_versions':raw['current_versions'],'missing_version_metadata':raw['missing_version_metadata']}))
