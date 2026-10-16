import json,sys,subprocess
from pathlib import Path
sys.path.insert(0, '/home/addaswsw/recommend-support/sid-extensions-p0/scripts')
from transport import Transport,SSH
base=Path('/home/addaswsw/recommend-support/grid-formal-20261015/collected-9a64fd31218d0c68')
entities={}
for domain in ['toys','sports']:
 output=base/f'gf-20261015-{domain}-mlp-formal-v1'/'output'
 result=json.loads((output/'training_result.json').read_text())
 entities[domain]={'best':result['best_checkpoint'],'recovery':result['final_recovery']['checkpoint'],'rng':result['final_recovery']['rng_files']}
remote=r"""import json,hashlib,datetime,stat
from pathlib import Path
entities=ENTITIES_PLACEHOLDER
result={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entities':{}}
for domain,records in entities.items():
 out={}
 for kind, recs in records.items():
  rows=[]
  for record in (recs if isinstance(recs,list) else [recs]):
   p=Path(record['path'])
   assert not p.is_symlink()
   before=p.stat();assert stat.S_ISREG(before.st_mode)
   h=hashlib.sha256()
   with p.open('rb') as stream:
    prefix=stream.read(4);h.update(prefix)
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
   after=p.stat()
   assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
   assert h.hexdigest()==record['sha256']
   if 'bytes' in record:assert before.st_size==record['bytes']
   rows.append({'path':str(p),'bytes':before.st_size,'sha256':h.hexdigest(),'mtime_ns':before.st_mtime_ns,'inode':before.st_ino,'header_hex':prefix.hex(),'regular_file':True,'symlink':False,'unchanged_during_read':True})
  out[kind]=rows
 result['entities'][domain]=out
print(json.dumps(result))
""".replace('ENTITIES_PLACEHOLDER',repr(entities))
t=Transport('new-server')
r=subprocess.run([SSH,*t.options,'-T',t.host,'python3 -I -S -B -'],input=remote.encode(),capture_output=True,timeout=45)
if r.returncode:print(r.stderr.decode(),file=sys.stderr);sys.exit(r.returncode)
x=json.loads(r.stdout);target=Path(__file__).parent/'checkpoint-entity-evidence.json';target.write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(x))
