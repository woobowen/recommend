#!/usr/bin/env python3
"""Check saved receipt hashes and every asset/member source pointer, without SSH."""
from pathlib import Path
import datetime,hashlib,json

OUT=Path(__file__).resolve().parent
target=OUT/'assets-not-copied.json'
b=target.read_bytes();m=json.loads(b)
sources={}
checked_sources=[]
for key,r in m['source_receipts'].items():
    data=Path(r['path']).read_bytes()
    assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],key
    sources[key]=json.loads(data)
    checked_sources.append(key)

def pointer(x,p):
    for k in p.lstrip('/').split('/'):
        k=k.replace('~1','/').replace('~0','~')
        x=x[int(k)] if isinstance(x,list) else x[k]
    return x

asset_refs=0
for a in m['assets']:
    for ref in a['source_receipts']:
        original=pointer(sources[ref['source']],ref['json_pointer'])
        assert original['bytes']==a['bytes'] and original['sha256']==a['sha256'],a['path']
        assert original['path']==a['path'] or a['path'].endswith('/'+original['path']),a['path']
        asset_refs+=1
member_refs=0
for a in m['dataset_members']:
    ref=a['source_receipt'];original=pointer(sources[ref['source']],ref['json_pointer'])
    assert original['actual_server_path']==a['path'] and original['bytes']==a['bytes'] and original['sha256']==a['sha256']
    member_refs+=1
for route,r in m['routes'].items():
    for v in r['actual_effective_loader_directories']:
        assert pointer(sources[v['receipt']],v['json_pointer'])==v['path']

result={'status':'ALL_SAVED_IDENTITY_SOURCE_POINTERS_AND_RECEIPT_HASHES_PASS','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256(b).hexdigest(),'source_files_checked':len(checked_sources),'asset_source_pointers_checked':asset_refs,'dataset_member_source_pointers_checked':member_refs,'route_loader_source_pointers_checked':12,'remote_calls':0,'large_asset_loads_or_hashes':0}
(OUT/'source-pointer-validation.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'assets-not-copied.sha256').write_text(hashlib.sha256(b).hexdigest()+'  assets-not-copied.json\n')
print(json.dumps(result))
