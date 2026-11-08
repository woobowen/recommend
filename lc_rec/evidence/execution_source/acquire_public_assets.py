#!/usr/bin/env python3
"""Bounded public-source relay preparation; no weights and no private credentials."""
import hashlib,json,pathlib,urllib.request,zipfile,time
root=pathlib.Path(__file__).resolve().parents[1];target=root/'public-relay';target.mkdir(exist_ok=True)
events=[];rev='4782ad278652c7c71b72204d462d6d01eaaf7549'
urls=[('Instruments.zip','https://drive.google.com/uc?export=download&id=19tsIxV_LbErh5FfYhT732M82y0V-3bxR',13612638)]
urls += [('llama-7b/'+rev+'/'+n,'https://huggingface.co/huggyllama/llama-7b/resolve/'+rev+'/'+n,None) for n in ['config.json','generation_config.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','LICENSE','README.md']]
for name,url,size in urls:
 p=target/name;p.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 try:
  if not p.exists():
   with urllib.request.urlopen(url,timeout=45) as r:
    b=r.read(32*1024**2+1)
    assert 'text/html' not in r.headers.get('Content-Type',''), 'HTML response'
   assert len(b)<32*1024**2
   if size is not None: assert len(b)==size
   p.write_bytes(b)
  h=hashlib.sha256(p.read_bytes()).hexdigest()
  if p.name=='tokenizer.model': assert h=='9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347'
  if p.suffix=='.zip':
   with zipfile.ZipFile(p) as z: assert z.testzip() is None
  events.append({'path':name,'url':url,'bytes':p.stat().st_size,'sha256':h,'elapsed_seconds':time.monotonic()-start,'status':'DOWNLOADED_OR_REUSED'})
 except Exception as e:
  events.append({'path':name,'url':url,'status':'FAILED','error':repr(e)});raise
 finally:(root/'evidence/public_relay.json').write_text(json.dumps(events,indent=2)+'\n')
print(json.dumps({'files':len(events),'bytes':sum(r.get('bytes',0) for r in events)}))
