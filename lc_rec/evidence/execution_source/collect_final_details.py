#!/usr/bin/env python3
import json,pathlib
import transport
root=pathlib.Path(__file__).resolve().parents[1];t=transport.Transport('new-server')
code='''import json,pathlib,subprocess,hashlib
new='/data/bwwu/envs/lc-rec-20261108/bin/python';old='/data/bwwu/envs/grid-20261009-e61baf3/bin/python'
probe="import importlib.metadata as m,json,sys,pathlib;print(json.dumps({'executable':sys.executable,'versions':{p:m.version(p) for p in ['torch','transformers','numpy','huggingface_hub','safetensors','tokenizers']},'local_distributions':[{'name':d.metadata['Name'],'version':d.version,'root':str(d.locate_file(''))} for d in m.distributions() if str(d.locate_file('')).startswith(sys.prefix+'/')]}))"
r={'environments':{}}
for label,exe in [('lc_rec',new),('old_grid',old)]:
 r['environments'][label]=json.loads(subprocess.check_output([exe,'-B','-c',probe],text=True))
inter=json.loads(pathlib.Path('/data/bwwu/lc-rec/assets/data/Instruments/Instruments.inter.json').read_text())
r['duplicate_item_user_count']=sum(len(v)!=len(set(map(str,v))) for v in inter.values())
r['event_timestamps_present']=False
r['interactions_count']=sum(map(len,inter.values()))
r['training_rows']=sum(len(v)-3 for v in inter.values())
print(json.dumps(r))
'''
result=json.loads(t.python(code,timeout=45));(root/'evidence/final_details.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
