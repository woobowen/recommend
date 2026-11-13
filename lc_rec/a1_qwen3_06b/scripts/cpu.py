"""Current asset and tokenizer gate, without loading any GPU model."""
import argparse, hashlib, importlib.metadata, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'qwen_preflight_20261110/scripts'))
from common import sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--output',required=True);p.add_argument('--run-commit',required=True);a=p.parse_args()
 root=Path(a.release);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 model=Path('/data/bwwu/lc-rec/qwen-preflight-20261110/models/Qwen3-0.6B-Base/da87bfb608c14b7cf20ba1ce41287e8de496c0cd')
 expected={'torch':'2.6.0','transformers':'4.57.6','peft':'0.21.2','accelerate':'1.12.0','deepspeed':'0.16.9','bitsandbytes':'0.45.5'}
 versions={k:importlib.metadata.version(k) for k in expected}
 assert all(versions[k].split('+')[0]==v for k,v in expected.items()),versions
 r={'RUN_COMMIT':a.run_commit,'argv':sys.argv,'python':sys.executable,'cwd':os.getcwd(),'versions':versions,'model_revision':model.name,'files':{}}
 for f in sorted(model.iterdir()):
  if f.is_file():r['files'][f.name]={'bytes':f.stat().st_size,'sha256':sha(f)}
 assert r['files']['model.safetensors']=={'bytes':1192135096,'sha256':'cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba'}
 cfg=json.loads((model/'config.json').read_text());assert cfg['model_type']=='qwen3' and cfg['tie_word_embeddings'] and cfg['num_hidden_layers']==28
 r['model_config']=cfg
 cmd=[sys.executable,str(root/'lc_rec/qwen_preflight_20261110/scripts/cpu_audit.py'),'--model',str(model),'--key','q06','--letter',str(root/'lc_rec/letter/vendor/LETTER'),'--output',str(out/'tokenizer_audit'),'--run-commit',a.run_commit]
 r['tokenizer_audit_argv']=cmd;start=time.monotonic();subprocess.run(cmd,check=True);r['tokenizer_audit_seconds']=time.monotonic()-start
 a0=Path('/data/bwwu/lc-rec/a0-final-audit-20261113/letter-a0-20261112-final-audit-20261113-offline-r1/primary')
 r['a0_pair_assets']={}
 for name,digest in [('per_user.csv','423309c6765395e42fc6a18a43706a4653d71692b184c65fe5755f78a34df6fb'),('per_user_metrics.npz','d46590f04dcd0f8438b67ccea636b3310a61ff081e146305032c614f444a1689')]:
  f=a0/name;actual=sha(f);assert actual==digest;r['a0_pair_assets'][name]={'path':str(f),'bytes':f.stat().st_size,'sha256':actual}
 r['status']='PASS';write(out/'assets_environment.json',r)
 print(json.dumps({'stage':'cpu','status':'PASS','versions':versions}))
if __name__=='__main__':main()
