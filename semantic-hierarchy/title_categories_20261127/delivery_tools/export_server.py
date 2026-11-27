"""CPU-only publication export; original experiments remain read-only."""
from pathlib import Path
import hashlib,json,zipfile,time
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
CONTROL=ROOT/'codex/title_categories_hierarchy_20261127'
RUN=CONTROL/'runs/20261127_014125'
FAILED=CONTROL/'runs/20261127_014041'
ALIGN=ROOT/'codex/instruments_letter_alignment_20261127'
OUT=ROOT/'codex/fusion_publication_20261127'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
def pack(name,files):
 dest=OUT/name;members=[]
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for rel,p in sorted(files.items()):
   s=dict(member=rel,source=str(p),bytes=p.stat().st_size,sha256=sha(p));z.write(p,rel);members.append(s)
  z.writestr('MANIFEST.json',json.dumps(members,indent=2))
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
 s=dict(name=name,bytes=dest.stat().st_size,sha256=sha(dest),members=members);write(OUT/(name+'.json'),s)
 print(json.dumps({k:v for k,v in s.items() if k!='members'}),flush=True)
 return s
def main():
 OUT.mkdir(exist_ok=True);assert json.loads((RUN/'status.json').read_text())['returncode']==0
 launch=json.loads((RUN/'launch.json').read_text())
 for n,h in launch['source_sha256'].items():assert sha(RUN/'source'/n)==h,n
 files={};vectors={d:{} for d in ['beauty','instruments']}
 for p in RUN.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.lock']:continue
  rel=p.relative_to(RUN).as_posix()
  if p.suffix=='.npy' and rel.split('/')[0] in vectors:vectors[rel.split('/')[0]]['run/'+rel]=p
  else:files['run/'+rel]=p
 for p in FAILED.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.json','.py','.md','.log','.csv']:
   files['attempts/failed_014041/'+p.relative_to(FAILED).as_posix()]=p
 for n in ['启动记录.md','launch_runtime_review.md','launch_review.md','review.md','inspect_run.py']:
  files['history/'+n]=CONTROL/n
 # Sufficient original identity chain, not merely a claim that counts match.
 for n in ['README.md','align_instruments.py','materialize_server.py','fetch_author_mapping.py','author_download_receipt.json','server_verification.json','final_verification.json','unresolved_diagnostic.json']:
  p=ALIGN/n
  if p.is_file():files['alignment/'+n]=p
 for sub in ['author_mapping','aligned']:
  for p in (ALIGN/sub).iterdir():
   if p.is_file() and p.name!='qwen_original_metadata_aligned.jsonl':files['alignment/'+sub+'/'+p.name]=p
 files['identity/beauty_mapping_audit.json']=ROOT/'codex/letter_upstream_transformer_20261115/inputs/beauty/mapping_audit.json'
 # Actual encoder model helper files and adapters; full base weights obtained from fixed external source.
 for name in ['Meta-Llama-3-8B-Instruct','LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised']:
  for p in (ROOT/'models'/name).iterdir():
   if p.is_file() and (p.suffix in ['.json','.py','.md'] or p.name in ['LICENSE','USE_POLICY.md','adapter_model.safetensors']):
    if p.name.startswith('model-'):continue
    files['encoder_assets/models/'+name+'/'+p.name]=p
 original=ROOT/'codex/llm2vec_sid_hierarchy_20261113'
 files['encoder_assets/original_experiment_config.json']=original/'experiment_config.json'
 model=json.loads((RUN/'model_verified.json').read_text())
 for d,rows in vectors.items():
  enc=json.loads((RUN/d/'encoding_complete.json').read_text())
  for rel,p in rows.items():assert sha(p)==enc['outputs'][p.name],rel
 pack('fusion-metadata-originals.zip',files)
 for d,rows in vectors.items():pack('fusion-'+d+'-vectors.zip',rows)
 write(OUT/'export_receipt.json',dict(run=str(RUN),failed_attempt=str(FAILED),time_unix=time.time(),frozen_sources_verified=len(launch['source_sha256']),
  original_files_modified=False,no_model_load=True,no_gpu=True,model_base_weights='not rehosted; fixed original hash manifest supplied',
  alignment_large_metadata='264MB original Qwen metadata omitted: exact ASIN mappings and actual encoded text supplied; server original retained'))
if __name__=='__main__':main()
