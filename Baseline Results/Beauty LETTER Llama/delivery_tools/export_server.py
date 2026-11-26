"""Read-only export of completed Beauty LETTER-LC-Rec training and predictions."""
from pathlib import Path
import hashlib,json,zipfile,time
ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
RUN=ROOT/'experiments/letter_beauty_seqrec_batch64_20261122_155523'
PREP=ROOT/'codex/letter_lc_rec_smoke_20261121'
OUT=ROOT/'codex/beauty_llama_publication_20261126'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def pack(name,files):
 dest=OUT/name;members=[]
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for member,p in sorted(files.items()):
   row=dict(member=member,source=str(p),bytes=p.stat().st_size,sha256=sha(p));z.write(p,member);members.append(row)
  z.writestr('MANIFEST.json',json.dumps(members,indent=2))
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
 spec=dict(name=name,bytes=dest.stat().st_size,sha256=sha(dest),members=members);write(OUT/(name+'.json'),spec)
 return {k:v for k,v in spec.items() if k!='members'}
def main():
 OUT.mkdir(exist_ok=True)
 assert json.loads((RUN/'Beauty/progress.json').read_text())['complete']
 launch=json.loads((RUN/'launch.json').read_text())
 for p,h in launch['source_sha256'].items():assert sha(RUN/p)==h,p
 result=json.loads((RUN/'Beauty/training/training_result.json').read_text());best=Path(result['best_checkpoint'])
 adapter=best/'adapter_model.safetensors';assert sha(adapter)==result['best_adapter_sha256']
 metadata={};predictions={}
 for p in RUN.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.lock']:continue
  rel=p.relative_to(RUN).as_posix()
  if '/checkpoints/' in rel:continue
  (predictions if '/testing/' in '/'+rel and p.name.startswith('part-') else metadata)['run/'+rel]=p
 for p in best.iterdir():
  if p.is_file() and p.name not in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','cuda_all_rng.pt','training_args.bin']:
   metadata['checkpoint/'+p.name]=p
 for p in (PREP/'vendor/LETTER/LETTER-LC-Rec').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:metadata['vendor/LETTER/LETTER-LC-Rec/'+p.relative_to(PREP/'vendor/LETTER/LETTER-LC-Rec').as_posix()]=p
 for p in (PREP/'vendor/LETTER/data/Beauty').iterdir():
  if p.is_file():metadata['vendor/LETTER/data/Beauty/'+p.name]=p
 for name in ['LICENSE','README.md']:
  p=PREP/'vendor/LETTER'/name
  if p.exists():metadata['vendor/LETTER/'+name]=p
 for name in ['assets_audit.json','runtime_provenance.json','model_download_verified.json','shuffle_grid_alignment.json','verified_smoke_summary.json']:
  p=PREP/name
  if p.exists():metadata['preparation/'+name]=p
 for p in (ROOT/'codex/beauty_seqrec_batch64_20261122').iterdir():
  if p.is_file() and p.suffix in ['.py','.json','.md']:metadata['change_history/'+p.name]=p
 outputs=[pack('beauty-llama-metadata.zip',metadata),pack('beauty-llama-predictions.zip',predictions)]
 chunks=[]
 with adapter.open('rb') as f:
  i=0
  while True:
   b=f.read(1024**3)
   if not b:break
   name=f'beauty-adapter.safetensors.part{i:02d}';p=OUT/name;p.write_bytes(b)
   row=dict(name=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest());write(OUT/(name+'.json'),row);chunks.append(row);i+=1
 write(OUT/'adapter_parts.json',dict(original=str(adapter),bytes=adapter.stat().st_size,sha256=sha(adapter),parts=chunks))
 write(OUT/'export_receipt.json',dict(run=str(RUN),time_unix=time.time(),source_files_verified=len(launch['source_sha256']),outputs=outputs,
  omitted=['optimizer/scheduler/RNG and last checkpoint: retained server-side; publication supports fixed-checkpoint evaluation and from-base rerun, not exact optimizer resume','capacity preflight checkpoint: disposable model; metadata retained','base model: fixed Hugging Face revision and original hash manifest supplied; weights not rehosted'],adapter=chunks))
 print(json.dumps(outputs),flush=True)
if __name__=='__main__':main()
