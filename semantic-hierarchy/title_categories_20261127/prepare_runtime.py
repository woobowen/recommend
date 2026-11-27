"""Prepare an independent reproduction from the actual saved text/SID inputs."""
from pathlib import Path
import argparse,hashlib,json,py_compile,shutil
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--base-model',type=Path);a=p.parse_args();archive=a.archive.resolve();out=a.output.resolve()
 assert not out.exists(),'Use an independent new output directory'
 assert (archive/'run/input_audit.json').exists(),'Restore metadata archive first'
 base=a.base_model.resolve() if a.base_model else None
 if base:
  model=json.loads((archive/'run/model_verified.json').read_text(encoding='utf8'))
  for row in model['files']:
   if not row['file'].startswith('models/Meta-Llama-3-8B-Instruct/'):continue
   file=base/Path(row['file']).name;assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],str(file)
 shutil.copytree(archive/'run/source',out/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copytree(archive/'run/inputs',out/'inputs')
 for name in ['input_audit.json','encoding_config.json','effective_config.json']:shutil.copy2(archive/'run'/name,out/name)
 shutil.copytree(archive/'encoder_assets/models',out/'runtime_root/models')
 for d in ['beauty','instruments']:
  (out/d).mkdir()
  for f in (archive/'run'/d).glob('*.npy'):shutil.copy2(f,out/d/f.name)
  for name in ['encoding_complete.json']:
   shutil.copy2(archive/'run'/d/name,out/d/name)
 runtime=out/'runtime_root';(runtime/'stage_gpu_locks').mkdir();(runtime/'codex/letter_lc_rec_smoke_20261121').mkdir(parents=True)
 common=out/'source/runtime_common.py';text=common.read_text(encoding='utf8')
 old="ROOT = Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')";assert old in text
 common.write_text(text.replace(old,'ROOT = Path('+repr(str(runtime))+')'),encoding='utf8')
 if base:
  encoder=out/'source/original_encoder.py';text=encoder.read_text(encoding='utf8');old="base = ROOT / 'models/Meta-Llama-3-8B-Instruct'";assert old in text
  encoder.write_text(text.replace(old,'base = Path('+repr(str(base))+')'),encoding='utf8')
 for f in (out/'source').glob('*.py'):py_compile.compile(str(f),doraise=True)
 manifest={str(f.relative_to(out)):sha(f) for f in (out/'source').iterdir() if f.is_file()}
 (out/'relocation.json').write_text(json.dumps(dict(original_archive=str(archive),source_sha256=manifest,changes=['runtime_common ROOT path','optional original_encoder base path'],research_parameters_changed=False,prepared_from_frozen_text_sid=True,new_gpu_run_executed=False),indent=2)+'\n',encoding='utf8')
 print(out)
if __name__=='__main__':main()
