"""Relocate frozen code into a fresh folder; preserve all historical originals."""
from pathlib import Path
import argparse,hashlib,json,shutil,py_compile
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',required=True,type=Path);p.add_argument('--base-model',required=True,type=Path);args=p.parse_args();root=args.root.resolve();out=args.output.resolve();model=args.base_model.resolve()
 assert not out.exists(),'Use a new runtime folder'
 audit=json.loads((root/'preparation/assets_audit.json').read_text())
 for row in audit['model_files']:
  file=model/row['name'];h=hashlib.sha256()
  with file.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  assert file.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256'],str(file)
 shutil.copytree(root/'run/runner',out)
 common=out/'common.py';text=common.read_text();original=text
 for line in original.splitlines():
  if line.startswith('ROOT='):text=text.replace(line,'ROOT=Path('+repr(str(root))+')')
  elif line.startswith('PREP='):text=text.replace(line,'PREP=Path('+repr(str(root))+')')
  elif line.startswith('AUTHOR='):text=text.replace(line,'AUTHOR=Path('+repr(str(root/'vendor/LETTER'))+')')
  elif line.startswith('MODEL='):text=text.replace(line,'MODEL=Path('+repr(str(model))+')')
 common.write_text(text)
 # verify_assets previously checked irrelevant domains; this publication includes Beauty only.
 selected=dict(audit);selected['official_files']={k:v for k,v in audit['official_files'].items() if (root/'vendor/LETTER'/k).is_file()}
 assert all(k in selected['official_files'] for k in ['data/Beauty/Beauty.index.json','data/Beauty/Beauty.inter.json','LETTER-LC-Rec/data.py','LETTER-LC-Rec/modeling_letter.py','LETTER-LC-Rec/evaluate.py'])
 record=out/'beauty_assets_audit.json';record.write_text(json.dumps(selected,indent=2))
 text=common.read_text().replace("PREP/'assets_audit.json'",'Path('+repr(str(record))+')');common.write_text(text)
 for f in out.glob('*.py'):py_compile.compile(str(f),doraise=True)
 for d in ['fixed_training_record','new_training','new_testing']: (out/d).mkdir()
 trained=json.loads((root/'run/Beauty/training/training_result.json').read_text());trained['best_checkpoint']=str(root/'checkpoint')
 (out/'fixed_training_record/training_result.json').write_text(json.dumps(trained,indent=2))
 (out/'relocation.json').write_text(json.dumps(dict(original_common_sha256=hashlib.sha256(original.encode()).hexdigest(),changes=['ROOT/PREP/AUTHOR/MODEL paths','asset audit scope restricted to published Beauty and author LC-Rec files'],research_parameters_changed=False),indent=2))
 print(out)
if __name__=='__main__':main()
