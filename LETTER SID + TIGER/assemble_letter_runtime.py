"""Assemble frozen sources and downloaded original assets into a new runtime."""
import argparse,difflib,hashlib,json,shutil,os
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','instruments','yelp'],required=True);p.add_argument('--assets',type=Path,required=True,help='Extracted letter-tiger-originals directory');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 here=Path(__file__).resolve().parent
 if os.name=='nt':here=Path('\\\\?\\'+str(here));a.output=Path('\\\\?\\'+str(a.output.resolve()));a.assets=Path('\\\\?\\'+str(a.assets.resolve()))
 runs=json.loads((here/'run_index.json').read_text(encoding='utf-8'));r=next(x for x in runs if x['dataset']==a.dataset)
 historical=here/'runs'/r['published_run'];manifest=json.loads((historical/'source_manifest.json').read_text(encoding='utf-8'))['files'];snapshot=here/'source_snapshots'/r['source_signature']
 assert not a.output.exists(),'New output directory required';a.output.mkdir(parents=True)
 for name,h in manifest.items():
  f=snapshot/name;assert sha(f)==h,name;out=a.output/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
 # Copy only the selected run's checkpoint, SID, official JSON and input shards.
 audit_data=json.loads((historical/'mapping_audit.json').read_text(encoding='utf-8'))
 root='/home/jcy/GRID_wmh_Qwen_TIGER_20261009/'
 required={r['checkpoint'].removeprefix(root),audit_data['sid_path'].removeprefix(root)}
 required.update(p.removeprefix(root) for p in audit_data['official_sources'])
 required.update(p.removeprefix(root) for p in audit_data['data_files'])
 for rel in sorted(required):
  f=a.assets/'project'/rel;assert f.is_file(),str(f)
  out=a.output/rel;out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists() or sha(out)==sha(f),str(out);shutil.copy2(f,out)
 (a.output/'.project-root').touch()
 oldroot='/home/jcy/GRID_wmh_Qwen_TIGER_20261009';newroot=a.output.resolve().as_posix();adapted=[]
 entry=a.output/r['entrypoint'];source=entry.read_text(encoding='utf-8');changed=source.replace(oldroot,newroot)
 for f in [entry,*a.output.glob('codex/**/mapping_audit.json')]:
  text=f.read_text(encoding='utf-8');new=text.replace(oldroot,newroot)
  if new!=text:
   adapted.append({'path':str(f.relative_to(a.output)),'before_sha256':sha(f),'diff':''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile='historical',tofile='derived'))});f.write_text(new,encoding='utf-8')
 # Add the exact config used by the historical runner's baseline assertions.
 reference=('experiments/sid_order_5seed_v1/beauty/shuffle/seed_42/.hydra/config.yaml' if a.dataset=='beauty' else 'experiments/letter_official_beauty_20261115_125008/formal/.hydra/config.yaml')
 refout=a.output/reference;refout.parent.mkdir(parents=True,exist_ok=True)
 refout.write_text((here/'reference_configs'/('qwen_beauty_seed42.yaml' if a.dataset=='beauty' else 'letter_beauty.yaml')).read_text(encoding='utf-8').replace(oldroot,newroot),encoding='utf-8')
 # Beauty historical audit contains original paths; add its exact captured audit at the runner's required input location.
 control=('letter_upstream_transformer_20261115' if a.dataset=='beauty' else 'letter_official_downstream_20261116')
 audit=a.output/'codex'/control/'inputs'/a.dataset/'mapping_audit.json';audit.parent.mkdir(parents=True,exist_ok=True)
 text=(historical/'mapping_audit.json').read_text(encoding='utf-8');audit.write_text(text.replace(oldroot,newroot),encoding='utf-8')
 cfg=(historical/'.hydra/config.yaml').read_text(encoding='utf-8').replace(oldroot,newroot)
 (a.output/'effective_config_path_adapted.yaml').write_text(cfg,encoding='utf-8')
 receipt={'dataset':a.dataset,'historical_source_signature':r['source_signature'],'adaptations':adapted,'mapping_audit':'historical input hashes retained; root paths mapped only','scope':'new derived runtime; no training/evaluation executed'}
 (a.output/'assembly_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'output':str(a.output),'entrypoint':str(entry),'source_files':len(manifest)}))
if __name__=='__main__':main()
