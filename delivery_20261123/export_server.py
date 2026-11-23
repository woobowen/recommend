"""Capture completed research originals and bounded active-run observations on CPU."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,shutil,subprocess,zipfile

ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
HERE=ROOT/'codex/experiment_publication_20261123'
META=HERE/'metadata'
SKIP={'.git','__pycache__','checkpoints','attention_analysis','lightning_logs'}
BIG={'.ckpt','.pt','.npy','.npz','.bin','.safetensors','.gz','.zip','.tar'}
records=[];archives={};assets={}


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def capture(src,member,active=False):
    if not src.is_file():return
    dest=META/member;dest.parent.mkdir(parents=True,exist_ok=True)
    size=src.stat().st_size
    with src.open('rb') as a,dest.open('wb') as b:
        left=size
        while left:
            raw=a.read(min(left,8*1024*1024))
            if not raw:break
            b.write(raw);left-=len(raw)
    records.append({'member':member,'source':str(src),'bytes':dest.stat().st_size,'sha256':digest(dest),'captured_size':size,'active_snapshot':active})
def tree(src,member,active=False):
    if not src.exists():return
    for p in sorted(src.rglob('*')):
        rel=p.relative_to(src)
        if not p.is_file() or any(x in SKIP for x in rel.parts) or p.suffix in BIG or p.suffix=='.pyc':continue
        capture(p,member+'/'+rel.as_posix(),active)
def asset(src,group,member):
    if src.is_file():assets.setdefault(group,{})[member]=src
def zip_assets(name,items):
    target=HERE/name
    members=[{'member':member,'source':str(src),'bytes':src.stat().st_size,'sha256':digest(src)} for member,src in sorted(items.items())]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for row in members:z.write(row['source'],row['member'])
        z.writestr('MANIFEST.json',json.dumps({'members':members},ensure_ascii=False,indent=2))
    manifest={'name':name,'bytes':target.stat().st_size,'sha256':digest(target),'members':members}
    (HERE/(name+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'asset':name,'bytes':manifest['bytes'],'members':len(members)}),flush=True)
    return manifest


def main():
    HERE.mkdir(exist_ok=True);META.mkdir(exist_ok=False)
    for folder in ['letter_upstream_transformer_20261115','letter_official_downstream_20261116','yelp_aligned_v1','yelp_diagnostics']:
        tree(ROOT/'codex'/folder,'control/'+folder)
    letter_runs=['letter_official_beauty_20261115_124859','letter_official_beauty_20261115_125008','letter_official_instruments_yelp_20261116_142754']
    for folder in letter_runs:tree(ROOT/'experiments'/folder,'runs/'+folder)
    for folder in ['yelp_rkmeans_qwen_aligned_v1','sid_order_yelp_5seed_v1','yelp_diagnostic_callback_check','yelp_diagnostic_patience40_seed43']:
        tree(ROOT/'experiments'/folder,'runs/'+folder)
    # Capture only closed-size current files; active model weights and outputs are not exported.
    for folder in ['letter_lc_rec_formal_20261121_123952','letter_beauty_seqrec_batch64_20261122_155523']:
        src=ROOT/'experiments'/folder
        for p in sorted(src.rglob('*')):
            rel=p.relative_to(src)
            if not p.is_file() or any(x in SKIP|{'capacity','capacity_v2','capacity_v3','capacity_micro8','test_preflight'} for x in rel.parts):continue
            if p.suffix in ['.py','.json','.jsonl','.md','.diff','.log']:capture(p,'active_observations/'+folder+'/'+rel.as_posix(),True)
    for folder in ['letter_lc_rec_formal_20261121','letter_lc_rec_smoke_20261121','beauty_seqrec_batch64_20261122']:
        src=ROOT/'codex'/folder
        if not src.exists():continue
        for p in sorted(src.iterdir()):
            if p.is_file() and p.suffix in ['.json','.md','.py','.log']:capture(p,'active_preparation/'+folder+'/'+p.name,True)
    # Completed LETTER checkpoints and unchanged official SID/data matched to each run.
    for run in [ROOT/'experiments/letter_official_beauty_20261115_125008/formal',ROOT/'experiments/letter_official_instruments_yelp_20261116_142754/instruments/formal',ROOT/'experiments/letter_official_instruments_yelp_20261116_142754/yelp/formal']:
        stage=json.loads((run/'stage_result.json').read_text());p=Path(stage['checkpoint']);assert digest(p)==stage['checkpoint_sha256']
        asset(p,'letter-tiger-originals.zip','project/'+p.relative_to(ROOT).as_posix())
        audit=json.loads((run/'mapping_audit.json').read_text())
        sid=Path(audit['sid_path']);asset(sid,'letter-tiger-originals.zip','project/'+sid.relative_to(ROOT).as_posix())
        for data in audit['data_files']:
            path=Path(data);path=ROOT/path if not path.is_absolute() else path
            if path.is_file():
                member=('project/'+path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else 'external_data/'+path.relative_to('/home/jcy/data').as_posix())
                asset(path,'letter-tiger-originals.zip',member)
    for ds in ['Beauty','Instruments','Yelp']:
        for p in (ROOT/'third_party/LETTER-8d0154e/data'/ds).glob('*.json'):asset(p,'letter-tiger-originals.zip','project/'+p.relative_to(ROOT).as_posix())
    upstream=ROOT/'experiments/yelp_rkmeans_qwen_aligned_v1'
    for n in ['item_embeddings.pt','semantic_ids.pt','quantizer_train/checkpoints/checkpoint_000_000030.ckpt']:
        p=upstream/n;asset(p,'qwen-yelp-upstream-originals.zip','project/'+p.relative_to(ROOT).as_posix())
    data=Path('/home/jcy/data/yelp_aligned_v1')
    for p in sorted(data.rglob('*')):
        if not p.is_file():continue
        if p.name.endswith('.tfrecord.gz'):asset(p,'qwen-yelp-upstream-originals.zip','external_data/yelp_aligned_v1/'+p.relative_to(data).as_posix())
        elif p.suffix in ['.json','.jsonl','.md'] and p.stat().st_size<40*1024*1024:capture(p,'qwen_yelp_data/'+p.relative_to(data).as_posix())
    for folder in ['sid_order_yelp_5seed_v1','yelp_diagnostic_patience40_seed43']:
        for p in (ROOT/'experiments'/folder).rglob('*.ckpt'):asset(p,'qwen-yelp-downstream-attempts.zip','project/'+p.relative_to(ROOT).as_posix())
    for p in (ROOT/'experiments/sid_order_yelp_5seed_v1').rglob('full_test_reference*.pt'):asset(p,'qwen-yelp-downstream-attempts.zip','project/'+p.relative_to(ROOT).as_posix())
    for p in (ROOT/'experiments/sid_order_yelp_5seed_v1/fixed_test_users').glob('*'):
        if p.is_file() and p.suffix in ['.pt','.json']:asset(p,'qwen-yelp-downstream-attempts.zip','project/'+p.relative_to(ROOT).as_posix())
    capture(ROOT/'requirements.txt','environment/original_requirements.txt')
    for env in ['grid','letter_lc_rec']:
        out=subprocess.check_output([str(ROOT/'environments'/env/'bin/python'),'-m','pip','freeze'],text=True)
        dest=META/'environment'/(env+'_pip_freeze.txt');dest.parent.mkdir(exist_ok=True);dest.write_text(out)
    exports=[]
    for name,items in assets.items():exports.append(zip_assets(name,items))
    (META/'server_capture.json').write_text(json.dumps({'time_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'server_root':str(ROOT),'records':records,'scope':'read-only completed originals plus fixed-size active observations; no model work','assets':[{k:v for k,v in x.items() if k!='members'} for x in exports]},ensure_ascii=False,indent=2))
    metadata=zip_assets('experiment-metadata-20261123.zip',{p.relative_to(META).as_posix():p for p in META.rglob('*') if p.is_file()})
    (HERE/'export_complete.json').write_text(json.dumps({'complete':True,'time_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'assets':[{k:v for k,v in x.items() if k!='members'} for x in exports+[metadata]],'metadata_files':len(records)},indent=2))
    print('EXPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
