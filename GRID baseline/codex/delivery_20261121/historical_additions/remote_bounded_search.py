"""Read-only bounded historical search; stdout is the complete structured receipt."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import base64
import hashlib
import json
import os
import re
import tarfile

ROOT=Path('/home/jcy/GRID_wmh_Qwen_TIGER_20261009')
OLD=Path('/home/jcy/Tiger_discarded_20261002')
ARCHIVE=Path('/home/jcy/.restore-archives/20261102')
out={'time_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
     'host':'ECNU-Server','user':'jcy','read_only':True,'gpu_used':False,
     'excluded':['Other accounts','credentials and SSH/config contents','environments/models/cache/.git','active LETTER work'],
     'inventories':[],'records':[],'name_candidates':[],'content_matches':[],'referenced_locations':[],'payloads':[]}


def receipt(p,include=False):
    raw=p.read_bytes()
    row={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if include:out['payloads'].append({**row,'base64':base64.b64encode(raw).decode()})
    out['records'].append(row)
    return raw


for rel in ['codex/restore_20261102/core_restore.json','codex/restore_20261102/restoration_complete.json',
            'codex/restore_20261102/historical_data_link.json','codex/restore_20261102/恢复说明与待补文件.md']:
    p=ROOT/rel
    if p.exists():receipt(p,True)
for name in ['备份说明.txt','backup_manifest.json','shared_inputs.json']:
    p=ARCHIVE/name
    if p.exists():receipt(p,True)
for p in sorted((ROOT/'codex/历史辅助脚本').glob('*.py')):receipt(p,True)

scopes=[('historical_restored',OLD),('current_entrypoints',ROOT/'_运行'),
        ('copy_receipts',ROOT/'codex/_复制说明'),('current_helpers',ROOT/'codex/历史辅助脚本'),
        ('baseline_sports',ROOT/'experiments/sports_rkmeans_qwen_20261010_093035'),
        ('baseline_toys',ROOT/'experiments/toys_rkmeans_qwen_20261010_093036'),
        ('order_completed',ROOT/'experiments/sid_order_5seed_v1'),
        ('order_plans',ROOT/'codex/plan_v1_1')]
pattern=re.compile(r'(asin|datamap|item.?map|convert|preprocess|candidate|recommend|predict|review|audit|verification|full_test_reference)',re.I)
content_pattern=re.compile(r'asin|datamap|raw.?amazon|convert.{0,30}tfrecord|tfrecord.{0,30}convert|independent|non.author|reviewer|审查者|非作者|独立审核',re.I)
for scope,root in scopes:
    count=0;candidate_count=0;scanned_text=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if d not in ['envs','environments','models','.git','__pycache__','checkpoints','source_snapshot']]
        for name in files:
            p=Path(directory)/name;count+=1;size=p.stat().st_size
            if pattern.search(name):
                candidate_count+=1
                out['name_candidates'].append({'scope':scope,'path':str(p),'bytes':size})
            # Text-only bound: no checkpoints, pickle execution or large artifact loads.
            if p.suffix in ['.py','.json','.yaml','.yml','.md','.txt','.csv'] and size<500000:
                scanned_text+=1
                for no,line in enumerate(p.read_text(encoding='utf-8',errors='replace').splitlines(),1):
                    if content_pattern.search(line):
                        out['content_matches'].append({'scope':scope,'path':str(p),'line':no,'text':line[:1000]})
    out['inventories'].append({'scope':scope,'root':str(root),'files_seen':count,'filename_candidates':candidate_count,'text_files_searched':scanned_text,'exclusions':['envs','environments','models','.git','__pycache__','checkpoints','source_snapshot']})

# Newly identified small original backup: inspect member names and collect only actual
# project helper/audit text, never user settings or keys. No extraction or execution.
archive=next((ARCHIVE/'quarantine_do_not_execute').glob('02_*.tar.gz'))
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers();out['small_backup_members']=[{'name':m.name,'bytes':m.size,'type':'file' if m.isfile() else 'other'} for m in members]
    out['small_backup_inventory']={'path':str(archive),'members':len(members),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    for m in members:
        if m.isfile() and m.size<500000 and re.search(r'(audit|review|verification)',m.name,re.I) and m.name.endswith(('.py','.json','.md','.txt')):
            raw=tar.extractfile(m).read()
            out['payloads'].append({'path':'tar:'+str(archive)+'!'+m.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()})

# Historical inventories are metadata, not payloads. Read only project-related rows.
for name in ['服务器home全量清单.tsv','quarantine_do_not_execute/文件清单.csv']:
    p=ARCHIVE/name
    if not p.exists():continue
    matches=[];rows=0
    with p.open(encoding='utf-8-sig',errors='replace') as f:
        for line in f:
            rows+=1
            if any(s in line for s in ['GRID_wmh_Qwen_TIGER','Tiger_discarded']) and not any(s in line for s in ['/envs/','/environments/','/models/','/.cache/','/__pycache__/']) and pattern.search(line):matches.append(line.rstrip())
    out['inventories'].append({'scope':'original_backup_inventory','path':str(p),'rows_read':rows,'project_relevant_name_matches':matches,'evidence_level':'inventory only; not original file content'})

for p in [Path('/data_a/mhwang/Rec/GRID/amazon_data'),Path('/data/jcyin/sid_order_reproduction')]:
    try:out['referenced_locations'].append({'path':str(p),'exists_on_this_host':p.exists(),'evidence':'Only existence on present ECNU host checked; original machine/other backups not checked'})
    except Exception as e:out['referenced_locations'].append({'path':str(p),'error':type(e).__name__})

# Full-test references are fixed sample records; report actual keys/counts without
# running model or treating naming as evidence of full-test predictions.
out['full_test_reference_inspection']=[]
import torch
for p in sorted((ROOT/'experiments/sid_order_5seed_v1').glob('*/*/seed_*/full_test_reference_*.pt')):
    j=torch.load(p,map_location='cpu',weights_only=True)
    row={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'type':type(j).__name__,'users':len(j),'loader':'torch.load CPU weights_only=True'}
    assert isinstance(j,dict) and len(j)==100
    row['user_ids']=sorted(j)
    row['all_record_keys']=sorted({tuple(sorted(v)) for v in j.values()})
    row['ids_shapes']=sorted({tuple(v['ids'].shape) for v in j.values()})
    row['scores_shapes']=sorted({tuple(v['scores'].shape) for v in j.values()})
    row['scope']='100 fixed sampled users, not all test users; contains candidate IDs and scores, no saved full-test targets'
    out['full_test_reference_inspection'].append(row)
    if len(out['full_test_reference_inspection'])<=1:receipt(p,True)
print(json.dumps(out,ensure_ascii=False))
