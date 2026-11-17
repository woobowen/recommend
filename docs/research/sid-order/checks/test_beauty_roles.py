#!/usr/bin/env python3
"""Negative fixtures mutate in-memory metadata only; no model or metrics recomputation."""
import copy,json,pathlib,subprocess
from beauty_roles import validate_beauty
root=pathlib.Path(__file__).resolve().parents[1];d=json.loads((root/'sources.json').read_text()); rows={x['condition']:x for x in d['conditions'] if x['id'].startswith('letter-beauty-a0-')}; ref='12eb8b176b156f3be7d7533ee8f443db02441303';bp='lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/evidence/'
def get(commit,p):return subprocess.check_output(['git','show',commit+':'+p])
def obj(p):return json.loads(get(ref,p))
state=obj(bp+'original/control/state.json');plan=obj(bp+'original/control/plan.json');tr=obj('lc_rec/a0_beauty/evidence/formal-launch-20261116/request.json');supp=obj(bp+'run/request.json');summary=obj(bp+'results/beauty_supplemental_final_summary.json');config=get('6e1b211d9c37f358c43b1540b40b17f6703150d2','lc_rec/a0_beauty/config/formal_beauty_seed42.json')
def check(row,primary=None):return validate_beauty(row,state,plan,tr,config,primary or obj(row['primary_metrics']['path']),summary,supp)
for r in rows.values():assert check(r)
fixtures=[]
r=copy.deepcopy(rows['normal']);r['run']['generation']['request']=r['training_run']['request'];fixtures.append(('train request as generation',r,None))
r=copy.deepcopy(rows['normal']);r['run']['generation']=copy.deepcopy(rows['s3']['run']['generation']);fixtures.append(('normal/s3 stage job hash swapped',r,None))
r=copy.deepcopy(rows['normal']);r['run']['commit']='0'*40;fixtures.append(('run altered',r,None))
r=copy.deepcopy(rows['normal']);r['config']['sha256']='0'*64;fixtures.append(('config SHA altered',r,None))
r=copy.deepcopy(rows['s4']);fixtures.append(('s4 uses s3 metrics and predictions',r,obj(rows['s3']['primary_metrics']['path'])))
r=copy.deepcopy(rows['s4']);r['run']['generation']['request_availability']='raw bytes SHA verified';fixtures.append(('unarchived request claimed verified',r,None))
results=[]
for name,row,primary in fixtures:
 try:check(row,primary)
 except ValueError as e:results.append({'fixture':name,'rejected':True,'reason':str(e)})
 else:raise AssertionError('negative fixture accepted: '+name)
print(json.dumps({'status':'PASS','source_snapshot':ref,'correct_cases':len(rows),'negative_cases':results},ensure_ascii=False,indent=2))
