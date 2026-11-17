#!/usr/bin/env python3
"""Offline checks against fixed Git objects. Does not execute research code."""
import csv,json,subprocess,pathlib,re,urllib.parse,collections
root=pathlib.Path(__file__).resolve().parents[1]; repo=root.parents[2];d=json.loads((root/'sources.json').read_text()); rows=d['conditions']; errors=[]; n=0
if len(rows)!=21 or len({r['id'] for r in rows})!=21:errors.append('condition count or duplicates')
expected={(route,domain,condition) for route,domain in [('GRID A','Beauty'),('GRID B','Beauty'),('GRID B','Toys'),('GRID B','Sports'),('LETTER A0','Instruments'),('LETTER A1','Instruments'),('LETTER A0','Beauty')] for condition in ('normal','shuffle3','shuffle4')}
def normalized(x):return {'s3':'shuffle3','s4':'shuffle4'}.get(x.lower(),x.lower())
if {(r['route'],r['domain'],normalized(r['condition'])) for r in rows}!=expected:errors.append('condition matrix mismatch')
def get(ref,path):return subprocess.check_output(['git','show',ref+':'+path],stderr=subprocess.DEVNULL)
def walk(x):
 global n
 if isinstance(x,dict):
  if set(x)=={'commit','path'}:
   n+=1
   if subprocess.run(['git','cat-file','-e',x['commit']+':'+x['path']],stderr=subprocess.DEVNULL).returncode:errors.append('missing '+str(x))
  else:
   for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
walk(rows)
for r in rows:
 try:
  m=json.loads(get(r['metric_origin']['commit'],r['metric_origin']['path'])) if 'metric_origin' in r else json.loads(get(r['result']['commit'],r['result']['path']))
  if r['id'].startswith('grid-beauty-'):
   z=next(x for x in m['job_map'] if x['job_id']==r['run']['job_id'])['metrics_as_recorded']['metrics']
  elif r['id'].startswith('grid-'):
   z=next(x for x in m['conditions'] if x['evaluation_job_id']==r['run']['job_id'])['metrics']
  else:
   z=m
   for k in r['result_key'].split('.'):z=z[k]
  if z!=r['metrics']:errors.append('metric mismatch '+r['id'])
 except Exception as e:errors.append('metric read '+r['id']+' '+repr(e))
 family='GRID' if r['route'].startswith('GRID') else 'LETTER'; keys=['recall@5','ndcg@5','recall@10','ndcg@10'] if family=='GRID' else ['hit@5','ndcg@5','hit@10','ndcg@10']; target=' | '.join(f"{r['metrics'][k]:.9f}" for k in keys)
 if target not in (root/(family+'.md')).read_text():errors.append('table mismatch '+r['id'])
# Verify phase identity and result semantics against the recorded requests and outputs.
for r in rows:
 if r['route'].startswith('GRID'):
  q=json.loads(get(r['run']['request']['commit'],r['run']['request']['path']))
  z=json.loads(get(r['result']['commit'],r['result']['path']))
  if q['job_id']!=r['run']['job_id'] or q['capture_version'].split('/')[0]!=r['run'].get('capture_commit',r['run'].get('commit')):errors.append('GRID request binding '+r['id'])
  if z.get('status')!='COMPLETE' or z.get('metrics',{}).get('metrics')!=r['metrics'] or z.get('shuffle_n')!={'normal':0,'shuffle3':3,'shuffle4':4}[r['condition']]:errors.append('GRID result binding '+r['id'])
  conf=get(r['config']['commit'],r['config']['path']).decode()
  if 'seed: 42' not in conf or 'test: true' not in conf:errors.append('GRID effective config '+r['id'])
 elif r['domain']=='Instruments' and r['condition']!='normal':
  loc=r['run']['request_or_normal_entry'];q=json.loads(get(loc['commit'],loc['path']));argv=q['argv'];
  if q['job_id']!=r['run']['job_id'] or q['capture_version']!=r['run']['commit'] or argv[argv.index('--shuffle-n')+1]!=r['condition'][-1] or argv[argv.index('--model-key')+1]!=r['route'][-2:].lower():errors.append('LETTER request binding '+r['id'])
 elif r['domain']=='Beauty':
  primary=json.loads(get(r['primary_metrics']['commit'],r['primary_metrics']['path']))
  if primary['metrics']!=r['metrics'] or primary['run_commit']!=r['run']['commit'] or primary['evaluation_commit']!=r['run']['commit']:errors.append('Beauty primary metrics binding '+r['id'])
  audit=get(r['result']['commit'],'lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/FINAL_AUDIT.md').decode()
  if r['run']['commit'] not in audit or 'TIMED_OUT/-15' not in audit or r['supplement']['commit'] not in audit:errors.append('Beauty stage binding '+r['id'])
for p in (root/'README.md',root/'GRID.md',root/'LETTER.md',repo/'README.md',repo/'lc_rec/README.md'):
 text=p.read_text()
 for u in re.findall(r'\]\(([^)]+)\)',text):
  if u.startswith('https://github.com/woobowen/recommend/blob/'):
   part=u.split('/blob/',1)[1].split('#',1)[0];ref,path=part.split('/',1);path=urllib.parse.unquote(path)
   if subprocess.run(['git','cat-file','-e',ref+':'+path],stderr=subprocess.DEVNULL).returncode:errors.append('missing link '+u)
  elif not (':' in u or u.startswith('#')):
   if not (p.parent/urllib.parse.unquote(u.split('#')[0])).exists():errors.append('missing relative '+u+' in '+str(p))
for p in ('README.md','lc_rec/README.md'):
 old=get(d['snapshots']['start'],p);new=(repo/p).read_bytes();first=old.index(b'\n')+1
 if not (new.startswith(old[:first]) and new.endswith(old[first:])):errors.append('historical readme not preserved '+p)
print(json.dumps({'status':'PASS' if not errors else 'FAIL','conditions':len(rows),'fixed_location_checks':n,'numeric_checks':len(rows)*4,'readme_original_bytes_preserved':not any('readme not' in x for x in errors),'errors':errors},ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
