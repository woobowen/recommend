#!/usr/bin/env python3
"""Read fixed Git objects only; construct condition index and path count receipt."""
import collections,csv,io,json,subprocess,pathlib
G='9767729a5114e21b9c949892af3d759baca6a98e'; L='12eb8b176b156f3be7d7533ee8f443db02441303'; B='700d5fb4560aca329442bf799c851a377693e75b'; I='3396d45a041424fe4af625bfc0b047467150c5c1'; BR='6e1b211d9c37f358c43b1540b40b17f6703150d2'; S='1934c01eee767b665ef1043995ffc741958e30e0'
root=pathlib.Path(__file__).resolve().parents[1]
def get(ref,path):return subprocess.check_output(['git','show',ref+':'+path])
def obj(ref,path):return json.loads(get(ref,path))
def loc(ref,path):return {'commit':ref,'path':path}
def paths(ref):
 out=subprocess.check_output(['git','ls-tree','-r','-z','--name-only',ref]);return [x.decode('utf8','surrogateescape') for x in out.split(b'\0') if x]
def rootnames(ref):
 b=subprocess.check_output(['git','ls-tree','-z','--name-only',ref]);return [x.decode('utf8','surrogateescape') for x in b.split(b'\0') if x]
def count(ps):return dict(sorted(collections.Counter(x.split('/')[0] for x in ps).items()))
synthetic=['space name/a','中 文/引号".txt','tab\tname/行.txt','"leading/a']
assert [x.decode() for x in ('\0'.join(synthetic)+'\0').encode().split(b'\0') if x]==synthetic
counts={}
for name,ref in [('grid',G),('lc_experiments',L),('start',B)]:
 ps=paths(ref); c=count(ps);assert len(ps)==len(set(ps))==sum(c.values());assert set(c)==set(rootnames(ref));counts[name]={'commit':ref,'total':len(ps),'top_level':c}
assert counts['grid']['top_level']['GRID baseline']==2194 and '"GRID baseline' not in counts['grid']['top_level']
# Preserve historical inventory and path-count receipts; this build writes only sources.json.
rows=[]
version=obj(G,'manifests/grid-formal-20261015/FINAL_VERSION_MAP.json')
for r in version['job_map']:
 jid=r['job_id']
 if jid not in [f'gf-20261015-{route}-{cond}-v1' for route in ('a','b') for cond in ('normal','shuffle3','shuffle4')]:continue
 _,_,route,cond,_=jid.split('-'); rp=r['repository_result']; req=r['repository_request']; conf=rp.replace('RESULT.json','effective_config.yaml'); result=obj(G,rp); request=obj(G,req); assert r['status']=='SUCCEEDED' and r['metrics_as_recorded']['status']=='PASS'
 rows.append({'id':f'grid-beauty-{route}-{cond}','family':'sid-order','route':'GRID '+route.upper(),'domain':'Beauty','condition':cond,'status':'formal completed','run':{'capture_commit':r['capture_version'].split('/')[0],'snapshot_sha256':r['capture_version'].split('/')[1],'job_id':jid,'request':loc(G,req)},'training_run':None,'training_gap':'inherited checkpoint; training RUN not asserted by condition map','implementation':loc(r['capture_version'].split('/')[0],'experiments/formal_20261015/run.py'),'config':loc(G,conf),'assets_recorded':r['input_assets'],'result':loc(G,rp),'metrics':r['metrics_as_recorded']['metrics'],'metric_origin':loc(G,'manifests/grid-formal-20261015/FINAL_VERSION_MAP.json'),'read_scope':'request and RESULT read; effective config path checked; asset identity from version map; server not reverified'})
summary=obj(G,'manifests/grid-shuffle-20261017/EXECUTION_SUMMARY.json')
for c in summary['conditions']:
 jid=c['evaluation_job_id']; p='manifests/grid-shuffle-20261017/runs/'+jid+'/';request=obj(G,p+'request.json');result=obj(G,p+'output/RESULT.json');assert c['source_commit']=='c8acac42d5e13ca80af75faac5d1f7bc33fb589f'
 rows.append({'id':f"grid-{c['domain']}-b-{c['condition']}",'family':'sid-order','route':'GRID B','domain':c['domain'].title(),'condition':c['condition'],'status':'formal completed','run':{'commit':c['source_commit'],'job_id':jid,'request':loc(G,p+'request.json')},'training_run':None,'training_gap':'inherited checkpoint; original training source binding not inferred','implementation':loc(c['source_commit'],'experiments/formal_20261015/run.py'),'config':loc(G,p+'output/effective_config.yaml'),'assets_recorded':{k:c[k] for k in ('config','sid','checkpoint','test_index','prediction_sha256')},'result':loc(G,p+'output/RESULT.json'),'metrics':c['metrics'],'metric_origin':loc(G,'manifests/grid-shuffle-20261017/EXECUTION_SUMMARY.json'),'read_scope':'request, RESULT and condition summary read; server not reverified'})
inst=obj(I,'lc_rec/shuffle/results/SUMMARY.json')
for model in ('a0','a1'):
 normal=inst['models'][model]['Normal']; normal_entry='lc_rec/letter/A0_NORMAL_FINAL_REVIEW_ENTRY.md' if model=='a0' else 'lc_rec/a1_qwen3_06b/A1_NORMAL_FINAL_REVIEW_ENTRY.md'; cfg='lc_rec/letter/config/formal_instruments_seed42.json' if model=='a0' else 'lc_rec/a1_qwen3_06b/config/formal_instruments_seed42.json'
 for cond in ('normal','S3','S4'):
  v=normal if cond=='normal' else inst['conditions'][model+'-'+cond]['metrics']; job=('letter-a0-20261112-normal-det-seed42-r0' if model=='a0' else 'letter-a1-q06-20261113-normal-seed42-r0') if cond=='normal' else inst['conditions'][model+'-'+cond]['job_id']; p=normal_entry if cond=='normal' else inst['conditions'][model+'-'+cond]['path']+'/job/request.json'
  req=obj(I,p) if cond!='normal' else None
  rows.append({'id':f'letter-instruments-{model}-{cond.lower()}','family':'sid-order','route':'LETTER '+model.upper(),'domain':'Instruments','condition':cond,'status':'formal completed','training_run':{'commit':normal['run_commit'],'job_id':('letter-a0-20261112-normal-det-seed42-r0' if model=='a0' else 'letter-a1-q06-20261113-normal-seed42-r0'),'selection':'checkpoint-8240','request':loc(I,'lc_rec/letter/evidence/formal_det_20261112/jobs/letter-a0-20261112-normal-det-seed42-r0/job/request.json' if model=='a0' else 'lc_rec/a1_qwen3_06b/evidence/jobs/letter-a1-q06-20261113-normal-seed42-r0/job/request.json')},'run':{'commit':v['run_commit'],'evaluation_commit':v['evaluation_commit'],'job_id':job,'request_or_normal_entry':loc(I,p)},'implementation':loc(I,'lc_rec/shuffle/entry.py' if cond!='normal' else ('lc_rec/letter/scripts/formal_evaluate.py' if model=='a0' else 'lc_rec/a1_qwen3_06b/scripts/evaluate.py')),'config':loc(I,cfg),'assets_recorded':{'evidence':loc(I,'lc_rec/shuffle/FINAL_REVIEW_ENTRY.md'),'sid_sha256':'33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd','inter_sha256':'0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743','item_sha256':'1d9938e576a8d964e6b977a2b63aa56402840a788eaa6685f26f0d910a7133f2','adapter_sha256':('f4651c5c6e6ed841cd3384c0c8ebfab30880e94597e515ffa21916b876001ed5' if model=='a0' else 'fe96ea74fc3c7253b0c3687239266b0f073e1a51ef9158e7d97bf30b4527bd03'),'status':'historical recorded identity; server not reverified'},'result':loc(I,'lc_rec/shuffle/results/SUMMARY.json'),'result_key':('models.'+model+'.Normal.metrics' if cond=='normal' else 'conditions.'+model+'-'+cond+'.metrics.metrics'),'metrics':v['metrics'],'read_scope':'normal final review and summary read; shuffle request and summary read; server not reverified'})
beautyp='lc_rec/a0_beauty/evidence/paired-s4-supplement-20261117/'
bs=obj(L,beautyp+'evidence/results/beauty_supplemental_final_summary.json')
state=obj(L,beautyp+'evidence/original/control/state.json')
plan=obj(L,beautyp+'evidence/original/control/plan.json')
train_request='lc_rec/a0_beauty/evidence/formal-launch-20261116/request.json'
train_obj=obj(L,train_request)
assert train_obj['beauty_stage']=='train' and plan['run_commit']==BR
assert __import__('hashlib').sha256(get(BR,'lc_rec/a0_beauty/config/formal_beauty_seed42.json')).hexdigest()==plan['config_sha256']
def phase(stage, role):
 found=[x for x in state['attempts'] if x['stage']==stage]; assert len(found)==1
 a=found[0]; argv=a['runner']['worker']['argv']; assert argv[argv.index('--stage')+1]==stage
 return {'role':role,'stage':stage,'job_id':a['job_id'],'attempt':a['attempt'],'status':a['status'],
 'request':None,'request_availability':'bytes not archived; path and sha recorded in state, not recomputed',
 'recorded_request_path':a['request_path'],'recorded_request_sha256':a['request_sha256'],
 'worker_argv':argv,'state':{'commit':L,'path':beautyp+'evidence/original/control/state.json','selector':'attempts[stage='+stage+']'}}
for cond in ('normal','s3','s4'):
 v=bs['conditions'][cond]; primary_path=beautyp+'evidence/original/'+cond+'-primary/metrics.json'; primary=obj(L,primary_path)
 training=phase('train','training'); training['request']=dict(loc(L,train_request),role='training');training['request_availability']='archived bytes; canonical JSON sha matched against state; raw bytes sha separately recorded';training['selection']=bs['best_selection']
 training['raw_bytes_sha256']=__import__('hashlib').sha256(get(L,train_request)).hexdigest()
 assert __import__('hashlib').sha256(json.dumps(train_obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==training['recorded_request_sha256']
 gen=phase(cond,'generation');gen['rank_reports']=[loc(L,beautyp+'evidence/original/'+cond+'-generation/rank'+str(rank)+'.json') for rank in (0,1)]; ev=phase(cond+'_eval','evaluation');ev['primary_metrics']=loc(L,primary_path);ev['evaluation_config_recorded']=primary['config'];ev['evaluation_script_recorded']=primary['script']
 paired=({'role':'no_shuffle_pair','scope':'Normal is the control'} if cond=='normal' else
 {'role':'original_s3_paired','result':loc(L,beautyp+'evidence/original/s3/paired_comparison.json')} if cond=='s3' else
 {'role':'s4_supplemental_pair','commit':S,'job_id':obj(L,beautyp+'evidence/run/request.json')['job_id'],'request':loc(L,beautyp+'evidence/run/request.json'),'scope':'S4 independent CPU supplement; original paired timed out and collect not run'})
 rows.append({'id':'letter-beauty-a0-'+cond,'family':'sid-order','route':'LETTER A0','domain':'Beauty','condition':cond,'status':'formal generation and primary evaluation completed; supplemental summary completed; external scientific audit pending',
 'training_run':training,'run':{'commit':BR,'generation':gen,'evaluation':ev},'pairing':paired,
 'primary_metrics':loc(L,primary_path),'predictions':primary['prediction_files'],
 'implementation':loc(BR,'lc_rec/a0_beauty/formal_evaluate.py' if cond=='normal' else 'lc_rec/a0_beauty/shuffle_entry.py'),
 'config':dict(loc(BR,'lc_rec/a0_beauty/config/formal_beauty_seed42.json'),sha256=plan['config_sha256'],role='training_config'),
 'plan':loc(L,beautyp+'evidence/original/control/plan.json'),
 'assets_recorded':{'data':bs['audit']['data_assets'],'best_checkpoint':bs['best_selection'],'input_manifest':loc(L,beautyp+'evidence/run/input-manifest.json')},
 'result':loc(L,beautyp+'evidence/results/beauty_supplemental_final_summary.json'),'result_key':'conditions.'+cond+'.metrics','metrics':v['metrics'],
 'read_scope':'state, plan, primary metrics and train request checked; child request bytes absent; server and LFS not revalidated'})
assert len(rows)==21 and len({r['id'] for r in rows})==21
(root/'sources.json').write_text(json.dumps({'schema':1,'method':'fixed Git objects; historical assets are recorded references, not revalidated','snapshots':{'grid':G,'lc':L,'start':B},'conditions':rows},ensure_ascii=False,indent=2)+'\n')
