from pathlib import Path
import copy, hashlib, json, sys
r=Path(sys.argv[1]); x=json.loads((r/'launch-readback.json').read_bytes())
known=Path('/home/addaswsw/recommend-support/letter-shuffle-v2-beauty/preparation-r1')
draft=json.loads((known/'formal-plan-reviewed.json').read_bytes())
ref='1182dfd914c20bd3dc11786587eae6879fba6b9d'
runner='cbb963fab68c34cc977653afc46f260e7ede962cb4a8f105458331a0e0e19a75'
h=lambda value:hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
a,b=copy.deepcopy(x['plan']),copy.deepcopy(draft['plan'])
clock_a,clock_b=a.pop('gpu_selection'),b.pop('gpu_selection')
assert a==b
pa,pb=copy.deepcopy(x['parent']),copy.deepcopy(draft['parent_request'])
plan_path=pa['argv'][pa['argv'].index('--plan')+1]
for payload in (pa,pb):
    i=[k['path'] for k in payload['inputs']].index(plan_path)
    payload['inputs'][i]['sha256']='LIVE_GPU_SELECTION_OBSERVATION'
assert pa==pb
assert any(v['path']==plan_path and v['sha256']==x['plan_sha256'] for v in x['parent']['inputs'])
assert x['child']==x['plan']['modes']['shuffle2']['gpu']
assert x['parent_stage']['formal_run_commit']==ref and x['parent_stage']['status']=='RUNNING'
assert x['parent_stage']['stages'][0]['job_id']==x['child']['job_id']
for kind in ('parent','child'):
    state=x[kind+'_query']; req=x[kind]; received=x[kind+'_accepted']
    assert req['capture_version']==ref and req['job_id']==state['job_id']
    assert state['status']=='RUNNING' and state['supervisor_alive'] and state['worker_alive'] and state['workgroup_live']
    assert h(req)==received['request_sha256']==state['request_sha256']
    assert received['runner_sha256']==runner
source=x['manifest']
assert source['RUN_COMMIT']==ref and source['runner_sha256']==runner
recorded={v['path']:v['sha256'] for v in source['files']}
assert all(recorded['lc_rec/shuffle_v2/'+k]==v for k,v in x['deployed_material'].items())
assert x['existing_formal_jobs']==sorted([x[k]['job_id'] for k in ('parent','child')])
assert not any(flag in x['child']['argv'] for flag in ('--preflight-users-per-rank','--preflight-manifest'))
receipts={kind:{'job_id':x[kind]['job_id'],'acceptance':x[kind+'_accepted'],'worker':x[kind+'_query']['worker'],
                'supervisor':x[kind+'_query']['supervisor'],'started_at':x[kind+'_query']['started_at']} for kind in ('parent','child')}
result={'status':'PASS_NONAUTHOR_FORMAL_LAUNCH','reviewer':'/root/formal_review','observed_utc':x['observed_utc'],
        'scope':'Fresh strict SSH connection read stored requests, runner-query identity and active DAG. No children submitted by reviewer.',
        'run_commit':ref,'runner_sha256':runner,'live_plan_sha256':x['plan_sha256'],
        'reviewed_construct_only_plan_sha256':draft['canonical_plan_sha256'],
        'consistent_live_and_reviewed_plan_except_gpu_selection_observation':True,
        'gpu_uuid':x['child']['env']['CUDA_VISIBLE_DEVICES'],'planned_gpu_conditions':list(x['plan']['modes']),
        'material_deployed_hashes':x['deployed_material'],'receipts':receipts}
(r/'launch-nonauthor.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='receipts'},indent=2))
