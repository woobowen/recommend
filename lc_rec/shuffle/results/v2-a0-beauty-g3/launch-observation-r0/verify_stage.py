"""Recheck the saved formal launch evidence and first six rows without CUDA or SSH."""
import argparse
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

from lc_rec.shuffle_v2 import core, g3_assets as ga, g3_evaluate as ge
from lc_rec.shuffle_v2 import g3_formal as gf

GPU = '86660be5a62c8325caf74cf2810581850ca12803'
CONTROL = 'cb3b1f36d69ef0c1a7f554ef5455026523635f78'
TRAINING = '21d495b0be5b606853ec7cec68a3eaa1b58acbd7'
NY = ZoneInfo('America/New_York')


def canonical_request(request):
    data=json.dumps(request,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    return hashlib.sha256(data).hexdigest()


def input_binding(path, expected):
    saved=Path(path)
    assert ga.sha(saved)==expected
    return saved


def read_requests(evidence):
    path=evidence/'formal-plan.json'
    plan=ga.read(path)
    gf.validate(plan)
    assert (plan['gpu_commit'],plan['evaluation_and_controller_commit'],plan['training_commit'])==(GPU,CONTROL,TRAINING)
    approval=ga.read(evidence/'authorization.json')
    auth_sha=ga.sha(evidence/'authorization.json')
    assert approval['plan_sha256']==ga.sha(path) and approval['run_commit']==GPU
    assert approval['scope']=='G3_BEAUTY_SHUFFLE_V2_FORMAL'
    assert approval['authorized_by']=='user' and approval['web_review_pass'] is True
    assert approval['single_gpu_uuid']==plan['card']
    sub=evidence/'server-startup'
    for which, phase in (('parent',plan['parent']),('s2',plan['modes']['shuffle2']['gpu'])):
        request=ga.read(sub/which/'request.json')
        assert request==ga.read(sub/which/'requests-copy.json')
        template=json.loads(json.dumps(phase['request']).replace(gf.APPROVAL_SHA,auth_sha))
        assert not phase['predecessor_members']
        template['inputs']=plan['release_inputs'][template['capture_version']]+template['inputs']
        if which=='parent':
            template['inputs']+=[{'path':plan['server_plan_path'],'sha256':ga.sha(path)},
                                {'path':plan['approval_path'],'sha256':auth_sha}]
            assert request==ga.read(evidence/'parent-request.json')
        assert request==template,which
        accepted=ga.read(sub/which/'accepted.json')
        attempt=ga.read(sub/which/'attempt.json')
        assert accepted['runner_sha256']==plan['runner_sha256']
        assert canonical_request(request)==accepted['request_sha256']==attempt['request_sha256']
    return plan


def verify_observations(evidence, plan):
    rows=[ga.read(evidence/'observations'/f'T{minute:02}.json') for minute in range(0,31,5)]
    ids={'parent':plan['parent']['name'],'child':plan['modes']['shuffle2']['gpu']['name']}
    request_hash={role:canonical_request(ga.read(evidence/'server-startup'/folder/'request.json'))
                  for role,folder in [('parent','parent'),('child','s2')]}
    first,last=rows[0],rows[-1]
    previous_count,previous_byte,previous_file_length,previous_time=0,0,0,0
    gpu_identity=None
    for point in rows:
        assert point['unix_ns']>previous_time and point['host']=='zwlab-01'
        previous_time=point['unix_ns']
        assert datetime.fromisoformat(point['utc']).astimezone(NY).isoformat()==point['america_new_york']
        for role,job in ids.items():
            assert point[role]['job_id']==job and point[role]['status']=='RUNNING'
            assert point[role]['request_sha256']==request_hash[role]
            assert point[role]['worker_alive'] and point[role]['supervisor_alive']
            assert point[role]['worker']==first[role]['worker']
            assert point[role]['supervisor']==first[role]['supervisor']
        assert point['controller']['status']=='RUNNING'
        assert [s['job_id'] for s in point['controller']['stages']]==[ids['child']]
        assert point['gpu']['uuid']==plan['card'] and not point['error_matches']
        rank,=point['gpu_rank_processes']
        identity={k:rank[k] for k in ('pid','start_ticks','boot_id','uid')}
        gpu_identity=gpu_identity or identity
        assert identity==gpu_identity and rank['proc_parent_pid']==point['child']['worker']['pid']
        env=rank['rank_env']
        assert [env[k] for k in ('WORLD_SIZE','RANK','LOCAL_RANK','CUDA_VISIBLE_DEVICES')]==['1','0','0',plan['card']]
        p=point['prediction']
        assert p['complete_users']-previous_count==p['delta_users']
        assert p['bytes']-previous_file_length==p['delta_bytes']
        if p['complete_users']:
            assert p['complete_users']>previous_count and p['complete_bytes']>previous_byte
            assert p['bytes']-p['complete_bytes']==p['buffered_suffix_bytes']
            assert p['delta_offset_start']==previous_byte
            assert point['rank']['assigned_users']==22363 and point['rank']['scope']=='FORMAL'
            assert point['rank']['RUN_COMMIT']==GPU
            assert point['rank']['completed_users']<=p['complete_users']
            assert [int(r['pid']) for r in point['compute_apps'] if r['uuid']==plan['card']]==[rank['pid']]
        previous_count,previous_byte,previous_file_length=p['complete_users'],p['complete_bytes'],p['bytes']
        for group in point['budget'].values():
            assert group['used_seconds']>=0 and group['remaining_seconds']>=0
            assert abs(group['used_seconds']+group['remaining_seconds']-group['limit_seconds'])<1e-7
    span=(last['unix_ns']-first['unix_ns'])/1e9
    assert span>=1800 and len(rows)==7 and previous_count>0
    assert last['elapsed_since_t0_seconds']==span
    return {'status':'PASS','points':len(rows),'span_seconds':span,'S2_complete_users':previous_count,'rank_identity':gpu_identity}


def verify_first_six(evidence,args):
    captured=ga.read(evidence/'first-six-capture.json')
    rank=captured['rank']
    normal=ga.read(args.normal_rank)
    binding=captured['binding']
    assert (rank['scope'],rank['world_size'],rank['rank'],rank['local_rank'],rank['assigned_users'])==('FORMAL',1,0,0,22363)
    assert rank['assigned_row_indices']==normal['assigned_row_indices']==list(dict.fromkeys(rank['assigned_row_indices']))
    assert core.canonical_sha256(rank['assigned_row_indices'])==rank['assignment_sha256']==normal['assignment_sha256']
    assert (rank['RUN_COMMIT'],rank['MODEL_TRAINING_COMMIT'])==(GPU,TRAINING)
    assert rank['adapter_identity']['scope']=='FORMAL'
    for field in ('model','model_identity_sha256','tokenizer_identity','strict_adapter_reload','adapter_files',
                  'adapter_identity','data_identity','allowed_callback_sha256','versions'):
        assert rank[field]==normal[field],field
    proof=rank['v2_model']
    assert (proof['base_dtype'],proof['lora_dtype'],proof['adapter_embedding_and_head_dtype'])==('torch.bfloat16','torch.float32','torch.bfloat16')
    assert proof['eval_mode'] and proof['requires_grad_false']
    assert rank['strict_adapter_reload']['keys']==450 and rank['strict_adapter_reload']['elements']==288571392
    assert (binding['V2_RUN_COMMIT'],binding['MODEL_TRAINING_COMMIT'],binding['scope'],binding['mode'])==(GPU,TRAINING,'FORMAL','shuffle2')
    assert binding['frozen_source_sha256']==normal['entry_sha256']
    assert ga.sha(evidence/'derived.rank0.py')==binding['derived_sha256']
    meta=ge.metadata(args.data_root)
    audit,cpu_rows=ge.bind_cpu(args.cpu_audit,meta)
    assert (audit['manifest_sha256'],ga.sha(args.cpu_audit))==(binding['cpu_manifest_sha256'],binding['cpu_audit_sha256'])
    raw=input_binding(evidence/'first-six.jsonl',captured['prediction_sha256']).read_bytes()
    assert raw.endswith(b'\n') and raw.count(b'\n')==6 and len(raw)==captured['end_offset_exclusive']
    assert captured['start_offset']==0
    result=[]
    for expected_idx, text in zip(rank['assigned_row_indices'][:6],raw.splitlines(),strict=True):
        row=json.loads(text)
        assert (row['row_idx'],row['scope'],row['condition'],row['rank'],row['world_size'],row['prompt_id'],
                row['V2_RUN_COMMIT'],row['model_identity_sha256'])==(expected_idx,'FORMAL','shuffle2',0,1,0,GPU,ge.MODEL_SHA)
        fields,history=ge.independent.identity_fields(meta,expected_idx)
        assert (row['user_id'],row['target_item_id'],row['target'])==(fields[1],fields[3],fields[4])
        identical=ge.verify_v2_input(row,cpu_rows[expected_idx],'shuffle2',audit['manifest_sha256'],history,meta['item_tokens'])
        sids=ge.independent.reconstruct_candidates(row,meta['symbol_by_id'],meta['allowed_ids'])
        result.append({'row_idx':expected_idx,'visible_groups':len(history),'candidates':len(sids),'natural_identity':identical,
                       'position_legal_outside_catalog':sum(sid not in meta['sizes'] for sid in sids)})
    assert sum(row['candidates'] for row in result)==120
    return {'status':'PASS','users':6,'candidates':120,'first_six_sha256':captured['prediction_sha256'],
            'rank_assignment_sha256':rank['assignment_sha256'],'rows':result}


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('evidence','data-root','cpu-audit','normal-rank'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--first-six-only',action='store_true')
    a=p.parse_args()
    plan=read_requests(a.evidence)
    out={'status':'PASS','requests':'parent and S2 accepted requests match reviewed plan and authorization',
         'first_six':verify_first_six(a.evidence,a)}
    if not a.first_six_only: out['observations']=verify_observations(a.evidence,plan)
    print(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
