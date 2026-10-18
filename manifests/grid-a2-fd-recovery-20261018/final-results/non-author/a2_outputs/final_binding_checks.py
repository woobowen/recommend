"""Close output provenance, acceptance fingerprints and descriptive selectors."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from numeric_core import ident,load,read_bundle
from verify_preparation import CAPTURE,EXPECTED_COHORT,EXPECTED_SID
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'terminal_outputs/raw/jobs'
REMOTE='/home/bwwu/grid-audits/sid-extensions-p0/jobs/'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def main():
    d=load(ROOT/'A2_OUTPUTS_NONAUTHOR_REVIEW.json');pixels=load(ROOT/'a2-png-pixel-consistency.json')
    assert d['status']=='A2_EVALUATION_ATTENTION_STATISTICS_NONAUTHOR_REVIEW_PASS'
    assert pixels['status']=='A2_ALL_PNG_RAW_MATRIX_CELL_CONSISTENCY_PASS'
    deployment_path=ROOT/'preparation-v2/raw/scientific_source/DEPLOYMENT.json';deployment=load(deployment_path)
    deployment_sha=ident(deployment_path)['sha256'];snapshot=deployment['snapshot_sha256']
    assert hashlib.sha256(json.dumps({'source_commit':deployment['source_commit'],'files':deployment['files']},sort_keys=True).encode()).hexdigest()==snapshot
    accepted=[];artifact_count=0
    for role in ['eval-0','eval-3','eval-4','attention-0','attention-3','attention-4','stats']:
        job=RAW/f'ga-20261018-sportsa2-{role}-fd-recovery-v1';request=load(job/'request.json');status=load(job/'status.json');accept=load(job/'accepted.json')
        fingerprint=hashlib.sha256(canonical(request)).hexdigest();assert fingerprint==accept['request_sha256']
        assert request['job_id']==status['job_id']==job.name and request['capture_version']==CAPTURE
        assert request['cwd'].endswith('/snapshots/22286966cd544773')
        inputs={x['path']:x['sha256'] for x in request['inputs']}
        assert inputs[request['cwd']+'/DEPLOYMENT.json']==deployment_sha
        invocation=load(job/'output/invocation.json');spec=invocation['effective_spec']
        assert invocation['code_identity']==CAPTURE and spec['sid']['sha256']==EXPECTED_SID
        assert invocation['template']['sha256']==inputs[invocation['template']['path']]
        stage=load(job/'output/stage-entry.json')
        assert stage['source_commit']==deployment['source_commit'] and stage['command'][0]=='/data/bwwu/envs/grid-20261009-e61baf3/bin/python'
        if role.startswith('attention-'):
            n=role.rsplit('-',1)[1];evaluation=RAW/f'ga-20261018-sportsa2-eval-{n}-fd-recovery-v1'
            assert spec['evaluation_result']['sha256']==ident(evaluation/'output/RESULT.json')['sha256']
            assert spec['evaluation_status']['sha256']==ident(evaluation/'status.json')['sha256']
            assert spec['attention_cohort']['sha256']==EXPECTED_COHORT
        for artifact in status['artifacts']:
            p=job/artifact['path']
            if p.is_file():
                actual=ident(p);assert actual['bytes']==artifact['bytes'] and actual['sha256']==artifact['sha256'];artifact_count+=1
        accepted.append({'job_id':job.name,'request_sha256':fingerprint,'request_acceptance_exact':True,
                         'actual_stage_command':stage['command'],'capture_version':request['capture_version']})
    stats=load(RAW/'ga-20261018-sportsa2-stats-fd-recovery-v1/output/RESULT.json');selector_checks=[];source_hashes=0
    for condition in stats['jobs']:
        n=condition['job_id'].split('attention-')[1].split('-')[0];independent=d['attention'][n]
        for item in condition['input_files']:
            assert item['path'].startswith(REMOTE)
            p=RAW/item['path'][len(REMOTE):];actual=ident(p)
            assert actual['sha256']==item['sha256'] and actual['bytes']==item['bytes'];source_hashes+=1
        for path in ['final_top1_cross','local_first_cross']:
            keys={r['sample_key'] for r in independent['all_query_pair_rows']};ranking=[]
            for key in keys:
                rows=[r for r in independent['all_query_pair_rows'] if r['sample_key']==key and r['path']==path];assert len(rows)==6
                ranking.append({'sample_key':key,'mean_pair_cosine':math.fsum(r['cosine'] for r in rows)/6,
                                'mean_pair_total_variation':math.fsum(r['total_variation'] for r in rows)/6})
            least=sorted(ranking,key=lambda x:(x['mean_pair_cosine'],x['sample_key']))[:3]
            old=condition['least_similar_three_under_declared_mean_cosine'][path]
            assert [r['sample_key'] for r in least]==[r['sample_key'] for r in old]
            for x,y in zip(least,old):
                assert abs(x['mean_pair_cosine']-y['mean_pair_cosine'])<1e-12
                assert abs(x['mean_pair_total_variation']-y['mean_pair_total_variation'])<1e-12
            selector_checks.append({'shuffle_n':int(n),'path':path,'least_three_exact_under_declared_rule':True,'independent_least_three':least})
        directory=RAW/condition['job_id']/'output'
        parameter_hashes=[]
        for b in range(4):
            bundle=read_bundle(directory/f'batch-{b:02d}/bundle/record.json');parameter_hashes.append(bundle['metadata']['parameter_sha256'])
            for step in bundle['steps']:
                scores=step['scores'];assert np.isfinite(scores).all() and np.all(np.diff(scores,axis=1)<=0)
            for mapping in bundle['position_map']:
                for m in mapping:
                    if m['role']=='sid_lookup_vector':assert m['padding_group_kept_at_input_positions'] is (not bool(m['visible']))
        assert len(set(parameter_hashes))==1
    report={'status':'A2_OUTPUT_SOURCE_BINDING_AND_SELECTOR_PASS','reviewer':'/root/toys_result_audit',
        'main_output_review':ident(ROOT/'A2_OUTPUTS_NONAUTHOR_REVIEW.json'),'PNG_pixel_review':ident(ROOT/'a2-png-pixel-consistency.json'),
        'accepted_requests':accepted,'status_artifact_hashes_independently_cross_checked':artifact_count,
        'statistics_raw_source_hashes_exact':source_hashes,'least_three_selectors':selector_checks,
        'all_raw_generation_step_scores_finite_and_sorted':True,'parameter_hash_constant_across_four_batches_per_condition':True,
        'scientific_capture':CAPTURE,'new_SID_sha256':EXPECTED_SID,'fixed32_evaluation_sha256':EXPECTED_COHORT,
        'checkpoint_payload_loads':0,'GPU_calls':0,'server_mutations':0,'original_evidence_mutations':0,
        'overall_final_internal_review_pass':False,'scope':'A2 formal outputs, not an unsigned replacement for full training/quantization acceptance',
        'program':ident(Path(__file__))}
    p=ROOT/'a2-final-output-binding-checks.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'requests':7,'status_artifacts':artifact_count,'stat_sources':source_hashes,'least_three_selectors':len(selector_checks)}))
if __name__=='__main__':main()
