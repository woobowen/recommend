"""Assemble the partial delivery from verified evidence; no model execution."""
import collections
import datetime
import hashlib
import json
from pathlib import Path


def main():
    repo=Path(__file__).resolve().parents[2]
    support=Path('/home/addaswsw/recommend-support/sid-extensions-p0/beauty-extensions')
    collection=Path(json.loads((support/'collection-receipt.json').read_text())['directory'])
    states=json.loads((support/'terminal-states.json').read_text())
    out=repo/'manifests/beauty-extensions'
    def load(job,relative):return json.loads((collection/job/relative).read_text())
    def optional(job,relative):
        p=collection/job/relative
        return json.loads(p.read_text()) if p.is_file() else None
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[];upper=0;preflight_upper=0
    for job,state in states.items():
        r=load(job,'request.json')
        seconds=(datetime.datetime.fromisoformat(state['ended_at'])-datetime.datetime.fromisoformat(state['started_at'])).total_seconds()
        if r.get('env',{}).get('CUDA_VISIBLE_DEVICES'):
            upper+=seconds
            if job not in ('beauty-mlp-formal-v1','beauty-mlp-formal-attention32-v1'):
                preflight_upper+=seconds
        result=collection/job/'output/RESULT.json'
        rows.append({**state,'job_id':job,'wall_seconds':seconds,
            'result':load(job,'output/RESULT.json') if result.exists() else None,
            'request_file_sha256':sha(collection/job/'request.json'),
            'source_capture':r['capture_version'],'cwd':r['cwd']})
    primary=['experiments/beauty_extensions/'+v for v in ['common.py','baseline.py','cli.py','training.py','observation.py','data_checks.py','mlp_execution.py','mlp_formal.py','mlp_attention.py','results.py','plot_training.py']]
    for directory in ['author','attention']:
        primary += [str(p.relative_to(repo)) for p in (repo/'experiments/beauty_extensions'/directory).glob('*.py')]
    identity=[]
    identity_jobs=['beauty-b-smoke-r3','beauty-a-metadata-r3','beauty-mlp-smoke-r3','beauty-a-attention-r3','beauty-mlp-attention-engineering-r3','beauty-mlp-accum64-fullval-v1','beauty-mlp-lifecycle-v1','beauty-mlp-formal-v1','beauty-mlp-formal-attention32-v1']
    for job in identity_jobs:
        if job not in states or not (collection/job/'output/invocation.json').is_file():
            continue
        inv=load(job,'output/invocation.json')
        deployment=inv.get('code_identity',inv.get('deployment'))
        recorded={v['path']:v['sha256'] for v in deployment['files']};comp=[]
        for rel in primary:
            if rel.endswith('/mlp_attention.py') and job!='beauty-mlp-formal-attention32-v1':
                continue  # this standalone collector is not executed by training/resource jobs
            if rel in recorded:
                actual=sha(repo/rel);assert actual==recorded[rel],(job,rel)
                comp.append({'path':rel,'sha256':actual})
        identity.append({'job_id':job,'snapshot':deployment['snapshot_sha256'],
            'all_compared_research_source_unchanged':True,'files':comp})
    formal=optional('beauty-mlp-formal-v1','output/training_result.json')
    mlp_complete=bool(formal and formal.get('training_complete') and formal.get('test_complete') and formal.get('status')=='COMPLETE')
    formal_attention=optional('beauty-mlp-formal-attention32-v1','output/RESULT.json')
    formal_results={x:{'status':'NOT_RUN','recall@5':None,'ndcg@5':None,'recall@10':None,'ndcg@10':None,'delta':None} for x in ['A-Normal','A-Shuffle3','A-Shuffle4','B-Normal','B-Shuffle3','B-Shuffle4','MLP-new-training']}
    if formal:
        formal_results['MLP-new-training'].update(status='COMPLETE' if mlp_complete else formal.get('status','INCOMPLETE'),
            comparison='Independent T5/RQ-VAE+STE condition; no matched Normal and no cross-SID strict delta')
        if mlp_complete:
            formal_results['MLP-new-training'].update(formal['test_metrics']['metrics'])
    checks={'status':'PARTIAL_BLOCKED','formal_runs_completed':int(mlp_complete),'formal_runs_started':int('beauty-mlp-formal-v1' in states),
        'authorization':{'A_B_asset_pairings':'USER_APPROVED_2026-10-14',
            'attention_first32_final_top1_and_local_first':'USER_APPROVED_2026-10-14',
            'MLP_single_GPU_formal_protocol':'USER_APPROVED_ORIGINAL_T5_RQVAE_STE_SCRATCH_FULL_PARAMETERS_RAW32_ACC64_VAL6400_HARD12H'},
        'rounds':{'initial':0,'repairs_used':[1,2,3],'maximum':3,'additional_research_repair_executed':False},
        'final_blocker':{'job':'beauty-b-smoke-r3','type':'PREFLIGHT_HARNESS_DEVICE_MISMATCH_AFTER_TRAINER_TEST',
            'cause':'Lightning strategy.teardown moves model to CPU; smoke reuses CUDA input for manual validation',
            'research_source_patched_after_round3':False,'formal_gate_passed':False},
        'coverage':{x:load(j,'output/test_coverage.json') for x,j in [('A','beauty-a-metadata-r3'),('B','beauty-b-smoke-r3'),('MLP','beauty-mlp-smoke-r3')]},
        'formal_results':formal_results,
        'preflight_B':{str(n):load('beauty-b-smoke-r3',f'output/trainer-test-{n}/metrics.json') for n in [0,3,4]},
        'preflight_B_metric_recomputation':load('beauty-preflight-metrics-verification-v1','output/RESULT.json'),
        'MLP':{'short':load('beauty-mlp-short-r2','output/short_training.json'),
            'independent_resume':load('beauty-mlp-resume-r2','output/short_training.json'),
            'original_class_cache_test':load('beauty-mlp-smoke-r3','output/off_and_cache.json'),
            'approved_accumulation64_resource_check':optional('beauty-mlp-accum64-fullval-v1','output/resource_check.json'),
            'evaluation_metadata':optional('beauty-mlp-evaluation-index-v1','output/evaluation_coverage.json'),
            'original_callbacks_lifecycle_check':optional('beauty-mlp-lifecycle-v1','output/training_result.json'),
            'formal_training_handler_implemented':True,'formal_result':formal,
            'formal_checkpoint':formal.get('best_checkpoint') if mlp_complete else None,
            'formal_training_curves':optional('beauty-mlp-training-curves-v1','output/RESULT.json'),
            'formal_nonauthor_gate':json.loads((out/'mlp-formal-gate.json').read_text()) if (out/'mlp-formal-gate.json').is_file() else None},
        'attention':{x:{'job_id':j,'status':'ENGINEERING_NORMAL_ONLY_PASS','samples':3,'shuffle_n':0,
            'formal_first32_collected':False,'result':load(j,'output/small-attention/RESULT.json'),
            'off_on':load(j,'output/small-attention/off_on_checks.json')} for x,j in [('A','beauty-a-attention-r3'),('B','beauty-b-attention-r2'),('MLP-3-update-engineering','beauty-mlp-attention-engineering-r3')]},
        'MLP_fixed8_engineering_attention':optional('beauty-mlp-fixed8-attention-v1','output/RESULT.json'),
        'MLP_formal_fixed32_attention':formal_attention,
        'remaining':['Repair post-Trainer CPU/CUDA lifecycle and revalidate affected checks after new repair authorization',
            'Complete A original OFF and true Shuffle3/4 Trainer/lifecycle/RNG gates',
            'Validate formal A worker/settings paths and full route gate',
            'Complete identity-bound formal nonauthor gates for A/B',
            'Run six admitted A/B formal evaluations; verify complete saved predictions',
            'Collect approved fixed32 Normal/Shuffle/effective-MLP attention and audit ancestry/permutations'],
        'resources':{'physical_gpu_used':1,'physical_gpu_uuid':'GPU-59cac3bd-3645-668b-79c8-839e0d878893',
            'GPU0_other_job_not_touched':True,'cuda_visible_job_wall_hours_conservative_upper_bound':upper/3600,
            'preflight_cuda_visible_job_wall_hours_conservative_upper_bound':preflight_upper/3600,
            'wall_upper_bound_includes_CPU_imports_asset_reads_and_metadata_not_a_GPU_compute_measurement':True,
            'separate_GPU_compute_integral_seconds':None,'preflight_budget_initial_GPU_hours':2,
            'preflight_budget_total_GPU_hours':6,'max_single_job_wall_seconds':max(r['wall_seconds'] for r in rows),
            'newly_recovered_asset_bytes':161660760,'installed_system_packages':[],
            'installed_language_packages':[],'installed_toolchains':[],'global_configuration_changes':[],
            'cleanup_performed':False,
            'local_engineering_checkpoint_export':{'job':'beauty-mlp-accum64-fullval-v1','bytes':161250202,
                'reason':'Accepted runner full evidence export included its engineering checkpoint; later collections exclude checkpoint payloads',
                'included_in_Git_or_review_ZIP':False,'source_server_copy_preserved':True}},
        'jobs':rows,'source_delivery_identity':identity,
        'non_author_reviewers':['/root/source_conditions','/root/author_adapters','/root/attention_audit'],
        'all_formal_nonauthor_gates':False,
        'delivery_only_failure':{'first_unsaved_summary_attempt':'TypeError: dict() got multiple values for request_sha256',
            'effect':'No summary or research output written; saved assembler distinguishes canonical request fingerprint from request file SHA256',
            'new_research_test_or_repair':False},
        'external_evidence':{'collection':json.loads((support/'collection-receipt.json').read_text()),'zip_required':True,
            'reason':'numeric matrices, PNGs, complete job receipts/logs and target indexes accompany delivery in a lightweight ZIP'}}
    if not mlp_complete:
        checks['remaining'].append('Complete the approved MLP fresh training, original validation selection and exact full test; current status: '+str(formal.get('status') if formal else 'NOT_RUN'))
    if not formal_attention or not formal_attention.get('formal') or formal_attention.get('status')!='PASS':
        checks['remaining'].append('Collect approved fixed32 attention from the valid new MLP best checkpoint after complete training/test')
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    runs=out/'runs';runs.mkdir(exist_ok=True)
    for job in states:
        dest=runs/job;dest.mkdir(exist_ok=True)
        for name in ['request.json','status.json']:(dest/name).write_bytes((collection/job/name).read_bytes())
        result=collection/job/'output/RESULT.json'
        if result.exists():(dest/'RESULT.json').write_bytes(result.read_bytes())
    print(json.dumps({'jobs':len(states),'terminal_counts':dict(collections.Counter(s['status'] for s in states.values())),
        'upper_gpu_hours':upper/3600,'summary_bytes':(out/'EXECUTION_SUMMARY.json').stat().st_size}))


if __name__=='__main__':main()
