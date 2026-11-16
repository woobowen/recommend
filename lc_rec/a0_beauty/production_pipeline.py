"""Fixed Beauty production stages. Test drivers may inject external compute only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from formal_control import durable_write
from production_identity import (need, load, identity, sha, check_file, scope_guard,
                                 verify_training, verify_predictions, SYNTHETIC)
from offline_metrics import ASSETS, KEYS

STAGES=('train','verify_best_adapter','normal','normal_eval','s3','s3_eval','s4','s4_eval','paired','collect')
GPU={'train','normal','s3','s4'}


def locations(plan, stage):
    """Successful dependency locations are immutable request outputs, not guessed names."""
    return Path(load(Path(plan['root'])/'state.json')['stages'][stage]['output'])


def training_root(plan):
    return Path(plan['root'])/'training'


def best_evidence(plan):
    # Revalidate on every model consumer; a saved PASS alone is not a source gate.
    return verify_training(training_root(plan),plan['run_commit'],plan['config_sha256'],plan['evidence_scope'])


def verify_prediction_stage(plan, condition):
    import formal_evaluate as original
    best=best_evidence(plan)
    output=locations(plan,condition);test=output/'test'
    proof=verify_predictions(test,plan['prediction_run_commit'],best,condition,plan['evidence_scope'])
    # A completed generation is an immutable candidate artifact. Reconcile its
    # saved hashes before any subsequent evaluator/model consumer uses it.
    saved=output/'stage_result.json'
    if saved.exists():
        record=load(saved)
        need(record['stage']==condition and record['run_commit']==plan['run_commit'] and
             record['scope']==plan['evidence_scope'],'saved generation identity')
        need(record['output']==str(output) and record['proof']['prediction_files']==proof['prediction_files'],
             'saved generation candidate identity changed')
        for row in record['proof']['prediction_files']:check_file(row)
    _,catalog,users,_=original.catalog_and_users(plan['data_root'])
    audit=original.aggregate_predictions([test/f'predictions.rank{rank}.jsonl' for rank in (0,1)],
        users,catalog,original.load_evaluator(original.author_source(plan['source_root'])),
        require_rank_done=True,run_commit=plan['prediction_run_commit'])
    need(audit['actual_test_users']==22363 and
         audit['expected_user_axis_sha256']=='abe0ed0fad612da1a4293927ba09698a47dff6687ecff33e83fadf05c918ea4c',
         'complete canonical generation user axis')
    proof['full_user_audit']=audit
    return proof


def evaluation(plan, condition, output):
    import formal_evaluate as original
    import offline_metrics
    test=locations(plan,condition)/'test';source=original.author_source(plan['source_root'])
    proof=verify_prediction_stage(plan,condition)
    _,catalog,users,_=original.catalog_and_users(plan['data_root'])
    # Both formal and synthetic acceptance supply scoped, validated rank evidence.
    stage=original.aggregate_predictions([test/f'predictions.rank{r}.jsonl' for r in (0,1)],
                users,catalog,original.load_evaluator(source),run_commit=plan['prediction_run_commit'])
    stage.update(RUN_COMMIT=plan['prediction_run_commit'],scope=plan['evidence_scope'])
    durable_write(output/'stage_a.json',stage)
    cfg=load(Path(plan['release'])/'lc_rec/letter/config/a0_evaluation_v1.json')
    cfg.update(protocol_version='letter_beauty_beam20_v1',dataset='Beauty',users=22363,
               original_run_commit=plan['prediction_run_commit'],sid_coverage_denominator=12088,
               asset_sha256=ASSETS,user_axis_sha256='7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f')
    durable_write(output/'evaluation_config.json',cfg)
    offline_metrics.evaluate(SimpleNamespace(output=output/'primary',test_dir=test,
        stage_a_report=output/'stage_a.json',config=output/'evaluation_config.json',
        official_cce=Path(plan['release'])/'lc_rec/letter/review/final_audit_20261113/external/official_cce.py',
        data_root=Path(plan['data_root']),run_commit=plan['prediction_run_commit'],evaluation_commit=plan['run_commit']))
    return {'prediction_proof':proof,'stage_a':identity(output/'stage_a.json'),'metrics':identity(output/'primary/metrics.json')}


def validate_worker_context(plan,stage,output,resume=None,*,job_id=None,job_dir=None):
    from formal_control import digest
    job_id=job_id or os.environ.get('GRID_JOB_ID')
    job_dir=job_dir or os.environ.get('GRID_JOB_DIR')
    need(job_id and job_dir and Path(job_dir)==Path(plan['job_root'])/job_id,'runner-owned worker context required')
    state=load(Path(plan['root'])/'state.json');record=state['stages'][stage]
    attempt=state['attempts'][record['attempt_index']]
    req=load(Path(job_dir)/'request.json');accepted=load(Path(job_dir)/'accepted.json')
    need(attempt['job_id']==job_id and digest(req)==attempt['request_sha256']==accepted['request_sha256'],'worker request identity')
    need(req['beauty_stage']==stage and req['capture_version']==plan['run_commit'],'worker stage/source identity')
    need(Path(record['output'])==Path(output) and Path(output).is_relative_to(Path(plan['root'])/'attempts'/stage),'worker output scope')
    if resume:
        from production_control import resume_point
        need(stage=='train' and Path(resume)==resume_point(training_root(plan),plan['run_commit'],plan['config_sha256']),'worker recovery source')
    return {'job_id':job_id,'request_sha256':digest(req),'output':str(output)}


def actual_source_root(plan):
    from formal_evaluate import author_source
    source=author_source(plan['source_root'])
    expected=Path(plan['release'])/'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'
    for name in ('data.py','collator.py','evaluate.py','test_ddp.py','prompt.py','utils.py','modeling_letter.py','config/ds_z2_bf16.json'):
        need(sha(source/name)==sha(expected/name),'frozen author source mismatch: '+name)
    return source


def actual_compute(plan, stage, output, resume=None):
    scope_guard(plan['root'],'FORMAL')
    scripts=Path(plan['release'])/'lc_rec/a0_beauty'
    source=actual_source_root(plan)
    common=['--source-root',str(source),'--data-root',plan['data_root'],'--model',plan['model']]
    argv=[plan['python'],'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2']
    if stage=='train':
        argv += [str(scripts/'formal_train.py'),*common,'--output',str(training_root(plan)),
                 '--config',plan['config'],'--run-commit',plan['run_commit'],
                 '--precision-approval',plan['precision_approval'],'--prelaunch-review',plan['prelaunch_review'],
                 '--precision-approval-sha256',plan['approval_files']['precision_approval']['sha256'],
                 '--prelaunch-review-sha256',plan['approval_files']['prelaunch_review']['sha256']]
        if resume:argv+=['--resume',resume]
    else:
        best_evidence(plan)
        if stage=='normal':argv += [str(scripts/'formal_evaluate.py')]
        else:
            audit=Path(plan['root'])/'shuffle_inputs'
            if not audit.exists():
                subprocess.run([plan['python'],str(scripts/'shuffle_cpu_inputs.py'),'--model-key','a0',
                    '--adapter',str(training_root(plan)/'best_adapter'),'--source-root',str(source),
                    '--data-root',plan['data_root'],'--output',str(audit),
                    '--normal-test-dir',str(locations(plan,'normal')/'test')],check=True)
            proof=load(audit/'audit.json');need(proof['status']=='PASS_CPU_FULL_INPUTS' and proof['asset_sha256']==ASSETS,'CPU shuffle audit')
            check_file(proof['manifest'])
            argv += [str(scripts/'shuffle_entry.py'),'--model-key','a0','--shuffle-n',stage[-1],'--cpu-audit',str(audit/'audit.json')]
        argv += [*common,'--adapter',str(training_root(plan)/'best_adapter'),'--output',str(output/'test'),
                 '--run-commit',plan['prediction_run_commit']]
    durable_write(output/'compute_argv.json',{'argv':argv,'python':plan['python'],'scope':'FORMAL'})
    # Same runner-owned markers and inherited process group; torchrun worker
    # sessions remain subject to the scoped runner's pidfd ownership checks.
    subprocess.run(argv,check=True,stdin=subprocess.DEVNULL)


def execute_stage(plan, stage, output, *, compute=actual_compute, resume=None):
    need(stage in STAGES,'unknown production stage')
    validate_worker_context(plan,stage,output,resume)
    output=Path(output);scope_guard(output,plan['evidence_scope']);output.mkdir(parents=True,exist_ok=False)
    if stage in GPU:
        compute(plan,stage,output,resume)
        if stage=='train':proof=verify_training(training_root(plan),plan['run_commit'],plan['config_sha256'],plan['evidence_scope'])
        else:proof=verify_prediction_stage(plan,stage)
    elif stage=='verify_best_adapter':proof=best_evidence(plan)
    elif stage.endswith('_eval'):proof=evaluation(plan,stage[:-5],output)
    elif stage=='paired':
        from paired_beauty import process
        results={}
        for condition in ('s3','s4'):
            for c in ('normal',condition):verify_prediction_stage(plan,c)
            args=SimpleNamespace(normal_primary=str(locations(plan,'normal_eval')/'primary'),
                  shuffle_primary=str(locations(plan,condition+'_eval')/'primary'),
                  normal_test=str(locations(plan,'normal')/'test'),shuffle_test=str(locations(plan,condition)/'test'),
                  output=str(output/condition))
            process(args,source_context={'scope':plan['evidence_scope'],'best':best_evidence(plan)})
            results[condition]=identity(output/condition/'paired_comparison.json')
        proof=results
    else:
        results={}
        for c in ('normal','s3','s4'):
            m=load(locations(plan,c+'_eval')/'primary/metrics.json')
            need(m['status']=='PASS' and list(m['metrics'])==KEYS,'metrics schema')
            results[c]={'metrics':m['metrics'],'file':identity(locations(plan,c+'_eval')/'primary/metrics.json')}
        paired=locations(plan,'paired')
        proof={'conditions':results,'paired':{c:identity(paired/c/'paired_comparison.json') for c in ('s3','s4')}}
        durable_write(output/'summary.json',{'scope':plan['evidence_scope'],'formal_launch_authorized':False,**proof})
    result={'stage':stage,'verified':True,'scope':plan['evidence_scope'],'run_commit':plan['run_commit'],
            'proof':proof,'output':str(output),'files':[identity(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.suffix in ('.json','.npz','.npy','.csv') and not p.name.startswith('predictions')]}
    durable_write(output/'stage_result.json',result)
    return result


def validate_stage(plan,stage,output):
    result=load(Path(output)/'stage_result.json')
    need(result['verified'] is True and result['stage']==stage and result['run_commit']==plan['run_commit'] and result['scope']==plan['evidence_scope'],'stage result identity')
    for row in result['files']:check_file(row)
    if stage in ('train','verify_best_adapter'):best_evidence(plan)
    elif stage in ('normal','s3','s4'):verify_prediction_stage(plan,stage)
    elif stage.endswith('_eval'):
        metrics=load(Path(output)/'primary/metrics.json')
        need(metrics['status']=='PASS' and metrics['users']==22363 and list(metrics['metrics'])==KEYS and metrics['official_cce_full_user_comparisons']==22363,'extended evaluation incomplete')
        for row in metrics['per_user_files']:check_file(row)
    return result


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for n in ('plan','stage','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--resume');p.add_argument('--authorization');p.add_argument('--authorization-sha256')
    a=p.parse_args();plan=load(a.plan)
    from production_control import validate_plan, authorize
    validate_plan(plan);need(plan['evidence_scope']=='FORMAL','production CLI refuses test scope')
    authorize(plan,a.authorization,a.authorization_sha256)
    execute_stage(plan,a.stage,a.output,resume=a.resume)

if __name__=='__main__':main()
