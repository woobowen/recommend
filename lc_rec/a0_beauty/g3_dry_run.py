#!/usr/bin/env python3
"""Validate the real formal control plan and argv; never submit model jobs."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from formal_control import durable_write, digest
from g3_identity import EXPERIMENT_ID, ASSETS, source_identity, verify_assets, verify_base_model, need, sha
from production_control import validate_plan, authorize, Control, STAGES
from production_pipeline import compute_argv


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for n in ('release','data-root','model','run-commit','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();release=Path(a.release);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    need(not out.exists(),'preserve dry-run evidence')
    dep=json.loads((release/'RUN_VERSION.json').read_text())
    need(dep['run_commit']==a.run_commit,'dry-run source commit')
    assets=verify_assets(a.data_root);base=verify_base_model(a.model)
    # Reuse tested production runner bytes and sidecar; this does not submit or
    # alter the historical ledger. G3 gets its own experiment and output root.
    runner='/data/bwwu/lc-rec/a0-beauty-20261115/production-closure-20261115/formal-candidate-5b2917d/runner-beauty.py'
    side=json.loads(Path(runner+'.deployment.json').read_text())
    config=release/'lc_rec/a0_beauty/config/formal_g3_beauty_seed42.json'
    precision=release/'lc_rec/letter/evidence/formal_20261111/precision_approval.json'
    root=Path('/data/bwwu/lc-rec/g3-formal-20261118')/a.run_commit[:12]
    env={'CUDA_VISIBLE_DEVICES':'0,1','CUDA_HOME':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4',
         'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH',''),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONOPTIMIZE':'0','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'}
    seconds={'train':108000,'verify_best_adapter':900,'normal':43200,'normal_eval':900,
             's3':16200,'s3_eval':900,'s4':16200,'s4_eval':900,'paired':1200,'collect':600}
    plan={'scope':'FORMAL_PREPARE','evidence_scope':'FORMAL','experiment_id':EXPERIMENT_ID,
        'experiment_series':EXPERIMENT_ID,'task_id':'g3-beauty-seed42','root':str(root),
        'run_commit':a.run_commit,'prediction_run_commit':a.run_commit,'release':str(release),
        'config':str(config),'config_sha256':sha(config),'asset_sha256':ASSETS,'inputs':dep['files'],
        'data_root':a.data_root,'source_root':str(release/'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'),
        'model':a.model,'python':sys.executable,'runner':runner,'runner_sha256':sha(runner),
        'job_root':side['job_root'],'namespace':{'host':side['hostname'],'user':side['user'],'uid':side['uid'],
            'namespace':side['namespace'],'deployment_sha256':sha(runner+'.deployment.json')},
        'precision_approval':str(precision),'prelaunch_review':None,
        'approval_files':{'precision_approval':source_identity([precision])[0]},
        'export_manifest':source_identity([release/'RUN_VERSION.json'])[0],
        'gpu_uuids':['GPU-dfff0e95-1db6-e2f4-3431-55bb852fa0ae','GPU-59cac3bd-3645-668b-79c8-839e0d878893'],
        'stage_seconds':seconds,'budget':{'train_normal':151200,'s3':16200,'s4':16200,'cpu':5400,'total':189000,'cleanup_reserve':120},
        'env':env,'formal_budget_status':'PLANNING_ONLY_REQUIRES_SEPARATE_USER_LAUNCH_AUTHORIZATION'}
    validate_plan(plan)
    plan_path=root/'preparation-plan.json';root.mkdir(parents=True,exist_ok=True)
    if plan_path.exists():need(json.loads(plan_path.read_text())==plan,'immutable prepared plan')
    else:durable_write(plan_path,plan)
    state={'status':'DRAFT_PREPARED_NOT_SEALED','attempts':[],'stages':{}}
    need(not (root/'plan.identity.json').exists() and not (root/'state.json').exists(),'dry-run must not seal final launch plan')
    negative=[]
    def reject(name,fn,expected):
        try:fn()
        except (ValueError,PermissionError) as e:
            need(expected in str(e),name+' reached unexpected guard: '+str(e))
            negative.append({'name':name,'expected':expected,'error':str(e),'passed':True})
        else:raise AssertionError('authorization guard accepted '+name)
    reject('prepare never submits',lambda:authorize(plan,None,None),'FORMAL_PREPARE never submits')
    submitted=copy.deepcopy(plan);submitted['scope']='FORMAL_SUBMIT'
    reject('missing new launch authorization',lambda:authorize(submitted,None,None),'separate user launch authorization')
    old=release/'lc_rec/a0_beauty/config/old_authorization_negative_fixture.json'
    # Explicit isolated synthetic receipt; no historical approval bytes changed.
    fixture=out.parent/'SYNTHETIC_OLD_EXPERIMENT_AUTHORIZATION.json'
    fixture.write_text(json.dumps({'scope':'FORMAL','authorization_kind':'USER_LAUNCH_AUTHORIZATION',
       'experiment_id':'beauty-a0-seed42-original','formal_launch_authorized':True,'asset_sha256':ASSETS}))
    reject('old experiment authorization',lambda:authorize(submitted,str(fixture),sha(fixture)),'different experiment')
    argv_plan=copy.deepcopy(plan)
    argv_plan['prelaunch_review']=str(root/'WEB_SECOND_REVIEW_REQUIRED.json')
    argv_plan['approval_files']['prelaunch_review']={'sha256':'WEB_REVIEW_CONTENT_SHA256_REQUIRED'}
    commands={s:compute_argv(argv_plan,s,root/'attempts'/s/'a0') for s in ('train','normal','s3','s4')}
    need('--boundary-preflight' not in commands['train'] and '--resume' not in commands['train'],'formal argv not preflight')
    need('--verify-normal-off' in commands['normal'],'formal same-process OFF check')
    for s in ('normal','s3','s4'):
        need('--preflight-users-per-rank' not in commands[s] and str(root/'training/best_adapter') in commands[s],'formal full-user planned model dependency')
    help_results=[]
    for name in ('formal_train.py','formal_evaluate.py','production_control.py','prepare_production_plan.py'):
        result=subprocess.run([sys.executable,str(release/'lc_rec/a0_beauty'/name),'--help'],capture_output=True,text=True)
        need(result.returncode==0,'production CLI parsing '+name)
        help_results.append({'script':name,'exit_code':result.returncode,'stdout_sha256':__import__('hashlib').sha256(result.stdout.encode()).hexdigest()})
    for s in STAGES:
        need(s in plan['stage_seconds'],'missing stage dependency')
    record={'status':'PASS','scope':'FORMAL_ENTRY_DRY_RUN_ONLY','experiment_id':EXPERIMENT_ID,
        'formal_experiment_started':False,'formal_launch_authorized':False,'argv':sys.argv,'cwd':os.getcwd(),
        'interpreter':sys.executable,'run_commit':a.run_commit,'plan':source_identity([plan_path])[0],
        'plan_sha256':digest(plan),'asset_identity':assets,'base_model_identity':base,'compute_argv':commands,
        'stage_order':list(STAGES),'future_stage_outputs':['training/checkpoint-*','training/best_adapter','attempts/*/test/predictions.rank*.jsonl'],
        'pending_before_formal_submit':['web second review with source/config identity','new G3 user authorization and pinned SHA','explicit formal budget approval'],
        'authorization_negatives':negative,'cli_parsing':help_results,
        'output_isolation':{'formal':str(root),'preflight':'/data/bwwu/lc-rec/g3-preparation-20261118'},
        'space_plan':{'preparation_new_limit_bytes':80*1024**3,'formal_future_checkpoint_peak_gib':54,'free_bytes':shutil.disk_usage(root).free},
        'state':state,'source_files':source_identity([__file__,release/'lc_rec/a0_beauty/production_control.py',release/'lc_rec/a0_beauty/production_pipeline.py'])}
    durable_write(out,record);print(json.dumps({'status':record['status'],'plan':str(plan_path),'jobs_submitted':0}))


if __name__=='__main__':main()
