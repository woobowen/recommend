"""Prepare a complete formal plan without approval creation or submission.

Budget JSON must be explicitly supplied following user confirmation.  The
result is an execution-capable blueprint; prepare/query never launch it.
"""
import argparse
from pathlib import Path
from formal_control import durable_write
from production_control import validate_plan, STAGES, INITIAL_SERIES
from production_identity import load, sha, identity, ASSETS, need


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('deployment','root','data-root','source-root','model','python','runner','precision-approval','prelaunch-review','budget','task-id'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--experiment-series',default=INITIAL_SERIES)
    p.add_argument('--gpu-uuids',nargs=2,required=True)
    a=p.parse_args();dep=load(a.deployment);release=dep['release'];budget=load(a.budget)
    side=load(a.runner+'.deployment.json')
    need(sha(a.runner)=='cc1a1e277faa9bc713a4d0d1b620248d47c398e8ed17d4ed3d7ab52805cc657e','tested Beauty Runner bytes')
    need(side['namespace'].startswith('beauty-') and side['job_root']!='/home/bwwu/grid-audits/sid-extensions-p0/jobs','Beauty isolated deployment required')
    root=Path(a.root);path=root/'plan.json'
    config=str(Path(release)/'lc_rec/a0_beauty/config/formal_beauty_seed42.json')
    plan={'scope':'FORMAL_SUBMIT','evidence_scope':'FORMAL','task_id':a.task_id,'root':str(root),
          'experiment_series':a.experiment_series,
          'run_commit':dep['run_commit'],'prediction_run_commit':dep['run_commit'],'release':release,
          'config':config,'config_sha256':sha(config),'asset_sha256':ASSETS,'inputs':dep['files'],
          'data_root':a.data_root,'source_root':a.source_root,'model':a.model,'python':a.python,
          'runner':a.runner,'runner_sha256':sha(a.runner),'job_root':side['job_root'],
          'namespace':{'host':side['hostname'],'user':side['user'],'uid':side['uid'],
                       'namespace':side['namespace'],'deployment_sha256':sha(a.runner+'.deployment.json')},
          'precision_approval':a.precision_approval,'prelaunch_review':a.prelaunch_review,
          'approval_files':{'precision_approval':identity(a.precision_approval),
                            'prelaunch_review':identity(a.prelaunch_review)},
          'export_manifest':identity(Path(release)/'RUN_VERSION.json'),
          'gpu_uuids':a.gpu_uuids,'stage_seconds':budget['stage_seconds'],'budget':budget['budget'],
          'env':{'CUDA_VISIBLE_DEVICES':'0,1','CUDA_HOME':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4',
                 'LD_LIBRARY_PATH':'/data/bwwu/lc-rec/offload-20261108/cuda-12.4/lib64:/data/bwwu/envs/grid-20261009-e61baf3/lib/python3.10/site-packages/torch/lib',
                 'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
                 'PYTHONOPTIMIZE':'0','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}}
    validate_plan(plan)
    if path.exists():need(load(path)==plan,'preserve existing immutable plan')
    else:durable_write(path,plan)
    print('PREPARED_NOT_SUBMITTED',path)

if __name__=='__main__':main()
