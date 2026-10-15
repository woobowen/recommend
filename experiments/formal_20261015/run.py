"""Formal thin orchestration: unchanged models, loaders, beam and evaluator."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from experiments.beauty_extensions.common import REPO, bind_source, instantiate, load_config, sha256, verify_asset, write_json
from experiments.beauty_extensions.authorization import canonical_plan, require_formal_gate


def admit(spec):
    if spec['route'] in ('A','B'):return require_formal_gate(spec)
    assert spec['route']=='MLP' and spec['domain'] in ('toys','sports') and spec['formal_approved']
    verify_asset(spec['nonauthor_gate']);gate=json.loads(Path(spec['nonauthor_gate']['path']).read_text())
    assert gate['kind']=='FORMAL_RUN_AUTHORIZED' and gate['explicit_user_approval_reference']
    plan=hashlib.sha256(json.dumps(canonical_plan(spec),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert gate['plan_sha256']==plan
    deployment=json.loads((REPO/'DEPLOYMENT.json').read_text())
    code={r['path']:r['sha256'] for r in deployment['files'] if r['path'].endswith('.py')}
    assert gate['source_python_sha256']==code
    for name,digest in code.items():assert sha256(REPO/name)==digest,name
    for name in ('config','sid'):
        verify_asset(spec[name]);assert gate['assets'][name]==spec[name]
    verify_asset(gate['independent_review']);review=json.loads(Path(gate['independent_review']['path']).read_text())
    assert review['status']=='PREFLIGHT_GATE_PASS' and review['plan_sha256']==plan and review['source_python_sha256']==code
    return gate


def formal_config(spec,output):
    cfg=load_config(spec,output)
    assert not any(name.startswith('limit_') for name in cfg.trainer)
    assert not cfg.trainer.get('fast_dev_run',False) and not cfg.trainer.get('overfit_batches',0)
    assert cfg.model.top_k_for_generation==10
    return cfg


def evaluate(spec,cfg,output):
    from experiments.beauty_extensions.cli import test_lifecycle
    from experiments.beauty_extensions.results import recompute
    cfg,model,dm=instantiate(spec,output,checkpoint=True)
    started=time.monotonic();error=None
    try:
        test_lifecycle(cfg,model,dm,output/'test')
    except BaseException as exc:error=repr(exc);raise
    finally:
        trace={'shuffle_n':spec['shuffle_n'],'encoder_calls':getattr(model,'encoder_calls',getattr(model,'encoder_call_count',0)),
            'encoder_role_counts':model.encoder_role_counts,'shuffle_summary':getattr(model,'shuffle_summary',None),
            'shuffle_role_counts':getattr(model,'shuffle_role_counts',None),'shuffle_call_count':getattr(model,'shuffle_call_count',None),
            'shuffle_events':getattr(model,'shuffle_events',None),'permutation_events':getattr(model,'permutation_events',None),
            'seed':spec['seed'],'loss_and_generate_sample_separately':True,'exception':error,
            'trace_scope':'existing bounded native observation events; full call counters; no extra sampler calls'}
        write_json(output/'intervention_trace.json',trace)
    metric=recompute(spec,cfg,output/'test/predictions.jsonl',output/'test',coverage_index=spec['test_index']['path'],trainer_metrics=output/'test/metrics.json')
    assert metric['coverage_exact'] and metric['samples']==22363
    actual=trace['shuffle_summary']['calls'] if spec['route']=='B' else trace['shuffle_call_count']
    assert actual==(trace['encoder_calls'] if spec['shuffle_n'] else 0)
    assert spec['shuffle_n']==0 or bool(trace['shuffle_events'] if spec['route']=='B' else trace['permutation_events'])
    return {'status':'COMPLETE','formal':True,'route':spec['route'],'shuffle_n':spec['shuffle_n'],
        'test_complete':True,'metrics':metric,'intervention_calls':actual,'elapsed_test_and_recompute_seconds':time.monotonic()-started,
        'checkpoint':spec['checkpoint'],'sid':spec['sid'],'config':spec['config'],'attention_collected':False}


def metadata(spec,cfg,output):
    from experiments.beauty_extensions.mlp_execution import evaluation_index
    from experiments.beauty_extensions.data_checks import scan_test_coverage,prepare_fixed_attention_samples
    val=evaluation_index(spec,cfg,output);test=scan_test_coverage(cfg,spec,output)
    cohort=prepare_fixed_attention_samples(cfg,spec,output)
    return {'status':'PASS','model_loaded':False,'evaluation':val,'test':test,
        'cohort':{'path':str(cohort),'sha256':sha256(cohort)}}


def attention(spec,output):
    import torch
    from experiments.beauty_extensions.common import batch_to
    from experiments.beauty_extensions.cli import clear_runtime
    from experiments.beauty_extensions.observation import capture_batch
    verify_asset(spec['evaluation_result']);verify_asset(spec['evaluation_status'])
    result=json.loads(Path(spec['evaluation_result']['path']).read_text());status=json.loads(Path(spec['evaluation_status']['path']).read_text())
    assert status['status']=='SUCCEEDED' and status['exit_code']==0 and status['cleanup']['verified']
    assert result['formal'] and result['test_complete'] and result['status']=='COMPLETE'
    assert all(result[name]==spec[name] for name in ('checkpoint','sid','config')) and result['shuffle_n']==spec['shuffle_n']
    verify_asset(spec['attention_cohort']);cohort=json.loads(Path(spec['attention_cohort']['path']).read_text())
    assert cohort['selected_users']==32 and len(set(cohort['sample_keys']))==32 and not cohort['model_outputs_inspected']
    assert cohort['sid']['sha256']==spec['sid']['sha256']
    _,model,_=instantiate(spec,output,checkpoint=True)
    from src.data.loading.components.interfaces import SequentialModelInputData,SequentialModuleLabelData
    model.to('cuda:0').eval();batches=[]
    try:
        for start in range(0,32,8):
            keys=cohort['sample_keys'][start:start+8]
            inputs=SequentialModelInputData(user_id_list=keys,transformed_sequences={'sequence_data':torch.tensor(cohort['input_ids'][start:start+8],dtype=torch.long)},mask=torch.tensor(cohort['attention_mask'][start:start+8],dtype=torch.long))
            labels=SequentialModuleLabelData(labels={'sequence_data':torch.tensor(cohort['targets'][start:start+8],dtype=torch.long)})
            destination=output/f'batch-{start//8:02d}'
            checks=capture_batch(spec,model,batch_to((inputs,labels),'cuda:0'),destination,engineering=False)
            batches.append({'sample_keys':keys,'bundle':str(destination/'bundle/record.json'),'bundle_sha256':sha256(destination/'bundle/record.json'),'observer_off_on_checks':checks})
    finally:
        clear_runtime(model)
        if hasattr(model,'capture_full_permutation'):model.capture_full_permutation=False
    return {'status':'COMPLETE','formal':True,'samples':32,'batches':batches,'checkpoint':spec['checkpoint'],'sid':spec['sid'],
        'config':spec['config'],'shuffle_n':spec['shuffle_n'],'cohort':spec['attention_cohort'],
        'rows':'four autoregressive SID query steps','columns':'encoder input positions; SID/SEP for A/B, item aggregation tokens for MLP',
        'attention_kind':'decoder cross-attention','paths':['actual generation final Top-1 ancestry','local first parent per step'],
        'display':'last decoder layer, head mean; all raw heads retained','observer_off_on_is_not_shuffle_off_on':True,'scientific_similarity_claim':None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['identity-check','evaluate','metadata','train','attention']);p.add_argument('--spec',type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.spec.read_text());output=Path(os.environ['GRID_JOB_DIR'])/'output'
    assert not (output/'invocation.json').exists();output.mkdir(exist_ok=True)
    deployment=json.loads((REPO/'DEPLOYMENT.json').read_text());job=Path(os.environ['GRID_JOB_DIR'])
    request=json.loads((job/'request.json').read_text());status=json.loads((job/'status.json').read_text())
    assert request['capture_version']==deployment['source_commit']+'/'+deployment['snapshot_sha256'] and Path(request['cwd']).resolve()==REPO.resolve()
    assert {r['path']:r['sha256'] for r in request['inputs']}[str(a.spec)]==sha256(a.spec)
    if a.mode!='metadata':admit(spec)
    write_json(output/'invocation.json',{'argv':sys.argv,'interpreter':sys.executable,'spec':spec,'spec_sha256':sha256(a.spec),
        'code_identity':deployment,'phase':a.mode,'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES']})
    bind_source(spec['route']);cfg=formal_config(spec,output)
    if a.mode=='train':
        from .train import configure
        cfg=configure(spec,output)
    from omegaconf import OmegaConf
    OmegaConf.save(cfg,output/'effective_config.yaml',resolve=False)
    if a.mode=='identity-check':
        from experiments.beauty_extensions.mlp_formal import read_index
        index,identity=read_index(spec['test_index']);assert sum(index.values())==22363
        value={'status':'PASS','model_loaded':False,'GPU_model_calls':0,'test_targets':sum(index.values()),'coverage_index':identity,
            'preflight_inherited_from':'93567ea7f61e970c448838443595242d0b9beef0','formal_result':False}
    elif a.mode=='evaluate':value=evaluate(spec,cfg,output)
    elif a.mode=='metadata':value=metadata(spec,cfg,output)
    elif a.mode=='attention':value=attention(spec,output)
    else:
        from .train import execute
        from experiments.beauty_extensions.mlp_formal import deadline_record
        value=execute('formal',spec,cfg,output,deadline_record(spec,request,status))
        value.update({'checkpoint':value['best_checkpoint'],'sid':spec['sid'],'config':spec['config'],'shuffle_n':0})
    write_json(output/'RESULT.json',value)


if __name__=='__main__':main()
