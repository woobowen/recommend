"""Bounded real checks, unchanged native loaders, model, Shuffle and lifecycle."""
import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from experiments.beauty_extensions.common import (
    REPO, instantiate, load_checkpoint, sha256, write_json,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True,type=Path)
    args=parser.parse_args();spec=json.loads(args.spec.read_text())
    assert spec['domain'] in ('toys','sports') and spec['route'] in ('A','B')
    assert spec['shuffle_n']==0 and spec['generation_k']==10
    assert spec['engineering_only'] is True
    job=Path(os.environ['GRID_JOB_DIR']);output=job/'output'
    output.mkdir(exist_ok=True);assert not (output/'invocation.json').exists()
    deployment=json.loads((REPO/'DEPLOYMENT.json').read_text())
    request=json.loads((job/'request.json').read_text())
    assert request['capture_version']==deployment['source_commit']+'/'+deployment['snapshot_sha256']
    assert Path(request['cwd']).resolve()==REPO.resolve()
    assert {x['path']:x['sha256'] for x in request['inputs']}[str(args.spec)]==sha256(args.spec)
    write_json(output/'invocation.json',{'argv':sys.argv,'interpreter':sys.executable,
        'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'spec':spec,
        'spec_sha256':sha256(args.spec),'code_identity':deployment,
        'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'engineering_only':True})
    start=time.monotonic()
    cfg,model,dm=instantiate(spec,output,checkpoint=True)
    from omegaconf import OmegaConf
    from experiments.beauty_extensions.cli import check_data,smoke,clear_runtime
    from experiments.beauty_extensions.data_checks import prepare_fixed_attention_samples,scan_test_coverage
    from experiments.beauty_extensions.results import recompute
    from experiments.preflight_20261015.run import tensor_tree_digest
    import torch
    saved=OmegaConf.load(output/'effective_config.yaml')
    assert OmegaConf.to_container(saved,resolve=True)==OmegaConf.to_container(cfg,resolve=True)
    assert cfg.model.top_k_for_generation==10
    batches=check_data(spec,cfg,output)
    prepare_fixed_attention_samples(cfg,spec,output)
    coverage=scan_test_coverage(cfg,spec,output)
    truncated=coverage.get('first_truncated_engineering_example')
    if truncated is not None:
        from src.data.loading.components.interfaces import SequentialModelInputData,SequentialModuleLabelData
        x=SequentialModelInputData(user_id_list=[truncated['sample_key']],
            transformed_sequences={'sequence_data':torch.tensor([truncated['input_ids']],dtype=torch.long)},
            mask=torch.tensor([truncated['attention_mask']],dtype=torch.long))
        y=SequentialModuleLabelData(labels={'sequence_data':torch.tensor(truncated['target'],dtype=torch.long)})
        batches.append((x,y))
    input_digest=tensor_tree_digest(batches)
    try:
        smoke(spec,cfg,model,dm,output,batches)
        assert tensor_tree_digest(batches)==input_digest,'Input, mask or labels changed'
        for n in (0,3,4):
            folder=output/f'trainer-test-{n}'
            recompute(spec,cfg,folder/'predictions.jsonl',folder,trainer_metrics=folder/'metrics.json')
            trace=json.loads((folder/'intervention_trace.json').read_text())
            assert bool(trace['events'])==(n!=0)
            delta=trace['role_count_delta']['encoder_role_counts']
            assert delta.get('loss',0)==len(batches) and delta.get('generate',0)==len(batches)
            if spec['route']=='A':
                assert trace['shuffle_call_count_delta']==(2*len(batches) if n else 0)
            else:
                changed=trace['role_count_delta']['shuffle_role_counts']
                assert changed.get('loss',0)==(len(batches) if n else 0)
                assert changed.get('generate',0)==(len(batches) if n else 0)
        state=load_checkpoint(spec['checkpoint']['path'])['state_dict']
        actual=model.state_dict()
        assert set(actual)==set(state) and all(torch.equal(actual[k].cpu(),v.cpu()) for k,v in state.items())
    finally:
        clear_runtime(model)
    write_json(output/'environment.json',{'versions':{p:importlib.metadata.version(p) for p in
        ('torch','transformers','lightning','hydra-core','tensorflow-cpu','numpy')},
        'device':torch.cuda.get_device_name(0),'visible_device_count':torch.cuda.device_count()})
    write_json(output/'RESULT.json',{'status':'PASS','formal':False,'engineering_only':True,
        'domain':spec['domain'],'route':spec['route'],'test_targets':coverage['targets'],
        'checkpoint_parameters_unchanged':True,'input_mask_labels_unchanged':True,
        'effective_config_roundtrip_exact':True,'separate_loss_generate_permutation_roles_verified':True,
        'test_index':{'path':str(output/'test_target_index.jsonl'),'sha256':sha256(output/'test_target_index.jsonl')},
        'attention_cohort':{'path':str(output/'fixed_attention_samples.json'),'sha256':sha256(output/'fixed_attention_samples.json')},
        'elapsed_seconds':time.monotonic()-start,'GPU_check_budget_seconds':request['timeout_seconds'],
        'test_protocol':'existing smoke checks Normal OFF vs original, Shuffle3/4 independently in real Trainer.test; loss/generate resample separately'})


if __name__=='__main__':main()
