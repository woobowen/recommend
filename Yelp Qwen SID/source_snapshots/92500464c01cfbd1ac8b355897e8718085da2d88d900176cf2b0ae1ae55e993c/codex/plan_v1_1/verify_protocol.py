"""Verify checkpoint compatibility and batched-versus-single-user test replay."""
import argparse
import json
import os
from pathlib import Path
import sys

ap=argparse.ArgumentParser()
ap.add_argument('output',type=Path)
ap.add_argument('--gpu',required=True)
args=ap.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OMP_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import rootutils
rootutils.setup_root(__file__,indicator='.project-root',pythonpath=True)
import torch
import hydra
from omegaconf import OmegaConf,open_dict
import src.utils.custom_hydra_resolvers
torch.use_deterministic_algorithms(True)
cfg=OmegaConf.load(args.output/'.hydra/config.yaml')
result=json.loads((args.output/'stage_result.json').read_text())
state=torch.load(result['checkpoint'],weights_only=False,map_location='cpu')['state_dict']
model=hydra.utils.instantiate(cfg.model)
model.load_state_dict(state,strict=True)
model.cuda().eval()
model._make_deterministic(False)
model.is_testing=True
dataset=result['dataset']
samples=torch.load(ROOT/f'experiments/sid_order_5seed_v1/fixed_test_users/{dataset}_100_users.pt',weights_only=False)[:8]
batch={model.feature_to_model_input_map.get(k,k):torch.cat([s['input'].transformed_sequences[k] for s in samples]).cuda()
       for k in samples[0]['input'].transformed_sequences}
mask=torch.cat([s['input'].mask for s in samples]).cuda()
modes=[('mlp',False,0)] if result['variant']=='mlp' else [('unshuffled',False,3),('shuffle_3',True,3),('shuffle_4',True,4)]
checks=[]
with torch.inference_mode():
    for mode,enabled,count in modes:
        if result['variant']=='shuffle':
            model.shuffle_embedding_on_test=enabled
            model.hierarchies_to_shuffle=count
        batch_ids,batch_scores=model.generate(attention_mask=mask,**batch)
        for i,sample in enumerate(samples):
            single={k:v[i:i+1] for k,v in batch.items()}
            ids,scores=model.generate(attention_mask=mask[i:i+1],**single)
            assert torch.equal(ids[0],batch_ids[i]),f'{mode}: changed ranking for {sample["user_id"]}'
            torch.testing.assert_close(scores[0],batch_scores[i],rtol=1e-4,atol=1e-6)
        checks.append({'mode':mode,'batch_size':8,'single_users_checked':8,'passed':True})
    # Adapter must add no trainable weights or change unshuffled inference.
    original_cfg=OmegaConf.create(OmegaConf.to_container(cfg.model,resolve=True))
    with open_dict(original_cfg):
        original_cfg._target_=f'codex.sid_order_launch.sid_order_{result["variant"]}_model.SemanticIDEncoderDecoder'
        del original_cfg['dataset_name']
        del original_cfg['eval_seed']
    original=hydra.utils.instantiate(original_cfg)
    original.load_state_dict(state,strict=True)
    original.cuda().eval()
    original._make_deterministic(False)
    if result['variant']=='shuffle':
        original.is_testing=True
        original.shuffle_embedding_on_test=False
        model.shuffle_embedding_on_test=False
    a=model.generate(attention_mask=mask,**batch)
    b=original.generate(attention_mask=mask,**batch)
    assert torch.equal(a[0],b[0])
    torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
report={'dataset':dataset,'variant':result['variant'],'batch_replay':checks,
        'original_architecture_state_dict_compatible':True,'unshuffled_original_predictions_exact':True}
(args.output/'protocol_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
