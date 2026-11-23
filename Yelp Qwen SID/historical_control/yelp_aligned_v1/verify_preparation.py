"""Final provenance and readiness audit, with no training."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'experiments/yelp_rkmeans_qwen_aligned_v1'
DATA=Path('/home/jcy/data/yelp_aligned_v1')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    # Match the normal launcher import order before unpickling dataclass configs.
    import src.utils.custom_hydra_resolvers
    status=json.loads((RUN/'pipeline_status.json').read_text())
    assert status['stage']=='all_preparation' and status['state']=='completed'
    assert status['formal_training_started'] is False
    original=json.loads((RUN/'original_code_hashes.json').read_text())
    assert all(sha(p)==h for p,h in original.items())
    manifest=json.loads((DATA/'file_manifest.json').read_text())
    assert all(sha(DATA/p)==h for p,h in manifest.items())
    info=json.loads((RUN/'upstream_audit.json').read_text())
    checkpoint=torch.load(info['quantizer_checkpoint'],map_location='cpu',weights_only=False)
    assert checkpoint['layers_initialized']==[True,True,True]
    assert checkpoint['global_step']==30
    vectors=torch.load(RUN/'item_embeddings.pt',map_location='cpu',weights_only=False)
    assert vectors.shape==(18775,3584) and vectors.dtype==torch.float32 and torch.isfinite(vectors).all()
    assert bool((vectors.norm(dim=1)>0).all())
    del vectors
    sample_path=ROOT/'experiments/sid_order_yelp_5seed_v1/fixed_test_users/yelp_aligned_v1_100_users.json'
    sample=json.loads(sample_path.read_text())
    users=[row['user_id'] for row in sample['users']]
    assert len(users)==len(set(users))==100 and sample['valid_unique_users']==28330
    checks={}
    for variant in ('shuffle','mlp'):
        d=RUN/f'readonly_{variant}'
        check=json.loads((d/'preparation_verified.json').read_text())
        assert check['passed'] and not check['fit_called'] and not check['test_called'] and check['optimizer_steps']==0
        assert json.loads((d/'sample_manifest.json').read_text())==sample
        assert not list(d.rglob('*.ckpt'))
        checks[variant]=check
    report=dict(ready_for_formal_training=True,formal_training_started=False,
        original_amazon_code_unchanged=True,dataset_all_file_hashes_verified=True,
        quantizer_all_3_layers_initialized=True,quantizer_global_step=checkpoint['global_step'],
        vector_shape=[18775,3584],sid_shape=info['sid_shape'],unique_sid_items=info['unique_items'],
        fixed_attention_users=users,samples_sha256=sample['samples_sha256'],
        downstream_readonly_checks=checks,
        artifact_sha256={str(p):sha(p) for p in [RUN/'item_embeddings.pt',RUN/'semantic_ids.pt',
            Path(info['quantizer_checkpoint']),DATA/'file_manifest.json',sample_path]})
    (RUN/'readiness_verified.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('downstream_readonly_checks','artifact_sha256','fixed_attention_users')},indent=2))


if __name__=='__main__':
    main()
