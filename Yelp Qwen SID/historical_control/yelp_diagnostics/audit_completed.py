"""Read-only diagnostic evidence for the completed Yelp run; CPU only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3',
                  TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.utils.custom_hydra_resolvers
import torch
torch.set_num_threads(4)
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main():
    data=Path('/home/jcy/data/yelp_aligned_v1')
    full=json.loads((data/'full_sequences.json').read_text())
    sequences=list(full.values())
    freq=Counter(i for s in sequences for i in s[:-2])
    test=Counter(s[-1] for s in sequences)
    val=Counter(s[-2] for s in sequences)
    report=dict(audited_utc=datetime.now(timezone.utc).isoformat(),read_only=True,
        nusers=len(full),nitems=18775,train_unseen_catalog_items=18775-len(freq),
        test_target_unseen_train_fraction=sum(v for i,v in test.items() if i not in freq)/len(full),
        val_target_unseen_train_fraction=sum(v for i,v in val.items() if i not in freq)/len(full))
    for k in (5,10):
        popular={i for i,_ in freq.most_common(k)}
        report[f'popularity_test_recall{k}']=sum(v for i,v in test.items() if i in popular)/len(full)
    texts=json.loads((data/'item_texts.json').read_text())
    counts=Counter(texts.values())
    report.update(unique_texts=len(counts),items_with_duplicate_text=sum(n for n in counts.values() if n>1),
                  max_identical_text_items=max(counts.values()))
    out=ROOT/'experiments/sid_order_yelp_5seed_v1/shuffle/seed_43'
    result=json.loads((out/'stage_result.json').read_text())
    sid=torch.load(ROOT/'experiments/yelp_rkmeans_qwen_aligned_v1/semantic_ids.pt',weights_only=False,map_location='cpu')
    samples=torch.load(ROOT/'experiments/sid_order_yelp_5seed_v1/fixed_test_users/yelp_aligned_v1_100_users.pt',weights_only=False,map_location='cpu')
    for s in samples:
        seq=full[str(s['user_id'])]
        expected=sid[:,seq[-1]].reshape(-1)
        assert all(torch.equal(x.reshape(-1).long(),expected) for x in s['labels'].labels.values())
        expected_history=sid[:,seq[:-1][-20:]].T.reshape(-1)
        x=s['input'].transformed_sequences['sequence_data'][0]
        mask=s['input'].mask[0]
        assert torch.equal(x[mask.bool()].long(),expected_history.long())
    report['all100_sample_labels_and_history_match_item_sid_mapping']=True
    refs={m:torch.load(out/f'full_test_reference_{m}.pt',map_location='cpu',weights_only=False)
          for m in ('unshuffled','shuffle_3','shuffle_4')}
    base=refs['unshuffled']
    lists=[tuple(v['ids'].reshape(-1).tolist()) for v in base.values()]
    firsts=[tuple(v['ids'][0].tolist()) for v in base.values()]
    sample_report=dict(users=len(base),unique_top10_lists=len(set(lists)),unique_top1_items=len(set(firsts)),
                       most_common_top1_count=Counter(firsts).most_common(1)[0][1])
    for mode in ('shuffle_3','shuffle_4'):
        sample_report[mode]=dict(same_ranked_ids=sum(torch.equal(x['ids'],refs[mode][u]['ids']) for u,x in base.items()),
            max_score_abs_difference=max(float((x['scores']-refs[mode][u]['scores']).abs().max()) for u,x in base.items()))
    report['sample100']=sample_report
    cfg=OmegaConf.load(out/'.hydra/config.yaml')
    model=instantiate(cfg.model)
    checkpoint=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    model.is_testing=True
    model.protocol_sample_ids=set()
    # Replay a small fixed sample on CPU; then exchange real histories between users.
    selected=samples[:8]
    inputs=torch.cat([s['input'].transformed_sequences['sequence_data'] for s in selected])
    users=torch.cat([s['input'].transformed_sequences['user_id'] for s in selected])
    masks=torch.cat([s['input'].mask for s in selected])
    with torch.inference_mode():
        pred,score=model.generate(attention_mask=masks,input_ids=inputs,user_id=users)
        swapped,swapped_score=model.generate(attention_mask=masks.roll(1,0),input_ids=inputs.roll(1,0),user_id=users)
    same_saved=[torch.equal(pred[i],base[s['user_id']]['ids']) for i,s in enumerate(selected)]
    assert all(same_saved)
    report['cpu_checkpoint_replay']=dict(users=8,all_match_saved_predictions=all(same_saved),
        exchanged_histories_same_top10=sum(torch.equal(a,b) for a,b in zip(pred,swapped)),
        exchanged_histories_max_score_abs_difference=float((score-swapped_score).abs().max()),
        checkpoint_step=int(checkpoint['global_step']),training_stop_step=result['global_step'])
    for dataset in ('beauty','toys'):
        f=ROOT/f'experiments/sid_order_5seed_v1/{dataset}/shuffle/seed_43/full_test_reference_unshuffled.pt'
        if f.exists():
            reference=torch.load(f,map_location='cpu',weights_only=False)
            report[f'{dataset}_seed43_sample_comparison']=dict(users=len(reference),
                unique_top10_lists=len({tuple(v['ids'].reshape(-1).tolist()) for v in reference.values()}),
                unique_top1_items=len({tuple(v['ids'][0].tolist()) for v in reference.values()}))
    destination=Path(__file__).resolve().parent/'completed_seed43_diagnosis.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
