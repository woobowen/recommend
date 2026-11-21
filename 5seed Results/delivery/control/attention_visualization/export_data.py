"""Export verified attention tensors for plotting and an offline viewer."""
import argparse
import base64
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/sid_order_5seed_v1'
MODES=('unshuffled','shuffle_3','shuffle_4','mlp')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_indices(payload):
    n=payload['history_items'];stride=payload['encoder_tokens_per_item']
    if stride==1:
        return np.arange(n)
    perm=payload['permutation'][0,:n].numpy()
    rows=[]
    for item in range(n):
        rows.extend((item*stride+np.argsort(perm[item])).tolist())
        if stride==5:rows.append(item*stride+4)
    return np.array(rows)


def extract(payload):
    keep=np.flatnonzero(payload['encoder_mask'][0].numpy())
    n=int(payload['history_items']);stride=int(payload['encoder_tokens_per_item'])
    assert np.array_equal(keep,np.arange(n*stride)), 'Unexpected non-contiguous valid tokens'
    arrays={}
    for name,values in payload['attention'].items():
        a=torch.stack(values).squeeze(1).float().numpy()
        assert a.shape[:2]==(4,6) and np.isfinite(a).all()
        if name=='encoder_self':a=a[:,:,keep,:][:,:,:,keep]
        elif name=='decoder_cross':a=a[:,:,:,keep]
        else:assert a.shape[-2:]==(4,4)
        assert np.max(np.abs(a.sum(-1)-1))<2e-4,(name,float(np.max(np.abs(a.sum(-1)-1))))
        arrays[name]=a
    return arrays


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    catalog=dict(created_utc=datetime.now(timezone.utc).isoformat(),datasets={},modes=list(MODES),
        layers=4,heads=6,display_quantization='uint16 attention / 65535; maximum rounding error 0.00000763; source raw tensors remain authoritative',
        source_root=str(SOURCE),aggregation='Each user equal weight within seed; each of five seeds equal weight. Recency aligns from most recent item; missing positions contribute zero.')
    aggregate={};audit=[];sources=[];selected={};case_count=0
    for dataset in ('beauty','toys'):
        sample=json.loads((SOURCE/f'fixed_test_users/{dataset}_100_users.json').read_text())
        users=sorted(sample['users'],key=lambda x:(x['history_items'],x['user_id']))
        assert len(users)==100
        representative=users[len(users)//2]['user_id']
        catalog['datasets'][dataset]=dict(users=users,representative_user=representative,seeds=list(range(42,47)))
        for seed in range(42,47):
            manifests={};folders={}
            for mode in MODES:
                variant='mlp' if mode=='mlp' else 'shuffle'
                folder=SOURCE/dataset/variant/f'seed_{seed}'/'attention_analysis'/mode
                manifest=json.loads((folder/'manifest.json').read_text())
                stage=json.loads((folder.parent.parent/'stage_result.json').read_text())
                assert stage['completed'] and manifest['users']==100
                assert manifest['checkpoint_sha256']==stage['checkpoint_sha256']
                manifests[mode]={x['user_id']:x for x in manifest['files']};folders[mode]=folder
                assert set(manifests[mode])=={u['user_id'] for u in users}
            seed_stats=defaultdict(list)
            for user in users:
                uid=user['user_id'];case=dict(dataset=dataset,seed=seed,user_id=uid,history_items=user['history_items'],modes={})
                packed=[];offset=0;full_arrays={};payloads={};items={}
                for mode in MODES:
                    f=folders[mode]/manifests[mode][uid]['file']
                    digest=sha(f);assert digest==manifests[mode][uid]['sha256']
                    with gzip.open(f,'rb') as stream:payload=torch.load(stream,map_location='cpu',weights_only=False)
                    assert payload['user_id']==uid and payload['history_items']==user['history_items']
                    arrays=extract(payload);full_arrays[mode]=arrays;payloads[mode]=payload
                    n=payload['history_items'];stride=payload['encoder_tokens_per_item'];idx=canonical_indices(payload)
                    meta=dict(stride=stride,permutation=payload['permutation'][0,:n].tolist(),
                        input_sids=payload['input_sids'][0,:n*4].reshape(n,4).tolist(),
                        target_sid=next(iter(payload['true_labels'].values())).reshape(-1).tolist(),
                        top10_ids=payload['top10_ids'][0].tolist(),top1_prefix=payload['top1_prefix'][0].tolist(),
                        checkpoint_sha256=payload['checkpoint_sha256'],source_sha256=digest,source_file=str(f),arrays={})
                    for name,a in arrays.items():
                        q=np.rint(np.clip(a,0,1)*65535).astype('<u2')
                        assert np.max(np.abs(a-q.astype(np.float32)/65535))<=8e-6
                        meta['arrays'][name]=dict(shape=list(a.shape),offset=offset,length=int(a.size))
                        offset+=a.size;packed.append(q.reshape(-1))
                    case['modes'][mode]=meta
                    canonical=arrays['decoder_cross'][:,:,:,idx]
                    item_attention=canonical.reshape(4,6,4,n,stride).sum(-1)
                    items[mode]=item_attention
                    recency=np.zeros((4,6,4,20),dtype=np.float64)
                    recency[:,:,:,-n:]=item_attention
                    seed_stats[mode+'_recency'].append(recency)
                    seed_stats[mode+'_decoder_self'].append(arrays['decoder_self'])
                    if stride>1:
                        seed_stats[mode+'_sid_mass'].append(canonical.reshape(4,6,4,n,stride).sum(-2))
                    entropy=-(item_attention*np.log(np.clip(item_attention,1e-30,None))).sum(-1)/np.log(n)
                    seed_stats[mode+'_item_entropy'].append(entropy)
                    if seed==42 and uid==representative:
                        selected[(dataset,mode)]=dict(arrays=arrays,payload=meta,n=n,uid=uid)
                    sources.append(dict(dataset=dataset,seed=seed,mode=mode,user_id=uid,source_file=str(f),sha256=digest))
                for mode in ('shuffle_3','shuffle_4'):
                    base=payloads['unshuffled'];other=payloads[mode]
                    same_prefix=torch.equal(base['top1_prefix'],other['top1_prefix'])
                    tv=.5*np.abs(items[mode]-items['unshuffled']).sum(-1)
                    seed_stats[mode+'_item_tv'].append(tv)
                    if same_prefix:seed_stats[mode+'_item_tv_matched_prefix'].append(tv)
                    audit.append(dict(dataset=dataset,seed=seed,user_id=uid,mode=mode,same_top1_prefix=same_prefix,
                        same_top1=torch.equal(base['top10_ids'][:,0],other['top10_ids'][:,0]),
                        same_top10=torch.equal(base['top10_ids'],other['top10_ids']),
                        last_layer_item_tv=float(tv[-1].mean())))
                raw=np.concatenate(packed).tobytes();compressed=gzip.compress(raw,compresslevel=3,mtime=0)
                js='window.attentionCaseLoaded('+json.dumps(case,separators=(',',':'))+','+json.dumps(base64.b64encode(compressed).decode())+');\n'
                (out/'data'/f'{dataset}_{seed}_{uid}.js').write_text(js,encoding='utf-8');case_count+=1
            for name,values in seed_stats.items():
                aggregate[f'{dataset}_{seed}_{name}']=np.mean(values,axis=0)
                if name.endswith('matched_prefix'):aggregate[f'{dataset}_{seed}_{name}_count']=np.array(len(values))
            print(f'Exported {dataset} seed {seed}: 100 users x 4 conditions',flush=True)
    np.savez_compressed(out/'aggregates.npz',**aggregate)
    # Small representative payloads for standard matplotlib figures.
    torch.save(selected,out/'representatives.pt')
    (out/'catalog.js').write_text('window.ATTENTION_CATALOG='+json.dumps(catalog,separators=(',',':'))+';\n',encoding='utf-8')
    (out/'catalog.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
    (out/'attention_comparisons.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (out/'source_manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    verification=dict(cases=case_count,source_user_files=len(sources),datasets=['beauty','toys'],seeds=list(range(42,47)),
        source_hashes_verified=True,all_four_layers_all_six_heads=True,all_padding_removed=True,
        valid_attention_row_sums_verified=True,display_quantization_max_error=8e-6,
        canonical_alignment='For each item, inverse of exported output-position -> original-SID permutation; SEP fixed. Encoder self aligns both axes; cross aligns key axis.',
        comparison_caveat='Cross-attention recomputed along each condition final Top1 prefix; differing prefixes are flagged and are not a controlled same-prefix causal comparison.',
        aggregated_item_keys='Sum SID and SEP key attention within each item; MLP already has one key per item.',
        interpretation='Attention weights describe allocation, not causal importance or proof of invariance.')
    (out/'export_verified.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()
