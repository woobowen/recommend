"""Build and validate the frozen outcome-blind Beauty preflight user axis."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(manifest, users, shards):
    ranks=manifest['ranks']
    if set(ranks)!={'0','1'} or len(users)!=22363 or len(shards)!=2:
        raise ValueError('bad user axis or ranks')
    selected=[]
    for rank in (0,1):
        records=ranks[str(rank)]
        if len(records)!=16: raise ValueError('expected 16 users per rank')
        eligible=set(shards[rank])
        for record in records:
            index=record['row_idx']
            if type(index) is not int or index not in eligible or not 0<=index<len(users):
                raise ValueError('row not assigned to rank')
            if any(record[k]!=users[index][k] for k in ('row_idx','user_id','target_item_id','target')):
                raise ValueError('user identity or target differs')
            selected.append(index)
    if len(set(selected))!=32: raise ValueError('duplicate or missing rows')
    return [[r['row_idx'] for r in ranks[str(rank)]] for rank in (0,1)]



def frozen_g3_manifest(index, inters, lengths=None):
    """Reuse the previously fixed user IDs; derive G3 fields before model outputs."""
    from g3_identity import ASSETS, EXPERIMENT_ID, INDEX_SHA, BRIDGE_SHA, G3_SHA
    old_path = Path(__file__).with_name('config')/'preflight_users_seed42.json'
    old = json.loads(old_path.read_text())
    ranks = {}
    for rank, records in old['ranks'].items():
        chosen = []
        for prior in records:
            uid = prior['user_id']
            seq = inters[uid]
            if str(seq[-1]) != prior['target_item_id']:
                raise ValueError('Frozen preflight user target-item identity changed')
            visible = seq[:-1][-20:]
            chosen.append({k:prior[k] for k in ('row_idx','user_id','target_item_id')})
            chosen[-1].update(target=''.join(index[str(seq[-1])]),history_length=len(visible),
                input_tokens=lengths[prior['row_idx']] if lengths is not None else prior['input_tokens'],
                fourth_position_zero_history_items=sum(index[str(i)][3]=='<d_0>' for i in visible),
                fourth_position_nonzero_history_items=sum(index[str(i)][3]!='<d_0>' for i in visible),
                repeated_visible_history_item=len(set(visible))<len(visible))
        ranks[rank] = chosen
    repeated = sum(len(set(seq[:-1][-20:]))<len(seq[:-1][-20:]) for seq in inters.values())
    return {'schema':'beauty_g3_preflight_users_v1','scope':'PREFLIGHT','experiment_id':EXPERIMENT_ID,
        'selection':'Reuse exact prior 32 user IDs, fixed before any G3 model outputs; derive G3 targets',
        'prior_manifest_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),
        'asset_sha256':ASSETS,'upstream_sha256':{'g3':G3_SHA,'bridge':BRIDGE_SHA,'index':INDEX_SHA},
        'ranks':ranks,'boundary_coverage':{'long_and_short_history':True,
            'fourth_position_zero_and_nonzero':True,'extra_left_padding_tokens_on_first_user_per_rank':3,
            'whole_fixed_corpus_repeated_visible_history_users':repeated,
            'repeated_history_note':'No real repeated visible-history item exists; isolated synthetic checks only'}}


def main():
    p=argparse.ArgumentParser()
    for x in ('source-root','data-root','model','output'):p.add_argument('--'+x,required=True)
    a=p.parse_args()
    import sys
    from types import SimpleNamespace
    from g3_identity import NEW_TOKEN_COUNT,VOCAB_SIZE,tokenizer_identity
    sys.path.insert(0,a.source_root)
    from data import SeqRecDataset
    from transformers import LlamaTokenizer
    from uneven_sampler import assignment,validate_assignments
    from formal_evaluate import catalog_and_users
    args=SimpleNamespace(dataset='Beauty',data_path=a.data_root,max_his_len=20,his_sep=', ',index_file='.index.json',add_prefix=False)
    data=SeqRecDataset(args,mode='test',sample_num=-1);data.set_prompt(0)
    tok=LlamaTokenizer.from_pretrained(a.model,local_files_only=True,model_max_length=2048,padding_side='left')
    tok.pad_token_id=0
    if tok.add_tokens(data.get_new_tokens())!=NEW_TOKEN_COUNT or len(tok)!=VOCAB_SIZE:
        raise ValueError('Fixed G3 tokenizer size')
    tokenizer_identity(tok,data.indices)
    index,_,users,identities=catalog_and_users(a.data_root)
    axis=list(data.inters)
    if axis!=[u['user_id'] for u in users]:raise ValueError('User axis mismatch')
    shards=[assignment(len(data),rank,2) for rank in (0,1)];validate_assignments(shards,len(data))
    wanted={r['row_idx'] for rows in json.loads((Path(__file__).with_name('config')/'preflight_users_seed42.json').read_text())['ranks'].values() for r in rows}
    lengths={i:len(tok(data[i]['input_ids'])['input_ids']) for i in wanted}
    manifest=frozen_g3_manifest(index,data.inters,lengths)
    validate(manifest,users,shards)
    path=Path(a.output)
    with path.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'users':32}))

if __name__=='__main__':main()
