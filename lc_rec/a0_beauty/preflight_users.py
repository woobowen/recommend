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


def main():
    p=argparse.ArgumentParser()
    for x in ('source-root','data-root','model','output'):p.add_argument('--'+x,required=True)
    a=p.parse_args()
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,a.source_root)
    from data import SeqRecDataset
    from transformers import LlamaTokenizer
    from uneven_sampler import assignment,validate_assignments
    from formal_evaluate import catalog_and_users
    args=SimpleNamespace(dataset='Beauty',data_path=a.data_root,max_his_len=20,his_sep=', ',index_file='.index.json',add_prefix=False)
    data=SeqRecDataset(args,mode='test',sample_num=-1);data.set_prompt(0)
    tok=LlamaTokenizer.from_pretrained(a.model,local_files_only=True,model_max_length=2048,padding_side='left')
    tok.pad_token_id=0
    assert tok.add_tokens(data.get_new_tokens())==965 and len(tok)==32965
    _,_,users,identities=catalog_and_users(a.data_root)
    axis=list(data.inters);assert axis==[u['user_id'] for u in users]
    shards=[assignment(len(data),rank,2) for rank in (0,1)];validate_assignments(shards,len(data))
    lengths=[len(tok(data[i]['input_ids'])['input_ids']) for i in range(len(data))]
    histories=[len(data.inters[uid][:-1][-20:]) for uid in axis]
    ranks={}
    for rank in (0,1):
        assigned=shards[rank]; chosen=[]
        def add(i):
            if i not in chosen: chosen.append(i)
        for lo,hi in ((0,5),(6,10),(11,20)):
            for i in [i for i in assigned if lo<=histories[i]<=hi][:2]:add(i)
        ordered=sorted(assigned,key=lambda i:(lengths[i],i))
        for i in (ordered[0],ordered[-1],ordered[-2],assigned[0],assigned[-1]):add(i)
        repeated=[i for i in assigned if len(set(data.inters[axis[i]][:-1][-20:]))<histories[i]]
        if repeated:add(repeated[0])
        for i in assigned:
            if len(chosen)>=16:break
            add(i)
        assert len(chosen)==16
        ranks[str(rank)]=[{**users[i], 'history_length':histories[i], 'input_tokens':lengths[i]} for i in chosen]
    manifest={'schema':'beauty_preflight_users_v1','selection':'fixed outcome-blind shuffle_cpu_inputs selection',
              'asset_sha256':{k:v['sha256'] for k,v in identities.items()},'ranks':ranks}
    validate(manifest,users,shards)
    path=Path(a.output)
    with path.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'users':32}))

if __name__=='__main__':main()
