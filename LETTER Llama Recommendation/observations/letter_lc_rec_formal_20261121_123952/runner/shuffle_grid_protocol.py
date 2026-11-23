"""Exact permutation mapping of GRID codex.plan_v1_1.protocol.ProtocolMixin.

All permutations, including identity; eval_seed=2026; lowercase dataset name;
actual user ID and zero-based visible history position. Padding is unchanged.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import itertools
import json
import operator

PROTOCOL_VERSION='GRID-PLAN-V1_1'


def integer(x):
    if isinstance(x,bool) or operator.index(x)<0:raise ValueError('Expected nonnegative integer')
    return int(x)


def digest(x):
    return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()


@lru_cache(maxsize=2)
def permutation_catalog(n):
    if n not in (3,4):raise ValueError('Shuffle count must be 3 or 4')
    return tuple(itertools.permutations(range(n)))


def permutation(n,uid,item_idx,dataset,seed=2026):
    n,uid,item_idx,seed=map(integer,(n,uid,item_idx,seed))
    choices=permutation_catalog(n)
    key=f'{seed}|{dataset.lower()}|{uid}|{item_idx}|shuffle_{n}'.encode()
    value=int.from_bytes(hashlib.sha256(key).digest(),'big')
    return choices[value%len(choices)]


def transform(input_ids,attention_mask,expected_groups,all_sid_ids,n,uid,dataset,seed=2026):
    n,uid,seed=map(integer,(n,uid,seed))
    if n not in (0,3,4):raise ValueError('Expected 0,3,4')
    ids=[integer(t) for t in input_ids];mask=list(attention_mask)
    if len(ids)!=len(mask) or any(x not in (0,1) for x in mask):raise ValueError('Invalid mask')
    vocab={integer(t) for t in all_sid_ids}
    if not vocab:raise ValueError('Empty SID vocabulary')
    groups=[[integer(t) for t in group] for group in expected_groups]
    if any(len(group)!=4 or any(t not in vocab for t in group) for group in groups):raise ValueError('Invalid group')
    positions=[i for i,t in enumerate(ids) if t in vocab]
    if len(positions)!=4*len(groups) or any(not mask[i] for i in positions):raise ValueError('SID boundary/padding mismatch')
    output=list(ids);records=[]
    for item_idx,group in enumerate(groups):
        pos=positions[4*item_idx:4*item_idx+4]
        if pos!=list(range(pos[0],pos[0]+4)) or [ids[i] for i in pos]!=group:raise ValueError('History group mismatch')
        full=permutation(n,uid,item_idx,dataset,seed)+tuple(range(n,4)) if n else (0,1,2,3)
        values=[group[j] for j in full]
        assert Counter(group)==Counter(values)
        assert n==4 or values[3]==group[3]
        for i,t in zip(pos,values):output[i]=t
        records.append({'visible_history_item_index':item_idx,'positions':pos,'permutation':list(full),
            'original_ids':group,'shuffled_ids':values,'nonidentity':full!=(0,1,2,3)})
    history=set(positions)
    assert all(output[i]==ids[i] for i in range(len(ids)) if i not in history)
    assert Counter(output)==Counter(ids)
    return output,{'protocol_version':PROTOCOL_VERSION,'seed':seed,'shuffle_n':n,'user_id':uid,
        'dataset':dataset.lower(),'history_group_count':len(groups),'identity_allowed':True,
        'input_changed':output!=ids,'nonhistory_unchanged':True,'attention_mask_unchanged':True,
        'sid_multiset_conserved':True,'original_sha256':digest(ids),'shuffled_sha256':digest(output),
        'groups':records}
