#!/usr/bin/env python3
"""Validate identity, missing, duplicate and wrong-rank rejection without weights."""
import copy
from preflight_users import validate
users=[{'row_idx':i,'user_id':str(i),'target_item_id':str(i),'target':'target'+str(i)} for i in range(22363)]
shards=[list(range(0,22363,2)),list(range(1,22363,2))]
manifest={'ranks':{'0':[users[i].copy() for i in shards[0][:16]],'1':[users[i].copy() for i in shards[1][:16]]}}
assert validate(manifest,users,shards)==[x[:16] for x in shards]
for kind in ('missing','duplicate','identity','rank'):
    m=copy.deepcopy(manifest)
    if kind=='missing':m['ranks']['0'].pop()
    if kind=='duplicate':m['ranks']['0'][1]=m['ranks']['0'][0].copy()
    if kind=='identity':m['ranks']['0'][0]['target']='bad'
    if kind=='rank':m['ranks']['0'][0]=users[1].copy()
    try: validate(m,users,shards)
    except ValueError: pass
    else: raise AssertionError('accepted '+kind)
print('PASS: manifest identity, missing, duplicate, wrong-rank rejected')
