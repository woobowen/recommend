"""Nonauthor shuffle_non_author_review program; executed by controller bridge."""
import argparse, collections, hashlib, itertools, json, pathlib, re, sys, time

VERSION = 'GRID-LETTER-SHUFFLE-V1'
def packed(x):
    return json.dumps(x,ensure_ascii=True,sort_keys=True,separators=(',', ':'),allow_nan=False).encode('utf-8')
def digest(x):return hashlib.sha256(packed(x)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def rows(paths):
    for path in paths:
        with open(path,encoding='utf-8') as f:
            for line_no,line in enumerate(f,1):
                if not line.strip():raise ValueError(f'Blank JSONL line: {path}:{line_no}')
                yield json.loads(line)
def draw(n,row,item):
    catalog=[p for p in itertools.permutations(range(n)) if p!=tuple(range(n))]
    ceiling=(2**256//len(catalog))*len(catalog);counter=0
    while True:
        value=int.from_bytes(hashlib.sha256(packed([VERSION,42,n,row,item,counter])).digest(),'big')
        if value<ceiling:
            number=value%len(catalog);return number,catalog[number],counter
        counter+=1

def main():
    if sys.flags.optimize:raise RuntimeError('Run without Python optimization')
    start=time.monotonic();p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('index','inter','audit','row-key','target-key','axis-key'):p.add_argument('--'+name,required=True)
    p.add_argument('--normal',nargs=2,required=True);p.add_argument('--shuffle',nargs=2,required=True)
    p.add_argument('--n',type=int,choices=(3,4),required=True);p.add_argument('--uid-key');a=p.parse_args()
    assert sha(a.index)=='33c87e28c52ae367985bdf494cb49d676d181e13a39497b4ad035f46111546dd'
    assert sha(a.inter)=='0a11d47791d90483005da93c36d826679e20e4592d1d2b86518d5f56f218e743'
    index,inter,audit=[json.loads(pathlib.Path(x).read_text()) for x in (a.index,a.inter,a.audit)]
    assert audit['status']=='PASS_CPU_FULL_INPUTS'
    axis,token_ids=list(inter),audit['token_ids']
    assert len(axis)==24772 and digest(axis)==audit[a.axis_key]
    assert len(index)==9922 and len({tuple(v) for v in index.values()})==9897
    assert len(token_ids)==len(set(token_ids.values()))==918
    assert set(token_ids)=={t for parts in index.values() for t in parts}
    reverse,vocabulary={v:k for k,v in token_ids.items()},set(token_ids.values());normal={}
    for row in rows(a.normal):
        i=row[a.row_key];assert type(i) is int and 0<=i<len(axis) and i not in normal
        if a.uid_key:assert str(row[a.uid_key])==axis[i]
        target=''.join(index[str(inter[axis[i]][-1])]);assert row[a.target_key]==target
        normal[i]=(row['input_token_ids'],target)
    assert len(normal)==24772
    frequency,lengths=collections.Counter(),collections.Counter()
    users=changed_users=group_count=decoded_count=0;symbol_pattern=re.compile(r'<[abcd]_[0-9]+>')
    for row in rows(a.shuffle):
        i=row[a.row_key];assert type(i) is int and 0<=i<len(axis)
        raw,target=normal.pop(i)
        if a.uid_key:assert str(row[a.uid_key])==axis[i]
        s,ev=row['shuffle'],row['shuffle']['evidence']
        assert (s['n'],s['seed'],s['protocol'],s['row_idx'])==(a.n,42,VERSION,i)
        assert s['cpu_manifest_sha256']==audit['manifest']['sha256']
        assert s['original_input_token_ids']==raw and s['attention_mask']==[1]*len(raw)
        assert row[a.target_key]==target
        visible=inter[axis[i]][:-1][-20:]
        groups=[[token_ids[t] for t in index[str(item)]] for item in visible]
        positions=[j for j,token in enumerate(raw) if token in vocabulary]
        assert len(positions)==4*len(groups) and len(ev['groups'])==len(groups)
        expected=list(raw)
        for j,old in enumerate(groups):
            pos=positions[4*j:4*j+4]
            assert len(old)==len(set(old))==4 and pos==list(range(pos[0],pos[0]+4))
            assert [raw[k] for k in pos]==old
            number,perm,counter=draw(a.n,i,j);full=perm+((3,) if a.n==3 else ())
            assert full!=(0,1,2,3);new=[old[k] for k in full]
            assert collections.Counter(new)==collections.Counter(old)
            assert a.n!=3 or new[3]==old[3]
            saved=ev['groups'][j]
            fields={'visible_history_item_index':j,'start':pos[0],'positions':pos,'permutation':list(full),
                    'permutation_id':number,'rejection_counter':counter,'original_ids':old,'shuffled_ids':new,'nonidentity':True}
            assert all(saved[k]==v for k,v in fields.items())
            for position,token in zip(pos,new):expected[position]=token
            frequency[str(list(perm))]+=1;group_count+=1
        actual=row['input_token_ids'];assert actual==expected and len(actual)==len(raw)
        history_positions=set(positions)
        assert all(actual[j]==raw[j] for j in range(len(raw)) if j not in history_positions)
        assert collections.Counter(actual)==collections.Counter(raw) and (not groups or actual!=raw)
        assert ev['original_sha256']==digest(raw) and ev['shuffled_sha256']==digest(actual)
        assert ev['attention_mask_sha256']==digest(s['attention_mask']) and ev['history_group_count']==len(groups)
        assert ev['input_changed']==(actual!=raw)
        assert all(ev[k] is True for k in ('nonhistory_unchanged','attention_mask_unchanged','sid_multiset_conserved'))
        assert (ev['protocol_version'],ev['seed'],ev['shuffle_n'],ev['row_idx'])==(VERSION,42,a.n,i)
        assert len(row['candidates'])==20
        history_symbols=[reverse[actual[k]] for k in positions]
        for candidate in row['candidates']:
            generated=candidate['generated_token_ids']
            assert len(generated)>=4 and all(t in reverse for t in generated[:4])
            assert symbol_pattern.findall(candidate['raw_decoded'])==history_symbols+[reverse[t] for t in generated[:4]]
            decoded_count+=1
        users+=1;changed_users+=actual!=raw;lengths[len(groups)]+=1
    assert not normal and users==24772 and decoded_count==495440
    assert sum(frequency.values())==group_count==172929
    paths=[a.index,a.inter,a.audit,*a.normal,*a.shuffle,__file__]
    result={'status':'PASS_INDEPENDENT_INPUT_REPLAY','users':users,'shuffle_n':a.n,'seed':42,'protocol':VERSION,
            'history_groups':group_count,'nonidentity_fraction':1.,'input_changed_fraction':changed_users/users,
            'permutation_frequencies':dict(sorted(frequency.items())),'history_lengths':dict(sorted(lengths.items())),
            'decoded_candidates_checked':decoded_count,'seconds':time.monotonic()-start,
            'files':[{'path':str(pathlib.Path(x).resolve()),'bytes':pathlib.Path(x).stat().st_size,'sha256':sha(x)} for x in paths]}
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=='__main__':main()
