#!/usr/bin/env python3
"""Read-only classification of author asset inconsistencies. Does not repair or regenerate labels."""
import argparse,collections,hashlib,json,pathlib,sys,time
p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();start=time.monotonic()
inter_path=a.data/'Instruments.inter.json';user_path=a.data/'Instruments.user.json'
inters=json.loads(inter_path.read_text());users=json.loads(user_path.read_text())
r={'kind':'REAL_ASSET_READ_ONLY_CLASSIFICATION','argv':sys.argv,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [inter_path,user_path]},'changes_to_assets':False,'sections':{},'limitations':['published inter file has no original timestamps, so raw chronological ordering and tie history cannot be reconstructed','matching item IDs/positions cannot prove the actual external model prompt or generated natural-language provenance','counts of held-out IDs in auxiliary fields establish inconsistency with this published split; historical generation cause remains unknown']}
for section,offset in [('train',-3),('test',-1)]:
    counts=collections.Counter();examples=collections.defaultdict(list)
    for uid,rec in users['user_vague_intention'][section].items():
        counts['users']+=1
        if uid not in inters:counts['user_absent_from_inter']+=1;continue
        seq=list(map(str,inters[uid]));target=str(rec['item']);history=list(map(str,rec['inters']));expected_h=seq[:offset]
        if target==seq[offset] and history==expected_h:counts['exact_expected_pair']+=1;continue
        counts['mismatching_expected_pair']+=1
        conditions={'target_expected':target==seq[offset],'history_expected':history==expected_h,'history_expected_last20':history==expected_h[-20:], 'target_in_training_items':target in seq[:-2], 'target_is_validation':target==seq[-2], 'target_is_test':target==seq[-1], 'target_absent_from_user_sequence':target not in seq, 'history_contains_validation':seq[-2] in history, 'history_contains_test':seq[-1] in history,'history_contains_items_outside_user_sequence':bool(set(history)-set(seq)),'history_is_ordered_subsequence_of_expected':False,'history_same_multiset_as_expected':collections.Counter(history)==collections.Counter(expected_h)}
        it=iter(expected_h);conditions['history_is_ordered_subsequence_of_expected']=all(any(x==y for y in it) for x in history)
        for key,value in conditions.items():
            if value:counts['mismatch_'+key]+=1
        categories=['mismatch_example']+[k for k in ['target_is_validation','target_is_test','history_contains_validation','history_contains_test','history_contains_items_outside_user_sequence'] if conditions[k]]
        example={'uid':uid,'full_interaction_item_ids':seq,'expected_target':seq[offset],'expected_history':expected_h,'actual_aux_target':target,'actual_aux_history':history,'target_positions':[i for i,v in enumerate(seq) if v==target],'conditions':conditions}
        for category in categories:
            if len(examples[category])<3:examples[category].append(example)
    r['sections'][section]={'counts':dict(counts),'example_rule':'first three mismatches in published auxiliary JSON insertion order per category; no metric selection','examples':dict(examples),'overlapping_flags':True}
r['elapsed_seconds']=time.monotonic()-start;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v['counts'] for k,v in r['sections'].items()},indent=2))
