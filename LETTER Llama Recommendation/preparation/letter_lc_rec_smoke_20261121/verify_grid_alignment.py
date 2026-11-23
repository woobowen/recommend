"""Audit token permutations, targets and output provenance of GRID replay."""
from pathlib import Path
import json
from shuffle_grid_protocol import PROTOCOL_VERSION, permutation

HERE=Path(__file__).resolve().parent
run=Path(json.loads((HERE/'latest_queue.json').read_text())['launch'])
assert json.loads((HERE/'fp16_grid_verified.json').read_text())['complete']
out={'complete':True,'protocol':PROTOCOL_VERSION,'shuffle_seed':2026,'identity_allowed':True,
     'training_repeated':False,'datasets':{}}
for ds in ['Instruments','Beauty']:
    data={}
    for mode,n in [('unshuffle',0),('shuffle_3',3),('shuffle_4',4)]:
        current=json.loads((run/ds/f'test_fp16_grid_{mode}.json').read_text())
        before=json.loads((run/ds/'test_fp16_unshuffle.json').read_text())
        assert current['shuffle_protocol']==PROTOCOL_VERSION and current['shuffle_seed']==2026
        assert current['adapter']==str(run/ds/'adapter') and current['users']==4
        assert [(r['user_row'],r['target']) for r in current['rows']]==[(r['user_row'],r['target']) for r in before['rows']]
        if not n:assert [r['candidates'] for r in current['rows']]==[r['candidates'] for r in before['rows']]
        groups=changed=0
        for row in current['rows']:
            audit=row['shuffle_audit']
            assert audit['protocol_version']==PROTOCOL_VERSION and audit['identity_allowed']
            assert audit['seed']==2026 and audit['dataset']==ds.lower()
            assert audit['nonhistory_unchanged'] and audit['attention_mask_unchanged'] and audit['sid_multiset_conserved']
            for g in audit['groups']:
                pos=g['visible_history_item_index']
                expected=list(permutation(n,audit['user_id'],pos,ds.lower(),2026))+list(range(n,4)) if n else [0,1,2,3]
                assert g['permutation']==expected
                assert g['shuffled_ids']==[g['original_ids'][i] for i in expected]
                assert g['nonidentity']==(expected!=[0,1,2,3])
                if n!=4:assert g['original_ids'][3]==g['shuffled_ids'][3]
            groups+=row['history_items'];changed+=row['changed_history_items']
        data[mode]={'passed':True,'users':4,'history_items':groups,'changed_history_items':changed,
            'unchanged_history_items':groups-changed,'metrics':current['metrics']}
    out['datasets'][ds]=data
(HERE/'shuffle_grid_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
