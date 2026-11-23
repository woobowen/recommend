"""Verify every visible history item changed under aligned S3/S4 inference."""
from pathlib import Path
import json
from shuffle_protocol import PROTOCOL_VERSION

HERE=Path(__file__).resolve().parent
run=Path(json.loads((HERE/'latest_queue.json').read_text())['launch'])
assert json.loads((HERE/'fp16_aligned_verified.json').read_text())['complete']
summary={'complete':True,'protocol':PROTOCOL_VERSION,'shuffle_seed':42,
    'identity_allowed':False,'training_repeated':False,'datasets':{}}
for ds in ['Instruments','Beauty']:
    results={}
    for mode,n in [('unshuffle',0),('shuffle_3',3),('shuffle_4',4)]:
        current=json.loads((run/ds/f'test_fp16_aligned_{mode}.json').read_text())
        assert current['shuffle_protocol']==PROTOCOL_VERSION and current['shuffle_seed']==42
        assert current['adapter']==str(run/ds/'adapter') and len(current['rows'])==4
        before=json.loads((run/ds/'test_fp16_unshuffle.json').read_text())
        assert [(r['user_row'],r['target']) for r in current['rows']]==[(r['user_row'],r['target']) for r in before['rows']]
        if not n:assert [r['candidates'] for r in current['rows']]==[r['candidates'] for r in before['rows']]
        groups=changed=0
        for row in current['rows']:
            audit=row['shuffle_audit']
            assert audit['protocol_version']==PROTOCOL_VERSION and audit['seed']==42 and audit['shuffle_n']==n
            assert audit['nonhistory_unchanged'] and audit['attention_mask_unchanged'] and audit['sid_multiset_conserved']
            assert audit['input_changed']==bool(n)
            assert row['changed_history_items']==(row['history_items'] if n else 0)
            for g in audit['groups']:
                assert g['nonidentity']==bool(n)
                assert (g['original_ids']!=g['shuffled_ids'])==bool(n)
                assert sorted(g['original_ids'])==sorted(g['shuffled_ids'])
                if n!=4:assert g['original_ids'][3]==g['shuffled_ids'][3]
            groups+=row['history_items'];changed+=row['changed_history_items']
        results[mode]={'passed':True,'users':4,'history_items':groups,'changed_history_items':changed,
                       'changed_percent':changed/groups*100,'metrics':current['metrics']}
    summary['datasets'][ds]=results
(HERE/'shuffle_aligned_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
