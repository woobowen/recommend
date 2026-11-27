from pathlib import Path
import json
a = Path('/data/bwwu/lc-rec/beauty-history-attention-20261127')
failed = json.loads((a/'training-regression-r1/training-regression.json').read_text())
assert failed['status']=='FAILED'
log = a/'train-r1/training/events.rank0.jsonl'
with log.open() as f:
    first = next(e for line in f if (e:=json.loads(line))['event']=='batch')
assert len(failed['real_batch']['sha256'])==4
for key in ('input_ids','labels','history_sid_mask'):
    match_key = {'input_ids':'input','labels':'label','history_sid_mask':'history_sid_mask'}[key]
    assert failed['real_batch']['sha256'][key] == first[match_key+'_sha256']
history = list((a/'train-r1/training/attempt_reports').glob('*.json'))
assert len(history)==1
rank = json.loads(history[0].read_text())
assert rank['status']=='BOUNDARY_PREFLIGHT_COMPLETE' and rank['global_step']==2
old = rank['parameter_update_validation']['before_sha256']
new = failed['initialized_trainable_sha256']
assert set(old)==set(new) and len(old)==450
mismatch=[k for k in old if old[k]!=new[k]]
print(json.dumps({'reproduced_initial_tensor_hashes':len(old)-len(mismatch),'mismatches':mismatch,
                  'same_real_first_batch_ids_labels_and_H':True,'training_evidence':str(history[0]),
                  'diagnostic_status':failed['status']}))
