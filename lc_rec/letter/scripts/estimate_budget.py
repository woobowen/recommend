"""Budget counts from completed CPU evidence; failed GPU runs cannot supply rates."""
import argparse
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--audit', required=True)
p.add_argument('--train-dir', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
audit_path = Path(a.audit)
j = json.loads(audit_path.read_text())
assert j['status'] == 'PASS'
c = j['budget_counts']
n = c['train_rows_per_epoch']
v = c['valid_two_prompt_rows_per_epoch']
t = c['test_prompt0_rows']
epochs, batch = c['epochs'], c['global_batch']
assert n == j['original_dataset_execution']['train_rows']
assert v == j['original_dataset_execution']['valid_rows_after_two_prompts']
assert t == j['original_dataset_execution']['test_rows']
assert t * j['seqrec_prompt_count'] == c['test_all_17_prompt_rows']
assert epochs == 4 and batch == 64
train = [json.loads(q.read_text()) for q in sorted(Path(a.train_dir).glob('rank[01].json'))]
assert len(train) == 2
assert all(x['status'] == 'FAIL' and not x['updates'] for x in train), 'This negative-result estimator requires the actual failed zero-update evidence'
updates = epochs * math.ceil(n / batch)
r = {
    'status': 'MISSING_GPU_MEASUREMENTS',
    'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    'count_source': 'Completed CPU audit budget_counts, cross-checked with actual original dataset execution',
    'observed_counts': {
        'train_per_epoch': n, 'epochs': epochs, 'train_presentations': n * epochs,
        'valid_texts_per_epoch': v, 'total_validation_texts': v * epochs,
        'test_users_per_prompt': t, 'test_prompt0_presentations': t,
        'test_17_prompt_presentations': c['test_all_17_prompt_rows'],
        'effective_batch': batch, 'updates_per_epoch': math.ceil(n / batch),
        'optimizer_updates_four_epochs': updates,
        'scheduler_warmup_steps': math.ceil(updates * .01),
    },
    'rounding_limit': c['update_rounding_note'],
    'actual_complete_optimizer_updates_by_rank': [len(x['updates']) for x in train],
    'publication_prompt_protocol': 'Unknown historically; public run_test_ddp.sh specifies prompt0. 17 prompts is a separate cost scenario.',
    'reason': 'Approved local precision adaptation completed one real forward/loss per rank, but first backward failed. No stable update throughput, validation timing, checkpoint I/O, adapter reload or beam20 measurements exist.',
    'time_limit_assessment': 'UNKNOWN; cannot certify <=96 hours or hardware execution; failure is not proof of OOM or >96h',
    'required_train_samples_per_second_if_other_hours': {
        str(h): n * epochs / ((96 - h) * 3600) for h in [0, 6, 12, 24]
    },
    'hours_formula': 'train_presentations/r_train/3600 + total_validation_texts/r_valid/3600 + test_users_per_prompt*P/r_gen/3600 + initialization + checkpoint_selection_save + adapter_reload + margin',
    'rates_definition': 'All r are dual-GPU aggregate actual samples or user-prompts per second; P=1 or17 are separate scenarios.',
    'identity_limit': 'Author supplement is not bound to historical checkpoint/argv; technical feasibility is not metric reproduction.',
}
Path(a.output).write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps({'status': r['status'], 'output': a.output}))
