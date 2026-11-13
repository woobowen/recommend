#!/usr/bin/env python3
"""Nonauthor checks of fixed A0 source and captured raw terminal receipts.

Standard library only. Reads local Git objects and published raw receipts; no
SSH, model loading, CUDA, mutation of formal assets, or producer metric imports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

RUN = '4c6a83d9ba42dc2fe737792759e410ef1874da76'
JOB = 'letter-a0-20261112-normal-det-seed42-r0'
CHECKS = {}


def check(name, value):
    CHECKS[name] = bool(value)
    if not value:
        raise AssertionError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base = a.repo / ('lc_rec/letter/evidence/formal_det_20261112/observations/'
                     + JOB + '/status_check_20261113T082550Z')
    read = lambda name: json.loads((base / name).read_text())
    raw = read('raw.json')
    states = read('original_status_receipts.json')
    ranks = read('original_training_events.json')
    cps = read('checkpoint_file_receipts.json')
    sources = {}
    for path in ('scripts/formal_train.py', 'scripts/formal_evaluate.py',
                 'scripts/formal_attention.py', 'config/formal_instruments_seed42.json',
                 'vendor/LETTER/LETTER-LC-Rec/data.py',
                 'vendor/LETTER/LETTER-LC-Rec/test_ddp.py',
                 'vendor/LETTER/LETTER-LC-Rec/evaluate.py',
                 'vendor/LETTER/LETTER-LC-Rec/utils.py',
                 'vendor/LETTER/LETTER-LC-Rec/collator.py',
                 'vendor/LETTER/LETTER-LC-Rec/modeling_letter.py',
                 'review/formal_det_20261112/independent_metrics.py'):
        full = 'lc_rec/letter/' + path
        data = subprocess.check_output(['git', 'show', RUN + ':' + full], cwd=a.repo)
        sources[full] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        check('local_fixed_source_' + path, data == (a.repo / full).read_bytes())
    config = json.loads((a.repo / 'lc_rec/letter/config/formal_instruments_seed42.json').read_text())
    check('config_hash_identity', sources['lc_rec/letter/config/formal_instruments_seed42.json']['sha256']
          == raw['config_sha256'] == '133e0e88f4158ab2806ebaa973eb87a0d98c68c98690bcc5cab62e8d126ae9b7')
    check('recipe', all(config[k] == v for k, v in {
        'seed':42, 'epochs':4, 'global_batch':64, 'world_size':2,
        'per_device_batch_size':8, 'gradient_accumulation_steps':4,
        'lora_r':8, 'lora_alpha':16, 'lora_dropout':0.05,
        'temperature':0.8, 'flash_backward_deterministic':True,
        'storage_save_total_limit':2, 'only_train_response':True}.items()))
    check('captured_run_identity', raw['RUN_COMMIT'] == RUN and raw['job_id'] == JOB)
    request = raw['raw_json']['request']
    check('request_version', request['capture_version'] == RUN and request['job_id'] == JOB)
    check('formal_only_flags', request['argv'][request['argv'].index('--kind')+1] == 'formal'
          and not set(request['argv']) & {'--diagnosis','--probe-stop','--preflight-interfaces'})
    check('runner_terminal', states['runner_status']['status'] == 'SUCCEEDED'
          and states['runner_status']['exit_code'] == 0 and states['cleanup']['verified'])
    check('recorded_processes_absent', all(not r['proc_directory_exists'] for r in raw['recorded_processes']))
    pipeline = states['pipeline']
    check('pipeline_terminal', pipeline['status'] == 'SUCCESS' and pipeline['exit_code'] == 0
          and pipeline['run_commit'] == RUN and pipeline['job_id'] == JOB)
    check('train_then_test', [s['stage'] for s in pipeline['stages']] == ['train','test']
          and all(s['exit_code'] == 0 for s in pipeline['stages']))
    attempts = states['ledger']['attempts']
    check('single_formal_attempt_no_resume', len(attempts) == 1 and attempts[0]['job_id'] == JOB
          and attempts[0]['run_commit'] == RUN and attempts[0]['resume'] is None
          and attempts[0]['status'] == 'SUCCESS' and attempts[0]['wall_seconds'] < 96*3600)
    best = states['best_selection']
    check('selection_identity', best['run_commit'] == RUN and best['metric'] == 'eval_loss'
          and best['greater_is_better'] is False and best['completed_training_step'] == 8240
          and best['loaded_best_checkpoint_step'] == 8240)
    validation = []
    for rank in ('0','1'):
        r = ranks[rank]
        report = r['report_excerpt']
        check('rank'+rank+'_complete', report['status'] == 'TRAINING_COMPLETE'
              and report['global_step'] == 8240 and report['epoch'] == 4.0
              and report['initial_global_step'] == 0 and report['run_commit'] == RUN)
        events = r['raw_milestones']
        epoch = [(e['step'],e['epoch']) for e in events if e['event']=='epoch_end']
        check('rank'+rank+'_epochs', epoch == [(2060,1.0),(4120,2.0),(6180,3.0),(8240,4.0)])
        vals = [e for e in events if 'eval_loss' in e.get('values',{})]
        check('rank'+rank+'_validations', [e['step'] for e in vals] == [2060,4120,6180,8240])
        check('rank'+rank+'_best', min(vals,key=lambda e:e['values']['eval_loss'])['step'] == 8240
              and vals[-1]['values']['eval_loss'] == best['value']
              and report['best_checkpoint'] == best['checkpoint'])
        check('rank'+rank+'_strict_restore', any(e['event']=='strict_ds_restored'
              and e['path']==best['checkpoint'] and e['global_step']==8240
              and e['scheduler']['last_epoch']==8240 for e in events))
        dispatch = [e for e in events if e['event']=='formal_fa_backward_dispatch']
        check('rank'+rank+'_actual_fa_true', {e['boundary'] for e in dispatch}
              == {'native_varlen','extension_varlen'} and all(e['deterministic'] is True for e in dispatch))
        calls = report['formal_attention']['calls']
        check('rank'+rank+'_fa_calls', calls['native_varlen'] == calls['extension_varlen'] == 1054720)
        check('rank'+rank+'_finite_events', not r['nonfinite_training_event_values'])
        validation.append([e['values']['eval_loss'] for e in vals])
    check('two_rank_validation_exact', validation[0] == validation[1])
    check('retained_complete_checkpoints', [c['complete_manifest']['global_step'] for c in cps['checkpoints']]
          == [6180,8240] and all(c['validation']=='PASS' and c['complete_marker_present'] for c in cps['checkpoints']))
    for cp in cps['checkpoints']:
        m = cp['complete_manifest']; step = m['global_step']; paths = {f['path'] for f in m['files']}
        required = {'adapter_model.bin','external_scheduler.pth','loader_state_0.pth','loader_state_1.pth',
            'rng_state_0.pth','rng_state_1.pth','trainer_state.json','latest',
            f'global_step{step}/mp_rank_00_model_states.pt',
            f'global_step{step}/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt',
            f'global_step{step}/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt'}
        check('checkpoint'+str(step)+'_members', required <= paths and m['collective_ranks']==2
              and m['run_commit']==RUN and m['config_sha256']==raw['config_sha256']
              and cp['latest']==f'global_step{step}' and all(f['bytes']>0 for f in m['files']))
    tests = [raw['raw_json']['test_rank'+str(r)] for r in range(2)]
    for rank, test in enumerate(tests):
        check('test_rank'+str(rank)+'_completed', test['status']=='PASS' and test['RUN_COMMIT']==RUN
              and test['completed_users']==test['assigned_users']==12386 and test['world_size']==2)
        g = test['generation']
        check('test_rank'+str(rank)+'_protocol', g['prompt_id']==0 and g['num_beams']==20
              and g['num_return_sequences']==20 and g['filter_items'] is False
              and g['oom_beam_reduction'] is False)
        reload = test['strict_adapter_reload']
        check('test_rank'+str(rank)+'_strict_adapter', reload['keys']==450 and reload['elements']==289652736
              and reload['no_missing_keys_tolerated'] and len(reload['tensor_checks'])==450
              and all(t['finite_and_exact_after_explicit_dtype_conversion'] for t in reload['tensor_checks'].values()))
        check('test_rank'+str(rank)+'_adapter_identity', test['adapter_files']['adapter_model.bin']['sha256']==best['adapter_sha256'])
    metrics = states['metrics']
    check('producer_complete_24772', metrics['status']=='PASS' and metrics['test_complete']
          and metrics['actual_test_users']==24772 and metrics['failed_users']==metrics['skipped_users']==0
          and metrics['RUN_COMMIT']==RUN and metrics['adapter_model_sha256']==best['adapter_sha256'])
    for rank in range(2):
        check('predictions_rank'+str(rank)+'_manifest', metrics['prediction_files'][rank]['sha256']==[
            'bdcbdde15d3ac6850708888e055aa67ae61212dbb675d9f915b9947b2a208bde',
            'f5d2ce538bfe86b024c38070f74cce3411a457554ae08ec610ec26987570c629'][rank])
    receipts = {f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
                for f in base.glob('*.json')}
    result = {'status':'PASS','scope':'Nonauthor fixed source and captured raw terminal receipt cross-check; '
        'direct local read only, no SSH/GPU/model load/full prediction recomputation',
        'RUN_COMMIT':RUN,'job_id':JOB,'checks':CHECKS,'source_identity':sources,
        'receipt_identity':receipts,'validation_loss':validation[0],
        'stage_a_independent_metrics':'PENDING_SEPARATE_RECEIPT',
        'limits':['Captured raw files were collected by controller on 2026-11-13 08:27 UTC',
        'Large checkpoint payload SHA hashes were not recomputed',
        'Producer strict adapter tensor checks were read, not rerun by this local check']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS','checks':len(CHECKS),'output':str(a.output)}))


if __name__ == '__main__':
    main()
