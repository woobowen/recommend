#!/usr/bin/env python3
"""Evaluate actual finite G3 predictions with the production metric/pairing code.

This CPU-only consumer requires genuine completed PREFLIGHT training and four
completed prediction conditions. It cannot accept or produce formal results.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from formal_control import durable_write
from g3_identity import (ASSETS, EXPERIMENT_ID, source_identity, verify_assets, need, sha,
                         EXECUTION_TOPOLOGY, EXECUTION_PROFILE, verify_execution_profile)
from production_identity import verify_training, verify_predictions
import formal_evaluate as original
import offline_metrics
import paired_beauty


# Inherited before observing G3 outputs: the existing BF16 cross-process rule.
NORMAL_RELOAD_ATOL = 1e-5
NORMAL_RELOAD_RTOL = 1e-5
NORMAL_RELOAD_TOLERANCE_SOURCE = {
    'commit':'8f36720a024be671a3ba893370a98126bc0f3b4a',
    'path':'lc_rec/a0_beauty/shuffle_runtime.py',
    'method':'torch.allclose(actual_float32_scores, reference_float32_scores, atol=1e-5, rtol=1e-5)',
    'sequences_and_order':'exact'}


def compare_normal_reload_rows(reference_rows, reloaded_rows):
    """Apply the inherited score criterion with exact actual inputs/outputs."""
    import torch
    need(len(reference_rows)==len(reloaded_rows)==32, 'cross-process Normal fixed32 population')
    exact=('row_idx','user_id','target_item_id','target','rank','prompt_id','scope','condition',
           'model_identity_sha256','input_token_ids','original_input_token_ids','attention_mask',
           'sorted_generation_order','relevance','actual_model_prefill')
    maximum=0.;exact_scores=0;seen=set()
    for reference,reloaded in zip(reference_rows,reloaded_rows):
        need(reference['row_idx'] not in seen,'cross-process Normal duplicate user')
        seen.add(reference['row_idx'])
        need(all(reference[key]==reloaded[key] for key in exact),'cross-process Normal identity/input/order mismatch')
        need(reference['condition']=='normal' and reference['scope']=='PREFLIGHT','cross-process Normal scope/condition')
        left,right=reference['candidates'],reloaded['candidates']
        need(len(left)==len(right)==20,'cross-process Normal beam20')
        candidate_keys=('generation_order','generated_token_ids','eos_position','sid')
        need(all(all(x[key]==y[key] for key in candidate_keys) for x,y in zip(left,right)),
             'cross-process Normal generated SID suffix/order mismatch')
        expected=torch.tensor([row['sequence_score'] for row in left],dtype=torch.float32)
        actual=torch.tensor([row['sequence_score'] for row in right],dtype=torch.float32)
        need(torch.isfinite(expected).all().item() and torch.isfinite(actual).all().item(),
             'cross-process Normal nonfinite score')
        need(torch.allclose(actual,expected,atol=NORMAL_RELOAD_ATOL,rtol=NORMAL_RELOAD_RTOL),
             'cross-process Normal score exceeds inherited fixed tolerance')
        maximum=max(maximum,float((actual-expected).abs().max()))
        exact_scores+=int(torch.equal(actual,expected))
    return {'status':'PASS','users':32,'candidate_suffixes_SIDs_order_and_actual_prefill_exact':True,
            'scores_within_inherited_tolerance':True,'scores_exact_users':exact_scores,
            'maximum_absolute_score_difference':maximum,'score_dtype':'torch.float32',
            'atol':NORMAL_RELOAD_ATOL,'rtol':NORMAL_RELOAD_RTOL,
            'criterion':'torch.allclose(actual, reference, atol=1e-5, rtol=1e-5)',
            'fixed_before_observation':True,'tolerance_source':NORMAL_RELOAD_TOLERANCE_SOURCE}


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    for name in ('release', 'data-root', 'training-root', 'generation-root', 'run-commit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    release, output = Path(args.release), Path(args.output)
    scripts = release / 'lc_rec/a0_beauty'
    generation, training = Path(args.generation_root), Path(args.training_root)
    need(not output.exists(), 'preserve previous finite evaluation receipt')
    output.mkdir(parents=True)
    dep = load(release / 'RUN_VERSION.json')
    need(dep['run_commit'] == args.run_commit, 'finite evaluation source commit')
    for row in dep['files']:
        need(sha(row['path']) == row['sha256'], 'finite evaluation deployment bytes')
    assets = verify_assets(args.data_root)
    config = scripts / 'config/formal_g3_beauty_seed42.json'
    best = verify_training(training, args.run_commit, sha(config), 'PREFLIGHT')
    manifest = scripts / 'config/preflight_g3_users_seed42.json'
    selected = load(manifest)
    need(selected['asset_sha256'] == ASSETS, 'finite evaluation user fixture asset identity')
    need(selected['execution_topology'] == EXECUTION_TOPOLOGY and set(selected['ranks']) == {'0'}, 'finite single-GPU user fixture topology')
    verify_execution_profile(selected['execution_profile'])
    indices = sorted(row['row_idx'] for rank in ('0',) for row in selected['ranks'][rank])
    need(len(indices) == len(set(indices)) == 32, 'finite evaluation fixed user coverage')
    _, catalog, users, _ = original.catalog_and_users(args.data_root)
    vendor = release / 'lc_rec/letter/vendor/LETTER/LETTER-LC-Rec'
    evaluator = original.load_evaluator(vendor)
    official = release / 'lc_rec/letter/review/final_audit_20261113/external/official_cce.py'
    conditions, metrics, raw = {}, {}, []
    for condition in ('normal','off','s3','s4'):
        test = generation / condition
        proof = verify_predictions(test, args.run_commit, best, condition, 'PREFLIGHT')
        stage = original.aggregate_predictions([test / f'predictions.rank{rank}.jsonl' for rank in range(1)],
            users, catalog, evaluator, run_commit=args.run_commit, scope='PREFLIGHT',
            expected_indices=indices, condition=condition)
        need(stage['actual_test_users'] == 32 and stage['preflight_only'] is True, 'finite original evaluation population')
        stage.update(RUN_COMMIT=args.run_commit, scope='PREFLIGHT', preflight_only=True,
                     condition=condition, adapter_model_sha256=best['adapter_files']['adapter_model.bin']['sha256'],
                     world_size=1, execution_topology=EXECUTION_TOPOLOGY, execution_profile=EXECUTION_PROFILE)
        condition_output = output / condition
        condition_output.mkdir()
        stage_path = condition_output / 'stage_a.json'
        durable_write(stage_path, stage)
        cfg = load(release / 'lc_rec/letter/config/a0_evaluation_v1.json')
        cfg.update(protocol_version='letter_beauty_beam20_v1', dataset='Beauty', users=32,
            original_run_commit=args.run_commit, sid_coverage_denominator=12101,
            asset_sha256=ASSETS, user_axis_sha256='7430079e2b917dfa5e4bb8ead2702fb6c622fb48ffb684dd2889780f9480af7f',
            scope='PREFLIGHT', preflight_only=True, condition=condition,
            world_size=1, execution_topology=EXECUTION_TOPOLOGY, execution_profile=EXECUTION_PROFILE)
        config_path = condition_output / 'evaluation_config.json'
        durable_write(config_path, cfg)
        receipt = offline_metrics.evaluate(SimpleNamespace(output=condition_output/'primary', test_dir=test,
            stage_a_report=stage_path, config=config_path, official_cce=official,
            data_root=Path(args.data_root), run_commit=args.run_commit, evaluation_commit=args.run_commit,
            scope='PREFLIGHT', preflight_users=manifest))
        actual = load(receipt)
        need(actual['scope'] == 'PREFLIGHT' and actual['users'] == 32 and
             actual['collision_catalog']['unique_sids'] == 12101 and actual['collision_catalog']['groups'] == 0,
             'finite G3 catalog/statistic scope')
        conditions[condition] = {'prediction_proof': proof, 'stage_a': source_identity([stage_path])[0],
                                 'metrics': source_identity([receipt])[0]}
        metrics[condition] = actual['metrics']
        raw += [test/f'predictions.rank{rank}.jsonl' for rank in range(1)]
        raw += [test/f'rank{rank}.json' for rank in range(1)]
    compared = 0
    model_process_proof = []
    for rank in range(1):
        reports = [load(generation/condition/f'rank{rank}.json') for condition in ('normal','off','s3','s4')]
        fields = ('host','pid','argv','RUN_COMMIT','adapter_files','generation','allowed_callback_sha256','world_size','execution_topology','execution_profile')
        need(all(r['world_size'] == 1 and r['execution_topology'] == EXECUTION_TOPOLOGY for r in reports), 'finite one-rank model process')
        for r in reports:verify_execution_profile(r['execution_profile'])
        need(all(all(row[field] == reports[0][field] for field in fields) for row in reports),
             'four conditions must share one actual rank process/model/config')
        checks = reports[0]['normal_off_same_process_exact']
        expected_rank = [row['row_idx'] for row in selected['ranks'][str(rank)]]
        need([row['row_idx'] for row in checks] == expected_rank and all(row['same_process_exact'] is True for row in checks),
             'all same-process Normal/OFF checks required')
        model_process_proof.append({'rank':rank,'host':reports[0]['host'],'pid':reports[0]['pid'],
                                    'same_process_checked_rows':expected_rank})
        left = [json.loads(line) for line in (generation/'normal'/f'predictions.rank{rank}.jsonl').read_text().splitlines()]
        right = [json.loads(line) for line in (generation/'off'/f'predictions.rank{rank}.jsonl').read_text().splitlines()]
        need(len(left) == len(right) == 32, 'normal/OFF fixed shard comparison')
        for normal, off in zip(left,right):
            keys = ('row_idx','user_id','target_item_id','target','input_token_ids','original_input_token_ids',
                    'attention_mask','candidates','sorted_generation_order','relevance','model_identity_sha256')
            need(all(normal[key] == off[key] for key in keys), 'normal/OFF actual candidate/input exact equivalence')
            need(normal['actual_model_prefill'] == off['actual_model_prefill'], 'normal/OFF actual prefill equivalence')
            compared += 1
    need(metrics['normal'] == metrics['off'], 'normal/OFF all inherited metrics exact')
    # A second independent inference process strictly loads the same saved adapter.
    reload_root=generation.parent/'generation_reload_normal'
    reload_proof=verify_predictions(reload_root,args.run_commit,best,'normal','PREFLIGHT')
    reload_stage=original.aggregate_predictions([reload_root/'predictions.rank0.jsonl'],users,catalog,evaluator,
        run_commit=args.run_commit,scope='PREFLIGHT',expected_indices=indices,condition='normal')
    need(reload_stage['actual_test_users']==32,'cross-process Normal complete original evaluation')
    first_report=load(generation/'normal/rank0.json');second_report=load(reload_root/'rank0.json')
    report_fields=('RUN_COMMIT','scope','condition','adapter_files','data_identity','tokenizer_identity',
        'model_identity_sha256','generation','allowed_callback_sha256','assigned_row_indices',
        'assignment_sha256','execution_topology','execution_profile','world_size','installed_model_sources','model','author_sources','entry_sha256','runtime_sources')
    need(all(first_report[key]==second_report[key] for key in report_fields),
         'cross-process Normal model/tokenizer/generation/user identity mismatch')
    need(first_report['host']==second_report['host'] and
         first_report['pid']!=second_report['pid'] and first_report['started_utc']!=second_report['started_utc'],
         'cross-process Normal requires a distinct actual inference process')
    reference_binding=second_report['cross_process_reference']
    need(reference_binding['files']==source_identity([generation/'normal/rank0.json',generation/'normal/predictions.rank0.jsonl']),
         'cross-process Normal reference raw-file identity mismatch')
    declared=reference_binding['criterion']
    need(reference_binding['source_scope_and_user_axis_verified_before_model_load'] is True and
         declared['score_atol']==NORMAL_RELOAD_ATOL and declared['score_rtol']==NORMAL_RELOAD_RTOL and
         declared['fixed_before_observation'] is True and
         declared['inherited_source_commit']==NORMAL_RELOAD_TOLERANCE_SOURCE['commit'] and
         declared['inherited_source_path']==NORMAL_RELOAD_TOLERANCE_SOURCE['path'],
         'cross-process Normal comparison criterion differs from predeclared legacy rule')
    left=[json.loads(line) for line in (generation/'normal/predictions.rank0.jsonl').read_text().splitlines()]
    right=[json.loads(line) for line in (reload_root/'predictions.rank0.jsonl').read_text().splitlines()]
    cross_process=compare_normal_reload_rows(left,right)
    cross_process.update(prediction_proof=reload_proof,original_evaluation=reload_stage,
        reference_process={key:first_report[key] for key in ('host','pid','started_utc')},
        reloaded_process={key:second_report[key] for key in ('host','pid','started_utc')})
    raw.extend([reload_root/'predictions.rank0.jsonl',reload_root/'rank0.json'])
    pairings = {}
    for condition in ('s3','s4'):
        paired_output = output / 'paired' / condition
        paired_output.parent.mkdir(exist_ok=True)
        pairing_args = SimpleNamespace(normal_primary=str(output/'normal/primary'),
            shuffle_primary=str(output/condition/'primary'), normal_test=str(generation/'normal'),
            shuffle_test=str(generation/condition), output=str(paired_output))
        paired_beauty.process(pairing_args, source_context={'scope':'PREFLIGHT','best':best})
        result_path = paired_output / 'paired_comparison.json'
        actual = load(result_path)
        need(actual['status'] == 'PASS' and actual['scope'] == 'PREFLIGHT' and actual['users'] == 32,
             'finite paired scope/population')
        pairings[condition] = source_identity([result_path])[0]
    result = {'status':'PASS','scope':'PREFLIGHT','preflight_only':True,'experiment_id':EXPERIMENT_ID,
        'formal_experiment_started':False,'formal_launch_authorized':False,'argv':sys.argv,
        'world_size':1,'execution_topology':EXECUTION_TOPOLOGY,'execution_profile':EXECUTION_PROFILE,
        'cwd':os.getcwd(),'python':sys.executable,'run_commit':args.run_commit,
        'assets':assets,'training_proof':best,'user_manifest':source_identity([manifest])[0],
        'conditions':conditions,'metrics':metrics,'pairings':pairings,
        'normal_off':{'users_compared':compared,'same_process_model_generation_proof':model_process_proof,
                      'actual_prefill_and_twenty_candidates_exact':True,'all_metrics_exact':True},
        'cross_process_normal_reload':cross_process,'raw_generation_files':source_identity(raw),
        'actual_import_files':source_identity([__file__,original.__file__,offline_metrics.__file__,paired_beauty.__file__,
            scripts/'production_identity.py',scripts/'g3_identity.py',official,
            release/'lc_rec/letter/review/final_audit_20261113/statistics_verify.py']),
        'elapsed_seconds':time.monotonic()-started,
        'limitations':'32 fixed users are interface validation only; no full-population research conclusion.'}
    durable_write(output/'summary.json',result)
    print(json.dumps({'status':'PASS','scope':'PREFLIGHT','users':32,'conditions':list(conditions),'summary':str(output/'summary.json')}),flush=True)


if __name__=='__main__':
    main()
