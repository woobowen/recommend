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
        'raw_generation_files':source_identity(raw),
        'actual_import_files':source_identity([__file__,original.__file__,offline_metrics.__file__,paired_beauty.__file__,
            scripts/'production_identity.py',scripts/'g3_identity.py',official,
            release/'lc_rec/letter/review/final_audit_20261113/statistics_verify.py']),
        'elapsed_seconds':time.monotonic()-started,
        'limitations':'32 fixed users are interface validation only; no full-population research conclusion.'}
    durable_write(output/'summary.json',result)
    print(json.dumps({'status':'PASS','scope':'PREFLIGHT','users':32,'conditions':list(conditions),'summary':str(output/'summary.json')}),flush=True)


if __name__=='__main__':
    main()
