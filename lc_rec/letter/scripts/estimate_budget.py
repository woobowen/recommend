"""Estimate one LETTER Instruments seed from actual bounded two-rank timings.

No failed/partial execution is promoted to a measured rate. Point estimates
and conservative scenarios are extrapolations, not complete-seed validation.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise AssertionError('Expected finite positive ' + label)
    return value


def read_ranks(directory, prefix):
    if not directory:
        return [], []
    paths = [Path(directory) / (prefix + str(rank) + '.json') for rank in range(2)]
    records = [json.loads(path.read_text()) for path in paths if path.is_file()]
    return records, [{'path': str(path), 'sha256': sha(path)} for path in paths if path.is_file()]


def paired_rows(records, field, selector):
    if len(records) != 2:
        return []
    selected = [[row for row in record.get(field, []) if selector(row)] for record in records]
    if len(selected[0]) != len(selected[1]):
        raise AssertionError('Rank measurement counts disagree for ' + field)
    rows = []
    for left, right in zip(*selected):
        phase = left.get('phase', left.get('selection'))
        if phase != right.get('phase', right.get('selection')) or left.get('update') != right.get('update'):
            raise AssertionError('Rank measurement identities disagree')
        seconds = max(positive(row['max_rank_seconds'], 'synchronized duration') for row in [left, right])
        item = dict(left)
        item['seconds'] = seconds
        item['data_preparation_seconds'] = max(float(row.get('cpu_batch_preparation_and_mask_verification_seconds',
                                                       row.get('cpu_collator_and_transfer_seconds', 0))) for row in [left, right])
        item['combined_true_lengths'] = left.get('rank_true_lengths', left.get('rank_lengths', [])) + right.get('rank_true_lengths', right.get('rank_lengths', []))
        item['wall_seconds_with_data'] = seconds + item['data_preparation_seconds']
        if field == 'timings' and left.get('row_idx') == right.get('row_idx'):
            raise AssertionError('Duplicate users across ranks cannot be counted as distinct throughput')
        rows.append(item)
    return rows


def stage_seconds(record, name):
    return next((float(stage['elapsed_seconds']) for stage in record.get('stages', []) if stage['stage'] == name), None)


def scenario(components, margin_fraction):
    missing = [key for key, value in components.items() if value is None]
    known = sum(value for value in components.values() if value is not None)
    margin = known * margin_fraction
    hours = {key: None if value is None else value / 3600 for key, value in components.items()}
    hours['environment_io_margin'] = margin / 3600
    total = (known + margin) / 3600 if not missing else None
    return {'component_hours': hours, 'margin_fraction_assumption': margin_fraction,
            'total_hours': total, 'remaining_to_96_hours': None if total is None else 96 - total,
            'known_components_plus_margin_hours': (known + margin) / 3600,
            'missing_components': missing}


def estimate(audit, train, inference, center_margin=.10, conservative_margin=.25):
    if audit['status'] != 'PASS':
        raise AssertionError('Budget denominators require retained completed CPU audit PASS')
    counts = audit['budget_counts']
    n, v, t = counts['train_rows_per_epoch'], counts['valid_two_prompt_rows_per_epoch'], counts['test_prompt0_rows']
    epochs, batch = counts['epochs'], counts['global_batch']
    actual = audit['original_dataset_execution']
    if (n, v, t, epochs, batch) != (actual['train_rows'], actual['valid_rows_after_two_prompts'], actual['test_rows'], 4, 64):
        raise AssertionError('Frozen counts/dataset/global64 identity mismatch')
    ceil_updates = epochs * math.ceil(n / batch)
    per_rank_rows = math.ceil(n / 2)
    per_rank_batches = math.ceil(per_rank_rows / 8)
    floor_updates = epochs * (per_rank_batches // 4)
    train_pass = len(train) == 2 and all(record['status'] == 'PASS' for record in train)
    inference_pass = len(inference) == 2 and all(record['status'] == 'RELOAD_BEAM20_TECHNICAL_PASS' for record in inference)
    updates = paired_rows(train, 'updates', lambda row: row['phase'] != 'warmup') if train_pass else []
    mixed = [row for row in updates if row['phase'] == 'mixed_steady']
    validation = paired_rows(train, 'validation_timings', lambda row: row['phase'] != 'warmup') if train_pass else []
    generation = paired_rows(inference, 'timings', lambda row: not row['warmup']) if inference_pass else []
    for records in [train, inference]:
        if len(records) == 2 and records[0].get('rank') == records[1].get('rank'):
            raise AssertionError('Two distinct rank records required')
    result = {'schema': 'letter_seed96h_bounded_estimate_v2', 'status': 'INCONCLUSIVE',
        'observed_counts': {'train_per_epoch': n, 'epochs': epochs, 'train_presentations': n * epochs,
            'validation_texts_per_epoch': v, 'total_validation_texts': v * epochs,
            'test_users_prompt0': t, 'effective_global_batch': batch,
            'conservative_optimizer_updates_four_epochs': ceil_updates,
            'trainer431_floor_scheduler_updates_four_epochs': floor_updates,
            'dataloader_batches_per_rank_per_epoch': per_rank_batches,
            'distributed_sampler_rows_per_rank': per_rank_rows,
            'distributed_sampler_total_rows_including_padding_per_epoch': per_rank_rows * 2},
        'dataloader_tail_semantics': {'binding': 'Transformers4.31 Trainer default drop_last=False; two-rank sample/batch accounting; micro8 accumulation4; actual sharding is runtime-bound',
            'scheduler': 'floor(len(dataloader)/4)*4 epochs; verified against training_definition when available',
            'wall_estimate': 'ceil(131837/64)*4 = 8240 global64-equivalent updates is conservative work accounting',
            'partial_accumulation_limit': 'Bounded nine full global64 updates do not execute an epoch tail. Scheduler arithmetic gives %d updates; conservative work accounting gives %d. DistributedSampler arithmetic alone would give last rank microbatch7/global accumulation62, while Accelerate even_batches may fill it; actual runtime shard/tail behavior has not been measured.' % (floor_updates, ceil_updates)},
        'actual_complete_optimizer_updates_by_rank': [len(record.get('updates', [])) for record in train],
        'train_statuses': [record['status'] for record in train],
        'inference_statuses': [record['status'] for record in inference],
        'scope': 'Conditional one-seed cost: four train epochs, two-template validation per epoch, four adapter serializations, one final reload and prompt0 beam20 for 24772 users; formal Trainer/DS state checkpoint/selection/best restore cost remains unmeasured',
        'rates_definition': 'Dual-GPU aggregate samples/users divided by synchronized MAX-rank wall duration. Rank totals are never added to duplicate the denominator.',
        'minimum_train_samples_per_second_if_other_hours': {str(h): n * epochs / ((96 - h) * 3600) for h in [0, 6, 12, 24]},
        'method_and_precision_limits': ['Training uses approved technical-only PEFT0.5 non-inplace residual plus LoRA-A-only BF16 adaptation.',
            'Author test_ddp.py LoRA inference reloads full BF16 base with vanilla HF attention; training is int8/flash-attention. These are separate measured paths.',
            'No complete epoch, best-checkpoint selection, sustained multi-hour run, or recommendation metric reproduction was performed.']}
    if train_pass:
        for record in train:
            definition = record['training_definition']
            if definition['formal_dataloader_batches_per_rank'] != per_rank_batches or definition['formal_scheduler_updates'] != floor_updates:
                raise AssertionError('Actual training scheduler/dataloader accounting differs')
    train_center = ceil_updates * statistics.mean(row['wall_seconds_with_data'] for row in mixed) if mixed else None
    train_slow = ceil_updates * max(row['wall_seconds_with_data'] for row in updates) if updates else None
    valid_mixed = [row for row in validation if row['phase'] == 'mixed']
    valid_center = (v * epochs * sum(row['wall_seconds_with_data'] for row in valid_mixed) /
                    sum(row['global_samples'] for row in valid_mixed)) if valid_mixed else None
    valid_slow = v * epochs * max(row['wall_seconds_with_data'] / row['global_samples'] for row in validation) if validation else None
    save = train[0].get('adapter_save', {}) if train_pass else {}
    save_center = epochs * float(save.get('write_seconds_rank0', save.get('seconds_rank0', 0))) if save else None
    save_slow = epochs * float(save['seconds_rank0']) if save else None
    reload_seconds = max(float(record['adapter_reload_seconds']) for record in inference) if inference_pass else None
    init_train_values = [stage_seconds(record, 'before_updates') for record in train] if train_pass else []
    init_train = max(init_train_values) if init_train_values and all(value is not None for value in init_train_values) else None
    init_inference = max(float(record['base_load_seconds']) for record in inference) if inference_pass else None
    initialization = init_train + init_inference if init_train is not None and init_inference is not None else None
    # Nearest measured prompt0 length stratum weights the point estimate with
    # the already-audited full user distribution. This is a timing proxy, not
    # a regression/claim that only length determines the generation runtime.
    beam_center, beam_slow, assignments = None, None, {}
    if generation:
        histogram = audit['length_distributions']['test']['input_per_prompt'][0]['histogram']
        if sum(map(int, histogram.values())) != t:
            raise AssertionError('Prompt0 histogram does not cover all test users')
        beam_center = 0.
        for length, count in histogram.items():
            nearest = min(generation, key=lambda row: (abs(row['input_length'] - int(length)), row['input_length']))
            per_user = nearest['wall_seconds_with_data'] / nearest['global_users']
            beam_center += int(count) * per_user
            assignments.setdefault(nearest['selection'], {'user_count': 0, 'measured_input_length': nearest['input_length'], 'seconds_per_user': per_user})['user_count'] += int(count)
        beam_slow = t * max(row['wall_seconds_with_data'] / row['global_users'] for row in generation)
    center = scenario({'train': train_center, 'validation': valid_center, 'adapter_checkpoint_serialization_proxy': save_center,
                       'adapter_reload': reload_seconds, 'beam20_prompt0_test': beam_center, 'initialization': initialization}, center_margin)
    conservative = scenario({'train': train_slow, 'validation': valid_slow, 'adapter_checkpoint_serialization_proxy': save_slow,
                       'adapter_reload': reload_seconds, 'beam20_prompt0_test': beam_slow, 'initialization': initialization}, conservative_margin)
    unquantified = ['Formal Trainer/DeepSpeed optimizer, scheduler, RNG and other state checkpoint writes.',
                   'Best-checkpoint selection/restore and final Trainer state/model saves. Full checkpoints were prohibited in this probe.',
                   'Validation loader/worker and distributed aggregation overhead beyond measured collator/forward; CPU recommendation filtering/metric aggregation were not executed.']
    for estimate in [center, conservative]:
        estimate['is_full_seed_cost_validation'] = False
        estimate['unquantified_formal_checkpoint_overheads'] = unquantified
        estimate['total_hours_definition'] = 'Conditional measured-component estimate plus stated margin; excludes additional formal checkpoint/selection/state restore costs'
    result['center_estimate'] = center
    result['conservative_estimate'] = conservative
    result['measurements'] = {'steady_mixed_update_count': len(mixed),
        'steady_mixed_update_seconds_with_data': [row['wall_seconds_with_data'] for row in mixed],
        'steady_mixed_global_samples_per_second': (sum(row['global_samples'] for row in mixed) / sum(row['seconds'] for row in mixed)) if mixed else None,
        'steady_mixed_effective_tokens_per_second': (sum(row['effective_input_tokens'] for row in mixed) / sum(row['seconds'] for row in mixed)) if mixed else None,
        'postwarmup_length_phases': [{'phase': row['phase'], 'seconds': row['seconds'], 'seconds_with_data': row['wall_seconds_with_data'],
                                    'global_samples': row['global_samples'], 'effective_tokens': row['effective_input_tokens'],
                                    'padded_tokens': row['padded_tokens']} for row in updates],
        'validation': [{'phase': row['phase'], 'seconds': row['seconds'], 'seconds_with_data': row['wall_seconds_with_data'],
                        'global_samples': row['global_samples']} for row in validation],
        'generation': [{'selection': row['selection'], 'seconds': row['seconds'], 'global_users': row['global_users'],
                       'input_length': row['input_length']} for row in generation],
        'beam_center_full_user_histogram_assignments': assignments}
    observed_lengths = [length for row in mixed for length in row['combined_true_lengths']]
    audited_lengths = audit['length_distributions']['train']['all_prompt_full_with_eos']
    result['train_length_representativeness'] = {'full_train_rows_x_uniform17_prompt_distribution': {key: audited_lengths[key] for key in ['count', 'min', 'max', 'mean', 'quantiles']},
        'steady_mixed_observed_length_count_including_repeats': len(observed_lengths),
        'steady_mixed_observed_mean': statistics.mean(observed_lengths) if observed_lengths else None,
        'steady_mixed_observed_min': min(observed_lengths) if observed_lengths else None,
        'steady_mixed_observed_max': max(observed_lengths) if observed_lengths else None,
        'limitation': 'The retained probe uses 128 fixed unique mixed train rows. Updates0/2/4 repeat one64-row half,1/3/5 the other. Three steady timings cover128 unique rows, not192 independent new rows. Short/middle/long are homogeneous length-envelope checks, excluded from center.'}
    if center['total_hours'] is not None:
        result['status'] = 'CONDITIONAL_ESTIMATE_WITH_UNMEASURED_FORMAL_CHECKPOINT_OVERHEAD'
        result['time_limit_assessment'] = 'INCONCLUSIVE_FORMAL_CHECKPOINT_AND_SUSTAINED_RUNTIME'
        result['conditional_measured_scenarios_within_96h'] = {'center': center['total_hours'] <= 96, 'conservative': conservative['total_hours'] <= 96}
        result['evidence_interpretation'] = ('Conditional center/conservative scenarios use actual steady mixed updates and real validation/adapter serialization/reload/beam20. '
            'Additional formal Trainer/DS checkpoint, selection and best restore costs are NOT TESTED. Remaining_to_96_hours is the combined capacity for these costs beyond the stated margin. '
            'Only three steady mixed updates and eight distinct measured generation users are available; no statistically calibrated 96h completion guarantee or sustained throughput validation is claimed.')
        other_center = sum(value for key, value in center['component_hours'].items() if key not in ['train', 'environment_io_margin'] and value is not None)
        allowed_train_hours = 96 / (1 + center_margin) - other_center
        result['minimum_train_samples_per_second_with_measured_other_costs_and_center_margin'] = (
            n * epochs / (allowed_train_hours * 3600)) if allowed_train_hours > 0 else None
        if center['total_hours'] > 96:
            result['time_limit_assessment'] = 'TIME_BUDGET_BLOCKED_AT_OBSERVED_CENTER_RATE'
        elif conservative['total_hours'] <= 96:
            result['conditional_measured_components_support_96h_feasibility'] = True
            result['remaining_risks'] = ['Unmeasured multi-hour thermal, scheduling, memory fragmentation and shared I/O changes.',
                'Only one adapter serialization/reload observation; best checkpoint selection itself was not exercised.',
                'Generation uses a small fixed length-stratified user sample; histogram-weighted center is a proxy.',
                'Guard/finite checks and synchronized component timers add probe overhead; formal dataloader workers/prefetch may differ.']
        else:
            result['conditional_measured_components_support_96h_feasibility'] = False
    else:
        result['time_limit_assessment'] = 'INCONCLUSIVE_MISSING_MEASUREMENTS'
        result['evidence_interpretation'] = 'Missing stages remain NOT TESTED. Known component time is a partial cost, never a full-seed estimate.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', required=True)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--inference-dir')
    parser.add_argument('--output', required=True)
    parser.add_argument('--center-margin', type=float, default=.10)
    parser.add_argument('--conservative-margin', type=float, default=.25)
    args = parser.parse_args()
    if not 0 <= args.center_margin <= args.conservative_margin < 1:
        raise ValueError('Margin fractions must satisfy 0 <= center <= conservative < 1')
    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text())
    train, train_refs = read_ranks(args.train_dir, 'rank')
    inference, inference_refs = read_ranks(args.inference_dir, 'inference_rank')
    result = estimate(audit, train, inference, args.center_margin, args.conservative_margin)
    result['evidence_sources'] = {'audit': {'path': str(audit_path), 'sha256': sha(audit_path)}, 'training': train_refs, 'inference': inference_refs}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'time_limit_assessment': result['time_limit_assessment'], 'output': args.output}))


if __name__ == '__main__':
    main()
