#!/usr/bin/env python3
"""CPU gate for exact DeepSpeed logging metadata encoding; no model or CUDA.

Uses the production helper with actual Accelerate DeepSpeedPlugin and actual
Transformers TrainingArguments. Synthetic negative cases must still fail.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata as md
import inspect
import json
import math
from pathlib import Path
import sys

from checkpoint import DEEPSPEED_LOGGING_SENTINEL, deepspeed_metadata, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Actual ds_z2_bf16.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from accelerate.utils import DeepSpeedPlugin
    from transformers import TrainingArguments
    assert not torch.cuda.is_initialized(), 'CPU metadata gate must not create a CUDA context'
    config = json.loads(Path(args.config).read_text())
    original = copy.deepcopy(config)
    plugin = DeepSpeedPlugin(hf_ds_config=copy.deepcopy(config),
                            gradient_accumulation_steps=4, gradient_clipping=1.0)
    actual = plugin.deepspeed_config
    assert type(actual['steps_per_print']) is float and math.isinf(actual['steps_per_print']) and actual['steps_per_print'] > 0
    actual_before = copy.deepcopy(actual)
    encoded = deepspeed_metadata(actual)
    assert encoded['steps_per_print'] == DEEPSPEED_LOGGING_SENTINEL
    assert actual == actual_before and config == original
    assert json.loads(json.dumps(encoded, allow_nan=False)) == encoded
    assert {key: value for key, value in actual.items() if key != 'steps_per_print'} == {
        key: value for key, value in encoded.items() if key != 'steps_per_print'}

    # Construct real TrainingArguments without querying a CUDA device or
    # launching distributed/DeepSpeed execution. use_cpu affects this metadata
    # gate only; it is never a training config or a formal compute path.
    training = TrainingArguments(output_dir=str(Path(args.output).parent / 'unused-training-output'),
        use_cpu=True, bf16=False, fp16=False, report_to=[],
        per_device_train_batch_size=8, gradient_accumulation_steps=4,
        deepspeed=copy.deepcopy(config))
    # HfDeepSpeedConfig deep-copies a dict: the ordinary requested TrainingArgs
    # can retain finite 2000 while its plugin's effective copy becomes inf.
    # Test both real paths rather than assuming an alias that does not exist.
    requested_dict = training.to_dict()
    assert requested_dict['deepspeed'] == config
    assert training.hf_deepspeed_config.config['steps_per_print'] == float('inf')
    requested_encoded = dict(requested_dict)
    requested_encoded['deepspeed'] = deepspeed_metadata(requested_dict['deepspeed'])
    assert requested_encoded['deepspeed'] == config
    json.dumps(requested_encoded, allow_nan=False)
    # A caller can also supply an already-resolved real plugin config. This
    # actual TrainingArguments object must safely encode its inherited inf.
    inherited_training = TrainingArguments(output_dir=str(Path(args.output).parent / 'unused-inherited-output'),
        use_cpu=True, bf16=False, fp16=False, report_to=[],
        per_device_train_batch_size=8, gradient_accumulation_steps=4,
        deepspeed=copy.deepcopy(actual))
    raw_training = inherited_training.to_dict()
    raw_deepspeed = raw_training['deepspeed']
    assert math.isinf(raw_deepspeed['steps_per_print']) and raw_deepspeed['steps_per_print'] > 0
    training_before = copy.deepcopy(inherited_training.deepspeed)
    encoded_training = dict(raw_training)
    encoded_training['deepspeed'] = deepspeed_metadata(raw_deepspeed)
    assert inherited_training.deepspeed == training_before
    assert inherited_training.deepspeed_plugin.deepspeed_config['steps_per_print'] == float('inf')
    json.dumps(encoded_training, allow_nan=False)

    finite = {'steps_per_print': 2000, 'zero_optimization': {'stage': 2}}
    assert deepspeed_metadata(finite) == finite
    negative_cases = {
        'steps_per_print_nan': {'steps_per_print': float('nan')},
        'steps_per_print_negative_inf': {'steps_per_print': -float('inf')},
        'other_top_level_positive_inf': {'steps_per_print': float('inf'), 'gradient_clipping': float('inf')},
        'nested_positive_inf': {'steps_per_print': float('inf'), 'optimizer': {'params': {'lr': float('inf')}}},
        'nested_nan': {'steps_per_print': 2000, 'zero_optimization': {'stage': float('nan')}},
    }
    rejected = {}
    for name, value in negative_cases.items():
        try:
            deepspeed_metadata(value)
        except ValueError as error:
            rejected[name] = {'rejected': True, 'type': type(error).__name__}
        else:
            raise AssertionError('Unexpected nonfinite config accepted: ' + name)
    observation_rejections = []
    for name, observation in [('loss_nan', {'loss': float('nan')}),
                               ('grad_norm_inf', {'grad_norm': float('inf')})]:
        try:
            # The production writer also retains allow_nan=False. Avoid
            # writing deliberate failure files while testing its JSON policy.
            json.dumps(observation, allow_nan=False)
        except ValueError:
            observation_rejections.append(name)
        else:
            raise AssertionError('Nonfinite training observation accepted: ' + name)
    assert not torch.cuda.is_initialized()
    source = inspect.getsourcefile(DeepSpeedPlugin)
    report = {
        'status': 'PASS_CPU_METADATA_CONTRACT', 'argv': sys.argv,
        'versions': {name: md.version(name) for name in ['torch', 'accelerate', 'transformers', 'deepspeed']},
        'cuda_context_created': False, 'scope': 'Metadata only; no model, optimizer or GPU computation',
        'config_sha256': sha(args.config), 'production_helper_source_sha256': sha(Path(__file__).with_name('checkpoint.py')),
        'plugin_source': {'path': source, 'sha256': sha(source)},
        'actual_plugin_positive_inf_sentinel': True, 'actual_training_arguments_positive_inf_sentinel': True,
        'ordinary_requested_training_arguments_finite_config_preserved': True,
        'inherited_effective_plugin_config_training_arguments_encoded': True,
        'encoded_sentinel': DEEPSPEED_LOGGING_SENTINEL,
        'plugin_and_training_configuration_unchanged': True, 'finite_values_unchanged': True,
        'negative_cases': rejected, 'nonfinite_training_observations_rejected': observation_rejections,
        'encoded_effective_ds': encoded,
        'encoded_training_deepspeed': encoded_training['deepspeed'],
        'encoded_training_arguments_sha256': hashlib.sha256(json.dumps(encoded_training,
            sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }
    write(args.output, report)
    print(json.dumps({'status': report['status'], 'cuda_context_created': False,
                      'negative_cases_rejected': len(rejected)}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
