"""G3 Beauty V2 generation derived from the immutable, single-rank G3 entry."""
import argparse
from collections import Counter
from copy import deepcopy
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

from lc_rec.shuffle_v2 import beauty, core, g3_assets, reference
from lc_rec.shuffle_v2.replay import dump


class CPUOracle:
    def __init__(self, path, expected_sha, config, normal_rank):
        self.audit = g3_assets.read(g3_assets.verified(path, expected_sha))
        audit = self.audit
        assert audit['status'] == 'PASS_G3_CPU_V2_FULL' and audit['protocol'] == core.PROTOCOL
        assert audit['identity_policy'] == core.IDENTITY_POLICY
        assert audit['training_commit'] == config['training_commit'] and audit['users'] == 22363
        assert audit['assets'] == normal_rank['data_identity']
        assert audit['normal_sha256'] == config['normal_predictions_sha256']
        assert audit['tokenizer_identity'] == normal_rank['tokenizer_identity']
        assert audit['assignment_sha256'] == config['full_assignment_sha256']
        assert {mode: {key: values[key] for key in core.mode_metadata(mode)}
                for mode, values in audit['modes'].items()} == {mode: core.mode_metadata(mode) for mode in core.MODES}
        manifest = g3_assets.verified(path.with_name('inputs.jsonl'), audit['manifest_sha256'])
        self.rows = []
        with manifest.open() as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                assert row['row_idx'] == i
                self.rows.append(row)
        assert len(self.rows) == audit['users']

    def validate_inputs(self, user, raw, mask, groups, actual, intervention):
        index = user['row_idx']
        row = self.rows[index]
        assert all(row[key] == val for key, val in user.items())
        assert len(raw) == len(actual) == len(mask) == row['encoded_length']
        assert core.canonical_sha256(raw) == row['original_sha256']
        assert core.canonical_sha256(mask) == row['attention_mask_sha256']
        proof = intervention['evidence']
        mode = intervention['mode']
        active = core.MODES[mode]
        wanted = row['conditions'][mode]
        assert core.canonical_sha256(actual) == wanted['input_sha256']
        assert wanted['input_changed'] == (raw != actual)
        for occurrence, (group, frozen, evidence, picked) in enumerate(
                zip(groups, row['groups'], proof['groups'], wanted['permutation_ids'], strict=True)):
            _, span, tokens = frozen
            draw, relative, counter = reference.relative(len(active), index, occurrence)
            assert group == tokens == evidence['original_ids'] == [raw[p] for p in span]
            assert span == evidence['positions'] and draw == picked == evidence['permutation_id']
            assert relative == evidence['relative_permutation'] and counter == evidence['rejection_counter']
            assert [actual[p] for p in span] == reference.remap(tokens, active, relative)
        return {'manifest_sha256': self.audit['manifest_sha256'], 'row_sha256': core.canonical_sha256(row),
                'original_sha256': row['original_sha256'], 'attention_mask_sha256': row['attention_mask_sha256'],
                'input_sha256': wanted['input_sha256'], 'mode': mode, 'row_idx': index}


def generation_kwargs(model, **kwargs):
    import torch

    assert not torch.is_grad_enabled() and not model.training
    assert kwargs['input_ids'].shape[0] == 1
    selected = {key: value for key, value in kwargs.items()
                if key not in ('input_ids', 'attention_mask', 'prefix_allowed_tokens_fn')}
    assert selected == {'max_new_tokens': 10, 'num_beams': 20, 'num_return_sequences': 20,
                        'output_scores': True, 'return_dict_in_generate': True, 'early_stopping': True}
    defaults = model.generation_config.to_dict()
    effective = deepcopy(model.generation_config)
    assert not effective.update(**selected)
    expected = {**selected, 'do_sample': False, 'use_cache': True, 'pad_token_id': 0, 'eos_token_id': 2}
    assert {key: getattr(effective, key) for key in expected} == expected
    result = model.generate(**kwargs)
    actual_defaults = model.generation_config.to_dict()
    assert actual_defaults == defaults
    assert result.sequences_scores.dtype == torch.float32
    return result, {'kwargs': selected, 'generation_config': actual_defaults, 'effective': expected,
                    'native_score_dtype': 'torch.float32'}


def observe_lookup(calls, expected, actual_ids, torch, output):
    assert len(calls) == 1 and output.shape[0] == 20
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, expected.expand_as(output))
    observed = output[0].detach().contiguous().view(torch.int16).cpu().numpy().tobytes()
    return {'shape': list(output.shape), 'dtype': str(output.dtype),
            'all_beams_exact': True, 'input_sha256': core.canonical_sha256(actual_ids),
            'first_beam_bf16_sha256': hashlib.sha256(observed).hexdigest()}


def parameters(model, report, normal_rank, audit, config):
    assert report['adapter_identity'] == normal_rank['adapter_identity']
    assert report['adapter_files'] == normal_rank['adapter_files']
    assert report['installed_model_sources'] == normal_rank['installed_model_sources']
    assert report['strict_adapter_reload'] == normal_rank['strict_adapter_reload']
    assert report['model'] == normal_rank['model'] and report['tokenizer_identity'] == audit['tokenizer_identity']
    model.requires_grad_(False)
    tensor_counts = Counter()
    for name, tensor in model.named_parameters():
        family = 'LoRA' if '.lora_' in name else 'base_embedding_or_head'
        expected = 'torch.float32' if family == 'LoRA' else 'torch.bfloat16'
        assert str(tensor.dtype) == expected, (name, tensor.dtype)
        assert tensor.device.type == 'cuda' and tensor.device.index == 0 and not tensor.requires_grad
        tensor_counts[family] += 1
    assert tensor_counts['LoRA'] == 448 and not model.training
    assert all(os.environ.get(key) == value for key, value in config['environment'].items())
    return {'actual_parameter_tensor_counts': dict(tensor_counts), 'base_dtype': 'torch.bfloat16',
            'lora_dtype': 'torch.float32', 'adapter_embedding_and_head_dtype': 'torch.bfloat16',
            'all_parameters_on_logical_cuda_0': True, 'eval_mode': True, 'requires_grad_false': True,
            'rank_environment': {**{key: os.environ[key] for key in config['environment']},
                                 **{key: os.environ[key] for key in ('CUDA_VISIBLE_DEVICES', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK')}}}


def derive(text):
    text = beauty.derive_g3(text)

    def change(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    change('scope = "PREFLIGHT" if args.preflight_users_per_rank else "FORMAL"\n    provenance',
           'scope = "FORMAL"\n    provenance')
    for old in (
        'provenance = verify_adapter_identity(adapter, scope=scope, run_commit=args.run_commit,',
        '    verify_adapter_identity(adapter, scope=scope, run_commit=args.run_commit,',
    ):
        change(old, old.replace('args.run_commit', 'v2_training_commit'))
    change('    device = next(model.parameters()).device',
           '''    v2_binding = v2_oracle.validate_inputs(user, raw_ids, raw_mask, groups, actual_ids, intervention)
    device = next(model.parameters()).device''')
    change('seen is None or any(x != actual_ids for x in seen.tolist())',
           'seen is None or tuple(seen.shape) != (BEAMS, len(actual_ids)) or any(x != actual_ids for x in seen.tolist())')
    change('seen_mask is None or any(x != raw_mask for x in seen_mask.tolist())',
           'seen_mask is None or tuple(seen_mask.shape) != tuple(seen.shape) or any(x != raw_mask for x in seen_mask.tolist())')
    change('positions is None or any(x != expected_positions[0].tolist() for x in positions.tolist())',
           'positions is None or tuple(positions.shape) != tuple(seen.shape) or any(x != expected_positions[0].tolist() for x in positions.tolist())')
    change("        past = named.get('past_key_values')", '''        assert named.get('use_cache') is True
        past = named.get('past_key_values')''')
    change("'beam_expansion': seen.shape[0]})", """'beam_expansion': seen.shape[0],
                      'effective_use_cache': named['use_cache'], 'all_beam_input_sha256': v2_hash(seen.tolist()),
                      'all_beam_mask_sha256': v2_hash(seen_mask.tolist()),
                      'all_beam_position_sha256': v2_hash(positions.tolist())})""")
    change('    handle = model.get_base_model().model.register_forward_pre_hook(before_forward, with_kwargs=True)',
           '''    handle = model.get_base_model().model.register_forward_pre_hook(before_forward, with_kwargs=True)
    lookups = []
    def after_embedding(module, args, result):
        if not lookups:
            if not args or args[0].tolist() != [actual_ids] * BEAMS:
                raise AssertionError('Actual embedding input does not match all prefill beams')
            lookups.append(v2_observe_lookup(calls, expected_lookup, actual_ids, torch, result))
    lookup_handle = embedding.register_forward_hook(after_embedding)''')
    change('        generated = model.generate(input_ids=actual, attention_mask=mask,\n                                   max_new_tokens=10, prefix_allowed_tokens_fn=allowed,\n                                   num_beams=BEAMS, num_return_sequences=BEAMS, output_scores=True,\n                                   return_dict_in_generate=True, early_stopping=True)',
           '''        with torch.no_grad():
            generated, effective_generation = v2_generate(model, input_ids=actual, attention_mask=mask,
                max_new_tokens=10, prefix_allowed_tokens_fn=allowed, num_beams=BEAMS,
                num_return_sequences=BEAMS, output_scores=True, return_dict_in_generate=True, early_stopping=True)''')
    change('        handle.remove()\n    if not calls:', '        handle.remove()\n        lookup_handle.remove()\n    if not calls or not lookups:')
    change("'actual_model_prefill': calls[0], 'lookup_equivalence_exact': True}",
           """'actual_model_prefill': {**calls[0], 'embedding_lookup': lookups[0]},
           'lookup_equivalence_exact': True, 'V2_RUN_COMMIT': v2_run_commit,
           'v2_cpu_binding': v2_binding, 'effective_generation': effective_generation}""")
    change('        torch.cuda.synchronize()\n        report["model_load_and_strict_verification_seconds"]',
           '''        report["v2_model"] = v2_parameter_evidence(model, report)
        report["MODEL_TRAINING_COMMIT"] = v2_training_commit
        report["v2_cpu"] = {key: v2_oracle.audit[key] for key in
                            ("manifest_sha256", "user_axis_sha256", "tokenizer_identity", "preflight_manifest_sha256")}
        torch.cuda.synchronize()
        report["model_load_and_strict_verification_seconds"]''')
    change('        original_collator = TestCollator(data_args, tokenizer)', '''        assert report["allowed_callback_sha256"] == v2_callback_sha
        original_collator = TestCollator(data_args, tokenizer)''')
    change('        report["assigned_users"] = len(assignment)', '''            assert assignment == v2_oracle.audit["preflight_assignment"]
        else:
            assert v2_hash(assignment) == v2_oracle.audit["assignment_sha256"]
        report["assigned_users"] = len(assignment)''')
    change("        report['status'] = 'PREDICTIONS_COMPLETE'", """        report['v2_model']['parameter_identity_storage_and_versions_unchanged_each_row'] = True
        for condition in conditions:
            condition_reports[condition]['v2_model'] = report['v2_model']
        report['status'] = 'PREDICTIONS_COMPLETE'""")
    return text


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-audit', type=Path, required=True)
    parser.add_argument('--cpu-audit-sha256', required=True)
    parser.add_argument('--run-commit', required=True)
    parser.add_argument('--mode', choices=core.MODES, required=True)
    parser.add_argument('--scope', choices=('PREFLIGHT', 'FORMAL'), required=True)
    parser.add_argument('--max-seconds', required=True, type=int)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--authorization-sha256')
    options = parser.parse_args()
    assert sys.flags.optimize == 0
    assert options.run_commit == g3_assets.read(Path.cwd() / 'RUN_VERSION.json')['RUN_COMMIT']
    assert (os.environ['WORLD_SIZE'], os.environ['RANK'], os.environ['LOCAL_RANK']) == ('1', '0', '0')
    config, _, normal_rank, adapter, paths = g3_assets.load()
    if options.scope == 'FORMAL':
        assert options.authorization and options.authorization_sha256
        approval = g3_assets.read(g3_assets.verified(options.authorization, options.authorization_sha256))
        assert approval['authorized_by'] == 'user' and approval['scope'] == 'G3_BEAUTY_SHUFFLE_V2_FORMAL'
        assert approval['run_commit'] == options.run_commit
    else:
        assert options.authorization is None and options.authorization_sha256 is None
    source = paths[g3_assets.ENTRY]
    oracle = CPUOracle(options.cpu_audit, options.cpu_audit_sha256, config, normal_rank)
    original = source.read_text()
    derived = derive(original)
    args = ['--source-root', str(Path(config['release']) / g3_assets.VENDOR),
            '--data-root', config['data_root'], '--model', config['model'], '--adapter', str(adapter),
            '--output', str(options.output), '--max-seconds', str(options.max_seconds),
            '--run-commit', options.run_commit, '--condition', options.mode]
    if options.scope == 'PREFLIGHT':
        args += ['--preflight-users-per-rank', '32', '--preflight-manifest', str(paths[g3_assets.MANIFEST])]
    options.output.mkdir(parents=True, exist_ok=False)
    dump(options.output / 'v2-binding.rank0.json', {
        **core.mode_metadata(options.mode), 'V2_RUN_COMMIT': options.run_commit,
        'MODEL_TRAINING_COMMIT': config['training_commit'], 'adapter_scope': 'FORMAL', 'scope': options.scope,
        'frozen_source': str(source), 'frozen_source_sha256': g3_assets.sha(source),
        'derived_sha256': hashlib.sha256(derived.encode()).hexdigest(),
        'effective_argv': args, 'cpu_audit_sha256': options.cpu_audit_sha256,
        'cpu_manifest_sha256': oracle.audit['manifest_sha256'],
    })
    (options.output / 'derived.rank0.py').write_text(derived)
    (options.output / 'derived.rank0.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), derived.splitlines(True), fromfile=str(source), tofile='V2-derived')))
    namespace = {
        '__name__': 'frozen_g3_v2', '__file__': str(source),
        'v2_transform_inputs': beauty.g3_transform, 'v2_mode_metadata': core.mode_metadata,
        'v2_modes': tuple(core.MODES), 'v2_oracle': oracle, 'v2_hash': core.canonical_sha256,
        'v2_observe_lookup': observe_lookup, 'v2_generate': generation_kwargs, 'v2_run_commit': options.run_commit,
        'v2_training_commit': config['training_commit'], 'v2_callback_sha': config['callback_sha256'],
        'v2_parameter_evidence': lambda model, report: parameters(model, report, normal_rank, oracle.audit, config),
    }
    sys.argv = [str(source), *args]
    exec(compile(derived, str(source), 'exec'), namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
