"""Single-rank A1 generation using the frozen Qwen evaluator and strict reload."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

from lc_rec.shuffle.entry import FROZEN, adapt
from lc_rec.shuffle_v2 import core
from lc_rec.shuffle_v2.a1_replay import vocab_hash
from lc_rec.shuffle_v2.runtime import Runtime


def derive(source):
    derived = adapt(source, 'a1')

    def replace(old, new):
        nonlocal derived
        assert derived.count(old) == 1, old
        derived = derived.replace(old, new)

    replace('args.preflight_users_per_rank not in (0, 16)', 'args.preflight_users_per_rank not in (0, 32)')
    replace('a separate fixed eight-user/rank preflight', 'the frozen 32-user single-rank preflight')
    replace('world != 2 or not 0 <= rank < 2', 'world != 1 or rank != 0 or local_rank != 0')
    replace('Requires two ranks', 'Requires rank 0 of a single rank')
    replace('len(assignment) != 12386', 'len(assignment) != 24772')
    replace('from runtime import ShuffleAllowed as Allowed', 'from common import Allowed')
    replace('hashlib.sha256(inspect.getsource(Allowed).encode()).hexdigest()',
            'hashlib.sha256(inspect.getsource(Allowed).encode()).hexdigest()\n'
            '        allowed = runtime.wrap_allowed(allowed, report)')
    replace('        report["assigned_users"] = len(assignment)\n'
            '        report["assignment_sha256"] = hashlib.sha256(json.dumps(assignment, separators=(",", ":")).encode()).hexdigest()\n'
            '        if args.preflight_users_per_rank:\n'
            '            assignment = runtime.preflight_assignment(rank)\n'
            '            report["preflight_assignment"] = assignment',
            '        assignment = runtime.check_assignment(assignment, rank, args.preflight_users_per_rank, report)')
    replace('        report["status"] = "PREDICTIONS_COMPLETE"\n',
            '        runtime.finish(report)\n        report["status"] = "PREDICTIONS_COMPLETE"\n')
    start = derived.index('        if rank == 0 and not args.preflight_users_per_rank:')
    stop = derived.index('        report["status"] = "PREFLIGHT_INTERFACE_PASS"', start)
    # The formal CPU stage reads the only prediction shard after runner cleanup.
    return derived[:start] + derived[stop:]


class A1Runtime(Runtime):
    def __init__(self, options, config):
        super().__init__(options, config)
        self.single_shard = json.loads(options.single_shard_audit.read_text())
        assert self.single_shard['status'] == 'PASS_SINGLE_SHARD_CPU'
        assert self.single_shard['manifest_sha256'] == self.digest
        assert self.audit['users'] == config['users'] and len(self.sid_ids) == config['sid_tokens']
        assert self.audit['user_axis_sha256'] == config['user_axis_sha256']
        assert self.audit['tokenizer_vocab_sha256'] == config['vocab_sha256']
        self.baseline = config['baseline']
        self.generation_calls = 0

    def check_assignment(self, assignment, rank, preflight, report):
        single = self.single_shard
        assert rank == 0 and assignment == single['assignment']
        assert core.canonical_sha256(assignment) == single['assignment_sha256']
        report['formal_assignment_sha256'] = single['assignment_sha256']
        if preflight:
            assignment = self.preflight_assignment(rank)
            assert len(assignment) == len(set(assignment)) == preflight == 32
            assert core.canonical_sha256(assignment) == single['preflight_assignment_sha256']
            report['preflight_assignment'] = assignment
        report['assigned_users'] = len(assignment)
        report['assignment_sha256'] = core.canonical_sha256(assignment)
        return assignment

    def wrap_allowed(self, original, report):
        assert report['allowed_callback_sha256'] == self.baseline['allowed_callback_sha256']
        assert [len(original.allowed[i]) for i in range(5)] == [150, 256, 256, 256, 1]
        runtime = self

        class Callback:
            def bind(self, prompt_ids, text, record):
                actual, _, orig, evidence, expected_record = runtime.context
                assert record is expected_record and prompt_ids == actual
                old = orig[0].tolist()
                original.bind(old, text)
                group_slots = {p for group in evidence['groups'] for p in group['positions']}
                assert all(a == b or i in group_slots for i, (a, b) in enumerate(zip(actual, old, strict=True)))
                assert actual[-len(original.sep):] == original.sep
                assert group_slots.isdisjoint(range(len(actual) - len(original.sep), len(actual)))
                original.prefix = list(actual)
                runtime.callback = self
                self.call_start = dict(original.calls)
                record['callback'] = {'original_validation': True, 'bound_prefix_sha256': core.canonical_sha256(actual),
                                      'separator_ids': original.sep, 'allowed_counts': [len(original.allowed[i]) for i in range(5)]}

            def generate_calls(self):
                return [original.calls.get(i, 0) - self.call_start.get(i, 0) for i in range(5)]

            def __call__(self, batch_id, sentence):
                return original(batch_id, sentence)

        return Callback()

    def loaded(self, model, tokenizer, args, report):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model

        normal = self.baseline
        assert vocab_hash(tokenizer.get_vocab()) == self.audit['tokenizer_vocab_sha256']
        assert all(tokenizer.encode(s, add_special_tokens=False) == [i] for s, i in self.audit['token_ids'].items())
        assert (len(tokenizer), tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id,
                tokenizer.padding_side) == (152587, None, 151643, 151643, 'left')
        assert report['tokenizer']['encodes_automatic_bos'] is False
        for key, expected in normal['rank_identity'].items():
            assert report[key] == expected, key
        assert core.canonical_sha256(report['strict_adapter_reload']) == normal['strict_reload_sha256']
        assert core.canonical_sha256(report['tokenizer']) == normal['tokenizer_receipt_sha256']
        assert torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0
        base = model.get_base_model()
        assert type(base) is Qwen3ForCausalLM and type(base.model) is Qwen3Model
        assert base.config._attn_implementation == 'sdpa'
        assert not getattr(base, 'is_loaded_in_4bit', False) and not getattr(base, 'is_loaded_in_8bit', False)
        emb, head = model.get_input_embeddings(), model.get_output_embeddings()
        assert emb.weight is head.weight and emb.weight.data_ptr() == head.weight.data_ptr()
        assert emb.weight.shape == (152587, 1024) and emb.weight.dtype == torch.bfloat16
        assert report['strict_adapter_reload']['keys'] == 394 and report['strict_adapter_reload']['lora_keys'] == 392
        assert all(p.dtype == torch.bfloat16 and p.device == torch.device('cuda', 0) for p in model.parameters())
        assert not model.training and not base.training and not base.model.training
        self.model, self.embedding, self.head = model, emb, head
        self.preflight = bool(args.preflight_users_per_rank)
        self.weight_versions = {name: p._version for name, p in model.named_parameters()}
        self.weight_hash = self.weight_digest() if self.preflight else None
        report['shuffle'] = {**core.mode_metadata(self.options.shuffle_mode), 'cpu_manifest_sha256': self.digest,
                             'actual_model_module': type(base.model).__name__, 'sdpa': True, 'quantized': False,
                             'loaded_generation_defaults': model.generation_config.to_dict(),
                             'active_parameter_count': len(self.weight_versions), 'parameter_sha256_before': self.weight_hash}
        native_generate = model.generate

        def checked_generate(**kwargs):
            actual, mask, original, evidence, record = self.context
            ids, mask_tensor = kwargs['input_ids'], kwargs['attention_mask']
            assert ids.tolist() == [actual] and mask_tensor.tolist() == [mask]
            assert kwargs['prefix_allowed_tokens_fn'] is self.callback
            assert set(kwargs).isdisjoint(('position_ids', 'past_key_values', 'cache_position', 'inputs_embeds'))
            expected_scalars = {'num_beams': 20, 'num_return_sequences': 20, 'max_new_tokens': 10,
                                'early_stopping': True, 'do_sample': False, 'use_cache': True,
                                'output_scores': True, 'return_dict_in_generate': True,
                                'eos_token_id': 151643, 'pad_token_id': 151643}
            actual_scalars = {k: v for k, v in kwargs.items() if isinstance(v, (int, bool, float, str))}
            assert actual_scalars == expected_scalars
            effective = deepcopy(model.generation_config)
            assert not effective.update(**actual_scalars)
            if self.generation_calls == 0:
                report['shuffle'].update(effective_generation_config=effective.to_dict(),
                                         generation_scalar_overrides=actual_scalars)
            assert not model.training and not torch.is_grad_enabled()
            record.update(eval_mode=True, grad_enabled=False)
            positions = mask_tensor.long().cumsum(-1) - 1
            positions.masked_fill_(mask_tensor == 0, 1)
            cache_positions = torch.arange(ids.shape[1], device=ids.device)
            boundary = {}

            def validate_prefill(module, positional, named):
                if boundary:
                    return
                beam_count = kwargs['num_beams']
                seen, seen_mask, seen_positions = (named[k] for k in ('input_ids', 'attention_mask', 'position_ids'))
                for observed, expected in ((seen, ids), (seen_mask, mask_tensor), (seen_positions, positions)):
                    assert observed.shape == (beam_count, ids.shape[1]) and torch.equal(observed, expected.expand_as(observed))
                seen_cache_positions = named['cache_position']
                assert torch.equal(seen_cache_positions, cache_positions) and named['use_cache'] is True
                cache = named['past_key_values']
                length = 0 if cache is None else int(cache.get_seq_length())
                layer_lengths = [] if cache is None else [int(cache.get_seq_length(i)) for i in range(len(cache))]
                assert length == 0 and all(n == 0 for n in layer_lengths)
                boundary.update(input_sha256=core.canonical_sha256(seen[0].tolist()),
                                attention_mask_sha256=core.canonical_sha256(seen_mask[0].tolist()),
                                position_ids_sha256=core.canonical_sha256(seen_positions[0].tolist()),
                                cache_length=length, beam_expansion=beam_count,
                                all_beams_ids_mask_positions_exact=True)
                if self.preflight:
                    boundary.update(input_ids=seen[0].tolist(), attention_mask=seen_mask[0].tolist(),
                                    position_ids=seen_positions[0].tolist(), cache_position=seen_cache_positions.tolist(),
                                    initial_cache_type=type(cache).__name__, initial_cache_layer_lengths=layer_lengths)

            if self.preflight:
                old_lookup = torch.nn.functional.embedding(original.to(ids.device), emb.weight)
                expected_lookup = old_lookup.clone()
                for group in evidence['groups']:
                    slots = group['positions']
                    expected_lookup[:, slots] = old_lookup[:, slots][:, group['permutation']]

                def validate_embedding(module, positional, output):
                    if 'lookup_exact' in boundary:
                        return
                    assert boundary and torch.equal(positional[0], ids.expand_as(positional[0]))
                    assert output.dtype == torch.bfloat16 and torch.equal(output, expected_lookup.expand_as(output))
                    boundary.update(lookup_exact=True, lookup_shape=list(output.shape), lookup_dtype=str(output.dtype),
                                    lookup_sha256=self.tensor_digest(output[0]),
                                    expected_lookup_sha256=self.tensor_digest(expected_lookup[0]))
                    assert boundary['lookup_sha256'] == boundary['expected_lookup_sha256']

                lookup_handle = emb.register_forward_hook(validate_embedding)
            handle = base.model.register_forward_pre_hook(validate_prefill, with_kwargs=True)
            try:
                output = native_generate(**kwargs)
            finally:
                handle.remove()
                if self.preflight:
                    lookup_handle.remove()
            assert boundary['input_sha256'] == evidence['shuffled_sha256']
            assert boundary['attention_mask_sha256'] == evidence['attention_mask_sha256']
            if self.preflight:
                assert boundary['lookup_exact']
                record.update(groups=evidence['groups'], original_input_ids=original[0].tolist())
            calls = self.callback.generate_calls()
            assert all(n > 0 for n in calls)
            record['callback']['generation_position_calls'] = calls
            assert output.sequences_scores.dtype == torch.float32
            record.update(actual_prefill=boundary, sequence_score_dtype=str(output.sequences_scores.dtype))
            self.generation_calls += 1
            return output

        model.generate = checked_generate

    @staticmethod
    def tensor_digest(tensor):
        import torch
        values = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        return hashlib.sha256(memoryview(values)).hexdigest()

    def weight_digest(self):
        return core.canonical_sha256({name: self.tensor_digest(p) for name, p in self.model.named_parameters()})

    def finish(self, report):
        assert self.generation_calls == report['completed_users']
        assert self.weight_versions == {name: p._version for name, p in self.model.named_parameters()}
        assert self.embedding.weight is self.head.weight and self.embedding.weight.data_ptr() == self.head.weight.data_ptr()
        after = self.weight_digest() if self.preflight else None
        assert after == self.weight_hash
        report['shuffle'].update(parameter_sha256_after=after, parameter_versions_unchanged=True,
                                 generate_calls=self.generation_calls, end_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cpu-manifest', type=Path, required=True)
    parser.add_argument('--mode', choices=core.MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-commit', required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--derived-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rel, expected = FROZEN['a1']
    source = Path(__file__).resolve().parents[1] / rel
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    common = source.parents[2] / 'qwen_preflight_20261110/scripts/common.py'
    assert hashlib.sha256(common.read_bytes()).hexdigest() == config['common_sha256']
    original = raw.decode('utf-8')
    derived = derive(original)
    code = compile(derived, str(source), 'exec')
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    assert int(os.environ.get('RANK', 0)) == 0
    with (args.output / 'derived.rank0.py').open('x') as stream:
        stream.write(derived)
    with (args.output / 'derived.rank0.diff').open('x') as stream:
        stream.writelines(difflib.unified_diff(original.splitlines(keepends=True), derived.splitlines(keepends=True),
                                              fromfile=rel, tofile='derived.rank0.py'))
    if args.derived_only:
        return 0
    assert os.environ['WORLD_SIZE'] == '1' and os.environ['LOCAL_RANK'] == '0'
    for key, value in config['gpu_environment'].items():
        assert os.environ.get(key) == value, key
    args.model_key, args.shuffle_mode = 'a1', args.mode
    args.cpu_audit = args.cpu_manifest / 'audit.json'
    args.single_shard_audit = args.cpu_manifest / 'single-shard.json'
    runtime = A1Runtime(args, config)
    namespace = {'__file__': str(source), '__name__': 'a1_v2_frozen_entry', 'runtime': runtime}
    exec(code, namespace)
    sys.argv = [str(source), '--output', str(args.output), '--run-commit', args.run_commit,
                '--max-seconds', str(args.max_seconds)]
    for key in ('source-root', 'data-root', 'model', 'adapter'):
        sys.argv.extend(('--' + key, config['baseline'][key]))
    if args.preflight:
        sys.argv.extend(('--preflight-users-per-rank', '32'))
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
