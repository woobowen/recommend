"""Bind the V2 CPU oracle to the frozen LETTER generation boundary."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from lc_rec.shuffle_v2 import core
from lc_rec.shuffle_v2.replay import CONFIG, frozen, semantic_hash


class Runtime:
    def __init__(self, options, config=None):
        assert sys.flags.optimize == 0
        self.options = options
        self.config = config
        self.assignment_hash = core.canonical_sha256
        self.audit = json.loads(options.cpu_audit.read_text())
        assert self.audit['status'] == 'PASS_CPU_V2_FULL' and self.audit['protocol'] == core.PROTOCOL
        self.digest = self.audit['manifest_sha256']
        manifest = options.cpu_audit.with_name('inputs.jsonl')
        assert frozen.sha(manifest) == self.digest
        with manifest.open() as stream:
            self.rows = [json.loads(line) for line in stream]
        assert len(self.rows) == self.audit['users']
        assert all(row['row_idx'] == i for i, row in enumerate(self.rows))
        self.sid_ids = set(self.audit['token_ids'].values())
        self.single_shard = None
        if options.model_key in ('a0', 'beauty_letter'):
            self.single_shard = json.loads(options.single_shard_audit.read_text())
            assert self.single_shard['status'] == 'PASS_SINGLE_SHARD_CPU'
            assert self.single_shard['manifest_sha256'] == self.digest

    def preflight_assignment(self, rank):
        if self.single_shard is not None:
            assert rank == 0
            return self.single_shard['preflight_assignment']
        return self.audit['preflight_assignments'][str(rank)]

    def intervene(self, inputs, index, data, tokenizer):
        import torch
        original_tensor = inputs['input_ids']
        original = original_tensor[0].tolist()
        mask = inputs['attention_mask'][0].tolist()
        replay = self.rows[index]
        assert list(data.inters)[index] == replay['user_id']
        seq = data.inters[replay['user_id']]
        groups = [[self.audit['token_ids'][token] for token in data.indices[str(i)]] for i in seq[:-1][-20:]]
        assert ''.join(data.indices[str(seq[-1])]) == replay['target']
        mode = self.options.shuffle_mode
        actual, evidence = core.transform(original, mask, groups, self.sid_ids, mode, index)
        expected = replay['conditions'][mode]
        assert (evidence['shuffled_sha256'], evidence['input_changed']) == (expected['input_sha256'], expected['input_changed'])
        assert [g['permutation_id'] for g in evidence['groups']] == expected['permutation_ids']
        assert evidence['original_sha256'] == replay['original_sha256']
        assert evidence['attention_mask_sha256'] == replay['attention_mask_sha256']
        inputs['input_ids'] = torch.tensor([actual], dtype=original_tensor.dtype, device=original_tensor.device)
        record = {**core.mode_metadata(mode), 'row_idx': index, 'input_changed': evidence['input_changed'],
                  'original_sha256': evidence['original_sha256'], 'shuffled_sha256': evidence['shuffled_sha256'],
                  'attention_mask_sha256': evidence['attention_mask_sha256'], 'cpu_manifest_sha256': self.digest,
                  'history_group_count': evidence['history_group_count']}
        self.context = (actual, mask, original_tensor, evidence, record)
        return inputs, record

    def allowed(self, tokenizer, indices):
        from common import Allowed
        original = Allowed(tokenizer, indices)

        class Callback:
            def bind(_, prompt_ids, text, record):
                actual, _, old, _, checked = self.context
                assert record == checked and prompt_ids == actual
                original.bind(old[0].tolist(), text)
                assert actual[-len(original.sep):] == original.sep
                original.prefix = actual

            def __call__(_, batch_id, sentence):
                return original(batch_id, sentence)

        return Callback()

    def loaded(self, model, tokenizer, args, report):
        import torch
        assert semantic_hash(tokenizer.get_vocab()) == self.audit['tokenizer_vocab_sha256']
        assert all(tokenizer.encode(symbol, add_special_tokens=False) == [i] for symbol, i in self.audit['token_ids'].items())
        if self.options.model_key == 'a0':
            baseline = json.loads(CONFIG.read_text())['baseline']
            assert report['adapter_files']['adapter_model.bin'] == baseline['adapter_identity']
            assert report['versions'] == {k: self.audit['versions'][k] for k in ('torch','transformers','peft','accelerate')}
        if self.options.model_key == 'beauty_letter':
            baseline = self.config['baseline']
            assert torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0
            assert self.audit['dataset'] == self.config['dataset'] and self.audit['users'] == self.config['users']
            assert report['adapter_files'] == baseline['adapter_files']
            assert report['model'] == baseline['model_identity'] and report['versions'] == baseline['versions']
            assert report['author_sources'] == baseline['author_sources']
            assert report['tokenizer']['vocab_size'] == 32965 and len(report['tokenizer']['added_tokens']) == 965
            assert report['tokenizer']['added_tokens'] == self.audit['token_ids']
            assert report['strict_adapter_reload']['keys'] == 450 and report['strict_adapter_reload']['elements'] == 290037760
            weights = model.get_base_model()
            assert not getattr(weights, 'is_loaded_in_4bit', False) and not getattr(weights, 'is_loaded_in_8bit', False)
            # PEFT 0.5 marks modules_to_save trainable even on its inference load path.
            model.requires_grad_(False)
            assert all(parameter.device.type == 'cuda' and parameter.device.index == 0 and not parameter.requires_grad
                       for parameter in model.parameters())
            assert weights.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
            historical_rank = json.loads((Path(baseline['normal_test_dir']) / 'rank0.json').read_text())
            assert report['strict_adapter_reload'] == historical_rank['strict_adapter_reload']
            report['beauty_normal_binding'] = baseline['run-commit']
        report['shuffle'] = {**core.mode_metadata(self.options.shuffle_mode), 'cpu_manifest_sha256': self.digest,
                             'mode_installation': 'frozen TestCollator then V2 transform then native generate',
                             'loaded_generation_defaults': model.generation_config.to_dict(),
                             'actual_model_module': type(model.get_base_model().model).__name__,
                             'capture_lookup': bool(args.preflight_users_per_rank)}
        assert model.training is False
        preflight = bool(args.preflight_users_per_rank)
        generate = model.generate
        embedding = model.get_input_embeddings()
        base_decoder = model.get_base_model().model

        def checked_generate(**kwargs):
            actual, mask, original, evidence, record = self.context
            ids = kwargs['input_ids']
            mask_tensor = kwargs['attention_mask']
            assert ids.tolist() == [actual] and mask_tensor.tolist() == [mask]
            assert kwargs['num_beams'] == kwargs['num_return_sequences'] == 20 and kwargs['max_new_tokens'] == 10
            assert all(k not in kwargs for k in ('position_ids', 'past_key_values', 'cache_position', 'inputs_embeds'))
            if self.options.model_key == 'beauty_letter':
                assert report['allowed_callback_sha256'] == self.config['baseline']['allowed_callback_sha256']
                assert callable(kwargs.get('prefix_allowed_tokens_fn'))
            if 'effective_generation_config' not in report['shuffle']:
                overrides = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
                effective = deepcopy(model.generation_config)
                assert not effective.update(**overrides)
                assert effective.do_sample is False and effective.num_beams == effective.num_return_sequences == 20
                if self.options.model_key == 'beauty_letter':
                    assert getattr(effective, 'use_cache', True) is True
                    assert (effective.bos_token_id, effective.eos_token_id, effective.pad_token_id) == (1, 2, 0)
                    assert effective.early_stopping is True and effective.max_new_tokens == 10
                report['shuffle']['effective_generation_config'] = effective.to_dict()
                report['shuffle']['generation_scalar_overrides'] = overrides
            expected_pos = mask_tensor.long().cumsum(-1) - 1
            expected_pos.masked_fill_(mask_tensor == 0, 1)
            boundary = {}

            def before_forward(module, positional, named):
                if boundary:
                    return
                seen = named.get('input_ids', positional[0] if positional else None)
                assert seen is not None and seen.shape[0] == kwargs['num_beams']
                seen_mask = named.get('attention_mask')
                positions = named.get('position_ids')
                assert torch.equal(seen, ids.expand_as(seen))
                assert seen_mask is not None and torch.equal(seen_mask, mask_tensor.expand_as(seen))
                assert positions is not None and torch.equal(positions, expected_pos.expand_as(positions))
                past = named.get('past_key_values')
                if past is None:
                    length = 0
                elif hasattr(past, 'get_seq_length'):
                    length = int(past.get_seq_length())
                else:
                    length = int(past[0][0].shape[-2]) if len(past) else 0
                assert length == 0
                boundary.update(input_sha256=core.canonical_sha256(seen[0].tolist()),
                                attention_mask_sha256=core.canonical_sha256(seen_mask[0].tolist()),
                                position_ids_sha256=core.canonical_sha256(positions[0].tolist()),
                                cache_length=length, beam_expansion=seen.shape[0])
                if preflight:
                    boundary.update(input_ids=seen[0].tolist(), attention_mask=seen_mask[0].tolist(),
                                    position_ids=positions[0].tolist())

            lookup_handle = None
            if preflight:
                old_lookup = embedding(original.to(ids.device))
                expected_lookup = old_lookup.clone()
                for group in evidence['groups']:
                    slots = group['positions']
                    expected_lookup[:, slots] = old_lookup[:, slots][:, group['permutation']]

                def verify_lookup(module, positional, output):
                    if 'lookup_exact' not in boundary:
                        assert boundary and torch.equal(output, expected_lookup.expand_as(output))
                        boundary['lookup_exact'] = True

                lookup_handle = embedding.register_forward_hook(verify_lookup)
            handle = base_decoder.register_forward_pre_hook(before_forward, with_kwargs=True)
            try:
                output = generate(**kwargs)
            finally:
                handle.remove()
                if lookup_handle is not None:
                    lookup_handle.remove()
            assert boundary.get('input_sha256') == evidence['shuffled_sha256']
            assert boundary['attention_mask_sha256'] == evidence['attention_mask_sha256']
            if preflight:
                assert boundary['lookup_exact']
                record['groups'] = evidence['groups']
                record['original_input_ids'] = original[0].tolist()
            record['actual_prefill'] = boundary
            return output

        model.generate = checked_generate
